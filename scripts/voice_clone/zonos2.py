"""
Fast local ZONOS2 runner for Apple Silicon / MLX.

Common use:
  python scripts/voice_clone/zonos2.py "Text to speak"
  python scripts/voice_clone/zonos2.py "Text to speak" --ref-audio speaker.wav

We use the local models/zonos2-mixed-q6-gate-up checkpoint (can be generated with
ZONOS2_CREATE_QUANTIZED=1), estimate the number of audio tokens needed from the
input text, run the compiled MLX decode step, decode with the local DAC cache,
and write zonos2.wav by default.

Most useful knobs:
  ZONOS2_OUTPUT=path.wav                Output file, or pass --output.
  ZONOS2_MODEL=path-or-hf-id            Override model; defaults to models/zonos2-mixed-q6-gate-up.
  ZONOS2_REF_AUDIO=path.wav             Reference voice audio, or pass --ref-audio.
  ZONOS2_MAX_TOKENS=N                   Override automatic token budgeting.
  ZONOS2_TOKEN_SECONDS_BUFFER=0.5       Extra seconds added to the estimate.
  ZONOS2_TOKEN_SAFETY_MULTIPLIER=1.2    Multiplier added to the estimate.
  ZONOS2_WORDS_PER_MINUTE=150           Speaking-rate assumption for auto tokens.
  ZONOS2_CHARS_PER_SECOND=24            Character-rate assumption for auto tokens.
  ZONOS2_MIN_TOKENS=64                  Lower bound for auto tokens.
  ZONOS2_MAX_TOKENS_CAP=2048            Upper bound for auto tokens.
  ZONOS2_TEMPERATURE=1.15               Sampling temperature.
  ZONOS2_SEED=0                         Deterministic sampling seed; empty disables.
  ZONOS2_PROFILE=1                      Print prompt/sample/decode timing.

Checkpoint creation:
  ZONOS2_CREATE_QUANTIZED=1 python scripts/voice_clone/zonos2.py

Advanced/debug knobs:
  ZONOS2_CREATE_ONLY=0                  Continue into generation after quantizing.
  ZONOS2_OVERWRITE_QUANTIZED=1          Recreate the quantized checkpoint.
  ZONOS2_QUANTIZE_GROUP_SIZE=64         Quantization group size.
  ZONOS2_QUANTIZE_SKIP_PATTERNS=...     Comma-separated module-name skips.
  ZONOS2_CLEAR_CACHE_AFTER_GENERATE=0   Keep MLX cache after a generation.
  ZONOS2_USE_LOCAL_DAC_CACHE=0          Let DAC loader call snapshot_download.
  ZONOS2_LAZY_LOAD=1                    Lazy-load model parameters; slower here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from math import ceil
from pathlib import Path
from typing import Any, Callable

SOURCE_MODEL = "mlx-community/Zyphra-ZONOS2"
DEFAULT_QUANTIZED_MODEL = Path("models/zonos2-mixed-q6-gate-up")
DEFAULT_DAC_REPO = "mlx-community/descript-audio-codec-44khz"
DEFAULT_QUANTIZE_SKIP_PATTERNS = ("speaker_",)
BASE_QUANT_BITS = 8
MIXED_QUANT_BITS = 6
MIXED_Q6_PATTERNS = (
    ".feed_forward.experts.gate_proj",
    ".feed_forward.experts.up_proj",
)

mx: Any = None
audio_write: Callable[..., Any] | None = None
load: Callable[..., Any] | None = None


def load_runtime() -> None:
    global audio_write, load, mx
    if mx is not None:
        return
    import mlx.core as mlx_core
    from mlx_audio.audio_io import write as mlx_audio_write
    from mlx_audio.tts import load as mlx_tts_load

    mx = mlx_core
    audio_write = mlx_audio_write
    load = mlx_tts_load


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return None if raw == "" else int(raw)


def env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast local ZONOS2 MLX generation with optional voice cloning."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize. Falls back to ZONOS2_TEXT or a sample sentence.",
    )
    parser.add_argument(
        "--ref-audio",
        "--speaker",
        dest="ref_audio",
        help="Optional reference audio path for voice cloning.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        help="Output WAV path. Falls back to ZONOS2_OUTPUT or zonos2.wav.",
    )
    parser.add_argument(
        "--create-quantized",
        action="store_true",
        help="Create the local mixed q6/q8 checkpoint before running.",
    )
    return parser.parse_args()


def estimate_max_tokens(text: str, prompt_tokens: int, model) -> int:
    if "ZONOS2_MAX_TOKENS" in os.environ:
        return env_int("ZONOS2_MAX_TOKENS", 220)

    words = re.findall(r"\w+(?:[-']\w+)?", text)
    word_count = max(1, len(words))
    non_space_chars = max(1, len(re.sub(r"\s+", "", text)))
    punctuation_pauses = len(re.findall(r"[.!?;:]", text)) * env_float(
        "ZONOS2_PUNCTUATION_PAUSE_SECONDS", 0.08
    )
    word_seconds = word_count / env_float("ZONOS2_WORDS_PER_MINUTE", 150.0) * 60.0
    char_seconds = non_space_chars / env_float("ZONOS2_CHARS_PER_SECOND", 24.0)
    estimated_seconds = (
        max(word_seconds, char_seconds)
        + punctuation_pauses
        + env_float("ZONOS2_TOKEN_SECONDS_BUFFER", 0.5)
    )
    estimated_seconds *= env_float("ZONOS2_TOKEN_SAFETY_MULTIPLIER", 1.2)

    # ZONOS2's DAC decode trims each generated code frame to 512 samples.
    tokens_per_second = model.sample_rate / 512
    estimated_tokens = ceil(estimated_seconds * tokens_per_second)

    min_tokens = env_int("ZONOS2_MIN_TOKENS", 64)
    configured_cap = env_int("ZONOS2_MAX_TOKENS_CAP", 2048)
    sequence_cap = max(1, int(model.config.max_seqlen) - int(prompt_tokens) - 16)
    max_tokens = max(min_tokens, min(configured_cap, sequence_cap))
    return max(min_tokens, min(estimated_tokens, max_tokens))


def create_quantized_checkpoint(
    source_model: str,
    output_dir: Path,
    *,
    group_size: int,
    mode: str,
    overwrite: bool,
) -> None:
    load_runtime()
    from mlx_audio.utils import get_model_path, load_config
    from mlx_lm.utils import quantize_model, save_config, save_model

    if output_dir.exists():
        if not overwrite:
            print(f"Quantized checkpoint already exists: {output_dir}", flush=True)
            return
        shutil.rmtree(output_dir)

    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Loading source model for quantization: {source_model}", flush=True)
    source_path = get_model_path(source_model)
    config = load_config(source_path)
    assert load is not None
    model = load(source_model, lazy=True)

    model_quant_predicate = getattr(
        model,
        "model_quant_predicate",
        lambda _p, _m: True,
    )
    skip_patterns = env_csv(
        "ZONOS2_QUANTIZE_SKIP_PATTERNS", DEFAULT_QUANTIZE_SKIP_PATTERNS
    )
    base_params = {"group_size": group_size, "bits": BASE_QUANT_BITS, "mode": mode}
    mixed_params = {"group_size": group_size, "bits": MIXED_QUANT_BITS, "mode": mode}

    def quant_predicate(path, module):
        if any(pattern in path for pattern in skip_patterns):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if not hasattr(module, "weight"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        if not model_quant_predicate(path, module):
            return False
        if any(pattern in path for pattern in MIXED_Q6_PATTERNS):
            return mixed_params
        return base_params

    print(
        "Quantizing ZONOS2 with q6 MoE gate/up projections and q8 elsewhere "
        f"({mode}, group size {group_size})...",
        flush=True,
    )
    model, config = quantize_model(
        model,
        config,
        group_size,
        BASE_QUANT_BITS,
        mode=mode,
        quant_predicate=quant_predicate,
    )

    tmp_dir.mkdir(parents=True, exist_ok=False)
    save_model(tmp_dir, model, donate_model=True)
    save_config(config, config_path=tmp_dir / "config.json")
    (tmp_dir / "README.md").write_text(
        "# ZONOS2 quantized MLX\n\n"
        f"Local quantized checkpoint converted from `{source_model}`.\n"
        "Quantization: q6 MoE expert gate/up projections, q8 elsewhere, "
        f"`{mode}` mode, group size {group_size}.\n",
        encoding="utf-8",
    )
    metadata = {
        "source_model": source_model,
        "base_bits": BASE_QUANT_BITS,
        "mixed_bits": MIXED_QUANT_BITS,
        "group_size": group_size,
        "mode": mode,
        "recipe": "mixed-q6-gate-up",
        "skip_patterns": skip_patterns,
        "mixed_patterns": MIXED_Q6_PATTERNS,
        "created_by": "scripts/voice_clone/zonos2.py",
    }
    (tmp_dir / "quantization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_dir.rename(output_dir)
    print(f"Wrote quantized checkpoint: {output_dir}", flush=True)


def install_dac_cache_shortcut() -> None:
    if not env_bool("ZONOS2_USE_LOCAL_DAC_CACHE", True):
        return
    from mlx_audio.codec.models.descript import dac as dac_module

    explicit_path = os.environ.get("ZONOS2_DAC_PATH")
    if explicit_path:
        dac_path = Path(explicit_path).expanduser()
    else:
        cache_root = (
            Path.home()
            / ".cache/huggingface/hub/models--mlx-community--descript-audio-codec-44khz/snapshots"
        )
        snapshots = sorted(cache_root.glob("*"), key=lambda p: p.stat().st_mtime)
        dac_path = snapshots[-1] if snapshots else None

    if not dac_path or not dac_path.exists():
        return

    original_fetch = dac_module.fetch_from_hub

    def fetch_from_local_cache(repo_id: str) -> Path:
        if repo_id == DEFAULT_DAC_REPO:
            return dac_path
        return original_fetch(repo_id)

    dac_module.fetch_from_hub = fetch_from_local_cache


def generate_sampled(
    model,
    *,
    text: str,
    max_tokens: int | None,
    ref_audio: str | None,
    text_normalization: bool,
):
    install_dac_cache_shortcut()
    if max_tokens is None:
        max_tokens = estimate_max_tokens(text, 64, model)
        print(f"Auto max tokens: {max_tokens}", flush=True)

    start = time.perf_counter()
    result = generate_sampled_compiled(
        model,
        text=text,
        ref_audio=ref_audio,
        max_tokens=max_tokens,
        temperature=env_float("ZONOS2_TEMPERATURE", 1.15),
        top_k=env_int("ZONOS2_TOP_K", 106),
        top_p=env_float("ZONOS2_TOP_P", 0.0),
        min_p=env_float("ZONOS2_MIN_P", 0.18),
        repetition_window=env_int("ZONOS2_REPETITION_WINDOW", 50),
        repetition_penalty=env_float("ZONOS2_REPETITION_PENALTY", 1.2),
        repetition_codebooks=env_int("ZONOS2_REPETITION_CODEBOOKS", 8),
        seed=env_optional_int("ZONOS2_SEED", 0),
        ignore_eos=env_bool("ZONOS2_IGNORE_EOS", False),
        text_normalization=text_normalization,
        clean_speaker_background=env_bool("ZONOS2_CLEAN_SPEAKER_BACKGROUND", False),
        accurate_mode=env_bool("ZONOS2_ACCURATE_MODE", True),
    )
    elapsed = time.perf_counter() - start
    if env_bool("ZONOS2_CLEAR_CACHE_AFTER_GENERATE", True):
        mx.clear_cache()
    result.setdefault("processing_time_seconds", elapsed)
    return result


def _frame_from_ids(ids, text_vocab: int):
    text_id = mx.array([int(text_vocab)], dtype=mx.int32)
    return mx.concatenate([ids.astype(mx.int32), text_id], axis=0)


class ExplicitKVCache:
    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset

    def make_mask(self, N: int, return_array: bool = False, window_size=None):
        del return_array
        capacity = self.keys.shape[2]
        queries = self.offset + mx.arange(N)
        keys = mx.arange(capacity)
        mask = keys[None, :] <= queries[:, None]
        if window_size is not None:
            mask = mask & (keys[None, :] > (queries[:, None] - int(window_size)))
        return mask[None, None, :, :]

    def update_and_fetch(self, keys, values):
        update_size = keys.shape[2]
        start = mx.stack(
            [
                mx.array(0, dtype=mx.int32),
                mx.array(0, dtype=mx.int32),
                self.offset.astype(mx.int32),
                mx.array(0, dtype=mx.int32),
            ]
        )
        self.keys = mx.slice_update(self.keys, keys, start, axes=(0, 1, 2, 3))
        self.values = mx.slice_update(self.values, values, start, axes=(0, 1, 2, 3))
        self.offset = self.offset + update_size
        return self.keys, self.values


def _expand_cache_array(array, filled: int, capacity: int):
    array = array[..., :filled, :]
    if filled >= capacity:
        return array
    pad_shape = list(array.shape)
    pad_shape[2] = capacity - filled
    return mx.concatenate(
        [array, mx.zeros(tuple(pad_shape), dtype=array.dtype)], axis=2
    )


def _make_explicit_cache_state(cache, max_tokens: int):
    prompt_tokens = int(cache[0].offset)
    capacity = prompt_tokens + int(max_tokens) + 1
    return [
        (
            _expand_cache_array(layer_cache.keys, int(layer_cache.offset), capacity),
            _expand_cache_array(layer_cache.values, int(layer_cache.offset), capacity),
            mx.array(int(layer_cache.offset), dtype=mx.int32),
        )
        for layer_cache in cache
    ]


def _caches_from_explicit_state(cache_state):
    return [
        ExplicitKVCache(keys, values, offset) for keys, values, offset in cache_state
    ]


def _state_from_explicit_caches(cache):
    return [(layer.keys, layer.values, layer.offset) for layer in cache]


def _decode_audio_timed(model, delayed_rows: list[list[int]] | list[Any] | Any, eos):
    start = time.perf_counter()
    audio = model._decode_audio(delayed_rows, eos)
    mx.eval(audio)
    return audio, time.perf_counter() - start


def generate_sampled_compiled(
    model,
    *,
    text: str,
    ref_audio: str | None,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_window: int,
    repetition_penalty: float,
    repetition_codebooks: int,
    seed: int | None,
    ignore_eos: bool,
    text_normalization: bool,
    clean_speaker_background: bool,
    accurate_mode: bool,
):
    from mlx_audio.tts.models.zonos2.generation import (
        TTSSamplingParams,
        Zonos2GenerationState,
        format_duration,
        sample_frame_ids,
    )
    from mlx_lm.models.cache import make_prompt_cache

    profile = env_bool("ZONOS2_PROFILE", False)
    start = time.perf_counter()
    normalized_text = model._normalize_text(
        text,
        language=os.environ.get("ZONOS2_LANG_CODE", "en_us"),
        text_normalization=text_normalization,
    )
    speaker_start = time.perf_counter()
    speaker_emb = model._resolve_speaker_embedding(
        speaker_embedding=None,
        ref_audio=ref_audio,
        ref_audio_sample_rate=None,
    )
    speaker_seconds = time.perf_counter() - speaker_start
    prompt, speaker_pos = model._build_prompt(
        normalized_text,
        speaking_rate_bucket=None,
        quality_buckets=None,
        speaker_embedding=speaker_emb,
        clean_speaker_background=clean_speaker_background,
        accurate_mode=accurate_mode,
    )
    mx.eval(prompt)

    cache = make_prompt_cache(model)
    prompt_start = time.perf_counter()
    logits = model(
        prompt,
        cache=cache,
        speaker_embedding=speaker_emb,
        speaker_token_position=speaker_pos,
    )
    last_logits = logits[0, -1]
    mx.eval(last_logits)
    prompt_seconds = time.perf_counter() - prompt_start

    params = TTSSamplingParams(
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        max_tokens=int(max_tokens),
        ignore_eos=bool(ignore_eos),
        repetition_window=int(repetition_window),
        repetition_penalty=float(repetition_penalty),
        repetition_codebooks=int(repetition_codebooks),
        seed=seed,
    )
    state = Zonos2GenerationState(
        n_codebooks=model.config.n_codebooks,
        eoa_id=model.config.eoa_id,
        text_vocab=int(model.config.text_vocab),
    )
    sample_start = time.perf_counter()
    explicit_cache_state = _make_explicit_cache_state(cache, int(max_tokens))

    def decode_step(next_ids, cache_state):
        explicit_cache = _caches_from_explicit_state(cache_state)
        logits = model(next_ids, cache=explicit_cache)
        return logits[0, -1], _state_from_explicit_caches(explicit_cache)

    decode_step = mx.compile(decode_step)
    for step in range(int(max_tokens)):
        sample_key = mx.random.key(int(seed) + step) if seed is not None else None
        ids = sample_frame_ids(last_logits, state, params, key=sample_key)
        row = [int(token) for token in ids.tolist()]  # type: ignore
        state.append(row + [state.text_vocab], ignore_eos=params.ignore_eos)
        if state.finished:
            break
        next_ids = _frame_from_ids(ids, state.text_vocab)[None, None, :]
        last_logits, explicit_cache_state = decode_step(next_ids, explicit_cache_state)
    generated_rows = state.generated
    token_count = len(state.generated)
    eos_frame = state.eos_frame

    mx.eval(last_logits)
    sample_seconds = time.perf_counter() - sample_start
    audio, decode_seconds = _decode_audio_timed(model, generated_rows, eos_frame)

    elapsed = time.perf_counter() - start
    samples = int(audio.shape[0])
    duration_s = samples / model.sample_rate if model.sample_rate else 0.0
    if profile:
        print(
            "Profile: "
            f"speaker={speaker_seconds:.2f}s, "
            f"prompt={prompt_seconds:.2f}s, "
            f"sample={sample_seconds:.2f}s, "
            f"decode={decode_seconds:.2f}s, "
            f"tokens/s={token_count / sample_seconds if sample_seconds else 0:.2f}",
            flush=True,
        )
    return {
        "audio": audio,
        "sample_rate": model.sample_rate,
        "token_count": token_count,
        "samples": samples,
        "audio_duration_s": duration_s,
        "decode_seconds": decode_seconds,
        "processing_time_seconds": elapsed,
        "real_time_factor": round(elapsed / duration_s, 3) if duration_s else 0.0,
        "audio_duration": format_duration(duration_s),
    }


def choose_model() -> str:
    if "ZONOS2_MODEL" in os.environ:
        return os.environ["ZONOS2_MODEL"]
    if DEFAULT_QUANTIZED_MODEL.exists():
        return str(DEFAULT_QUANTIZED_MODEL)
    raise FileNotFoundError(
        f"Missing {DEFAULT_QUANTIZED_MODEL}. Run with ZONOS2_CREATE_QUANTIZED=1 "
        "to create the optimized local checkpoint."
    )


def main() -> None:
    args = parse_args()
    load_runtime()
    quantized_path = Path(
        os.environ.get("ZONOS2_QUANTIZED_PATH", DEFAULT_QUANTIZED_MODEL)
    )
    if args.create_quantized or env_bool("ZONOS2_CREATE_QUANTIZED", False):
        create_quantized_checkpoint(
            os.environ.get("ZONOS2_SOURCE_MODEL", SOURCE_MODEL),
            quantized_path,
            group_size=env_int("ZONOS2_QUANTIZE_GROUP_SIZE", 64),
            mode=os.environ.get("ZONOS2_QUANTIZE_MODE", "affine"),
            overwrite=env_bool("ZONOS2_OVERWRITE_QUANTIZED", False),
        )
        if env_bool("ZONOS2_CREATE_ONLY", True):
            return

    text = (
        args.text
        or os.environ.get("ZONOS2_TEXT")
        or "Hello, this is ZONOS two running locally with MLX audio."
    )
    output_path = args.output or os.environ.get("ZONOS2_OUTPUT", "zonos2.wav")
    max_tokens = (
        env_int("ZONOS2_MAX_TOKENS", 220) if "ZONOS2_MAX_TOKENS" in os.environ else None
    )
    ref_audio = args.ref_audio or os.environ.get("ZONOS2_REF_AUDIO")
    model_name = choose_model()

    start = time.perf_counter()
    print(f"Loading {model_name}...", flush=True)
    assert load is not None
    model = load(model_name, lazy=env_bool("ZONOS2_LAZY_LOAD", False))
    if env_bool("ZONOS2_PRELOAD_DAC", False):
        print("Loading DAC decoder...", flush=True)
        install_dac_cache_shortcut()
        model._load_dac()  # type: ignore

    if max_tokens is None:
        print("Generating with automatic token budget...", flush=True)
    else:
        print(f"Generating {max_tokens} tokens...", flush=True)
    result = generate_sampled(
        model,
        text=text,
        max_tokens=max_tokens,
        ref_audio=ref_audio,
        text_normalization=env_bool("ZONOS2_TEXT_NORMALIZATION", True),
    )

    assert audio_write is not None
    audio_write(output_path, result["audio"], result["sample_rate"])
    elapsed = time.perf_counter() - start
    print(
        f"Wrote {output_path} in {elapsed:.2f}s "
        f"({result['token_count']} tokens, "
        f"{result['audio_duration_s']:.2f}s audio, "
        f"RTF {result['real_time_factor']}"
        + (
            f", decode {result['decode_seconds']:.2f}s)"
            if result["decode_seconds"] is not None
            else ")"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
