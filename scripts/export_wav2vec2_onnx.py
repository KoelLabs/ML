#!/usr/bin/env python3
"""Export a Hugging Face Wav2Vec2 CTC model to ONNX and quantized ONNX."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoProcessor, Wav2Vec2ForCTC

DEFAULT_SOURCE = "KoelLabs/wav2vec2-large-xlsr-53-english-100"
DEFAULT_TARGET = "KoelLabs/xlsr-english-private-01"


class Wav2Vec2OnnxWrapper(torch.nn.Module):
    def __init__(self, model: Wav2Vec2ForCTC) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_values=input_values, attention_mask=attention_mask
        ).logits


def start_memory_guard(max_rss_gb: float | None, min_available_gb: float) -> None:
    process = psutil.Process(os.getpid())
    total_gb = psutil.virtual_memory().total / (1024**3)
    limit_gb = max_rss_gb if max_rss_gb else min(total_gb * 0.70, 24.0)

    def watch() -> None:
        while True:
            rss_gb = process.memory_info().rss / (1024**3)
            available_gb = psutil.virtual_memory().available / (1024**3)
            if rss_gb > limit_gb or available_gb < min_available_gb:
                print(
                    f"Memory guard stopping export: rss={rss_gb:.1f}GB, "
                    f"available={available_gb:.1f}GB, rss_limit={limit_gb:.1f}GB",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(137)
            time.sleep(2)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    print(
        f"Memory guard active: rss_limit={limit_gb:.1f}GB, min_available={min_available_gb:.1f}GB",
        flush=True,
    )


def copy_runtime_files(source_dir: Path, out_dir: Path) -> None:
    names = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "normalizer.json",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def write_model_card(out_dir: Path, source_repo: str, target_repo: str) -> None:
    readme = f"""---
language:
- en
license: apache-2.0
library_name: onnx
pipeline_tag: automatic-speech-recognition
private: true
tags:
- wav2vec2
- ctc
- onnx
- int8
---

# XLSR English ONNX

This repository contains graph-optimized ONNX exports of the IPA transcription model `{source_repo}`:

- `model.onnx`: fp32 ONNX export, optimized during export and with ONNX Runtime graph optimizations
- `model_quantized.onnx`: dynamic int8 ONNX Runtime quantization of the optimized fp32 graph

Tokenizer, vocabulary, processor, and model config files are included so the model can be run directly with `transformers` processors plus `onnxruntime`.

## Install

```bash
pip install huggingface_hub transformers onnxruntime soundfile numpy
```

## IPA transcription without timestamps

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

repo_id = "{target_repo}"
model_dir = snapshot_download(repo_id, token=True)

processor = AutoProcessor.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)
session = ort.InferenceSession(f"{{model_dir}}/model_quantized.onnx", providers=["CPUExecutionProvider"])

audio, sr = sf.read("audio.wav")
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

inputs = processor(audio, sampling_rate=sr, return_tensors="np", padding=True)
ort_inputs = {{
    "input_values": inputs["input_values"].astype(np.float32),
    "attention_mask": inputs.get("attention_mask", np.ones_like(inputs["input_values"], dtype=np.int64)).astype(np.int64),
}}
logits = session.run(["logits"], ort_inputs)[0]
predicted_ids = np.argmax(logits, axis=-1)
text = processor.batch_decode(predicted_ids)[0]
print(text)
```

## IPA transcription with token timestamps

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

repo_id = "{target_repo}"
model_dir = snapshot_download(repo_id, token=True)

processor = AutoProcessor.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)
session = ort.InferenceSession(f"{{model_dir}}/model_quantized.onnx", providers=["CPUExecutionProvider"])

audio, sr = sf.read("audio.wav")
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

inputs = processor(audio, sampling_rate=sr, return_tensors="np", padding=True)
ort_inputs = {{
    "input_values": inputs["input_values"].astype(np.float32),
    "attention_mask": inputs.get("attention_mask", np.ones_like(inputs["input_values"], dtype=np.int64)).astype(np.int64),
}}
logits = session.run(["logits"], ort_inputs)[0]
predicted_ids = np.argmax(logits, axis=-1)

decoded = processor.decode(predicted_ids[0], output_char_offsets=True)
time_offset = config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
tokens = [
    {{
        "token": item["char"],
        "start": round(item["start_offset"] * time_offset, 3),
        "end": round(item["end_offset"] * time_offset, 3),
    }}
    for item in decoded.char_offsets
]

print(decoded.text)
print(tokens)
```

Use `model.onnx` instead of `model_quantized.onnx` in `InferenceSession(...)` to run the fp32 export.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def export_onnx(model_dir: Path, out_dir: Path) -> Path:
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    model.eval()
    wrapper = Wav2Vec2OnnxWrapper(model).eval()

    input_values = torch.randn(1, 16000, dtype=torch.float32)
    attention_mask = torch.ones(1, 16000, dtype=torch.long)
    raw_onnx_path = out_dir / "model.raw.onnx"
    onnx_path = out_dir / "model.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_values, attention_mask),
            raw_onnx_path,
            input_names=["input_values", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "time"},
            },
            opset_version=17,
            dynamo=False,
            external_data=False,
            optimize=True,
            do_constant_folding=True,
        )

    del model, wrapper, input_values, attention_mask
    gc.collect()

    onnx.checker.check_model(str(raw_onnx_path))
    optimize_onnx(raw_onnx_path, onnx_path)
    raw_onnx_path.unlink()
    return onnx_path


def optimize_onnx(input_path: Path, output_path: Path) -> None:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    session_options.optimized_model_filepath = str(output_path)
    session = ort.InferenceSession(
        str(input_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    del session
    gc.collect()
    onnx.checker.check_model(str(output_path))


def verify_export(onnx_path: Path, quantized_path: Path, model_dir: Path) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    audio = 0.05 * np.sin(2 * np.pi * 440 * t)
    inputs = processor(
        audio, sampling_rate=sample_rate, return_tensors="np", padding=True
    )
    input_values = inputs["input_values"].astype(np.float32)
    attention_mask = inputs.get(
        "attention_mask", np.ones_like(input_values, dtype=np.int64)
    ).astype(np.int64)

    for path in [onnx_path, quantized_path]:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logits = session.run(
            ["logits"], {"input_values": input_values, "attention_mask": attention_mask}
        )[0]
        if logits.ndim != 3 or logits.shape[-1] != config.vocab_size:
            raise RuntimeError(f"Unexpected logits shape for {path}: {logits.shape}")


def upload_folder(api: HfApi, target: str, out_dir: Path) -> None:
    expected = [
        "README.md",
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "model.onnx",
        "model_quantized.onnx",
    ]
    missing = [name for name in expected if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing upload files in {out_dir}: {missing}")

    try:
        api.model_info(target)
    except RepositoryNotFoundError:
        api.create_repo(target, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=target,
        repo_type="model",
        folder_path=str(out_dir),
        commit_message="Add ONNX and dynamic int8 quantized exports",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--out-dir", default="models/xlsr-english-private-01")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--reuse-onnx", action="store_true")
    parser.add_argument("--max-rss-gb", type=float, default=None)
    parser.add_argument("--min-available-gb", type=float, default=3.0)
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    out_dir = Path(args.out_dir).resolve()
    if args.upload_only:
        upload_folder(HfApi(token=token), args.target, out_dir)
        print(f"Uploaded artifacts from {out_dir}")
        return

    start_memory_guard(args.max_rss_gb, args.min_available_gb)

    source_dir = Path(
        snapshot_download(
            args.source,
            token=token,
            ignore_patterns=["*.onnx", "*.tflite", "*.msgpack", "README.md"],
        )
    )
    if out_dir.exists() and not args.reuse_onnx:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_runtime_files(source_dir, out_dir)
    write_model_card(out_dir, args.source, args.target)

    onnx_path = out_dir / "model.onnx"
    if args.reuse_onnx:
        if not onnx_path.exists():
            raise RuntimeError(f"--reuse-onnx was set but {onnx_path} does not exist")
        onnx.checker.check_model(str(onnx_path))
    else:
        onnx_path = export_onnx(source_dir, out_dir)

    quantized_path = out_dir / "model_quantized.onnx"
    if quantized_path.exists():
        quantized_path.unlink()
    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    onnx.checker.check_model(str(quantized_path))
    verify_export(onnx_path, quantized_path, source_dir)

    if not args.skip_upload:
        upload_folder(HfApi(token=token), args.target, out_dir)

    print(f"Wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
