#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.audio import audio_array_to_wav_file, audio_array_play

# ===== workarounds =====
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import importlib_resources
import importlib.resources

importlib.resources.files = importlib_resources.files


with open("venv/lib/python3.8/site-packages/f5_tts/model/dataset.py", "r") as f:
    # replace all occurences of "nn.Module | None" with the quoted version
    content = f.read()
with open("venv/lib/python3.8/site-packages/f5_tts/model/dataset.py", "w") as f:
    f.write(
        content.replace("nn.Module | None", '"None | nn.Module"')
        .replace("Sampler[list[int]]", "Sampler[list]")
        .replace("CustomDataset | HFDataset", '"HFDataset | CustomDataset"')
    )
# ===== workarounds =====

from f5_tts.api import F5TTS

f5tts = F5TTS()


def f5tts_clone_voice(
    voice_reference_audio_path, voice_reference_text: str, target_text: str
):
    wav, sr, spect = f5tts.infer(
        ref_file=voice_reference_audio_path,
        ref_text=voice_reference_text,
        gen_text=target_text,
        file_wave=None,
        file_spect=None,
        seed=42,
    )
    return wav, sr, spect


def main(args):
    try:
        assert len(args) >= 3, "Not enough arguments"
        voice_ref = args[0]
        text_ref = args[1]
        target_text = args[2]
        save_file = args[3] if len(args) >= 4 else None

        wav, sr, spect = f5tts_clone_voice(voice_ref, text_ref, target_text)
        if save_file:
            audio_array_to_wav_file(wav, save_file)
        else:
            audio_array_play(wav)
    except Exception as e:
        print(e)
        print(
            "Usage: python ./scripts/voice_clone/f5tts.py <ref_wav_path> <ref_text> <target_text> [<save_file>]"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

# example: python ./scripts/voice_clone/f5tts.py data/alexIsConfused.wav "hello. alex is extra confused." "hello. alex is extra confused"
