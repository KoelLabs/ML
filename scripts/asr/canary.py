#!/usr/bin/env python3

#  Canary-Qwen2.5b: https://huggingface.co/nvidia/canary-canary-2.5b

#  Nemo requirements:
# "git+https://github.com/NVIDIA/NeMo.git@6294bc7708ce67522b92f5e9b6917ea0b2e23429#egg=nemo_toolkit[asr]"
# tqdm
# soundfile
# librosa
# huggingface_hub>=0.24
# IPython # Workaround for https://github.com/NVIDIA/NeMo/pull/9890#discussion_r1701028427
# cuda-python>=12.4 # Used for fast TDT and RNN-T inference
# sacrebleu
# bitsandbytes

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from nemo.collections.speechlm2.models import SALM
from core.audio import audio_array_to_wav_file

from tempfile import NamedTemporaryFile


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_id = "nvidia/canary-qwen-2.5b"
salm = SALM.from_pretrained(model_id).to(DEVICE)


def canary_transcribe_from_array(wav_array):
    """
    wav_array is an int16 16kHz wav pcm array or a normalized float32 16kHz pcm array
    """

    if wav_array.dtype != np.float32:  # canary expects normalized float32 at 16 kHz
        wav_array = wav_array.astype(np.float32) / 32768

    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)

    ids = salm.generate(
        prompts=[
            [
                {
                    "role": "user",
                    "content": f"Transcribe the speech in English: {salm.audio_locator_tag}",
                    "audio": [f.name],
                }
            ]
        ],
        max_new_tokens=128,
    )
    response = salm.tokenizer.ids_to_text(ids[0]).to(DEVICE)

    return response


def canary_transcribe_from_file(input_path: str):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768
    return canary_transcribe_from_array(wav_array)


def canary_transcribe_from_mic():
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return canary_transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/canary.py <audio file>")
        print("Usage: python ./scripts/asr/canary.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(canary_transcribe_from_mic())
    else:
        print(canary_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
