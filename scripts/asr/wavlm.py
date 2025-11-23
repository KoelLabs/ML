#!/usr/bin/env python3

# WavLM is a series of pretrained speech models by Microsoft: https://github.com/microsoft/unilm/tree/master/wavlm
# To be used for ASR, they must be fine-tuned. We use a checkpoint by Patrick since he was the one to
# implement WavLM in transformers and is very smart: https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from transformers import AutoProcessor, WavLMForCTC

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_id = "patrickvonplaten/wavlm-libri-clean-100h-base"
# model_id = "microsoft/wavlm-large"
# model_id = "microsoft/wavlm-base-plus"
# model_id = "microsoft/wavlm-base"

processor = AutoProcessor.from_pretrained(model_id)
model = WavLMForCTC.from_pretrained(model_id).to(DEVICE)  # type: ignore


def wavlm_transcribe_from_array(wav_array):
    """
    wav_array is an int16 16kHz wav pcm array or a normalized float32 16kHz pcm array
    Language is an ISO 639-3 code from SUPPORTED_LANGUAGES
    """

    if wav_array.dtype != np.float32:  # wavlm expects normalized float32 at 16 kHz
        wav_array = wav_array.astype(np.float32) / 32768

    inputs = processor(
        wav_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]


def wavlm_transcribe_from_file(input_path: str):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768  # type: ignore
    return wavlm_transcribe_from_array(wav_array)


def wavlm_transcribe_from_mic():
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return wavlm_transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/wavlm.py <audio file>")
        print("Usage: python ./scripts/asr/wavlm.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(wavlm_transcribe_from_mic())
    else:
        print(wavlm_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
