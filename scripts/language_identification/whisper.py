#!/usr/bin/env python3

import torch
import numpy as np
from langcodes import standardize_tag
from faster_whisper import WhisperModel

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_file_to_array, audio_record_to_array

# model_size = "large-v3"
model_size = "small"

if torch.cuda.is_available():
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    # Run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")


def whisper_detect_language_from_array(
    wav_array: np.ndarray,
) -> "tuple[str, float, list[tuple[str, float]]]":
    if wav_array.dtype != np.float64:
        wav_array = wav_array.astype(np.float64) / 32768

    language, language_probabiliy, all_probabilities = model.detect_language(wav_array)
    return (
        standardize_tag(language),
        language_probabiliy,
        [(standardize_tag(l), p) for l, p in all_probabilities],
    )


def whisper_detect_language_from_file(
    input_path: str,
) -> "tuple[str, float, list[tuple[str, float]]]":
    wav_array = audio_file_to_array(input_path).astype(np.float64) / 32768
    return whisper_detect_language_from_array(wav_array)


def whisper_detect_language_from_mic():
    wav_array = audio_record_to_array().astype(np.float64) / 32768
    return whisper_detect_language_from_array(wav_array)


def main(args):
    if len(args) != 1:
        print("Usage: python ./scripts/language_identification/whisper.py <audio file>")
        print("Usage: python ./scripts/language_identification/whisper.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        language, language_probabiliy, all_probabilities = (
            whisper_detect_language_from_mic()
        )
    else:
        language, language_probabiliy, all_probabilities = (
            whisper_detect_language_from_file(input_path)
        )

    print((language, language_probabiliy, all_probabilities[:3]))


if __name__ == "__main__":
    main(sys.argv[1:])
