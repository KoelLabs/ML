#!/usr/bin/env python3

import torch
import numpy as np
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
    language, language_probabiliy, all_probabilities = model.detect_language(wav_array)
    return language, language_probabiliy, all_probabilities


def whisper_detect_language_from_file(
    input_path: str,
) -> "tuple[str, float, list[tuple[str, float]]]":
    wav_array = audio_file_to_array(input_path)
    return whisper_detect_language_from_array(wav_array)


def whisper_detect_language_from_mic():
    wav_array = audio_record_to_array()
    return whisper_detect_language_from_array(wav_array)


def main(args):
    if len(args) != 1:
        print("Usage: python ./scripts/language_identification/whisper.py <audio file>")
        print("Usage: python ./scripts/language_identification/whisper.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(whisper_detect_language_from_mic()[:2])
    else:
        print(whisper_detect_language_from_file(input_path)[:2])


if __name__ == "__main__":
    main(sys.argv[1:])
