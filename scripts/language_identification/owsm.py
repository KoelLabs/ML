#!/usr/bin/env python3

# OWSM: Open Whisper-style Speech Model by the CMU WAVLab
# https://www.wavlab.org/activities/2024/owsm/

import torch
import numpy as np
from langcodes import standardize_tag
from espnet2.bin.s2t_inference_language import Speech2Language

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_file_to_array, audio_record_to_array

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet

s2l = Speech2Language.from_pretrained(
    model_tag="espnet/owsm_v3.1_ebf",
    device=DEVICE,
    nbest=3,  # return nbest prediction and probability
)


def owsm_detect_language_from_array(
    wav_array: np.ndarray,
):
    if wav_array.dtype != np.float64:
        wav_array = wav_array.astype(np.float64) / 32768

    result = s2l(wav_array)
    return (
        standardize_tag(result[0][0].replace("<", "").replace(">", "")),
        result[0][1],
        [(standardize_tag(l.replace("<", "").replace(">", "")), p) for l, p in result],
    )


def owsm_detect_language_from_file(
    input_path: str,
):
    wav_array = audio_file_to_array(input_path).astype(np.float64) / 32768
    return owsm_detect_language_from_array(wav_array)


def owsm_detect_language_from_mic():
    wav_array = audio_record_to_array().astype(np.float64) / 32768
    return owsm_detect_language_from_array(wav_array)


def main(args):
    if len(args) != 1:
        print("Usage: python ./scripts/language_identification/owsm.py <audio file>")
        print("Usage: python ./scripts/language_identification/owsm.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(owsm_detect_language_from_mic())
    else:
        print(owsm_detect_language_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
