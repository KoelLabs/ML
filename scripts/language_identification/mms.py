#!/usr/bin/env python3

# MMS: Massively Multilingual Speech models by facebook: https://github.com/facebookresearch/fairseq/tree/main/examples/mms

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from langcodes import standardize_tag
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_id = "facebook/mms-lid-4017"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(DEVICE)  # type: ignore

SUPPORTED_LANGUAGES = set(model.config.id2label.values())  # type: ignore


def mms_detect_language_from_array(wav_array, topk=3):
    """wav_array is an int16 16kHz wav pcm array or a normalized float32 16kHz pcm array"""

    if wav_array.dtype != np.float32:  # mms expects normalized float32 at 16 kHz
        wav_array = wav_array.astype(np.float32) / 32768

    inputs = processor(
        wav_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs).logits

    prob = torch.softmax(outputs, dim=-1)
    top = torch.topk(prob, topk, dim=-1, sorted=True)
    return (
        standardize_tag(model.config.id2label[top.indices[0][0].item()]),  # type: ignore
        top.values[0][0].item(),
        [
            (standardize_tag(model.config.id2label[ix.item()]), val.item())  # type: ignore
            for ix, val in zip(top.indices[0], top.values[0])
        ],
    )


def mms_detect_language_from_file(input_path: str, language="eng"):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768  # type: ignore
    return mms_detect_language_from_array(wav_array)


def mms_detect_language_from_mic(language="eng"):
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return mms_detect_language_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/language_identification/mms.py <audio file>")
        print("Usage: python ./scripts/language_identification/mms.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(mms_detect_language_from_mic())
    else:
        print(mms_detect_language_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
