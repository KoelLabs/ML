#!/usr/bin/env python3

# MMS: Massively Multilingual Speech models by facebook: https://github.com/facebookresearch/fairseq/tree/main/examples/mms

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, AutoProcessor

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_id = "facebook/mms-1b-all"
# model_id = "mms-meta/mms-zeroshot-300m"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)  # type: ignore

SUPPORTED_LANGUAGES = processor.tokenizer.vocab.keys()


def mms_transcribe_from_array(wav_array, language="eng"):
    """
    wav_array is an int16 16kHz wav pcm array or a normalized float32 16kHz pcm array
    Language is an ISO 639-3 code from SUPPORTED_LANGUAGES
    """

    if wav_array.dtype != np.float32:  # mms expects normalized float32 at 16 kHz
        wav_array = wav_array.astype(np.float32) / 32768

    processor.tokenizer.set_target_lang(language)
    model.load_adapter(language)

    inputs = processor(
        wav_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    return transcription


def mms_transcribe_from_file(input_path: str, language="eng"):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768
    return mms_transcribe_from_array(wav_array, language)


def mms_transcribe_from_mic(language="eng"):
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return mms_transcribe_from_array(wav_array, language)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/mms.py <audio file>")
        print("Usage: python ./scripts/asr/mms.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(mms_transcribe_from_mic())
    else:
        print(mms_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
