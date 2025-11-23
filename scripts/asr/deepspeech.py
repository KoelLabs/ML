#!/usr/bin/env python3

# https://github.com/mozilla/DeepSpeech --continued as--> https://github.com/coqui-ai/STT

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array

if not os.path.exists("models/deepspeech"):
    # Download model files from https://github.com/coqui-ai/STT-models/releases/tag/english%2Fcoqui%2Fv1.0.0-huge-vocab
    os.makedirs("models/deepspeech")
    from urllib.request import urlretrieve

    urlretrieve(
        r"https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/model.tflite",
        "models/deepspeech/model.tflite",
    )
    urlretrieve(
        r"https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/huge-vocabulary.scorer",
        "models/deepspeech/huge-vocabulary.scorer",
    )

from stt import Model

model = Model("models/deepspeech/model.tflite")
model.enableExternalScorer("models/deepspeech/huge-vocabulary.scorer")


def deepspeech_transcribe_from_array(wav_array):
    return model.stt(wav_array)


def deepspeech_transcribe_from_file(input_path: str):
    wav_array = audio_file_to_array(input_path)
    return deepspeech_transcribe_from_array(wav_array)


def deepspeech_transcribe_from_mic():
    wav_array = audio_record_to_array()
    return deepspeech_transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/deepspeech.py <audio file>")
        print("Usage: python ./scripts/asr/deepspeech.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(deepspeech_transcribe_from_mic())
    else:
        print(deepspeech_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
