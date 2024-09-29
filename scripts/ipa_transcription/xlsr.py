#!/usr/bin/env python3

from transformers import pipeline

import sys
import os
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.audio import audio_record_to_file

# set espeak library path for macOS
if sys.platform == "darwin":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    _ESPEAK_LIBRARY = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib"
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

MODEL_IDS = [
    "facebook/wav2vec2-lv-60-espeak-cv-ft",  # best for English so far
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
    "ginic/wav2vec-large-xlsr-en-ipa",
    "ginic/data_seed_4_wav2vec2-large-xlsr-buckeye-ipa",
    "speech31/XLS-R-300m-english-ipa",
]

pipelines = {}


def xlsr_transcribe(input_path, model_id=MODEL_IDS[0]):
    pipelines[model_id] = pipelines.get(model_id) or pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device="cpu",
    )
    return pipelines[model_id](input_path).get("text", "")


def xlsr_transcribe_from_mic(model_id=MODEL_IDS[0]):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return xlsr_transcribe(f.name, model_id)


def main(args):
    if args[0] == "mic":
        print(xlsr_transcribe_from_mic())
    else:
        try:
            input_path = args[0]
            print(xlsr_transcribe(input_path))
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/ipa_transcription/xlsr.py mic")
            print("Usage: python ./scripts/ipa_transcription/xlsr.py <input_wav_path>")


if __name__ == "__main__":
    main(sys.argv[1:])
