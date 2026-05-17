#!/usr/bin/env python3

# https://github.com/mozilla/DeepSpeech --continued as--> https://github.com/coqui-ai/STT

import os
import shutil
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "deepspeech")
MODEL_PATH = os.path.join(MODEL_DIR, "model.tflite")
SCORER_PATH = os.path.join(MODEL_DIR, "huge-vocabulary.scorer")

if not os.path.exists(MODEL_DIR):
    # Download model files from https://github.com/coqui-ai/STT-models/releases/tag/english%2Fcoqui%2Fv1.0.0-huge-vocab
    os.makedirs(MODEL_DIR)
    from urllib.request import urlretrieve

    urlretrieve(
        r"https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/model.tflite",
        MODEL_PATH,
    )
    urlretrieve(
        r"https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/huge-vocabulary.scorer",
        SCORER_PATH,
    )

_configured_stt_binary = os.environ.get("STT_BINARY")
STT_BINARY = (
    _configured_stt_binary
    if _configured_stt_binary and os.path.exists(_configured_stt_binary)
    else shutil.which("stt")
)
model = None


def _get_python_model():
    global model
    if model is None:
        from stt import Model

        model = Model(MODEL_PATH)
        model.enableExternalScorer(SCORER_PATH)
    return model


def _run_stt_binary(input_path: str):
    completed = subprocess.run(
        [
            STT_BINARY,  # type: ignore
            "--model",
            MODEL_PATH,
            "--scorer",
            SCORER_PATH,
            "--audio",
            input_path,
        ],
        capture_output=True,
        text=True,
    )  # type: ignore
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"stt command failed with exit code {completed.returncode}: {detail}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def deepspeech_transcribe_from_array(wav_array):
    return _get_python_model().stt(wav_array)


def deepspeech_transcribe_from_file(input_path: str):
    if STT_BINARY:
        return _run_stt_binary(input_path)
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
