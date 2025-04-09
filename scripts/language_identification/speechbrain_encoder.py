#!/usr/bin/env python3

import os, sys
import warnings
from speechbrain.dataio.encoder import logger
from tempfile import NamedTemporaryFile

warnings.filterwarnings("ignore", category=FutureWarning)
logger.setLevel("ERROR")

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.core.audio import audio_record_to_file

import torch
import numpy as np
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data")

assert (
    len(torchaudio.list_audio_backends()) > 0
), "Torchaudio backend not found. Try `pip install soundfile==0.13.1`"

language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=os.path.join(DATA_DIR, "speechbrain/lang-id-voxlingua107-ecapa"),
)
assert language_id, "Failed to load the model. Check the path and try again."


def identify_language_from_array(wav_array: np.ndarray) -> str:
    # add batch dimension
    signal = torch.from_numpy(wav_array).unsqueeze(0)
    prediction = language_id.classify_batch(signal)  # type: ignore
    return prediction[3][0]


def identify_language_from_file(audio_path: str) -> str:
    signal = language_id.load_audio(audio_path)  # type: ignore
    return identify_language_from_array(signal)


def identify_language_from_mic() -> str:
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return identify_language_from_file(f.name)


def main(args):
    if len(args) != 1:
        print(
            "Usage: python ./scripts/language_identification/speechbrain_encoder.py <audio file>"
        )
        print(
            "Usage: python ./scripts/language_identification/speechbrain_encoder.py mic"
        )
        return

    input_path = args[0]
    if input_path == "mic":
        print(identify_language_from_mic())
    else:
        print(identify_language_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
