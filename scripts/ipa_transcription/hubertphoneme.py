#!/usr/bin/env python3

# https://github.com/bootphon/spokenlm-phoneme

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array
from core.codes import ARPABET2IPA

import torch
import numpy as np
from phonslm import HuBERTPhoneme


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model = (
    HuBERTPhoneme.from_pretrained("coml/hubert-phoneme-classification")
    .to(DEVICE)
    .eval()
)


# fmt:off
PHONEMES = {
    "SIL": 0, "AA": 1, "AE": 2, "AH": 3, "AO": 4, "AW": 5, "AY": 6, "B": 7,
    "CH": 8, "D": 9, "DH": 10, "EH": 11, "ER": 12, "EY": 13, "F": 14, "G": 15,
    "HH": 16, "IH": 17, "IY": 18, "JH": 19, "K": 20, "L": 21, "M": 22, "N": 23,
    "NG": 24, "OW": 25, "OY": 26, "P": 27, "R": 28, "S": 29, "SH": 30, "T": 31,
    "TH": 32, "UH": 33, "UW": 34, "V": 35, "W": 36, "Y": 37, "Z": 38, "ZH": 39,
}
TOKEN2ID = PHONEMES | {"<pad>": len(PHONEMES)}
ID2TOKEN = {v: k for k, v in TOKEN2ID.items()}
# fmt:on
def decode(tokens):
    return (ID2TOKEN[int(token)] for token in tokens if token < len(PHONEMES))


def transcribe_from_array(wav_array_16khz_float32_or_int16):
    if wav_array_16khz_float32_or_int16.dtype != np.float32:
        wav_array_16khz_float32_or_int16 = (
            wav_array_16khz_float32_or_int16.astype(np.float32) / 32768
        )
    audio = torch.from_numpy(wav_array_16khz_float32_or_int16)

    arpabet = []
    with torch.inference_mode():
        output, _ = model.inference(audio.to(DEVICE).unsqueeze(0))
        predictions = output.argmax(dim=-1).squeeze().cpu()
        arpabet = decode(predictions.unique_consecutive())
    return "".join(ARPABET2IPA[p] for p in arpabet if p != "SIL")


def transcribe_from_file(input_path):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768  # type: ignore
    return transcribe_from_array(wav_array)


def transcribe_from_mic():
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/ipa_transcription/hubertphoneme.py <audio file>")
        print("Usage: python ./scripts/ipa_transcription/hubertphoneme.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(transcribe_from_mic())
    else:
        print(transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
