#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array

import torch
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text

MODEL_ID = "espnet/powsm"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )  # temp: use "cpu" because powsm has memory leak with "mps"
)
TASK = "<pr>"  # phone recognition <pr>, automatic speech recognition <asr>, audio guided grapheme to phoneme <g2p>, audio guided phoneme to grapheme <p2g>
LANGUAGE = "<eng>"  # ISO 639-3; set to <unk> for unseen languages
PROMPT = (
    "<na>"  # G2P: set to ASR transcript; P2G: set to phone transcription with slashes
)

s2t = Speech2Text.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    lang_sym=LANGUAGE,
    task_sym=TASK,
)


def transcribe_from_array(wav_array_16khz_float32_or_int16):
    if wav_array_16khz_float32_or_int16.dtype != np.float32:
        wav_array_16khz_float32_or_int16 = (
            wav_array_16khz_float32_or_int16.astype(np.float32) / 32768
        )

    pred = s2t(wav_array_16khz_float32_or_int16, text_prev=PROMPT)[0][0]

    # post-processing for better format
    pred = pred.split("<notimestamps>")[1].strip()  # type: ignore
    if TASK == "<pr>" or TASK == "<g2p>":
        pred = pred.replace("/", "")

    return pred


def transcribe_from_file(input_path):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768  # type: ignore
    return transcribe_from_array(wav_array)


def transcribe_from_mic():
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/ipa_transcription/powsm.py <audio file>")
        print("Usage: python ./scripts/ipa_transcription/powsm.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(transcribe_from_mic())
    else:
        print(transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
