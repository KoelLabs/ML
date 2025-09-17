#!/usr/bin/env python3

# OWSM: Open Whisper-style Speech Model by the CMU WAVLab
# https://www.wavlab.org/activities/2024/owsm/

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_file_to_array, audio_record_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# valid model ids:
# espnet/owsm_v3 - 889M params, 180K audio hours
# espnet/owsm_v3.1_ebf_base - 101M params, 180K audio hours
# espnet/owsm_v3.1_ebf_small - 367M params, 180K audio hours
# espnet/owsm_v3.1_ebf - 1.02B params, 180K audio hours
# espnet/owsm_v3.1_ebf_small_lowrestriction - 367M params, 70K audio hours
# pyf98/owsm_ctc_v3.1_1B - 1.01B params, 180K audio hours
# espnet/owsm_v3.2 - 367M params, 180K audio hours

s2t = Speech2Text.from_pretrained(
    model_tag="espnet/owsm_v3.1_ebf",
    device=DEVICE.replace("mps", "cpu"),
    beam_size=5,
    ctc_weight=0.0,
    maxlenratio=0.0,  # if it seems to not terminate, set this to a small value like 0.05
    # below are default values which can be overwritten in __call__
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False,
)
if DEVICE == "mps":
    s2t.s2t_model.to(device=DEVICE, dtype=torch.float32)
    s2t.beam_search.to(device=DEVICE, dtype=torch.float32).eval()
    for scorer in s2t.beam_search.scorers.values():
        if isinstance(scorer, torch.nn.Module):
            scorer.to(device=DEVICE, dtype=torch.float32).eval()
    s2t.dtype = "float32"
    s2t.device = DEVICE


def _naive_decode_long(wav_array, config, chunk_size=30 * TARGET_SAMPLE_RATE):
    predictions = []
    for chunk in range(0, len(wav_array), chunk_size):
        audio_chunk = wav_array[chunk : chunk + chunk_size]
        predictions.append(s2t(audio_chunk, **config)[0][-2])
    if config["predict_time"]:
        return [(start, end, text) for pred in predictions for start, end, text in pred]
    return " ".join(predictions)


def owsm_transcribe_from_array(
    wav_array: np.ndarray,
    text_prompt: "str | None" = None,
    naive_long=True,  # the proper long-form decoding is super resource intensive on CPU
    timestamps=False,
    translate: "tuple[str, str]" = ("eng", "eng"),
):
    if wav_array.dtype != np.float64:
        wav_array = wav_array.astype(np.float64) / 32768

    # enable long-form decoding for audio longer than 30s
    long = len(wav_array) > 30 * TARGET_SAMPLE_RATE

    config = {}
    if text_prompt is not None:
        config["text_prev"] = text_prompt
    config["lang_sym"] = f"<{translate[0]}>"
    config["task_sym"] = (
        "<asr>" if translate[0] == translate[1] else f"<st_{translate[1]}>"
    )
    config["predict_time"] = timestamps

    if long:
        if naive_long:
            return _naive_decode_long(wav_array, config)
        else:
            del config["predict_time"]
            result = s2t.decode_long(wav_array, **config)
            if timestamps:
                return result
            else:
                return " ".join(res[2] for res in result)
    else:
        return s2t(wav_array, **config)[0][-2]


def owsm_transcribe_from_file(
    input_path: str,
    text_prompt: "str | None" = None,
    naive_long=True,
    timestamps=False,
    translate: "tuple[str, str]" = ("eng", "eng"),
):
    wav_array = audio_file_to_array(input_path).astype(np.float64) / 32768  # type: ignore
    return owsm_transcribe_from_array(
        wav_array, text_prompt, naive_long, timestamps, translate
    )


def owsm_transcribe_from_mic(
    text_prompt: "str | None" = None,
    naive_long=True,
    timestamps=False,
    translate: "tuple[str, str]" = ("eng", "eng"),
):
    wav_array = audio_record_to_array().astype(np.float64) / 32768
    return owsm_transcribe_from_array(
        wav_array, text_prompt, naive_long, timestamps, translate
    )


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/owsm.py <audio file>")
        print("Usage: python ./scripts/asr/owsm.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(owsm_transcribe_from_mic())
    else:
        print(owsm_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
