#!/usr/bin/env python3

# Setup (tested with MacOS 15 and python 3.9.13):
#   - pip install numpy==1.23.5 datasets==3.1.0 torch==2.4.1 torchaudio==2.4.1 soundfile==0.13.1 librosa==0.9.2 espnet==202412 espnet-model-zoo==0.1.7

# Trouble shooting:
#   - Try pip installing without the version numbers
#   - Try logging in with the Hugging Face CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cli
#   - Do a debugging dance while chanting magic encantations, then restart your computer
#   - On MacOS you may need brew install espeak ffmpeg portaudio pyaudio
#   - On Linux you may need sudo apt-get update && sudo apt-get install ffmpeg espeak-ng libportaudio2 python3-pyaudio
#   - On Windows you may need to pray to the Microsoft gods
#   - This sample script loads the model twice (both for lang ID and ASR), you may want to comment one out or use a smaller model ID
#   - You can optionally `pip install flash_attn` if supported on your platform for faster inference

######################## Hide Extraneous Warnings #####################
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)  # fmt: skip
#######################################################################

import torch
import librosa
import soundfile as sf
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.bin.s2t_inference_language import Speech2Language
from langcodes import standardize_tag

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet
TARGET_SAMPLE_RATE = 16_000  # 16 kHz
MODEL_ID = "espnet/owsm_v3.1_ebf"
# valid model ids:
# espnet/owsm_v3 - 889M params, 180K audio hours
# espnet/owsm_v3.1_ebf_base - 101M params, 180K audio hours
# espnet/owsm_v3.1_ebf_small - 367M params, 180K audio hours
# espnet/owsm_v3.1_ebf - 1.02B params, 180K audio hours
# espnet/owsm_v3.1_ebf_small_lowrestriction - 367M params, 70K audio hours
# pyf98/owsm_ctc_v3.1_1B - 1.01B params, 180K audio hours
# espnet/owsm_v3.2 - 367M params, 180K audio hours

#######################################################################
####################### Language Identification #######################
s2l = Speech2Language.from_pretrained(
    model_tag=MODEL_ID,
    device=DEVICE,
    nbest=3,  # return nbest prediction and probability
)


def owsm_detect_language_from_array(
    wav_array: np.ndarray,
):
    assert wav_array.dtype == np.float64, "owsm expects float64 pcm array"
    if np.abs(wav_array).max() > 1:
        wav_array /= np.abs(wav_array).max()

    result = s2l(wav_array)
    return standardize_tag(result[0][0].replace("<", "").replace(">", ""))


#######################################################################
################################ ASR ##################################

s2t = Speech2Text.from_pretrained(
    model_tag=MODEL_ID,
    device=DEVICE,
    beam_size=5,
    ctc_weight=0.0,
    maxlenratio=0.0,
    # below are default values which can be overwritten in __call__
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False,
)


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
    assert wav_array.dtype == np.float64, "owsm expects float64 pcm array"
    if np.abs(wav_array).max() > 1:
        wav_array /= np.abs(wav_array).max()

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


#######################################################################
########################## Data Processing ############################


def hugging_face_sample_to_array(sample):
    audio = sample["audio"]
    wav_array = np.array(audio["array"])
    sample_rate = audio["sampling_rate"]

    return librosa.resample(
        wav_array.astype(np.float64), orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
    )


def file_to_array(filename):
    array, sample_rate = sf.read(filename)
    return librosa.resample(array, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)


#######################################################################
############################ Example usage ############################

if __name__ == "__main__":
    from datasets import load_dataset

    print("Analyzing first sample from Common Accent")
    ds = load_dataset("DTU54DL/common-accent")
    testset = ds["test"]  # type: ignore

    sample = testset[0]
    array = hugging_face_sample_to_array(sample)
    print("Language", owsm_detect_language_from_array(array))
    print("Transcription", owsm_transcribe_from_array(array))

    print("Analyzing a file from TIMIT")
    filename = "/Users/alex/Desktop/CS/Startups/Koel/ML/data/ExamplesWithComments/TIMIT_sample_0.wav"
    array = file_to_array(filename)
    print("Language", owsm_detect_language_from_array(array))
    print("Transcription", owsm_transcribe_from_array(array))
