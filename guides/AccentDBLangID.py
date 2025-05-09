#!/usr/bin/env python3

# Setup (tested with MacOS 15 and python 3.9.13):
#   - pip install numpy==1.23.5 datasets==3.1.0 torch==2.4.1 torchaudio==2.4.1 soundfile==0.13.1 librosa==0.9.2 speechbrain==1.0.2

# Trouble shooting:
#   - Try pip installing without the version numbers
#   - Try logging in with the Hugging Face CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cli
#   - Do a debugging dance while chanting magic encantations, then restart your computer
#   - On MacOS you may need brew install espeak ffmpeg portaudio pyaudio
#   - On Linux you may need sudo apt-get update && sudo apt-get install ffmpeg espeak-ng libportaudio2 python3-pyaudio
#   - On Windows you may need to pray to the Microsoft gods

################## Hide Extraneous Warnings ####################
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)  # fmt: skip
import os; os.environ["SB_LOG_LEVEL"] = "10000"  # fmt: skip
###############################################################

import torch
import librosa
import numpy as np
from datasets import load_dataset
from speechbrain.inference.classifiers import EncoderClassifier

MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"  # hugging face id or local folder

model = EncoderClassifier.from_hparams(source=MODEL_ID)
assert model, "Failed to load the model. Check the path and try again."


def identify_language(sample):
    audio = sample["audio"]
    wav_array = audio["array"]
    sample_rate = audio["sampling_rate"]

    # speechbrain expects float32 normalized pcm array
    if wav_array.dtype != np.float32 or np.abs(wav_array).max() > 1:
        wav_array = wav_array.astype(np.float32) / 32768

    # resample to 16 kHz
    TARGET_SAMPLING_RATE = 16_000
    if sample_rate != TARGET_SAMPLING_RATE:
        wav_array = librosa.resample(
            wav_array, orig_sr=sample_rate, target_sr=TARGET_SAMPLING_RATE
        )

    # add batch dimension
    signal = torch.from_numpy(wav_array).unsqueeze(0)
    prediction = model.classify_batch(signal)  # type: ignore
    return prediction[3][0]


ds = load_dataset("DTU54DL/common-accent")
testset = ds["test"]  # type: ignore

sample = testset[0]
print(identify_language(sample))
