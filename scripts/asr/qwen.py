#!/usr/bin/env python3

# Qwen2-Audio: https://huggingface.co/Qwen/Qwen2-Audio-7B

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from transformers import Qwen2AudioProcessor, Qwen2AudioForConditionalGeneration

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu" if torch.backends.mps.is_available() else "cpu"
)

model_id = "Qwen/Qwen2-Audio-7B"

processor: Qwen2AudioProcessor = Qwen2AudioProcessor.from_pretrained(model_id)  # type: ignore
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id).to(DEVICE)  # type: ignore


def qwen_transcribe_from_array(wav_array):
    """
    wav_array is an int16 16kHz wav pcm array or a normalized float32 16kHz pcm array
    """

    if wav_array.dtype != np.float32:  # qwen expects normalized float32 at 16 kHz
        wav_array = wav_array.astype(np.float32) / 32768

    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech in English:"
    inputs = processor(
        text=prompt,
        audios=wav_array,
        return_tensors="pt",
        sampling_rate=TARGET_SAMPLE_RATE,
    ).to(DEVICE)

    generated_ids = model.generate(**inputs, max_length=np.inf)
    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


def qwen_transcribe_from_file(input_path: str):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768  # type: ignore
    return qwen_transcribe_from_array(wav_array)


def qwen_transcribe_from_mic():
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return qwen_transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/qwen.py <audio file>")
        print("Usage: python ./scripts/asr/qwen.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(qwen_transcribe_from_mic())
    else:
        print(qwen_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
