#!/usr/bin/env python3

# Wav2vec2: https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/, https://huggingface.co/collections/facebook/wav2vec-20-651e865258e3dee2586c89f5

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array, TARGET_SAMPLE_RATE

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_IDS = [
    "facebook/wav2vec2-base-960h",  # samir model
    "facebook/wav2vec2-large-960h-lv60-self",  # best one from the initial wav2vec2
]


def load_model(model_id, device=DEVICE):
    processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(model_id)  # type: ignore
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)  # type: ignore
    return model, processor


def decode_timestamps(processor, input_values, predicted_ids):
    predicted_ids = predicted_ids[0].tolist()
    duration_sec = input_values.shape[1] / processor.feature_extractor.sampling_rate

    ids_w_time = [
        (i / len(predicted_ids) * duration_sec, _id)
        for i, _id in enumerate(predicted_ids)
    ]

    current_token_id = processor.tokenizer.pad_token_id
    current_start_time = 0
    decoded_tokens_with_time = []
    for time, _id in ids_w_time:
        if current_token_id != _id:
            if current_token_id != processor.tokenizer.pad_token_id:
                decoded_tokens_with_time.append(
                    (processor.decode(current_token_id), current_start_time, time)
                )
            current_start_time = time
            current_token_id = _id

    return decoded_tokens_with_time


def wav2vec2_transcribe_from_array(
    model, processor, wav_array, include_timestamps=False
):
    """wav_array is an int16 16kHz wav pcm array or a normalized float32 16kHz pcm array"""

    if wav_array.dtype != np.float32:  # expects normalized float32 at 16 kHz
        wav_array = wav_array.astype(np.float32) / 32768

    inputs = processor(
        wav_array,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding="longest",
    ).input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs).logits

    predicted_ids = torch.argmax(outputs, dim=-1)

    if include_timestamps:
        return decode_timestamps(processor, inputs, predicted_ids)
    else:
        transcriptions = processor.batch_decode(predicted_ids)
        return transcriptions[0]


def wav2vec2_transcribe_from_file(
    model, processor, input_path: str, include_timestamps=False
):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768
    return wav2vec2_transcribe_from_array(
        model, processor, wav_array, include_timestamps
    )


def wav2vec2_transcribe_from_mic(model, processor, include_timestamps=False):
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return wav2vec2_transcribe_from_array(
        model, processor, wav_array, include_timestamps
    )


def main(args):
    if len(args) < 1:
        print(
            "Usage: python ./scripts/asr/wav2vec2.py <audio file> [model_id] [--timestamped]"
        )
        print("Usage: python ./scripts/asr/wav2vec2.py mic [model_id] [--timestamped]")
        return

    input_path = args[0]
    model_id = args[1] if len(args) > 1 and args[1] != "--timestamped" else MODEL_IDS[0]
    include_timestamps = "--timestamped" in args
    model, processor = load_model(model_id)
    if input_path == "mic":
        print(wav2vec2_transcribe_from_mic(model, processor, include_timestamps))
    else:
        print(
            wav2vec2_transcribe_from_file(
                model, processor, input_path, include_timestamps
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])
