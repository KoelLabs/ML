#!/usr/bin/env python3

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.streaming import (
    Source,
    run_file_source,
    run_array_source,
    run_microphone_source,
)

import numpy as np

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def calculate_cnn_window(model: Wav2Vec2ForCTC):
    receptive_field = 1
    stride = 1
    for conv_layer in model.wav2vec2.feature_extractor.conv_layers:
        assert hasattr(conv_layer, "conv")
        conv = conv_layer.conv
        assert isinstance(conv, torch.nn.Conv1d)
        receptive_field += (conv.kernel_size[0] - 1) * stride
        stride *= conv.stride[0]
    return receptive_field, stride


# ============================ Transcription Utils ============================
def decode_timestamps(
    predicted_ids,
    processor: Wav2Vec2Processor,
    duration_per_id_sec=0.020,
    time_offset=0,
):
    ids_w_time = [
        (time_offset + i * duration_per_id_sec, _id)
        for i, _id in enumerate(predicted_ids)
    ]

    current_phoneme_id = processor.tokenizer.pad_token_id  # type: ignore
    current_start_time = 0
    phonemes_with_time = []
    for time, _id in ids_w_time:
        if current_phoneme_id != _id:
            if current_phoneme_id != processor.tokenizer.pad_token_id:  # type: ignore
                phonemes_with_time.append(
                    (
                        processor.decode(current_phoneme_id),
                        current_start_time,
                        time,
                    )
                )
            current_start_time = time
            current_phoneme_id = _id

    return phonemes_with_time


def transcribe(
    wav_array: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    duration_per_id_sec=0.020,
    time_offset=0,
):
    processed = processor(
        wav_array,
        sampling_rate=processor.feature_extractor.sampling_rate,  # type: ignore
        return_tensors="pt",
        padding=True,
    )
    processed["input_values"] = processed["input_values"].to(model.device)
    processed["attention_mask"] = processed["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(**processed).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    return decode_timestamps(predicted_ids, processor, duration_per_id_sec, time_offset)


# ============================= Streaming Methods =============================
def stream_naive(
    ws: Source,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    duration_per_id_sec=0.020,
):
    buffer = np.array([], dtype=np.float32)

    while True:
        data = ws.receive(format="float32")
        is_stop = isinstance(data, str) and data == "stop"
        if not is_stop:
            buffer = np.concatenate([buffer, data])

        transcript = transcribe(
            buffer, processor, model, duration_per_id_sec=duration_per_id_sec
        )
        ws.send(transcript)
        yield "".join(p for p, _, _ in transcript)

        if is_stop:
            break


def stream_naive_chunked(
    ws: Source,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    chunk_size=4_000,
    duration_per_id_sec=0.020,
):
    buffer = np.array([], dtype=np.float32)
    full_transcript = []
    total_time = 0

    while True:
        data = ws.receive(format="float32")
        is_stop = isinstance(data, str) and data == "stop"
        if not is_stop:
            buffer = np.concatenate([buffer, data])

        if is_stop or len(buffer) >= chunk_size:
            chunk_transcript = transcribe(
                buffer,
                processor,
                model,
                duration_per_id_sec=duration_per_id_sec,
                time_offset=total_time,
            )
            total_time += len(buffer) / processor.feature_extractor.sampling_rate  # type: ignore
            full_transcript.extend(chunk_transcript)
            ws.send(full_transcript)
            yield "".join(p for p, _, _ in full_transcript)
            buffer = np.array([], dtype=np.float32)

        if is_stop:
            break


# ==================================== CLI ====================================
def main(args):
    if len(args) < 1:
        print(
            "Usage: python ./scripts/ipa_transcription/wav2vec2_streaming.py <source> <method> [--slow-down-to-realtime]"
        )
        print(
            "Example: python ./scripts/ipa_transcription/wav2vec2_streaming.py mic stream_naive_chunked"
        )
        print(
            "Example: python ./scripts/ipa_transcription/wav2vec2_streaming.py timit stream_naive"
        )
        print(
            "Example: python ./scripts/ipa_transcription/wav2vec2_streaming.py data/ExamplesWithComments/TIMIT_sample_0.wav stream_naive_chunked --slow-down-to-realtime"
        )
        return

    # Load model
    print("Loading model...", end=" ", flush=True)
    model_name = "KoelLabs/xlsr-english-01"
    model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(model_name)  # type: ignore
    model.to(DEVICE)  # type: ignore
    processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(model_name)  # type: ignore
    receptive_field, stride = calculate_cnn_window(model)
    duration_per_id_sec = stride / processor.feature_extractor.sampling_rate  # type: ignore
    print("Done!")

    # Define source
    print("Connecting to audio source...", end=" ", flush=True)
    source_type = args[0]
    slow_down_to_realtime = "--slow-down-to-realtime" in args
    if source_type == "mic":
        ws = run_microphone_source()
    elif source_type == "timit":
        from data_loaders.TIMIT import TIMITDataset

        dataset = TIMITDataset(split="test")
        sample = dataset[0]
        audio: np.ndarray = sample[1]  # type: ignore
        ws = run_array_source(audio, slow_down_to_realtime=slow_down_to_realtime)
    else:
        audio_path = source_type
        ws = run_file_source(audio_path, slow_down_to_realtime=slow_down_to_realtime)
    print("Done!")

    # Run streaming
    print("---")
    assert args[1].startswith("stream_")
    start = time.perf_counter()
    method = globals()[args[1]]
    try:
        for update in method(
            ws, processor, model, duration_per_id_sec=duration_per_id_sec
        ):
            print("\r" + update, end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    print("---")
    print(f"Overall Time: {time.perf_counter() - start:.6f} seconds")
    (
        audio_duration,
        stream_duration,
        computation_time,
        warmup_latency,
        average_first_guess_latency,
        average_final_guess_latency,
    ) = ws.evaluate()
    print("---")
    print(f"Audio Duration: {audio_duration:.6f} seconds")
    print(f"Stream Duration: {stream_duration:.6f} seconds")
    print(f"Computation Time: {computation_time:.6f} seconds")
    print("---")
    print(f"Warmup Latency: {warmup_latency:.6f} seconds")
    print(f"Average Guess Latency: {average_first_guess_latency:.6f} seconds")
    print(f"Average Final Latency: {average_final_guess_latency:.6f} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
