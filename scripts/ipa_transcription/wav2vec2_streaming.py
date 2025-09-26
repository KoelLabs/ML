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
    receptive=400,
    duration_per_id_sec=0.020,
    time_offset=0,
):
    processed = processor(
        wav_array,
        sampling_rate=processor.feature_extractor.sampling_rate,  # type: ignore
        return_tensors="pt",
        padding="max_length",
        max_length=receptive,
    )
    processed["input_values"] = (
        processed["input_values"].type(torch.float32).to(model.device)
    )
    processed["attention_mask"] = processed["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(**processed).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    return decode_timestamps(predicted_ids, processor, duration_per_id_sec, time_offset)


def extract_features_only(processor, model, receptive, audio: np.ndarray):
    """Extract CNN features and project to encoder hidden size (transformer-ready), return hidden states."""
    inputs = processor(
        audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=receptive,
    )

    input_values = inputs.input_values.type(torch.float32).to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        extract_features = model.wav2vec2.feature_extractor(input_values)  # (B, C, T')
        extract_features = extract_features.transpose(1, 2)  # (B, T', C)
        attention_masks = model.wav2vec2._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )
        hidden_states, _ = model.wav2vec2.feature_projection(extract_features)
        hidden_states = model.wav2vec2._mask_hidden_states(
            hidden_states, attention_mask=attention_masks
        )
    return hidden_states, attention_masks


def run_transformer_on_features(
    model, features: torch.Tensor, attention_mask: torch.LongTensor
):
    """Run transformer from features and get predicted ids"""
    encoder_outputs = model.wav2vec2.encoder(features, attention_mask=attention_mask)
    hidden_states = model.lm_head(encoder_outputs[0])
    predicted_ids = torch.argmax(hidden_states, dim=-1)[0].tolist()
    return predicted_ids


class WelfordsAccumulator:
    """Numerically stable online mean and standard deviation calculation"""

    def __init__(self):
        self.n_a, self.avg_a, self.M2_a = 0, 0, 0

    def update_stats(self, sample: np.ndarray):
        """Returns the accumulated mean and standard deviation given the new samples"""

        n_b, avg_b, M2_b = (
            len(sample),
            sample.mean(),
            sample.std(),  # sample.var() * (len(sample) - 1)
        )

        n = self.n_a + n_b
        delta = avg_b - self.avg_a
        M2 = self.M2_a + M2_b + delta**2 * self.n_a * n_b / n
        var_ab = M2 / (n - 1)

        self.n_a, self.avg_a, self.M2_a = (
            n,
            (self.avg_a * self.n_a + avg_b * n_b) / n,
            M2,
        )

        return self.avg_a, np.sqrt(var_ab)


# ============================= Streaming Methods =============================
def stream_naive(
    ws: Source,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    receptive=400,
    stride=320,
    duration_per_id_sec=0.020,
):
    buffer = np.array([], dtype=np.float32)
    last_length = 0
    accumulation_size = receptive

    while True:
        data = ws.receive(format="float32")
        is_stop = isinstance(data, str) and data == "stop"
        if not is_stop:
            buffer = np.concatenate([buffer, data])
            if len(buffer) - last_length < accumulation_size:
                continue
            last_length = len(buffer)

        start = time.perf_counter()  # adaptive accumulation size

        transcript = transcribe(
            buffer, processor, model, receptive, duration_per_id_sec=duration_per_id_sec
        )
        ws.send(transcript)
        yield "".join(p for p, _, _ in transcript)

        # adaptive accumulation size
        seconds = time.perf_counter() - start
        accumulation_size = seconds * processor.feature_extractor.sampling_rate  # type: ignore

        if is_stop:
            break


def stream_naive_chunked(
    ws: Source,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    receptive=400,
    stride=320,
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
                receptive,
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


def stream_cnn_chunked_transformer(
    ws: Source,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    receptive=400,
    stride=320,
    duration_per_id_sec=0.020,
):
    chunk_size = receptive * np.dtype(np.float32).itemsize  # 1600 bytes
    stride_interval = stride * np.dtype(np.float32).itemsize  # 1280 bytes
    overlap_interval = chunk_size - stride_interval  # 320 bytes (80 samples)
    buffer = b""
    feature_list = []
    attention_list = []
    time_offset = 0  # NOTE: this value should be getting updated if you don't run the full audio to transformer
    full_transcription = []
    current_length = 0
    last_length = 0
    accumulation_size = chunk_size
    num_new_features = 0

    # We'll manually handle normalization using Welford's since we are chunking up the audio
    processor.feature_extractor.do_normalize = False  # type: ignore
    stats_accumulator = WelfordsAccumulator()

    while True:
        data: bytes = ws.receive()  # type: ignore
        is_stop = isinstance(data, str) and data == "stop"
        if not is_stop:
            buffer += data
            current_length += len(data)

        start = time.perf_counter()
        if len(buffer) > chunk_size:
            num_chunks = (len(buffer) - chunk_size) // stride_interval + 1
            num_full_chunk_bytes = num_chunks * stride_interval + overlap_interval
            audio = np.frombuffer(buffer[:num_full_chunk_bytes], dtype=np.float32)
            mean, std = stats_accumulator.update_stats(audio)
            audio = (audio - mean) / std
            buffer = buffer[num_full_chunk_bytes - overlap_interval :]

            features, attention_mask = extract_features_only(
                processor, model, num_chunks * stride + receptive - stride, audio
            )
            assert num_chunks == features.shape[1]
            num_new_features += features.shape[1]
            feature_list.append(features)
            attention_list.append(attention_mask)

        if not is_stop and current_length - last_length < accumulation_size:
            continue
        last_length = current_length

        # accumulate features based on dynamic accumulation size
        if num_new_features >= 1:
            predicted_ids = run_transformer_on_features(
                model,
                torch.cat(feature_list, dim=1),
                torch.cat(attention_list, dim=1),  # type: ignore
            )
            full_transcription = decode_timestamps(
                predicted_ids, processor, duration_per_id_sec, time_offset
            )

            ws.send(full_transcription)
            yield "".join(p for p, _, _ in full_transcription)

            # calculate new accumulation size
            seconds = time.perf_counter() - start
            accumulation_size = seconds * processor.feature_extractor.sampling_rate * np.dtype(np.float32).itemsize  # type: ignore
            num_new_features = 0

        if is_stop:
            break

    processor.feature_extractor.do_normalize = True  # type: ignore


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
            "Example: python ./scripts/ipa_transcription/wav2vec2_streaming.py timit stream_cnn_chunked_transformer"
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
    receptive, stride = calculate_cnn_window(model)
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

    # Warmup model
    transcribe(
        torch.rand((receptive,)).numpy(),
        processor,
        model,
        receptive=receptive,
        duration_per_id_sec=duration_per_id_sec,
        time_offset=0,
    )

    # Run streaming
    print("---")
    assert args[1].startswith("stream_")
    start = time.perf_counter()
    method = globals()[args[1]]
    try:
        for update in method(
            ws,
            processor,
            model,
            receptive=receptive,
            stride=stride,
            duration_per_id_sec=duration_per_id_sec,
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
