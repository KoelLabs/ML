#!/usr/bin/env python3

import sherpa_ncnn

import wave
import numpy as np

import sys

import sounddevice as sd


def create_recognizer(folder):
    recognizer = sherpa_ncnn.Recognizer(
        tokens=f"{folder}/tokens.txt",
        encoder_param=f"{folder}/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin=f"{folder}/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param=f"{folder}/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin=f"{folder}/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param=f"{folder}/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin=f"{folder}/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    return recognizer


MODEL_IDS = [
    "./models/sherpa-models/sherpa-ncnn-pruned-transducer-stateless7-streaming-id",  # Chinese IPA
    "./models/sherpa-models/sherpa-ncnn-streaming-zipformer-20M-2023-02-17",  # English
]


def ncnn_transcribe(input_path, model_id=MODEL_IDS[0]):
    recognizer = create_recognizer(model_id)

    with wave.open(input_path) as f:
        assert f.getframerate() == recognizer.sample_rate, (
            f.getframerate(),
            recognizer.sample_rate,
        )
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

    recognizer.accept_waveform(recognizer.sample_rate, samples_float32)

    tail_paddings = np.zeros(int(recognizer.sample_rate * 0.5), dtype=np.float32)
    recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)

    recognizer.input_finished()

    return recognizer.text


def ncnn_transcribe_from_mic(model_id=MODEL_IDS[0]):
    recognizer = create_recognizer(model_id)

    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    print("Started! Please speak")
    last_result = ""
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)
            result = recognizer.text
            if last_result != result:
                last_result = result
                yield result


def main(args):
    if args[0] == "mic":
        for result in ncnn_transcribe_from_mic():
            print(result)
    else:
        try:
            input_path = args[0]
            print(ncnn_transcribe(input_path))
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/ipa_transcription/sherpa.py mic")
            print(
                "Usage: python ./scripts/ipa_transcription/sherpa.py <input_wav_path>"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
