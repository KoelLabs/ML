import sherpa_ncnn

import wave
import numpy as np

FOLDER = "./sherpa-models/sherpa-ncnn-pruned-transducer-stateless7-streaming-id"  # Chinese IPA
# FOLDER = "./sherpa-models/sherpa-ncnn-streaming-zipformer-20M-2023-02-17"  # English


def create_recognizer():
    recognizer = sherpa_ncnn.Recognizer(
        tokens=f"{FOLDER}/tokens.txt",
        encoder_param=f"{FOLDER}/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin=f"{FOLDER}/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param=f"{FOLDER}/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin=f"{FOLDER}/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param=f"{FOLDER}/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin=f"{FOLDER}/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    return recognizer


def main():
    print("Started! Please speak")
    recognizer = create_recognizer()

    with wave.open("./alexIsConfused16kHz.wav") as f:
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

    print(recognizer.text)


if __name__ == "__main__":
    main()
