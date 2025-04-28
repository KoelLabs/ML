#!/usr/bin/env python3

# Audio processing utilities
# Convert between audio formats, play audio, record audio, etc.

import ffmpeg
import sys
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import requests
import os
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.load_secrets import load_secrets

load_secrets()

WAV_HEADER_SIZE = 44
TARGET_SAMPLE_RATE = 16000


def audio_convert(input_path, output_path, output_sample_rate=TARGET_SAMPLE_RATE):
    ffmpeg.input(input_path).output(output_path, ar=output_sample_rate).run()


def audio_bytes_to_wav_array(bytes, format, output_sample_rate=TARGET_SAMPLE_RATE):
    wav_bytes = (
        ffmpeg.input("pipe:0", format=format)
        .output("pipe:1", format="wav", ar=output_sample_rate)
        .run(input=bytes, capture_stdout=True)
    )
    return audio_bytes_to_array(wav_bytes[0], output_sample_rate)


def audio_array_to_bytes(array, sample_rate=TARGET_SAMPLE_RATE):
    with BytesIO() as f:
        wavfile.write(f, sample_rate, array)
        return f.getvalue()


def audio_array_to_wav_file(
    input_array, output_path, output_sample_rate=TARGET_SAMPLE_RATE
):
    wavfile.write(output_path, output_sample_rate, input_array)


def audio_resample(array, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    if src_sample_rate == target_sample_rate:
        return array
    return np.interp(
        np.linspace(
            0,
            len(array),
            int(len(array) * target_sample_rate / src_sample_rate),
        ),
        np.arange(len(array)),
        array,
    ).astype(np.int16)


def audio_bytes_to_array(
    data, src_sample_rate=None, target_sample_rate=TARGET_SAMPLE_RATE
):
    # TODO: rename to make clear this requires WAV format
    assert data[:4] == b"RIFF", "Not a WAV file, first 4 bytes are not RIFF: " + data[
        :4
    ].decode("utf-8")
    if src_sample_rate == None:
        # read 32 bit integer from bytes 25-28 in header
        src_sample_rate = int.from_bytes(data[24:28], byteorder="little")
        assert src_sample_rate == 44100
    # read bits per sample from bytes 35-36 in header
    bits_per_sample = int.from_bytes(data[34:36], byteorder="little")
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    # read number of channels from bytes 23-24 in header
    num_channels = int.from_bytes(data[22:24], byteorder="little")
    data = data[WAV_HEADER_SIZE:]
    audio = np.frombuffer(data, dtype=dtype).astype(np.int16)
    # average in chunks of num_channels
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels)
        audio = np.mean(audio, axis=1).astype(np.int16)
    audio = audio_resample(audio, src_sample_rate, target_sample_rate)
    return audio


def audio_dual_channel_to_mono(input_array):
    if input_array.ndim == 2 and input_array.shape[1] == 2:
        return np.mean(input_array, axis=1).astype(np.int16)
    return input_array


def audio_file_to_array(input_path, desired_sample_rate=TARGET_SAMPLE_RATE):
    rate, data = wavfile.read(input_path)
    data = audio_dual_channel_to_mono(data)
    data = audio_resample(data, rate, desired_sample_rate)
    return data


def audio_array_play(input_array, sample_rate=TARGET_SAMPLE_RATE):
    sd.play(input_array, sample_rate)
    sd.wait()


def audio_wav_file_play(input_path, start_sec=None, end_sec=None):
    print(start_sec, end_sec)
    rate, data = wavfile.read(input_path)
    start = int(float(start_sec) * rate) if start_sec else 0
    end = int(float(end_sec) * rate) if end_sec else len(data)
    data = data[start:end]
    audio_array_play(data, rate)


def audio_wav_file_crop(input_path, start_sec, end_sec, output_path):
    rate, data = wavfile.read(input_path)
    start = int(float(start_sec) * rate)
    end = int(float(end_sec) * rate)
    data = data[start:end]
    audio_array_to_wav_file(data, output_path, rate)


def audio_record_to_array(output_sample_rate=TARGET_SAMPLE_RATE):
    print("Recording, please speak and press Ctrl+C when done")
    samples = np.array([], dtype=np.int16)
    try:
        with sd.InputStream(
            channels=1, dtype="int16", samplerate=output_sample_rate
        ) as s:
            while True:
                sample, _ = s.read(output_sample_rate)
                samples = np.append(samples, sample.reshape(-1))
    except KeyboardInterrupt:
        print("Recording stopped")
    return samples


def audio_record_to_file(output_path, output_sample_rate=TARGET_SAMPLE_RATE):
    samples = audio_record_to_array(output_sample_rate)
    audio_array_to_wav_file(samples, output_path, output_sample_rate)


def audio_file_from_text(text, output_path):
    assert os.environ.get("GOOEY_API_KEY"), "GOOEY_API_KEY environment variable not set"

    response = requests.post(
        "https://api.gooey.ai/v2/TextToSpeech",
        headers={
            "Authorization": "bearer " + os.environ["GOOEY_API_KEY"],
        },
        json={
            "text_prompt": text,
            "tts_provider": "AZURE_TTS",
            "azure_voice_name": "en-US-KaiNeural",
        },
    )
    assert response.ok, response.content

    result = response.json()
    audio_url = result["output"]["audio_url"]
    response = requests.get(audio_url)
    with open(output_path, "wb") as f:
        f.write(response.content)


def main(args):
    if args[0] == "record":
        audio_record_to_file(args[1])
    elif args[0] == "convert":
        if len(args) > 3:
            audio_convert(args[1], args[2], int(args[3]))
        else:
            audio_convert(args[1], args[2])
    elif args[0] == "play":
        start, end = None, None
        if len(args) > 2:
            start, end = args[2].split(":")
            start, end = float(start), float(end)
        audio_wav_file_play(args[1], start, end)
    elif args[0] == "crop":
        audio_wav_file_crop(args[1], float(args[2]), float(args[3]), args[4])
    elif args[0] == "text":
        audio_file_from_text(args[1], args[2])
    else:
        print("Invalid command")
        print("Usage: python ./scripts/core/audio.py record <output_wav_path>")
        print("Usage: python ./scripts/core/audio.py play <input_wav_path> [start:end]")
        print(
            "Usage: python ./scripts/core/audio.py convert <input_path> <output_path>"
        )
        print(
            "Usage: python ./scripts/core/audio.py convert <input_path> <output_path> <output_sample_rate>"
        )
        print("Usage: python ./scripts/core/audio.py text <text> <output_wav_path>")
        print(
            "Usage: python ./scripts/core/audio.py crop <input_wav_path> <start> <end> <output_wav_path>"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
