#!/usr/bin/env python3

import ffmpeg
import sys
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))


def audio_convert(input_path, output_path, output_sample_rate=16000):
    ffmpeg.input(input_path).output(output_path, ar=output_sample_rate).run()


def audio_array_to_wav_file(input_array, output_path, output_sample_rate=16000):
    wavfile.write(output_path, output_sample_rate, input_array)


def audio_array_play(input_array, sample_rate=16000):
    sd.play(input_array, sample_rate)
    sd.wait()


def audio_wav_file_play(input_path):
    rate, data = wavfile.read(input_path)
    audio_array_play(data, rate)


def audio_record_to_array(output_sample_rate=16000):
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


def audio_record_to_file(output_path, output_sample_rate=16000):
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
        audio_wav_file_play(args[1])
    elif args[0] == "text":
        audio_file_from_text(args[1], args[2])
    else:
        print("Invalid command")
        print("Usage: python ./scripts/audio.py record <output_wav_path>")
        print("Usage: python ./scripts/audio.py play <input_wav_path>")
        print("Usage: python ./scripts/audio.py convert <input_path> <output_path>")
        print(
            "Usage: python ./scripts/audio.py convert <input_path> <output_path> <output_sample_rate>"
        )
        print("Usage: python ./scripts/audio.py text <text> <output_wav_path>")


if __name__ == "__main__":
    main(sys.argv[1:])
