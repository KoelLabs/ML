#!/usr/bin/env python3

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import os
import sys
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_file
from core.secrets import load_secrets

load_secrets()

# Instantiates a client
client = SpeechClient()


def google_transcribe(input_path):
    # Reads a file as bytes
    with open(input_path, "rb") as f:
        audio_content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="short",  # Chosen model
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{os.environ.get('GOOGLE_PROJECT_ID')}/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    # Transcribes the audio into text
    return client.recognize(request=request)


def google_transcribe_from_mic():
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return google_transcribe(f.name)


def main(args):
    if args[0] == "mic":
        response = google_transcribe_from_mic()
        for result in response.results:
            print(f"Transcript: {result.alternatives[0].transcript}")
    else:
        try:
            input_path = args[0]
            response = google_transcribe(input_path)
            for result in response.results:
                print(f"Transcript: {result.alternatives[0].transcript}")
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/asr/google_speech.py mic")
            print("Usage: python ./scripts/asr/google_speech.py <input_wav_path>")


if __name__ == "__main__":
    main(sys.argv[1:])
