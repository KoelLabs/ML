#!/usr/bin/env python3

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.gcs import get_gcs_uri, create_temp_object_from_bytes
from core.audio import audio_record_to_array, audio_array_to_bytes, TARGET_SAMPLE_RATE
from core.load_secrets import load_secrets

load_secrets()

# Instantiates a client
client = SpeechClient()


def google_transcribe_from_bytes(input_bytes, longer_than_one_minute=True, timeout=120):
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="long" if longer_than_one_minute else "short",  # Chosen model
    )
    recognizer = (
        f"projects/{os.environ.get('GOOGLE_PROJECT_ID')}/locations/global/recognizers/_"
    )

    if longer_than_one_minute:
        with create_temp_object_from_bytes(input_bytes) as key:
            url = get_gcs_uri(key)
            file_metadata = cloud_speech.BatchRecognizeFileMetadata(
                uri=url,
            )

            request = cloud_speech.BatchRecognizeRequest(
                recognizer=recognizer,
                config=config,
                files=[file_metadata],
                recognition_output_config=cloud_speech.RecognitionOutputConfig(
                    inline_response_config=cloud_speech.InlineOutputConfig(),
                ),
            )

            operation = client.batch_recognize(request=request)
            response = operation.result(timeout=timeout)
            return response.results[url].transcript  # type: ignore
    else:
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=input_bytes,
        )

        # Transcribes the audio into text
        return client.recognize(request=request)


def google_transcribe_from_file(input_path, longer_than_one_minute=True, timeout=120):
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    return google_transcribe_from_bytes(
        input_bytes, longer_than_one_minute=longer_than_one_minute, timeout=timeout
    )


def google_transcribe_from_array(input_array, force_long_model=False, min_timeout=120):
    duration = len(input_array) / TARGET_SAMPLE_RATE
    timeout = int(max(min_timeout, duration * 2))
    input_bytes = audio_array_to_bytes(input_array)
    return google_transcribe_from_bytes(
        input_bytes,
        longer_than_one_minute=force_long_model or duration > 60,
        timeout=timeout,
    )


def google_transcribe_from_mic():
    input_array = audio_record_to_array()
    return google_transcribe_from_array(input_array)


def main(args):
    if args[0] == "mic":
        response = google_transcribe_from_mic()
        for result in response.results:  # type: ignore
            print(f"Transcript: {result.alternatives[0].transcript}")
    else:
        try:
            input_path = args[0]
            response = google_transcribe_from_file(input_path)
            for result in response.results:  # type: ignore
                print(f"Transcript: {result.alternatives[0].transcript}")
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/asr/google_speech.py mic")
            print("Usage: python ./scripts/asr/google_speech.py <input_wav_path>")


if __name__ == "__main__":
    main(sys.argv[1:])
