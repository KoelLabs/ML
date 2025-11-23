#!/usr/bin/env python3

import os
import sys
import requests
from time import sleep
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.s3 import get_presigned_url, create_temp_object
from core.audio import audio_record_to_file, audio_array_to_wav_file
from core.load_secrets import load_secrets

load_secrets()

API_URL = os.environ.get("SALAD_TRANSCRIBE_API_URL")
API_KEY = os.environ.get("SALAD_API_KEY")

assert API_URL, "SALAD_TRANSCRIBE_API_URL is not set"
assert API_KEY, "SALAD_API_KEY is not set"


def salad_transcribe(audio_path):
    print(f"Status: uploading audio")
    with create_temp_object(audio_path) as key:  # type: ignore
        url = get_presigned_url(key, expiration=7200)
        response = requests.post(
            API_URL,  # type: ignore
            headers={
                "Salad-Api-Key": API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input": {
                    "url": url,
                    "return_as_file": True,
                    "language_code": "en",
                    "translate": "to_eng",
                    "sentence_level_timestamps": True,
                    "word_level_timestamps": True,
                    "diarization": True,
                    "sentence_diarization": True,
                    # "srt": True,
                    # "summarize": 100,
                    # "llm_translation": "german, italian, french, spanish, english, portuguese, hindi, thai",
                    # "srt_translation": "german, italian, french, spanish, english, portuguese, hindi, thai",
                    # "custom_vocabulary": "terms devided by comma",
                }
            },
        )
        response.raise_for_status()
        job_id = response.json()["id"]

        status = "pending"
        print(f"Status: {status}")
        while status not in ["succeeded", "failed"]:
            response = requests.get(
                f"{API_URL}/{job_id}",
                headers={"Salad-Api-Key": API_KEY},
            )
            response.raise_for_status()
            response = response.json()
            status = response["status"]
            print(f"Status: {status}")
            sleep(0.5)

        if status == "succeeded":
            output = response["output"]  # type: ignore
            print(output)
            response = requests.get(output["url"])
            response.raise_for_status()
            output = response.json()
            return output
        else:
            return None


def salad_transcribe_from_array(input_array):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(input_array, f.name)
        return salad_transcribe(f.name)


def salad_transcribe_from_mic():
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return salad_transcribe(f.name)


def display_salad_output(output):
    print("----------------")
    print("Duration (seconds):", output["duration_in_seconds"])
    print("Processing time (seconds):", output["processing_time"])
    print()
    print("Transcript:", output["text"])
    print()
    print("Word segments:")
    for word in output["word_segments"]:
        print(word["word"], word["timestamp"], word["speaker"], sep="\t\t")
    print()
    print("Sentence segments:")
    for sentence in output["sentence_level_timestamps"]:
        print(sentence["text"], sentence["timestamp"], sentence["speaker"], sep="\t\t")


def main(args):
    if args[0] == "mic":
        response = salad_transcribe_from_mic()
        display_salad_output(response)
    else:
        try:
            input_path = args[0]
            response = salad_transcribe(input_path)
            display_salad_output(response)
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/asr/salad.py mic")
            print("Usage: python ./scripts/asr/salad.py <input_wav_path>")


if __name__ == "__main__":
    main(sys.argv[1:])
