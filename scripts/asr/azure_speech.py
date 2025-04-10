#!/usr/bin/env python3

import azure.cognitiveservices.speech as speechsdk

import os
import sys
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.load_secrets import load_secrets
from core.audio import audio_array_to_wav_file

load_secrets()

speech_config = speechsdk.SpeechConfig(
    subscription=os.environ.get("AZURE_SPEECH_KEY"),
    region=os.environ.get("AZURE_SPEECH_REGION"),
)


def azure_transcribe(input_path):
    audio_config = speechsdk.AudioConfig(filename=input_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # listen for all utterances and return them as a list
    speech_recognition_results = []
    while True:
        speech_recognition_result = speech_recognizer.recognize_once_async().get()
        if (
            not speech_recognition_result
            or speech_recognition_result.reason
            != speechsdk.ResultReason.RecognizedSpeech
        ):
            break
        speech_recognition_results.append(speech_recognition_result.text)
    return speech_recognition_results


def azure_transcribe_from_array(input_array):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(input_array, f.name)
        return azure_transcribe(f.name)


def azure_transcribe_from_mic():
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    if not speech_recognition_result:
        return ""
    assert (
        speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech
    ), speech_recognition_result
    return speech_recognition_result.text


def main(args):
    if args[0] == "mic":
        print(azure_transcribe_from_mic())
    else:
        try:
            input_path = args[0]
            print(azure_transcribe(input_path))
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/asr/azure_speech.py mic")
            print("Usage: python ./scripts/asr/azure_speech.py <input_wav_path>")


if __name__ == "__main__":
    main(sys.argv[1:])
