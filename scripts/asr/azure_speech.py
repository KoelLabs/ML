#!/usr/bin/env python3

import azure.cognitiveservices.speech as speechsdk
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.load_secrets import load_secrets

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

    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    if not speech_recognition_result:
        return ""
    assert (
        speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech
    ), speech_recognition_result
    return speech_recognition_result.text


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
