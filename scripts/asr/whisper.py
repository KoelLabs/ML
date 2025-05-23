#!/usr/bin/env python3

import torch
from faster_whisper import WhisperModel

import sys
import os
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_file, audio_array_to_wav_file

# model_size = "large-v3"
model_size = "small"

if torch.cuda.is_available():
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    # Run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")


def whisper_transcribe(input_path):
    return model.transcribe(input_path, language="en")


def whisper_transcribe_timestamped(input_path):
    return model.transcribe(input_path, language="en", word_timestamps=True)


def whisper_transcribe_from_array(wav_array):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)
        return whisper_transcribe(f.name)


def whisper_transcribe_timestamped_from_array(wav_array):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)
        return whisper_transcribe_timestamped(f.name)


def whisper_transcribe_from_mic():
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return whisper_transcribe(f.name)


def whisper_transcribe_timestamped_from_mic():
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return whisper_transcribe_timestamped(f.name)


def display_whisper_result(segments, info, timestamped):
    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    if timestamped:
        for segment in segments:
            for word in segment.words:
                print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    else:
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def main(args):
    timestamped = len(args) > 1 and args[1] == "--timestamped"
    if args[0] == "mic":
        if timestamped:
            segments, info = whisper_transcribe_timestamped_from_mic()
        else:
            segments, info = whisper_transcribe_from_mic()
        display_whisper_result(segments, info, timestamped)
    else:
        try:
            input_path = args[0]

            if timestamped:
                segments, info = whisper_transcribe_timestamped(input_path)
            else:
                segments, info = whisper_transcribe(input_path)

            display_whisper_result(segments, info, timestamped)
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/asr/whisper.py mic [--timestamped]")
            print(
                "Usage: python ./scripts/asr/whisper.py <input_wav_path> [--timestamped]"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
