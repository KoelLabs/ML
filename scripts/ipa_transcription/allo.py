#!/usr/bin/env python3

from typing import Mapping
from collections import abc

abc.Mapping = Mapping

from allosaurus.app import read_recognizer

from tempfile import NamedTemporaryFile
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.audio import audio_record_to_file
model_name = "eng2102"
model = read_recognizer(model_name)


def allosaurus_transcribe(input_path):
    return model.recognize(input_path)


def allosaurus_transcribe_timestamped(input_path):
    return list(
        map(
            lambda t: (t[2], t[1], t[0]),
            map(
                lambda x: x.split(" "),
                model.recognize(input_path, timestamp=True).split("\n"),
            ),
        )
    )


def allosaurus_transcribe_from_mic():
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return allosaurus_transcribe(f.name)


def allosaurus_transcribe_timestamped_from_mic():
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return allosaurus_transcribe_timestamped(f.name)


def main(args):
    if len(args) == 0:
        print(
            "Usage: python ./scripts/ipa_transcription/allo.py <audio file> [--timestamped]"
        )
        print("Usage: python ./scripts/ipa_transcription/allo.py mic [--timestamped]")
        return

    input_path = args[0]
    if input_path == "mic":
        if "--timestamped" in args:
            print(allosaurus_transcribe_timestamped_from_mic())
        else:
            print(allosaurus_transcribe_from_mic())
    else:
        if "--timestamped" in args:
            print(allosaurus_transcribe_timestamped(input_path))
        else:
            print(allosaurus_transcribe(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
