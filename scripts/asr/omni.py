#!/usr/bin/env python3

# Facebook Omnilingual ASR: https://github.com/facebookresearch/omnilingual-asr

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_file, audio_array_to_wav_file, TARGET_SAMPLE_RATE

from tempfile import NamedTemporaryFile

from omnilingual_asr.models.inference.pipeline import (
    ASRInferencePipeline,
    ContextExample,
)

USE_LLM = True
USE_CONTEXT = False
SIZE = "7B"  # 300M, 1B, 3B, 7B
assert (
    not USE_CONTEXT or SIZE == "7B"
), "only the 7B variant supports in-context learning"

pipeline = ASRInferencePipeline(
    model_card=f"omniASR_{'LLM' if USE_LLM else 'CTC'}_{SIZE}{'_ZS' if USE_CONTEXT else ''}"
)


def omni_transcribe_from_file(
    input_path: str,
    examples: list[
        tuple[str, str]
    ] = [],  # (audio_file_path, ground_truth_transcription)
    languages: list[str] = ["eng_Latn"],
):
    if len(examples) > 0:
        assert USE_LLM, "cannot use few shot prompting without LLM"
        assert USE_CONTEXT, "cannot use few shot prompting without ZS model type"
        context_examples = [
            ContextExample(audio_path, transcript)
            for audio_path, transcript in examples
        ]
        return pipeline.transcribe_with_context(
            [input_path], context_examples=[context_examples], batch_size=1
        )[0]
    else:
        return pipeline.transcribe([input_path], lang=languages, batch_size=1)[0]


def omni_transcribe_from_array(
    wav_array,
    examples: list[
        tuple[str, str]
    ] = [],  # (audio_file_path, ground_truth_transcription)
    languages: list[str] = ["eng_Latn"],
):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)
        return omni_transcribe_from_file(f.name, examples=examples, languages=languages)


def omni_transcribe_from_mic(
    examples: list[
        tuple[str, str]
    ] = [],  # (audio_file_path, ground_truth_transcription)
    languages: list[str] = ["eng_Latn"],
):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return omni_transcribe_from_file(f.name, examples=examples, languages=languages)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/omni.py <audio file>")
        print("Usage: python ./scripts/asr/omni.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(omni_transcribe_from_mic())
    else:
        print(omni_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
