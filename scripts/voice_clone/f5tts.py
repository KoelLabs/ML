#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_array_to_wav_file, audio_array_play

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from f5_tts.api import F5TTS

f5tts = F5TTS()


def f5tts_clone_voice(
    voice_reference_audio_path, voice_reference_text: str, target_text: str
):
    wav, sr, spect = f5tts.infer(
        ref_file=voice_reference_audio_path,
        ref_text=voice_reference_text,
        gen_text=target_text,
        file_wave=None,
        seed=42,
    )
    return wav, sr, spect


def main(args):
    try:
        assert len(args) >= 3, "Not enough arguments"
        voice_ref = args[0]
        text_ref = args[1]
        target_text = args[2]
        save_file = args[3] if len(args) >= 4 else None

        wav, sr, spect = f5tts_clone_voice(voice_ref, text_ref, target_text)
        if save_file:
            audio_array_to_wav_file(wav, save_file)
        else:
            audio_array_play(wav)
    except Exception as e:
        print(e)
        print(
            "Usage: python ./scripts/voice_clone/f5tts.py <ref_wav_path> <ref_text> <target_text> [<save_file>]"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

# example: python ./scripts/voice_clone/f5tts.py data/ArunaSpeech/L1Suitcase.wav "There is a man and a woman walking with green suitcases and then suddenly they bump into each other. The man and woman appear very dizzy and when they stand up they grab their suitcases and walk away. But then the man arrives at a hotel and sees that there is a red dress in his suitcase and he looks very confused and embarrassed. And likewise the woman opens her suitcase and finds a striped yellow and black tie. The woman also seems very surprised." "Alex is awesome."
