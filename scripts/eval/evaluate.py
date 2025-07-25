#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.ipa import simplify_ipa
from eval.metrics import fer, per
from forced_alignment.common import IPA_SYMBOLS, group_phonemes


def preprocess_ipa(ipa_string):
    """Make ipa_string processable by panphon and standardize representation for fair comparison."""
    # remove <unk> tokens
    ipa_string = ipa_string.replace("<unk>", "")

    # remove white space
    ipa_string = ipa_string.replace(" ", "")

    # some combined symbols are only supported by ipapy and panphon when separated:
    # - replace syllabic ŋ (syllabic marker above) with syllabic ŋ (syllabic marker below): U+014B U+030D => U+014B U+0329
    # - exteme care is needed with ĩ (U+0129) and ĩ (U+0069 U+0303): they look identical in most fonts but are different: ĩ should map to ɪ̰, while ĩ is just the nasal i as the already separate symbols indicate
    # - ä (U+00E4) and ä (U+0061 U+0308) mean the same and look the same, but ipapy and panphon need them as separate symbols
    ipa_string = ipa_string.replace("ŋ̍", "ŋ̩").replace("ĩ", "ɪ̰").replace("ä", "ä")

    # standardize representations and identify phonemes not in panphon (IPA_SYMBOLS)
    phonemes = group_phonemes(ipa_string)

    # panphon has some extra symbols that must be separated:
    # - r-colored schwas: ɚ (U+025A) => ə (U+0259) ˞ (U+02DE); ɝ (U+025D) => ɜ (U+025C) ˞ (U+02DE)
    # - ç (U+00E7) => ç (U+0063 U+0327) which are tricky since they look the same
    # - we also remove lonesome ties/syllabic markers that are not part of any symbols
    phonemes = [
        p.replace("ɚ", "ə˞").replace("ɝ", "ɜ˞").replace("ç", "ç")
        for p in phonemes
        if p not in ["͡", "̩", "̃", "ʲ", "̥"]
    ]

    # print warning about phonemes not in panphon if any
    unsupported_phonemes = set(phonemes) - set(IPA_SYMBOLS)
    if len(unsupported_phonemes) > 0:
        print(
            f"Warning: unsupported phonemes found in input: {unsupported_phonemes}.",
            file=sys.stderr,
        )

    # remove phonemes not in panphon
    return "".join(p for p in phonemes if p in IPA_SYMBOLS)


def evaluate(label, predicted, simplify=False):
    label_sequence = simplify_ipa(label) if simplify else preprocess_ipa(label)
    pred_sequence = simplify_ipa(predicted) if simplify else preprocess_ipa(predicted)

    fer_score = fer(pred_sequence, label_sequence)
    per_score = per(predicted, label)

    return per_score, fer_score


def usage():
    print("Usage: python ./scripts/eval/evaluate.py eval <label> <predicted>")
    print("Usage: python ./scripts/eval/evaluate.py process <ipa_string>")


def main(args):
    if len(args) < 2:
        usage()
        return
    command = args[0]
    if command == "eval":
        if len(args) < 3:
            usage()
            return
        label = args[1]
        predicted = args[2]
        per_score, fer_score = evaluate(label, predicted)
        print(f"PER: {per_score}")
        print(f"FER: {fer_score}")
    elif command == "process":
        ipa_string = args[1]
        print(preprocess_ipa(ipa_string))
    else:
        usage()


if __name__ == "__main__":
    main(sys.argv[1:])
