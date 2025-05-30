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
    # - replace combined syllabic ŋ with separate syllabic marker and ŋ
    # - exteme care is needed with ĩ and ĩ: they look identical in most fonts but are different: ĩ=ɪ̰ and ĩ=nasal i, both libraries support the latter as combined but needs the former separated
    # - "ä" and "ä" mean the same and look the same but are different unicode symbols
    ipa_string = ipa_string.replace("ŋ̍", "ŋ̩").replace("ĩ", "ɪ̰").replace("ä", "ä")

    # standardize representations and identify phonemes not in panphon (IPA_SYMBOLS)
    phonemes = group_phonemes(ipa_string)

    # now apply some substitutions to combined symbols because panphon separates them:
    # - replace combined r-colored vowels with separate symbols as supported by panphon
    # - remove lonesome ties/syllabic markers that are not part of any symbols
    phonemes = [p.replace("ɚ", "ə˞").replace("ɝ", "ɜ˞").replace("ç", "ç") for p in phonemes if p not in ["͡", '̩', '̃', 'ʲ', '̥']]

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
