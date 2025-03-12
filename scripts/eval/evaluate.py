#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.ipa import filter_chars
from eval.metrics import fer, per
from forced_alignment.common import IPA_SYMBOLS, group_phonemes


def simplify_ipa(ipa_string):
    # remove all whitespace and numbers
    ipa_string = "".join(c for c in ipa_string if not c.isspace() and not c.isdigit())
    # remove anything that is not a vowel/consonant including tie bars
    return filter_chars(ipa_string, "letters_rmv_tie")


def preprocess_ipa(ipa_string):
    # remove white space and phonemes not in panphon (IPA_SYMBOLS)
    phonemes = group_phonemes(ipa_string.replace(" ", ""))
    unsupported_phonemes = set(phonemes) - set(IPA_SYMBOLS)
    if len(unsupported_phonemes) > 0:
        print(
            f"Warning: unsupported phonemes found in input: {unsupported_phonemes}.",
            file=sys.stderr,
        )
    return "".join(p for p in phonemes if p in IPA_SYMBOLS)


def evaluate(label, predicted):
    label_sequence = preprocess_ipa(label)
    pred_sequence = preprocess_ipa(predicted)

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
