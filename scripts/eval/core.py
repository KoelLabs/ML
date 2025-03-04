import os
import sys

from .metrics import fer, per

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.ipa import filter_chars


def preprocess_ipa(ipa_string):
    return filter_chars(ipa_string, "letters_rmv_tie")


def evaluate(label, predicted):
    label_sequence = preprocess_ipa(label)
    pred_sequence = preprocess_ipa(predicted)
    # TODO: check vocab compatibility, show warning

    fer_score = fer(pred_sequence, label_sequence)
    per_score = per(predicted, label)

    return per_score, fer_score
