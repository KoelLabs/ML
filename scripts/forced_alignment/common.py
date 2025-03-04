import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.codes import string2symbols

from yaml import warnings

warnings({"YAMLLoadWarning": False})

import panphon
import panphon.distance

# Create a panphon feature table
ft = panphon.FeatureTable()
panphon_dist = panphon.distance.Distance()

IPA_SYMBOLS = [ipa for ipa, *_ in ft.segments]


def group_phonemes(phoneme_string):
    return string2symbols(phoneme_string, IPA_SYMBOLS)[0]


# Convert a phoneme to a numerical feature vector
def phoneme_to_vector(phoneme):
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Take the first vector if multiple exist
    else:
        return None  # Invalid phoneme


# Convert sequences of phonemes to sequences of vectors
def sequence_to_vectors(seq):
    return [phoneme_to_vector(p) for p in seq if phoneme_to_vector(p) is not None]


def weighted_substitution_cost(x, y):
    return -abs(panphon_dist.weighted_substitution_cost(x, y))


def weighted_insertion_cost(x):
    return -abs(panphon_dist.weighted_insertion_cost(x))


def weighted_deletion_cost(x):
    return -abs(panphon_dist.weighted_deletion_cost(x))
