import numpy as np
import panphon
import panphon.distance

# Create a panphon feature table
ft = panphon.FeatureTable()


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
    return -abs(panphon.distance.Distance().weighted_substitution_cost(x, y))


def weighted_insertion_cost(x):
    return -abs(panphon.distance.Distance().weighted_insertion_cost(x))


def weighted_deletion_cost(x):
    return -abs(panphon.distance.Distance().weighted_deletion_cost(x))
