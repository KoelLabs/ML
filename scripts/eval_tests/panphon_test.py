from dtw import dtw
import panphon
import numpy as np

# Initialize the panphon feature table
ft = panphon.FeatureTable()

def phoneme_feature_vector(phoneme):
    """Get the feature vector for a phoneme using panphon."""
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Return the vector for the single phoneme
    else:
        print(f"Warning: No vector found for phoneme '{phoneme}'")
        return None

def word_to_feature_sequence(word_phonemes):
    """Convert a list of phonemes to a sequence of feature vectors."""
    return [phoneme_feature_vector(phoneme) for phoneme in word_phonemes if phoneme_feature_vector(phoneme) is not None]

# Define the phoneme sequences
correct_phoneme = ["ɛ", "s", "p", "r", "ɛ", "s", "oʊ"]

incorrect_phoneme = ["ɛ", "k", "s", "p", "r", "ɛ", "s", "oʊ"]
# Convert the phoneme sequences to feature vector sequences
wednesday1_vectors = word_to_feature_sequence(correct_phoneme)
wednesday2_vectors = word_to_feature_sequence(incorrect_phoneme)

# Define a custom distance function (Euclidean distance)
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# Compute DTW distance without using 'dist' parameter
alignment = dtw(wednesday1_vectors, wednesday2_vectors, keep_internals=True, step_pattern="symmetric2", dist_method=euclidean_distance)
dist = alignment.distance
print("Distance between pronunciations:", dist)
