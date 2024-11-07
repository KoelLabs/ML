import panphon
import numpy as np

from fastdtw import fastdtw
from scipy.spatial.distance import cosine

ft = panphon.FeatureTable()

def phoneme_feature_vector(phoneme):
    """Get the feature vector for a phoneme."""
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    return np.array(vectors[0]) if vectors else None

def dtw_phonemic_distance(target, predicted):
    """Calculate DTW phonemic distance between two transcriptions."""
    
    # Convert each phoneme in both transcriptions to a feature vector using the panphone featuremap
    target_vectors = [phoneme_feature_vector(p) for p in target]
    predicted_vectors = [phoneme_feature_vector(p) for p in predicted]
    
    # Filter out any None values where phonemes had no feature representation
    target_vectors = [v for v in target_vectors if v is not None]
    predicted_vectors = [v for v in predicted_vectors if v is not None]

    # Calculate DTW on the sequences of vectors
    distance, _ = fastdtw(target_vectors, predicted_vectors, dist=cosine)
    return distance

def simple_phonemic_distance(target, predicted):
    """Calculate phonemic distance by directly comparing each phoneme pair."""
    distances = []
    min_length = min(len(target), len(predicted))

    for i in range(min_length):
        t_vec = phoneme_feature_vector(target[i])
        p_vec = phoneme_feature_vector(predicted[i])

        # Skip if either phoneme has no feature vector
        if t_vec is None or p_vec is None:
            distances.append(1)  # Arbitrary high distance for unrecognized phonemes
        else:
            # Calculate cosine distance between the feature vectors
            distances.append(cosine(t_vec, p_vec))
    
    # Average or sum the distances
    return np.mean(distances)


# Example usage
target_transcription = "ælɛks ɪz ɛkstrə kənfjuzd".split()  # Adjust to IPA list as needed
predicted_transcription = "æligz ɪz ɛkstknfjuzd".split()  # Adjust as needed

distance_score = dtw_phonemic_distance(target_transcription, predicted_transcription)
print("Phonemic DTW distance score:", distance_score)
print("Normalized score:", distance_score / len(target_transcription))
distance_score2 = simple_phonemic_distance(target_transcription, predicted_transcription)
print("Phonemic distance score without DTW:", distance_score2)

