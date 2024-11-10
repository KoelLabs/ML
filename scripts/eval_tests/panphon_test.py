import numpy as np
import panphon
from fastdtw import fastdtw

# Initialize the panphon feature table
ft = panphon.FeatureTable()
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def phoneme_feature_vector(phoneme):
    """Get the feature vector for a phoneme using panphon."""
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Return the vector for the single phoneme
    
    else:
        # Handle missing vectors, either by skipping or providing a default vector
        print(f"Warning: No vector found for phoneme '{phoneme}'")
        return np.zeros(22)  # 22 is the feature vector length used by panphon

def word_to_feature_sequence(word_phonemes):
    """Convert a list of phonemes to a sequence of feature vectors."""
    return [phoneme_feature_vector(phoneme) for phoneme in word_phonemes if phoneme_feature_vector(phoneme) is not None]

def compute_dtw_distance(label_embeddings, predicted_embeddings):
    """Compute the Dynamic Time Warping distance between two sequences of feature vectors."""
    # FastDTW returns a tuple (distance, path), but we are interested only in the distance
    distance, _ = fastdtw(label_embeddings, predicted_embeddings, dist=euclidean_distance)
    return distance

def cer(pred, label):
    """Compute the Character Error Rate (CER) between two sequences."""
    distances = np.zeros((len(pred) + 1, len(label) + 1))

    for t1 in range(len(pred) + 1):
        distances[t1][0] = t1

    for t2 in range(len(label) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(pred) + 1):
        for t2 in range(1, len(label) + 1):
            if (pred[t1-1] == label[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(pred)][len(label)] / len(label)


# Define test cases
test_cases = [
    (['b', 'æ', 't'], ['p', 'æ', 't']),   # "bat" vs "pat"
    (['b', 'æ', 't'], ['m', 'æ', 't']),   # "bat" vs "mat"
    (['b', 'æ', 't'], ['k', 'æ', 't']),   # "bat" vs "mat"
    (['b', 'æ', 't'], ['m', 'æ', 't', 't']),   # "bat" vs "mat"
    (['b', 'æ', 't'], ['b', 'æ', 'θ']),   # "bat" vs "bath"
    (['r', 'æ', 't'], ['b', 'æ', 't']),   # "rat" vs "bat"
        (['θ', 'ɪ', 'ŋ', 'k'], ['s', 'ɪ', 'ŋ', 'k']),  # "think" vs "sink"
    (['θ', 'ɪ', 'ŋ', 'k'], ['s', 'ɪ', 'ŋ', 'k', 'n']),  # "think" vs "sinkn"
    (['θ', 'ɪ', 'ŋ', 'k'], ['θ', 'æ', 'ŋ', 'k']), # "think" vs "thank"
    (['ɹ', 'aɪ', 't'], ['w', 'r', 'aɪ', 't']),   # "right" vs "write"
    (['ʃ', 'ɪ', 'p'], ['s', 'ɪ', 'p']),   # "ship" vs "sip"
]

# Iterate through test cases and calculate distances
for i, (label_phonemes, predicted_phonemes) in enumerate(test_cases):
    # Convert to embeddings
    label_embeddings = word_to_feature_sequence(label_phonemes)
    predicted_embeddings = word_to_feature_sequence(predicted_phonemes)

    # Equalize lengths by padding the shorter sequence with zeros
    max_len = max(len(label_embeddings), len(predicted_embeddings))
    label_embeddings = label_embeddings + [np.zeros(22)] * (max_len - len(label_embeddings))
    predicted_embeddings = predicted_embeddings + [np.zeros(22)] * (max_len - len(predicted_embeddings))

    # Calculate non-DTW (Euclidean) distance for each corresponding phoneme
    non_dtw_distances = [np.linalg.norm(l - p) for l, p in zip(label_embeddings, predicted_embeddings)]

    # Calculate total non-DTW distance
    total_non_dtw_dist = np.sum(non_dtw_distances)

    # Calculate DTW distance using panphon embeddings
    dist, _ = fastdtw(label_embeddings, predicted_embeddings, dist=euclidean_distance)

    # Calculate CER
    cer_score = cer(predicted_phonemes, label_phonemes)

    # Print results for this test case
    print(f"Test Case {i+1}:")
    print(f"Label Phonemes: {label_phonemes}")
    print(f"Predicted Phonemes: {predicted_phonemes}")
    print(f"Non-DTW distances (per phoneme): {non_dtw_distances}")
    print(f"Total Non-DTW distance: {total_non_dtw_dist:.4f}")
    print(f"DTW distance: {dist:.4f}")
    print(f"CER: {cer_score:.4f}")
    print("=" * 50)