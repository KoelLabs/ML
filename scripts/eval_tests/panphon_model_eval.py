import numpy as np
import panphon
from fastdtw import fastdtw

# Initialize the panphon feature table
ft = panphon.FeatureTable()
# global var called embedding length
EMBEDDING_LEN = 22
def cosine_distance(x, y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0  # or return some default value (1 for similarity, etc.)
    score = np.dot(x, y) / (norm_x * norm_y)
    # print(f"cosine similarity: {score}")
    return 1-score

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)
def hamming_distance(v1, v2):
    return np.sum(np.array(v1) != np.array(v2))


def phoneme_feature_vector(phoneme):
    """Get the feature vector for a phoneme using panphon."""
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Return the vector for the single phoneme
    
    else:
        # Handle missing vectors, either by skipping or providing a default vector
        print(f"Warning: No vector found for phoneme '{phoneme}'")
        return np.zeros(EMBEDDING_LEN)  # 22 is the feature vector length used by panphon

def word_to_feature_sequence(word_phonemes):
    """Convert a list of phonemes to a sequence of feature vectors."""
    list_emb = [phoneme_feature_vector(phoneme) for phoneme in word_phonemes if phoneme_feature_vector(phoneme) is not None]
    return list_emb

def compute_dtw_distance(label_embeddings, predicted_embeddings):
    """Compute the Dynamic Time Warping distance between two sequences of feature vectors."""
    # FastDTW returns a tuple (distance, path), but we are interested only in the distance
    distance, path = fastdtw(label_embeddings, predicted_embeddings, dist=hamming_distance)

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
def preprocess_ipa(ipa_string):
    """Preprocess an IPA string by removing unsupported symbols. Suggestions by David Mortensen, creator of panphon."""
    replacement_map = {
        'ɚ': 'ɹ̩',   # Convert /ɚ/ to /ɹ/ (non-syllabic r)
        'ɝ': 'ɹ',   # Convert /ɝ/ to /ɹ/ (non-syllabic r)
        'ː': '',    # Remove length mark (or duplicate previous vowel if length is important)
        '͡': '',     # Remove tie bar (or split into components if part of an affricate)
    }
    
    # Replace unsupported symbols
    processed_string = ''.join(
        replacement_map.get(char, char) for char in ipa_string
    )
    
    return processed_string

def panphon_model_eval(label, predicted):
    """
    Generalized evaluation function for phoneme-based transcription.
    Arguments:
    - label: Ground truth transcription (list of IPA phonemes).
    - predictedipa: Predicted transcription (list of IPA phonemes).
    """
    # Convert to feature embeddings
    
    # Initialize the FeatureTable class
    ft = panphon.FeatureTable()


    #preprocess the `ipa_sequence` to remove unsupported symbols
    label_sequence = preprocess_ipa(label)
    pred_sequence = preprocess_ipa(predicted)

    # calculate distances between the two feature arrays using panphone distance metrics
    feature_dist = panphon.distance.Distance().feature_edit_distance(label_sequence, pred_sequence)
    weighted_feature_dist =  panphon.distance.Distance().weighted_feature_edit_distance(label_sequence, pred_sequence)
    hamming_feature_dist = panphon.distance.Distance().hamming_feature_edit_distance(label_sequence, pred_sequence)
    normalized_by_len = (weighted_feature_dist / max(len(label_sequence), len(pred_sequence)))
    # calculate hamming distance with fastdtw
    # dist, path = fastdtw(feature_array, second_feature_array, dist=hamming_distance)
    # CER calculation
    cer_score = cer(predicted, label)
    print("Ground truth: ", label)
    print("Predicted: ", predicted)
    print(f"Weighted Feature Edit Distance: {weighted_feature_dist}")
    print(f"Normalized Weighted Feature Edit Distance: {normalized_by_len}") # this does not work...
    print(f"CER (no noise): {cer_score}")

 

    # Print results
    print("Phoneme Transcription Evaluation:")

    
    return {
        "dtw_distance_no_pad": dist_no_pad,
        "dtw_distance": dist,
        "dtw_distance_percent": dist_percent,
        "cer_score": cer_score
    }