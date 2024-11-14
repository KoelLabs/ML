import numpy as np
import panphon
from fastdtw import fastdtw
import os
import sys
import panphon.distance
import yaml

# # Initialize the Panphon Distance class
# dist = panphon.distance.Distance()
# Custom loader function
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.ipa import filter_chars
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


vectors = ft.word_to_vector_list('ɹ̩', numeric=True)
print(vectors)


# Define test cases
test_cases = [
    (['b', 'æ', 't'], ['p', 'æ', 't', 'ɚ', 'ː']),   # "bat" vs "pat"
    (['b', 'æ', 't'], ['m', 'æ', 't']),   # "bat" vs "mat"
    (['b', 'æ', 't'], ['k', 'æ', 't']),   # "bat" vs "mat"
    (['b', 'æ', 't'], ['m', 'æ', 't', 't']),   # "bat" vs "mat"
    (['b', 'æ', 't'], ['b', 'æ', 'θ', 'θ', 'θ']),   # "bat" vs "bath"
    (['r', 'æ', 't'], ['b', 'æ', 't']),   # "rat" vs "bat"
    (['θ', 'ɪ', 'ŋ', 'k'], ['s', 'ɪ', 'ŋ', 'k']),  # "think" vs "sink"
    (['θ', 'ɪ', 'ŋ', 'k'], ['s', 'ɪ', 'ŋ', 'k', 'n']),  # "think" vs "sinkn"
    (['θ', 'ɪ', 'ŋ', 'k'], ['θ', 'æ', 'ŋ', 'k']), # "think" vs "thank"
    (['ɹ', 'aɪ', 't'], ['w', 'r', 'aɪ', 't']),   # "right" vs "write"
    (['ʃ', 'ɪ', 'p'], ['s', 'ɪ', 'p']),   # "ship" vs "sip"
]

# Initialize the FeatureTable class
ft = panphon.FeatureTable()

# Define the continuous IPA sequence (e.g., a sentence)
ipa_sequence = "ðə"
second_ipa_sequence = "ðɨ"
second_noisy_ipa_sequence = "ðahdty"

#preprocess the `ipa_sequence` to remove unsupported symbols
ipa_sequence = preprocess_ipa(ipa_sequence)
second_ipa_sequence = preprocess_ipa(second_ipa_sequence)
second_noisy_ipa_sequence = preprocess_ipa(second_noisy_ipa_sequence)
# im using all of the features in this library but you can get a compressed feature representation by using the following
names = ft.names 

# Get the full feature array for the continuous IPA sequence
feature_array = ft.word_array(names, ipa_sequence)
second_feature_array = ft.word_array(names, second_ipa_sequence)
second_noisy_feature_array = ft.word_array(names, second_noisy_ipa_sequence)

# calculate distances between the two feature arrays using panphone distance metrics
feature_dist = panphon.distance.Distance().feature_edit_distance(ipa_sequence, second_ipa_sequence)
weighted_feature_dist =  panphon.distance.Distance().weighted_feature_edit_distance(ipa_sequence, second_ipa_sequence)
hamming_feature_dist = panphon.distance.Distance().hamming_feature_edit_distance(ipa_sequence, second_ipa_sequence)
# calculate hamming distance with fastdtw
dist, path = fastdtw(feature_array, second_feature_array, dist=hamming_distance)
print("Ground truth: ", ipa_sequence)
print("Sequence (no noise): ", second_ipa_sequence)
print(f"Weighted Feature Edit Distance: {weighted_feature_dist}")
normalized_by_len = weighted_feature_dist / (len(ipa_sequence) + len(second_ipa_sequence))
print(f"Normalized Weighted Feature Edit Distance: {1-normalized_by_len}")
print(f"CER (no noise): {cer(second_ipa_sequence, ipa_sequence)}")
# calculate distance between first feature array and second noise feature array
feature_dist = panphon.distance.Distance().feature_edit_distance(ipa_sequence, second_noisy_ipa_sequence)
weighted_feature_dist =  panphon.distance.Distance().weighted_feature_edit_distance(ipa_sequence, second_noisy_ipa_sequence)
hamming_feature_dist = panphon.distance.Distance().hamming_feature_edit_distance(ipa_sequence, second_noisy_ipa_sequence)
# calculate hamming distance with fastdtw
dist, path = fastdtw(feature_array, second_noisy_feature_array, dist=hamming_distance)
print("Ground truth: ", ipa_sequence)
print("Sequence (noisy): ", second_noisy_ipa_sequence)
print(f"Weighted Feature Edit Distance: {weighted_feature_dist}")
normalized_by_len = weighted_feature_dist / (len(ipa_sequence) + len(second_ipa_sequence))
print(f"Normalized Weighted Feature Edit Distance: {1-normalized_by_len}")
print(f"CER (noise): {cer(second_ipa_sequence, ipa_sequence)}")







