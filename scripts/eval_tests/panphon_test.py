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
print(f"Feature Edit Distance: {feature_dist}")
print(f"Weighted Feature Edit Distance: {weighted_feature_dist}")
# print(f"Hamming Feature Edit Distance: {hamming_feature_dist}")
# print(f"hamming distance fastdtw: {dist}")
# normalize the weighted feature edit distance by the length of the feature array
print(f"Normalized Weighted Feature Edit Distance: {weighted_feature_dist/2}")
print(f"exponetiated norm:", np.exp(-weighted_feature_dist))
print("/n")
# calculate distance between first feature array and second noise feature array
feature_dist = panphon.distance.Distance().feature_edit_distance(ipa_sequence, second_noisy_ipa_sequence)
weighted_feature_dist =  panphon.distance.Distance().weighted_feature_edit_distance(ipa_sequence, second_noisy_ipa_sequence)
hamming_feature_dist = panphon.distance.Distance().hamming_feature_edit_distance(ipa_sequence, second_noisy_ipa_sequence)
# calculate hamming distance with fastdtw
dist, path = fastdtw(feature_array, second_noisy_feature_array, dist=hamming_distance)
print("Ground truth: ", ipa_sequence)
print("Sequence (noisy): ", second_noisy_ipa_sequence)
print(f"Feature Edit Distance: {feature_dist}")
print(f"Weighted Feature Edit Distance: {weighted_feature_dist}")
# print(f"Hamming Feature Edit Distance: {hamming_feature_dist}")
# print(f"hamming distance fastdtw: {dist}")
print(f"Normalized Weighted Feature Edit Distance: {weighted_feature_dist/6}")
print(f"exponetiated norm:", np.exp(-weighted_feature_dist))
print(f"signoided: ", 1/(1+np.exp(-weighted_feature_dist)))








# # Iterate through test cases and calculate distances
# for i, (label_phonemes, predicted_phonemes) in enumerate(test_cases):
#     # Preprocess the phonemes
#     label_phonemes = preprocess_ipa(label_phonemes)
#     predicted_phonemes = preprocess_ipa(predicted_phonemes)
#     print(predicted_phonemes)

#     print(f"label_phonemes: {label_phonemes}")
#     print(f"predicted_phonemes: {predicted_phonemes}")
#     # Convert to embeddings
#     label_embeddings = word_to_feature_sequence(label_phonemes)
#     predicted_embeddings = word_to_feature_sequence(predicted_phonemes)
#     print(f"label_embeddings: {label_embeddings}")
#     print(f"predicted_embeddings: {predicted_embeddings}")
#     # <TODO> use different panphone distance metrics  
#     # Compute the distance using different Panphon distance metrics
#     feature_dist = panphon.distance.Distance().feature_edit_distance(label_phonemes, predicted_phonemes)
#     weighted_feature_dist =  panphon.distance.Distance().weighted_feature_edit_distance(label_phonemes, predicted_phonemes)
#     hamming_feature_dist = panphon.distance.Distance().hamming_feature_edit_distance(label_phonemes, predicted_phonemes)
    
#     # Print the results
#     print(f"Feature Edit Distance: {feature_dist}")
#     print(f"Weighted Feature Edit Distance: {weighted_feature_dist}")
#     print(f"Hamming Feature Edit Distance: {hamming_feature_dist}")

#     # Calculate non-DTW distances: get per phoneme distance and then just average them
#     # Equalize lengths by padding the shorter sequence with zeros
#     max_len = max(len(label_embeddings), len(predicted_embeddings))
#     label_embeddings = label_embeddings + [np.zeros(EMBEDDING_LEN)] * (max_len - len(label_embeddings))
#     predicted_embeddings = predicted_embeddings + [np.zeros(EMBEDDING_LEN)] * (max_len - len(predicted_embeddings))
#     # print(f"label_embeddings: {label_embeddings}")
#     # print(f"predicted_embeddings: {predicted_embeddings}")
#     # print the embeddings that are different between label and predicted
#     # compute levenstein distance between the two embeddings and then divide by the length of the embeddings
    


#     dist = compute_dtw_distance(label_embeddings, predicted_embeddings)
#     dist_percent = dist / (max_len*EMBEDDING_LEN)


#     # Calculate CER
#     cer_score = cer(predicted_phonemes, label_phonemes)

#     # Print results for this test case
#     print(f"Test Case {i+1}:")
#     print(f"Label Phonemes: {label_phonemes}")
#     print(f"Predicted Phonemes: {predicted_phonemes}")
#     print(f"DTW distance (number of differences in embeddings): {dist:.4f}")
#     print(f"DTW distance percent: {dist_percent:.4f}")

#     print(f"CER: {cer_score:.4f}")
#     print("=" * 50)