import numpy as np
import panphon
from fastdtw import fastdtw
import os
import sys
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

# Iterate through test cases and calculate distances
for i, (label_phonemes, predicted_phonemes) in enumerate(test_cases):

    # quick TEST
    ipa_string = "ˈkætəˌɡɔːri"  # Example IPA string with stress markers
    filtered_string = filter_chars(ipa_string, filter_type="cns_vwl_str_len_wb_sb")
    print("TETSW ETESTWEIGYQWDGILYDEQ")
    print(filtered_string)
    # clean up the label and predicted 
    # label_phonemes = filter_chars(label_phonemes)
    # predicted_phonemes = filter_chars("".join(predicted_phonemes), "cns_vwl_str_len_wb_sb")
    print(f"label_phonemes: {label_phonemes}")
    print(f"predicted_phonemes: {predicted_phonemes}")
    # Convert to embeddings
    label_embeddings = word_to_feature_sequence(label_phonemes)
    predicted_embeddings = word_to_feature_sequence(predicted_phonemes)
    

    # Calculate non-DTW distances: get per phoneme distance and then just average them
    # Equalize lengths by padding the shorter sequence with zeros
    max_len = max(len(label_embeddings), len(predicted_embeddings))
    label_embeddings = label_embeddings + [np.zeros(EMBEDDING_LEN)] * (max_len - len(label_embeddings))
    predicted_embeddings = predicted_embeddings + [np.zeros(EMBEDDING_LEN)] * (max_len - len(predicted_embeddings))
    # print(f"label_embeddings: {label_embeddings}")
    # print(f"predicted_embeddings: {predicted_embeddings}")
    # print the embeddings that are different between label and predicted
    # compute levenstein distance between the two embeddings and then divide by the length of the embeddings
    


    dist = compute_dtw_distance(label_embeddings, predicted_embeddings)
    dist_percent = dist / (max_len*EMBEDDING_LEN)


    # Calculate CER
    cer_score = cer(predicted_phonemes, label_phonemes)

    # Print results for this test case
    print(f"Test Case {i+1}:")
    print(f"Label Phonemes: {label_phonemes}")
    print(f"Predicted Phonemes: {predicted_phonemes}")
    print(f"DTW distance (number of differences in embeddings): {dist:.4f}")
    print(f"DTW distance percent: {dist_percent:.4f}")

    print(f"CER: {cer_score:.4f}")
    print("=" * 50)