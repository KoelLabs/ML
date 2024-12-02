import numpy as np
import panphon.distance
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
    return 1 - score


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def hamming_distance(v1, v2):
    return np.sum(np.array(v1) != np.array(v2))


def compute_dtw_distance(label_embeddings, predicted_embeddings):
    """Compute the Dynamic Time Warping distance between two sequences of feature vectors."""
    # FastDTW returns a tuple (distance, path), but we are interested only in the distance
    distance, path = fastdtw(
        label_embeddings, predicted_embeddings, dist=hamming_distance
    )

    return distance


def cer(prediction, ground_truth):
    """
    Compute the Character Error Rate (CER) between prediction and ground truth sequences
    using Levenshtein distance.

    Args:
        prediction (str): The predicted sequence
        ground_truth (str): The ground truth sequence

    Returns:
        float: Character Error Rate as a value between 0 and 1
    """
    # Convert input lists to strings if they're lists
    if isinstance(prediction, list):
        prediction = "".join(prediction)
    if isinstance(ground_truth, list):
        ground_truth = "".join(ground_truth)
    print(f"prediction: {prediction}")
    print(f"ground_truth: {ground_truth}")

    # Handle empty strings
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) > 0 else 0.0

    # Initialize the matrix
    matrix = np.zeros((len(prediction) + 1, len(ground_truth) + 1))

    # Fill first row and column
    for i in range(len(prediction) + 1):
        matrix[i, 0] = i
    for j in range(len(ground_truth) + 1):
        matrix[0, j] = j

    # Fill in the rest of the matrix
    for i in range(1, len(prediction) + 1):
        for j in range(1, len(ground_truth) + 1):
            if prediction[i - 1] == ground_truth[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1]
            else:
                substitution = matrix[i - 1, j - 1] + 1
                insertion = matrix[i, j - 1] + 1
                deletion = matrix[i - 1, j] + 1
                matrix[i, j] = min(substitution, insertion, deletion)

    # Calculate CER
    distance = matrix[len(prediction), len(ground_truth)]
    return distance / len(ground_truth)


def preprocess_ipa(ipa_string):
    """Preprocess an IPA string by removing unsupported symbols. Suggestions by David Mortensen, creator of panphon."""
    replacement_map = {
        "ɚ": "ɹ̩",  # Convert /ɚ/ to /ɹ/ (non-syllabic r)
        "ɝ": "ɹ",  # Convert /ɝ/ to /ɹ/ (non-syllabic r)
        "ː": "",  # Remove length mark (or duplicate previous vowel if length is important)
        "͡": "",  # Remove tie bar (or split into components if part of an affricate)
        "g": "ɡ",  # replace two versions of g, TIMIT uses ascii g and STANDARD IPA uses unicode g
    }

    # Replace unsupported symbols
    processed_string = "".join(replacement_map.get(char, char) for char in ipa_string)

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

    # preprocess the `ipa_sequence` to remove unsupported symbols
    label_sequence = preprocess_ipa(label)
    pred_sequence = preprocess_ipa(predicted)

    # calculate distances between the two feature arrays using panphone distance metrics
    feature_dist = panphon.distance.Distance().feature_edit_distance(
        label_sequence, pred_sequence
    )
    weighted_feature_dist = panphon.distance.Distance().weighted_feature_edit_distance(
        label_sequence, pred_sequence
    )
    hamming_feature_dist = panphon.distance.Distance().hamming_feature_edit_distance(
        label_sequence, pred_sequence
    )
    # calculate hamming distance with fastdtw
    # dist, path = fastdtw(feature_array, second_feature_array, dist=hamming_distance)

    # CER calculation
    cer_score = cer(predicted, label)

    return {
        "label_sequence": label_sequence,
        "pred_sequence": pred_sequence,
        "feature_dist": feature_dist,
        "weighted_feature_dist": weighted_feature_dist,
        "hamming_feature_dist": hamming_feature_dist,
        "cer_score": cer_score,
    }


# TEST CASE
# ground_truth = "ðɨaɪɹeɪtʔækɚstɑmpəweɪʔɨɾiɑɾɨkli"
# test = "aɪ ɹ eɪ t ʔ æ k t ɚ s t ʌ m p ð ə w eɪ ʔ ɨ ɾ i ɑ ɾ ɨ k l i"
# # Call panphon_model_eval with label and predictedipa
# results = panphon_model_eval(test, ground_truth)

# # Output results
# print("Evaluation Results:")
# print(f"Feature edit distance: {results['feature_dist']}")
# print(f"Weighted feature edit distance: {results['weighted_feature_dist']}")
# print(f"Hamming distance: {results['hamming_feature_dist']}")
# print(f"CER: {results['cer_score']}")
# print
