from .common import IPA_SUBSTITUTIONS

import torch

import panphon
import panphon.distance


def transcribe_batch(batch, model, processor):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return [processor.decode(ids) for ids in predicted_ids]


def preprocess_ipa(ipa_string):
    """Preprocess an IPA string by removing unsupported symbols. Suggestions by David Mortensen, creator of panphon."""
    replacement_map = {
        "ː": "",  # Remove length mark (or duplicate previous vowel if length is important)
        "͡": "",  # Remove tie bar (or split into components if part of an affricate)
        **IPA_SUBSTITUTIONS,
    }
    processed_string = "".join(replacement_map.get(char, char) for char in ipa_string)

    return processed_string


def evaluate(label, predicted):
    label_sequence = preprocess_ipa(label)
    pred_sequence = preprocess_ipa(predicted)

    fer_score = fer(pred_sequence, label_sequence)
    per_score = per(predicted, label)

    return per_score, fer_score


# ================= METRICS =================
panphon_dist = panphon.distance.Distance()


def per(prediction, ground_truth):
    """
    Phoneme Error Rate: the number of edits (substitutions, insertions, deletions)
    needed to transform the prediction into the ground truth divided by the length of the ground truth.
    """
    return panphon_dist.fast_levenshtein_distance(prediction, ground_truth) / len(
        ground_truth
    )


def fer(prediction, ground_truth):
    """
    Feature Error Rate: the edits weighted by their acoustic features summed up and divided by the length of the ground truth.
    """
    return panphon_dist.weighted_feature_edit_distance(ground_truth, prediction) / len(
        ground_truth
    )
