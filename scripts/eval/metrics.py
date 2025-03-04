import panphon
import panphon.distance

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
