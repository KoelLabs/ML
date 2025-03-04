#!/usr/bin/env python3

import sys

from yaml import warnings

warnings({"YAMLLoadWarning": False})

import panphon
import panphon.distance

ft = panphon.FeatureTable()
panphon_dist = panphon.distance.Distance()
inverse_double_weight_sum = 1 / (sum(ft.weights) * 2)


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
    return (
        inverse_double_weight_sum
        * panphon_dist.weighted_feature_edit_distance(ground_truth, prediction)
        / len(ground_truth)
    )


def usage():
    print("Usage: python ./scripts/eval/metrics.py <per|fer> <predicted> <label>")
    return


def main(args):
    if len(args) < 3:
        usage()
        return
    metric = args[0]
    predicted = args[1]
    label = args[2]
    if metric == "per":
        print(per(predicted, label))
    elif metric == "fer":
        print(fer(predicted, label))
    else:
        usage()
        return


if __name__ == "__main__":
    main(sys.argv[1:])
