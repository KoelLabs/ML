#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from forced_alignment.common import (
    sequence_to_vectors,
    weighted_substitution_cost,
    weighted_insertion_cost,
    weighted_deletion_cost,
)


def needleman_wunsch(
    seq1,
    seq2,
    substitution_func=lambda x, y: 0 if x == y else -1,
    deletetion_func=lambda _: -1,
    insertion_func=lambda _: -1,
):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))

    # Initialize DP table
    for i in range(n + 1):
        dp[i][0] = i * deletetion_func(seq1[i - 1]) if i > 0 else 0
    for j in range(m + 1):
        dp[0][j] = j * insertion_func(seq2[j - 1]) if j > 0 else 0

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + substitution_func(seq1[i - 1], seq2[j - 1])
            delete = dp[i - 1][j] + deletetion_func(seq1[i - 1])
            insert = dp[i][j - 1] + insertion_func(seq2[j - 1])
            dp[i][j] = max(match, delete, insert)

    # Traceback to get alignment
    i, j = n, m
    aligned_seq1, aligned_seq2 = [], []

    while i > 0 or j > 0:
        current = dp[i][j]
        if (
            i > 0
            and j > 0
            and current
            == dp[i - 1][j - 1] + substitution_func(seq1[i - 1], seq2[j - 1])
        ):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and current == dp[i - 1][j] + insertion_func(seq1[i - 1]):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    return reversed(aligned_seq1), reversed(aligned_seq2)


def main(args):
    if len(args) < 3:
        print(
            "Usage: python ./scripts/forced_alignment/needleman_wunsch.py <weighted|unweighted> <seq1> <seq2>"
        )
        return
    weighted = args[0] == "weighted"
    seq1 = args[1]
    seq2 = args[2]
    if weighted:
        vector_seq1 = sequence_to_vectors(seq1)
        vector_seq2 = sequence_to_vectors(seq2)
        aligned_seq1, aligned_seq2 = needleman_wunsch(
            [(s, v) for s, v in zip(seq1, vector_seq1)],
            [(s, v) for s, v in zip(seq2, vector_seq2)],
            lambda x, y: weighted_substitution_cost(list(x[1]), list(y[1])),
            lambda x: weighted_deletion_cost(list(x[1])),
            lambda x: weighted_insertion_cost(list(x[1])),
        )
        aligned_seq1 = "".join([s if s == "-" else s[0] for s in aligned_seq1])
        aligned_seq2 = "".join([s if s == "-" else s[0] for s in aligned_seq2])
    else:
        aligned_seq1, aligned_seq2 = needleman_wunsch(seq1, seq2)
        aligned_seq1 = "".join(aligned_seq1)
        aligned_seq2 = "".join(aligned_seq2)
    print(aligned_seq1)
    print(aligned_seq2)


if __name__ == "__main__":
    main(sys.argv[1:])
