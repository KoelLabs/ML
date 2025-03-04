#!/usr/bin/env python3

import os
import sys

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from forced_alignment.common import sequence_to_vectors


def dtw_phoneme_alignment(seq1, seq2):
    seq1_vectors = list(range(len(seq1)))
    seq2_vectors = list(range(len(seq2)))

    distance, path = fastdtw(
        seq1_vectors,
        seq2_vectors,
        dist=lambda x, y: 0 if seq1[int(x)] == seq2[int(y)] else 1,
    )

    aligned_seq1 = []
    aligned_seq2 = []
    for i, j in path:
        aligned_seq1.append(seq1[i] if i < len(seq1) else "-")
        aligned_seq2.append(seq2[j] if j < len(seq2) else "-")

    return "".join(aligned_seq1), "".join(aligned_seq2)


def dtw_phoneme_alignment_weighted(seq1, seq2):
    # Convert phoneme sequences to feature vector sequences
    seq1_vectors = sequence_to_vectors(seq1)
    seq2_vectors = sequence_to_vectors(seq2)

    if not seq1_vectors or not seq2_vectors:
        raise ValueError(
            "One or both sequences could not be converted to feature vectors."
        )

    # Use FastDTW with Euclidean distance on the vectors
    distance, path = fastdtw(seq1_vectors, seq2_vectors, dist=euclidean)

    # Align the original phoneme sequences based on the path
    aligned_seq1 = []
    aligned_seq2 = []
    for i, j in path:
        aligned_seq1.append(seq1[i] if i < len(seq1) else "-")
        aligned_seq2.append(seq2[j] if j < len(seq2) else "-")

    return "".join(aligned_seq1), "".join(aligned_seq2)


def main(args):
    if len(args) < 3:
        print(
            "Usage: python ./scripts/forced_alignment/dtw.py <weighted|unweighted> <seq1> <seq2>"
        )
        return
    weighted = args[0] == "weighted"
    seq1 = args[1]
    seq2 = args[2]
    if weighted:
        aligned_seq1, aligned_seq2 = dtw_phoneme_alignment_weighted(seq1, seq2)
    else:
        aligned_seq1, aligned_seq2 = dtw_phoneme_alignment(seq1, seq2)
    print(aligned_seq1)
    print(aligned_seq2)


if __name__ == "__main__":
    main(sys.argv[1:])
