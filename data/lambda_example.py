import numpy as np


def needleman_wunsch(
    seq1,
    seq2,
    substitution_func=lambda x, y: 0 if x == y else -1,
    deletion_func=lambda _: -1,
    insertion_func=lambda _: -1,
):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))

    # Initialize DP table
    for i in range(n + 1):
        dp[i][0] = i * deletion_func(seq1[i - 1]) if i > 0 else 0
    for j in range(m + 1):
        dp[0][j] = j * insertion_func(seq2[j - 1]) if j > 0 else 0

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + substitution_func(seq1[i - 1], seq2[j - 1])
            delete = dp[i - 1][j] + deletion_func(seq1[i - 1])
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

    return list(reversed(aligned_seq1)), list(reversed(aligned_seq2))


print(needleman_wunsch("aruna", "arna"))
# outputs: (['a', 'r', 'u', 'n', 'a'],
#           ['a', 'r', '-', 'n', 'a'])

print(
    needleman_wunsch(
        "aruna",
        [("a", 1), ("r", 2), ("n", 3), ("a", 5)],
        substitution_func=lambda x, y: 0 if x == y[0] else -1,
    )
)
# outputs: (['a',       'r',     'u',  'n',      'a'],
#           [('a', 1), ('r', 2), '-', ('n', 3), ('a', 5)])
