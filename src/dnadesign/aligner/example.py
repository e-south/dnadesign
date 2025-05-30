"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/example.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from dnadesign.aligner.metrics import compute_alignment_scores, mean_pairwise, score_pairwise

# Example sequences (all expected to be of equal length)
sequences = [
    {"sequence": "ACGTACGT"},
    {"sequence": "ACAGATCGTA"},
    "ACAAAGTACGA",
]

# Compute and print the mean pairwise normalized similarity score.
mean_score = mean_pairwise(sequences, verbose=True)
print("Mean Normalized Similarity Score:", mean_score)

# Compute multiple output formats (mean and condensed vector),
# with the scores normalized.
results = compute_alignment_scores(sequences=sequences, return_formats=("mean", "condensed"), verbose=True)
print("Batch results:", results)

# Compute a pairwise global alignment, including the full alignment string,
# normalized similarity, and dissimilarity.
pair_result = score_pairwise(
    "ACCGTACGT", "ACGTCGTA", return_raw=True, return_alignment_str=True, return_dissimilarity=True
)
print("Pairwise Alignment:")
print(pair_result["alignment"])
print("Normalized Similarity:", pair_result["normalized"])
print("Dissimilarity:", pair_result["dissimilarity"])
