## aligner

![aligner banner](assets/aligner-banner.svg)

**aligner** computes Needleman–Wunsch global alignment scores between nucleotide sequences. It is designed to integrate into sibling pipelines such as **libshuffle**, **clustering**, and **billboard**. It uses Biopython's [PairwiseAligner](https://biopython.org/docs/dev/Tutorial/chapter_pairwise.html#chapter-pairwise) class and offers different output formats, normalization options, and lightweight caching.

```python
aligner/
├── README.md
├── __init__.py
├── align.py
├── cache.py
├── matrix.py
├── metrics.py
└── utils.py
```

#### Features

- **Global Alignment Scoring:**
  Computes pairwise global alignment scores via Biopython's implementation of the Needleman–Wunsch algorithm.


  - Compute normalized similarity as:
    $$
    Normalized Similarity = \frac{Alignment\ Score}{match\_score \times L}
    $$

  - Compute dissimilarity as:

    $$
    Dissimilarity = 1 - Normalized\ Similarity
    $$



- **Flexible Output:**
  Supports multiple output formats:
  - **Mean:** Returns the mean pairwise score.
  - **Matrix:** Returns the full N×N score matrix.
  - **Condensed:** Returns a SciPy-style upper-triangular vector.

- **Configurable Parameters:**
  Alignment parameters (match score, mismatch penalty, gap penalties) and normalization options are fully configurable with sensible defaults.

#### Basic Alignment Scoring
```python
from dnadesign.aligner.metrics import mean_pairwise_sw, compute_alignment_scores, score_pairwise

# Example sequences: list of dictionaries or raw strings.
sequences = [
    {"sequence": "ACGTACGT"},
    {"sequence": "ACGTCGTA"},
    "ACGTACGA",
]

# Compute and print the mean pairwise SW score.
mean_score = mean_pairwise_sw(sequences, verbose=True)
print("Mean SW Score:", mean_score)

# Get multiple output formats (mean and condensed vector).
results = compute_alignment_scores(
    sequences=sequences,
    return_formats=("mean", "condensed"),
    verbose=True
)
print(results)

# Pairwise alignment of two sequences with detailed output.
pair_result = score_pairwise("ACGTACGT", "ACGTCGTA", return_raw=True)
print(pair_result)
```

#### Caching
By default, caching is enabled. Cache files are written to ./swcache with a filename that includes the number of sequences, normalization status, and alignment parameters. You can disable caching via use_cache=False if needed.

#### Parallel Processing
If you pass a large batch (e.g., more than 1000 sequences), a warning will be printed and parallel processing will be attempted. You can control this behavior with the parallel and num_workers parameters.
