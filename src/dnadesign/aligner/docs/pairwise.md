# Pairwise Alignment

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20

Use `dnadesign.aligner.pairwise` when a workflow needs global pairwise
nucleotide alignment scores.

## Public API

```python
from dnadesign.aligner import compute_alignment_scores, mean_pairwise, score_pairwise
from dnadesign.aligner.pairwise import global_alignment
```

The root `dnadesign.aligner` API is the supported import surface for sibling
tools. Do not import flat implementation modules such as
`dnadesign.aligner.metrics`; those modules are not part of the modern package
layout.

## Outputs

`compute_alignment_scores` supports:

- `mean`: mean normalized pairwise score
- `matrix`: full square score matrix
- `condensed`: upper-triangular condensed vector

Pairwise scores preserve the historical normalization behavior:

```text
normalized_similarity = alignment_score / (match_score * reference_length)
dissimilarity = 1 - normalized_similarity
```

## Example

```python
from dnadesign.aligner import compute_alignment_scores, score_pairwise

result = compute_alignment_scores(
    ["ACGTACGT", "ACGTCGTA", "ACGTACGA"],
    return_formats=("mean", "condensed"),
    use_cache=False,
)

pair = score_pairwise("ACGTACGT", "ACGTCGTA", return_raw=True)
```
