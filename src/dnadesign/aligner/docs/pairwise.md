# Pairwise Alignment

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-26

Use `dnadesign.aligner.pairwise` when a workflow needs global pairwise
nucleotide alignment scores.

## Public API

```python
from dnadesign.aligner import compute_alignment_scores, mean_pairwise, score_pairwise
from dnadesign.aligner.pairwise import global_alignment
```

The root `dnadesign.aligner` API is the supported import surface for other tools.

## Outputs

`return_formats` selects a non-empty tuple of distinct outputs:

- `mean`: mean normalized pairwise score
- `matrix`: full square score matrix
- `condensed`: upper-triangular condensed vector

A mean-only request returns a scalar. Other combinations return a dictionary;
`return_dissimilarity=True` includes `dissimilarity` alongside the mean.
`return_raw=True` returns `normalized` outputs and the `raw` score matrix.
Batch execution is serial. A mean-only request accumulates pair scores without allocating a full score matrix.
Scoring options after `sequence_key` are keyword-only.

## Cache

`use_cache=True` stores numeric NPZ results in `cache_dir` (default `./swcache`).
Cache identity includes sequence content, scoring parameters, requested outputs,
and the scoring/cache version; it does not depend on the date. Reads disable
pickle, and invalid cache contents raise an error. Writes replace files atomically.

Pairwise scores use one supported `max_score` contract:

```text
normalized_similarity = alignment_score / (match_score * longer_sequence_length)
dissimilarity = 1 - normalized_similarity
```

Using the longer sequence makes the default `max_score` result symmetric and
keeps the scalar and batch APIs on the same denominator contract.
Other normalization names fail rather than silently changing or ignoring the
requested rule.

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
