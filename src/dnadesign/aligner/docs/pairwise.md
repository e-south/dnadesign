# Pairwise Alignment

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-21

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
