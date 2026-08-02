"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/performance/test_hot_path_correctness.py

Reference-equivalence contracts for deterministic TriJunction hot paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import combinations

from dnadesign.trijunction.design.matching import _matching_score
from dnadesign.trijunction.design.randomness import StablePrng
from dnadesign.trijunction.sequence import longest_common_substring_length
from dnadesign.trijunction.tests.performance._factories import candidate


def test_stable_prng_has_a_golden_cross_runtime_stream() -> None:
    stream = StablePrng(0)

    assert [stream.next_u64() for _ in range(5)] == [
        16294208416658607535,
        7960286522194355700,
        487617019471545679,
        17909611376780542444,
        1961750202426094747,
    ]


def test_duplicate_substring_matching_equals_pairwise_lcs_semantics() -> None:
    candidates = (
        candidate(0, "ACGATTCGGT"),
        candidate(1, "TTACGATACC"),
        candidate(2, "CGGTACTGAA"),
        candidate(3, "GATCAGGTCA"),
    )
    barcodes = ("ACGTAC", "TGCACT", "GATTAC", "CCTAGG")
    combined = tuple(item.sequence + barcode for item, barcode in zip(candidates, barcodes, strict=True))
    reference = max(longest_common_substring_length(left, right) for left, right in combinations(combined, 2))

    assert _matching_score(candidates, barcodes) == reference
