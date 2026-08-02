"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_barcode_sampling.py

Resource and reproducibility tests for barcode subset sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from dnadesign.trijunction.design import barcodes as barcode_module
from dnadesign.trijunction.design.randomness import StablePrng


@pytest.mark.parametrize(
    ("population_size", "selected_count", "seed"),
    [
        (1, 1, 0),
        (5, 0, 17),
        (5, 5, 17),
        (10_000, 1, 42),
        (10_000, 17, 1_234_567),
    ],
)
def test_sparse_barcode_index_sampling_preserves_stable_prng_semantics(
    population_size: int,
    selected_count: int,
    seed: int,
) -> None:
    sparse_rng = StablePrng(seed)
    reference_rng = StablePrng(seed)

    observed = barcode_module._sample_candidate_indices(
        sparse_rng,
        population_size=population_size,
        selected_count=selected_count,
    )
    expected = tuple(reference_rng.sample(range(population_size), selected_count))

    assert observed == expected
    assert sparse_rng.next_u64() == reference_rng.next_u64()


def test_legal_one_locus_wide_pool_never_materializes_the_candidate_index_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_count = 10_000
    iterations = 4
    alphabet = "ACGT"

    def candidate_for(index: int) -> str:
        bases: list[str] = []
        for _ in range(7):
            bases.append(alphabet[index & 0b11])
            index >>= 2
        return "".join(bases)

    candidates = tuple(candidate_for(index) for index in range(candidate_count))
    real_guard = barcode_module.guard_barcode_subset_search
    guarded_shape: dict[str, int] = {}

    def reject_population_copy(
        _rng: StablePrng,
        _values: object,
        _count: int,
    ) -> list[object]:
        raise AssertionError("barcode subset sampling must not copy the candidate population")

    def record_guard(**shape: int) -> None:
        guarded_shape.update(shape)
        real_guard(**shape)

    monkeypatch.setattr(StablePrng, "sample", reject_population_copy)
    monkeypatch.setattr(barcode_module, "guard_barcode_subset_search", record_guard)
    monkeypatch.setattr(barcode_module, "_BarcodeDistanceCache", lambda _candidates: object())
    monkeypatch.setattr(barcode_module, "_subset_score_indices", lambda _indices, _cache: (0, Fraction(0)))

    observed = barcode_module.select_barcodes(
        candidates,
        count=1,
        iterations=iterations,
        seed=17,
        forbidden_toehold_k=3,
        forbidden_barcode_k=4,
    )

    assert guarded_shape == {
        "candidate_count": candidate_count,
        "selected_count": 1,
        "sequence_length": 7,
        "iterations": iterations,
    }
    assert observed.candidates_generated == candidate_count
    assert 1 <= observed.subsets_evaluated <= iterations + 1
