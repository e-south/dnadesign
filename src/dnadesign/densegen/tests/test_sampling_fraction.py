from __future__ import annotations

from dnadesign.densegen.src.core.pipeline.stage_b import _compute_sampling_fraction, _compute_sampling_fraction_pairs


def test_sampling_fraction_bounded_with_duplicates() -> None:
    library = ["AAA", "AAA", "CCC"]
    fraction = _compute_sampling_fraction(library, input_tfbs_count=2, pool_strategy="subsample")
    assert fraction == 1.0


def test_sampling_fraction_pairs_bounded() -> None:
    library = ["AAA", "AAA", "CCC"]
    regulators = ["TF1", "TF1", "TF2"]
    fraction = _compute_sampling_fraction_pairs(
        library,
        regulators,
        input_pair_count=2,
        pool_strategy="subsample",
    )
    assert fraction == 1.0
