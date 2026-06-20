from __future__ import annotations

import numpy as np
import pytest

from dnadesign import aligner
from dnadesign.aligner import pairwise


def test_root_public_api_preserves_pairwise_scoring() -> None:
    assert aligner.score_pairwise("ACGT", "ACGT") == 1.0
    assert aligner.mean_pairwise(["ACGT", "ACGA"], use_cache=False) == 0.625

    result = aligner.compute_alignment_scores(
        ["ACGT", "ACGA"],
        use_cache=False,
        return_formats=("mean", "condensed"),
    )

    assert result["mean"] == 0.625
    np.testing.assert_allclose(result["condensed"], np.array([0.625], dtype=np.float32))


def test_pairwise_package_exports_public_functions() -> None:
    assert pairwise.score_pairwise("ACGT", "ACGT") == 1.0
    assert pairwise.global_alignment("ACGT", "ACGT") == 8.0


def test_invalid_nucleotide_sequence_fails_fast() -> None:
    with pytest.raises(ValueError, match="Invalid character"):
        aligner.score_pairwise("ACGT", "ACGU")
