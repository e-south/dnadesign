"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/pairwise/test_public_api.py

Module support for dnadesign.aligner.tests.pairwise.test_public_api.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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


def test_pairwise_cache_key_depends_on_sequence_content(tmp_path) -> None:
    first = aligner.mean_pairwise(["AAAA", "AAAA"], cache_dir=tmp_path)
    second = aligner.mean_pairwise(["AAAA", "TTTT"], cache_dir=tmp_path)

    assert first == 1.0
    assert second != first
    assert len(list(tmp_path.glob("*.pkl"))) == 2


def test_compute_alignment_scores_normalized_matrix_is_symmetric_for_unequal_lengths() -> None:
    forward = aligner.compute_alignment_scores(
        ["AAAA", "AAAAAA"],
        use_cache=False,
        return_formats=("matrix", "mean", "condensed"),
    )
    reverse = aligner.compute_alignment_scores(
        ["AAAAAA", "AAAA"],
        use_cache=False,
        return_formats=("matrix", "mean", "condensed"),
    )

    assert forward["matrix"][0, 1] == forward["matrix"][1, 0]
    assert reverse["matrix"][0, 1] == reverse["matrix"][1, 0]
    assert forward["mean"] == reverse["mean"]
    np.testing.assert_allclose(forward["condensed"], reverse["condensed"])
