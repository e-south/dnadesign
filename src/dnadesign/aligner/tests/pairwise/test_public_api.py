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


def test_pairwise_cache_key_depends_on_output_shape_flags(tmp_path) -> None:
    sequences = ["AAAA", "AAAT"]

    plain = aligner.compute_alignment_scores(
        sequences,
        cache_dir=tmp_path,
        return_formats=("mean", "condensed"),
    )
    raw = aligner.compute_alignment_scores(
        sequences,
        cache_dir=tmp_path,
        return_formats=("mean", "condensed"),
        return_raw=True,
    )
    dissimilarity = aligner.compute_alignment_scores(
        sequences,
        cache_dir=tmp_path,
        return_formats=("mean", "condensed"),
        return_dissimilarity=True,
    )

    assert isinstance(plain, dict)
    assert isinstance(raw, dict)
    assert "raw" in raw
    assert isinstance(dissimilarity, dict)
    assert "dissimilarity" in dissimilarity
    assert len(list(tmp_path.glob("*.pkl"))) == 3


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


def test_scalar_and_batch_max_score_normalization_match_for_unequal_lengths() -> None:
    sequences = ["AAAA", "AAAAAA"]

    scalar = aligner.score_pairwise(
        *sequences,
        normalization="max_score",
        gap_open=0,
        gap_extend=0,
    )
    batch = aligner.compute_alignment_scores(
        sequences,
        normalization="max_score",
        gap_open=0,
        gap_extend=0,
        use_cache=False,
        return_formats=("mean",),
    )

    assert scalar == pytest.approx(2 / 3)
    assert batch == pytest.approx(2 / 3)


def test_public_pairwise_paths_reject_unsupported_normalization() -> None:
    with pytest.raises(ValueError, match="Only 'max_score' normalization is supported"):
        aligner.score_pairwise("AAAA", "AAAAAA", normalization="alignment_length")

    with pytest.raises(ValueError, match="Only 'max_score' normalization is supported"):
        aligner.compute_alignment_scores(
            ["AAAA", "AAAAAA"],
            normalization="alignment_length",
            use_cache=False,
        )
