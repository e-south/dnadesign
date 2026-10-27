"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/pairwise/test_public_api.py

Module support for dnadesign.aligner.tests.pairwise.test_public_api.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pickle
from datetime import datetime
from unittest.mock import Mock

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
    assert len(list(tmp_path.iterdir())) == 2


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
    assert len(list(tmp_path.iterdir())) == 3


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


@pytest.mark.parametrize("kwargs", [{"output": "mean"}, {"parallel": True}, {"num_workers": 2}])
def test_batch_rejects_unsupported_controls(kwargs) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        aligner.compute_alignment_scores(["AAAA", "AAAT"], use_cache=False, **kwargs)


@pytest.mark.parametrize("formats", [(), ("unknown",), ("mean", "mean"), ["mean"]])
def test_batch_rejects_invalid_return_formats(formats) -> None:
    with pytest.raises(ValueError, match="return_formats"):
        aligner.compute_alignment_scores(["AAAA", "AAAT"], use_cache=False, return_formats=formats)


def test_mean_and_dissimilarity_request_returns_both_values() -> None:
    result = aligner.compute_alignment_scores(
        ["ACGT", "ACGA"], use_cache=False, return_formats=("mean",), return_dissimilarity=True
    )
    assert result == {"mean": 0.625, "dissimilarity": 0.375}


def test_mean_pairwise_returns_mean_when_raw_values_are_requested() -> None:
    assert aligner.mean_pairwise(["ACGT", "ACGA"], use_cache=False, return_raw=True) == 0.625


def test_mean_only_does_not_build_a_full_matrix(monkeypatch) -> None:
    from dnadesign.aligner.pairwise import matrix

    def unexpected_matrix(*args, **kwargs):
        pytest.fail("mean-only scoring must not allocate a full matrix")

    monkeypatch.setattr(matrix, "build_score_matrix", unexpected_matrix)
    assert aligner.mean_pairwise(["ACGT", "ACGA"], use_cache=False) == 0.625


def test_score_only_does_not_materialize_alignments(monkeypatch) -> None:
    from dnadesign.aligner.pairwise import scoring

    class ScoreOnlyAligner(scoring.PairwiseAligner):
        def align(self, *args, **kwargs):
            pytest.fail("score-only calls must not materialize alignments")

    monkeypatch.setattr(scoring, "PairwiseAligner", ScoreOnlyAligner)
    assert aligner.score_pairwise("ACGT", "ACGA") == 0.625


def test_identical_computation_reuses_cache_on_a_later_day(tmp_path, monkeypatch) -> None:
    from dnadesign.aligner.pairwise import cache, scoring

    clock = Mock()
    clock.now.return_value = datetime(2026, 1, 1)
    monkeypatch.setattr(cache, "datetime", clock, raising=False)
    first = aligner.mean_pairwise(["ACGT", "ACGA"], cache_dir=tmp_path)
    clock.now.return_value = datetime(2026, 1, 2)

    def unexpected_alignment(*args, **kwargs):
        pytest.fail("an identical cached computation must not be repeated")

    monkeypatch.setattr(scoring, "global_alignment", unexpected_alignment)
    from dnadesign.aligner.pairwise import matrix

    monkeypatch.setattr(matrix, "global_alignment", unexpected_alignment)
    assert aligner.mean_pairwise(["ACGT", "ACGA"], cache_dir=tmp_path) == first
    assert len(list(tmp_path.iterdir())) == 1


class _ExecutableCacheValue:
    def __reduce__(self):
        return print, ("CACHE_OBJECT_EXECUTED",)


def test_cache_rejects_pickled_objects_without_execution(tmp_path, capsys) -> None:
    from dnadesign.aligner.pairwise.cache import load_cache

    (tmp_path / "untrusted.npz").write_bytes(pickle.dumps(_ExecutableCacheValue()))
    with pytest.raises(ValueError):
        load_cache(tmp_path, "untrusted.npz")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("return_raw", [False, True])
def test_cache_roundtrip_preserves_output_arrays_and_scalars(tmp_path, return_raw) -> None:
    kwargs = dict(
        cache_dir=tmp_path,
        return_formats=("mean", "matrix", "condensed"),
        return_raw=return_raw,
        return_dissimilarity=True,
    )
    first = aligner.compute_alignment_scores(["ACGT", "ACGA"], **kwargs)
    second = aligner.compute_alignment_scores(["ACGT", "ACGA"], **kwargs)
    if return_raw:
        np.testing.assert_array_equal(second["raw"], [[8, 5], [5, 8]])
        first, second = first["normalized"], second["normalized"]
    assert first["mean"] == second["mean"] == 0.625
    assert first["dissimilarity"] == second["dissimilarity"] == 0.375
    np.testing.assert_array_equal(second["matrix"], [[1, 0.625], [0.625, 1]])
    np.testing.assert_array_equal(second["condensed"], [0.625])


@pytest.mark.parametrize("sequences", [[], ["ACGT"], ["AAAA", "AAAAAA", "AAAT"]])
@pytest.mark.parametrize("normalize", [False, True])
def test_streamed_mean_matches_matrix_values(sequences, normalize) -> None:
    matrix_result = aligner.compute_alignment_scores(
        sequences, use_cache=False, normalize=normalize, return_formats=("mean", "matrix")
    )
    assert aligner.mean_pairwise(sequences, use_cache=False, normalize=normalize) == matrix_result["mean"]


def test_dissimilarity_requires_a_mean_output() -> None:
    with pytest.raises(ValueError, match="requires.*mean"):
        aligner.compute_alignment_scores(
            ["ACGT", "ACGA"], use_cache=False, return_formats=("matrix",), return_dissimilarity=True
        )
