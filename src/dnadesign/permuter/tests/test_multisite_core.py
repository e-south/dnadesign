"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_multisite_core.py

Focused unit tests for multisite_select core helpers:

  • geometry: angular distance + medoids
  • scoring: robust median/MAD scaling + composite scores
  • utils:   parsing, validation, embedding coercion, and sequence decoration

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.permuter.src.protocols.multisite_select.geometry import (
    angular_distance,
    l2_normalize_rows,
    medoid_index,
    min_angular_distance_to_set,
    pairwise_angular,
)
from dnadesign.permuter.src.protocols.multisite_select.scoring import (
    compute_scaled_scores,
    robust_z,
)
from dnadesign.permuter.src.protocols.multisite_select.utils import (
    _as_int_list,
    build_mutated_aa_sequences,
    coerce_vector1d,
    compute_mutation_window_indices_nt,
    extract_embedding_matrix,
    filter_valid_source_rows,
    is_numeric_vector1d,
    parse_aa_combo_to_map,
    uppercase_mutated_codons,
)

# ---------------------------------------------------------------------------
# geometry.py
# ---------------------------------------------------------------------------


def test_l2_normalize_rows_and_angular_distance_basics():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    U = l2_normalize_rows(X)

    # Unit norms
    norms = np.linalg.norm(U, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms))

    # 90 degrees between orthogonal unit vectors
    theta = angular_distance(U[0], U[1])
    assert np.isclose(np.degrees(theta), 90.0, atol=1e-6)

    # 0 degrees to self
    assert np.isclose(angular_distance(U[0], U[0]), 0.0, atol=1e-8)

    # 180 degrees to opposite vector
    theta_pi = angular_distance(U[0], -U[0])
    assert np.isclose(np.degrees(theta_pi), 180.0, atol=1e-6)


def test_l2_normalize_rows_rejects_zero_norm_and_nonfinite():
    X_zero = np.array([[0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError):
        l2_normalize_rows(X_zero)

    X_nan = np.array([[np.nan, 1.0]], dtype=float)
    with pytest.raises(ValueError):
        l2_normalize_rows(X_nan)


def test_pairwise_angular_symmetry_and_zero_diagonal():
    angles = np.deg2rad([0.0, 60.0, 120.0])
    X = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    U = l2_normalize_rows(X)
    A = pairwise_angular(U)

    assert A.shape == (3, 3)
    # Symmetric and zero diagonal
    np.testing.assert_allclose(A, A.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(A), np.zeros(3), atol=1e-8)


def test_medoid_index_picks_central_point_on_circle():
    # Three points at 0°, 60°, 120°; the 60° point should be the medoid.
    angles = np.deg2rad([0.0, 60.0, 120.0])
    X = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    U = l2_normalize_rows(X)
    m = medoid_index(U)
    assert m == 1


def test_min_angular_distance_to_set_handles_empty_and_nonempty():
    x = np.array([1.0, 0.0], dtype=float)
    empty = np.zeros((0, 2), dtype=float)
    d_empty = min_angular_distance_to_set(x, empty)
    assert d_empty == float("inf")

    S = l2_normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
    d = min_angular_distance_to_set(S[0], S)
    # Set contains identical vector → minimum distance 0
    assert np.isclose(d, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# scoring.py
# ---------------------------------------------------------------------------


def test_robust_z_gaussian_consistent_matches_manual():
    x = np.array([-1.0, 0.0, 1.0, 4.0])
    z, med, mad_raw = robust_z(x, gaussian_consistent=True, winsor_mads=None)

    assert med == np.nanmedian(x)
    expected_mad = np.nanmedian(np.abs(x - med))
    assert np.isclose(mad_raw, expected_mad)

    scale = mad_raw * 1.4826
    if scale == 0.0 or not np.isfinite(scale):
        expected = np.zeros_like(x)
    else:
        expected = (x - med) / scale
    np.testing.assert_allclose(z, expected)


def test_robust_z_degenerate_mad_returns_zeros():
    x = np.ones(5)
    z, med, mad_raw = robust_z(x, gaussian_consistent=False, winsor_mads=None)
    assert mad_raw == 0.0
    assert np.all(z == 0.0)


def test_robust_z_winsorization_clips_extreme_values():
    x = np.array([-10.0, 0.0, 10.0])
    z_no_clip, _, _ = robust_z(x, gaussian_consistent=False, winsor_mads=None)
    z_clip, _, _ = robust_z(x, gaussian_consistent=False, winsor_mads=1.0)

    # With winsorization, z-scores are clipped in magnitude to <= winsor_mads
    assert np.max(np.abs(z_clip)) <= 1.0 + 1e-8
    # Central value should be unchanged
    assert np.isclose(z_no_clip[1], z_clip[1])


def test_compute_scaled_scores_combines_llr_and_epi():
    llr = np.array([0.0, 1.0, 2.0])
    delta = np.array([0.0, 2.0, 4.0])

    z_llr, z_epi, score, summary = compute_scaled_scores(
        llr_obs=llr,
        delta=delta,
        gaussian_consistent=False,
        winsor_mads=None,
        w_llr=1.0,
        w_epi=0.5,
    )

    # Summary is well-formed
    assert isinstance(summary.median_llr, float)
    assert isinstance(summary.mad_llr, float)
    assert isinstance(summary.median_epi, float)
    assert isinstance(summary.mad_epi, float)

    # Composite score = α z_llr + β z_epi
    np.testing.assert_allclose(score, 1.0 * z_llr + 0.5 * z_epi)


# ---------------------------------------------------------------------------
# utils.py – parsing & embedding coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        (["3", "1", "2"], [1, 2, 3]),
        ("['5','7','5']", [5, 7]),
        ("", []),
        ("  ", []),
    ],
)
def test_as_int_list_defensive_parsing(s, expected):
    assert _as_int_list(s) == expected


def test_parse_aa_combo_to_map_roundtrip_and_validation():
    m = parse_aa_combo_to_map("G16F|L17I|N21H")
    assert m == {16: "F", 17: "I", 21: "H"}

    for bad in ["16F", "G16", "G16FI", "X-1Z"]:
        with pytest.raises(ValueError):
            parse_aa_combo_to_map(bad)


def test_build_mutated_aa_sequences_applies_tokens_and_checks_range():
    ref = "ACDEFGHIK"  # length 9
    combos = ["A1V|H8Y", "C2G", ""]
    seqs = build_mutated_aa_sequences(ref, combos)

    seq0 = list(ref)
    seq0[0] = "V"
    seq0[7] = "Y"

    seq1 = list(ref)
    seq1[1] = "G"

    assert seqs[0] == "".join(seq0)
    assert seqs[1] == "".join(seq1)
    assert seqs[2] == ref

    # Out-of-range AA position should raise
    with pytest.raises(ValueError):
        build_mutated_aa_sequences(ref, ["Z999X"])


def test_coerce_vector1d_and_is_numeric_vector1d_behaviour():
    v = coerce_vector1d([1, 2, 3])
    np.testing.assert_allclose(v, np.array([1.0, 2.0, 3.0]))

    # 2D array is flattened into 1D
    v2 = coerce_vector1d(np.array([[1.0, 2.0, 3.0]]))
    assert v2.shape == (3,)

    # None is not acceptable
    with pytest.raises(TypeError):
        coerce_vector1d(None)

    assert is_numeric_vector1d([0, 1, 2])
    assert not is_numeric_vector1d("not-a-vector")


def test_extract_embedding_matrix_enforces_consistent_dimension():
    s = pd.Series([[0.0, 1.0], [2.0, 3.0]], name="emb")
    M = extract_embedding_matrix(s)
    assert M.shape == (2, 2)

    # Inconsistent lengths should raise
    s_bad = pd.Series([[0.0, 1.0], [2.0, 3.0, 4.0]], name="emb")
    with pytest.raises(ValueError):
        extract_embedding_matrix(s_bad)


# ---------------------------------------------------------------------------
# utils.py – row-level validation and DNA decoration
# ---------------------------------------------------------------------------


def test_compute_mutation_window_indices_nt_basic_and_validation():
    # 10 codons → 30 nt; AA positions 3..7 should span a window of 5 codons.
    aa_pos_lists = [[3], [5, 6], "['7']"]
    nt_start, nt_end = compute_mutation_window_indices_nt(aa_pos_lists, seq_length_nt=30)
    assert nt_start == 3 * (3 - 1)
    assert nt_end == 3 * 7
    assert nt_end - nt_start == 3 * (7 - 3 + 1)

    # Non-codon-length sequence should raise
    with pytest.raises(ValueError):
        compute_mutation_window_indices_nt(aa_pos_lists, seq_length_nt=31)

    # Positions beyond the coding region should raise
    with pytest.raises(ValueError):
        compute_mutation_window_indices_nt([[100]], seq_length_nt=30)

    # Empty AA position lists should raise
    with pytest.raises(ValueError):
        compute_mutation_window_indices_nt([], seq_length_nt=30)


def test_filter_valid_source_rows_keeps_only_rows_passing_all_checks():
    df = pd.DataFrame(
        {
            "llr": [1.0, np.nan, 2.0, 3.0, 4.0],
            "epi": [0.0, 1.0, -0.5, 2.0, 3.0],
            "aa_list": [[1, 2], [1], [2], None, [3]],
            "emb": [
                [0.0, 1.0],  # valid
                [0.0, 1.0],  # llr NaN
                [0.0, 1.0],  # epi negative
                [0.0, 1.0],  # missing aa_list
                "not-a-vector",  # invalid embedding
            ],
        }
    )

    filtered, info = filter_valid_source_rows(df, emb_col="emb", aa_col="aa_list", llr_col="llr", epi_col="epi")

    assert info["n_total"] == 5
    assert info["n_kept"] == 1
    assert len(filtered) == 1

    row = filtered.iloc[0]
    assert row["llr"] == 1.0
    assert row["epi"] == 0.0
    assert row["aa_list"] == [1, 2]

    drops = info["drops_by_cause"]
    assert drops["nan_llr"] == 1
    assert drops["neg_epistasis"] == 1
    assert drops["missing_aa_pos_list"] == 1
    assert drops["bad_embedding"] == 1


def test_filter_valid_source_rows_requires_numeric_epistasis():
    df = pd.DataFrame(
        {
            "llr": [1.0, 2.0],
            "epi": ["not-numeric", "still-bad"],
            "aa_list": [[1], [2]],
            "emb": [[0.0, 1.0], [1.0, 0.0]],
        }
    )
    with pytest.raises(TypeError):
        filter_valid_source_rows(df, emb_col="emb", aa_col="aa_list", llr_col="llr", epi_col="epi")


def test_uppercase_mutated_codons_marks_expected_codons_and_validates_input():
    # 4 codons → positions 1..4 are valid
    seq = "acg" * 4  # length 12
    aa_positions = [1, 3]
    out = uppercase_mutated_codons(seq, aa_positions)

    # Codon 1 and 3 should contain uppercase bases
    assert out[0:3].isupper()
    assert out[6:9].isupper()

    # Codon 2 and 4 should be all lowercase
    assert out[3:6].islower()
    assert out[9:12].islower()

    # Non-multiple-of-3 sequence should raise
    with pytest.raises(ValueError):
        uppercase_mutated_codons("acgt", [1])

    # Non-positive or out-of-range AA positions should raise
    with pytest.raises(ValueError):
        uppercase_mutated_codons(seq, [0])
    with pytest.raises(ValueError):
        uppercase_mutated_codons(seq, [100])
