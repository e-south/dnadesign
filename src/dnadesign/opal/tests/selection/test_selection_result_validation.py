"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/selection/test_selection_result_validation.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.selection import normalize_selection_result, validate_selection_result


def test_validate_selection_result_rejects_fractional_order_idx() -> None:
    with pytest.raises(OpalError, match="integral"):
        validate_selection_result(
            {"order_idx": np.array([0.0, 1.5, 2.0], dtype=float), "score": np.array([1.0, 0.8, 0.3], dtype=float)},
            plugin_name="test_plugin",
            expected_len=3,
        )


def test_validate_selection_result_rejects_non_numeric_score_payload() -> None:
    with pytest.raises(OpalError, match="numeric"):
        validate_selection_result(
            {"order_idx": np.array([0, 1, 2], dtype=int), "score": np.array(["1.0", "0.8", "0.3"], dtype=str)},
            plugin_name="test_plugin",
            expected_len=3,
        )


def test_normalize_selection_result_rejects_non_1d_order_idx() -> None:
    with pytest.raises(ValueError, match="order_idx must be 1D"):
        normalize_selection_result(
            {"order_idx": np.array([[0, 1, 2]], dtype=int)},
            ids=np.array(["a", "b", "c"]),
            scores=np.array([1.0, 0.8, 0.3]),
            top_k=1,
            tie_handling="competition_rank",
            objective="maximize",
        )


def test_normalize_selection_result_rejects_non_permutation_order_idx() -> None:
    with pytest.raises(ValueError, match="order_idx must be a permutation"):
        normalize_selection_result(
            {"order_idx": np.array([0, 0, 1], dtype=int)},
            ids=np.array(["a", "b", "c"]),
            scores=np.array([1.0, 0.8, 0.3]),
            top_k=1,
            tie_handling="competition_rank",
            objective="maximize",
        )


def test_validate_selection_result_accepts_empty_payload_when_expected_len_zero() -> None:
    out = validate_selection_result(
        {"order_idx": np.array([], dtype=int), "score": np.array([], dtype=float)},
        plugin_name="test_plugin",
        expected_len=0,
    )
    assert out.order_idx.size == 0
    assert out.score.size == 0
