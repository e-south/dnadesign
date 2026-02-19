"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_objective_contract_v2.py

Validates strict v2 objective and selection uncertainty contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from scipy.stats import norm

from dnadesign.opal.src.core.objective_result import ObjectiveResultV2, validate_objective_result_v2
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.objectives import list_objectives, register_objective
from dnadesign.opal.src.registries.selection import normalize_selection_result
from dnadesign.opal.src.selection.expected_improvement import ei
from dnadesign.opal.src.selection.top_n import top_n


class _ObjResult:
    def __init__(self, score: np.ndarray, diagnostics: Dict[str, Any] | None = None) -> None:
        self.score = np.asarray(score, dtype=float)
        self.scalar_uncertainty = None
        self.diagnostics = diagnostics or {}


def test_register_objective_rejects_legacy_signature() -> None:
    name = "test_obj_v2_bad_signature"
    if name in list_objectives():
        pytest.skip("already registered in current process")

    with pytest.raises(ValueError, match="y_pred_std"):

        @register_objective(name)
        def _bad(*, y_pred, params, ctx=None, train_view=None):  # pragma: no cover - signature guard
            return _ObjResult(score=np.ones(len(y_pred), dtype=float))


def test_register_objective_accepts_optional_extension_kwargs() -> None:
    name = "test_obj_v2_extensible_signature"
    if name in list_objectives():
        pytest.skip("already registered in current process")

    @register_objective(name)
    def _ok(*, y_pred, params, ctx, train_view, y_pred_std, extra_debug=None):  # pragma: no cover - registry only
        _ = y_pred, params, ctx, train_view, y_pred_std, extra_debug
        return _ObjResult(score=np.ones(len(y_pred), dtype=float))

    assert name in list_objectives()


def test_expected_improvement_requires_uncertainty() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        ei(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.2, 0.4], dtype=float),
            scalar_uncertainty=None,
            top_k=1,
            objective="maximize",
            tie_handling="competition_rank",
        )


def test_expected_improvement_uses_uncertainty_as_std() -> None:
    scores = np.array([0.4, 0.6], dtype=float)
    std = np.array([0.1, 0.2], dtype=float)
    out = ei(
        ids=np.array(["a", "b"], dtype=str),
        scores=scores,
        scalar_uncertainty=std,
        top_k=1,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    incumbent = float(np.max(scores))
    improvement = scores - incumbent
    z = improvement / std
    expected_raw = (improvement * norm.cdf(z)) + (std * norm.pdf(z))
    expected = (expected_raw - float(np.min(expected_raw))) / (
        float(np.max(expected_raw)) - float(np.min(expected_raw))
    )
    np.testing.assert_allclose(np.asarray(out["score"], dtype=float), expected, rtol=1e-9, atol=1e-12)


def test_objective_result_v2_requires_explicit_mode_per_score_channel() -> None:
    result = ObjectiveResultV2(
        scores_by_name={"scalar": np.array([0.1, 0.2], dtype=float)},
        uncertainty_by_name={},
        diagnostics={},
        modes_by_name={},
    )
    with pytest.raises(OpalError, match="modes_by_name"):
        validate_objective_result_v2(
            result=result,
            objective_name="scalar_identity_v1",
            n_rows=2,
        )


def test_top_n_rejects_invalid_objective_mode() -> None:
    with pytest.raises(ValueError, match="objective"):
        top_n(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.1, 0.2], dtype=float),
            scalar_uncertainty=None,
            top_k=1,
            objective="max",
            tie_handling="competition_rank",
        )


def test_selection_normalizer_rejects_invalid_tie_handling() -> None:
    with pytest.raises(ValueError, match="tie_handling"):
        normalize_selection_result(
            raw={},
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.1, 0.2], dtype=float),
            top_k=1,
            tie_handling="bad_mode",
            objective="maximize",
        )
