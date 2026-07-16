"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/objectives/test_observed_replay_contract.py

Validates objective declarations for replaying observed rows in history plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.core.objective_result import ObjectiveResultV2
from dnadesign.opal.src.registries.objectives import (
    get_objective_observed_replay_contract,
    list_objectives,
    register_objective,
)


def test_only_pointwise_objectives_declare_observed_replay_support() -> None:
    assert get_objective_observed_replay_contract("response_magnitude_feasibility_v1") == "pointwise_params_v1"
    assert get_objective_observed_replay_contract("sfxi_v1") is None


def test_registry_rejects_unknown_observed_replay_contracts() -> None:
    name = "test_unknown_observed_replay_contract"
    if name not in list_objectives():

        @register_objective(name)
        def _objective(*, y_pred, params, ctx, train_view, y_pred_std):
            del params, ctx, train_view, y_pred_std
            return ObjectiveResultV2(
                scores_by_name={"score": np.zeros(len(y_pred), dtype=float)},
                modes_by_name={"score": "maximize"},
            )

        _objective.__opal_observed_replay_contract__ = "unknown_v1"

    with pytest.raises(ValueError, match="unsupported observed replay contract"):
        get_objective_observed_replay_contract(name)
