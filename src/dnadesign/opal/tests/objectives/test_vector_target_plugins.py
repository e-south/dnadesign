"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/objectives/test_vector_target_plugins.py

Regression tests for vector target plugins OPAL objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.config.plugin_schemas import validate_params
from dnadesign.opal.src.core.objective_result import validate_objective_result_v2
from dnadesign.opal.src.objectives.vector_channel_v1 import vector_channel_v1
from dnadesign.opal.src.objectives.vector_target_similarity_v1 import vector_target_similarity_v1
from dnadesign.opal.src.transforms_y.vector_from_table_v1 import vector_from_table_v1


def test_vector_from_table_builds_finite_numeric_y_with_custom_id() -> None:
    df = pd.DataFrame(
        {
            "design_id": ["a", "b"],
            "sequence": ["AAAA", "CCCC"],
            "lexA": [1, 0],
            "cpxR": [0, 2],
        }
    )

    out = vector_from_table_v1(
        df,
        {
            "id_column": "design_id",
            "value_columns": ["lexA", "cpxR"],
        },
    )

    assert out.to_dict(orient="list") == {
        "id": ["a", "b"],
        "sequence": ["AAAA", "CCCC"],
        "y": [[1.0, 0.0], [0.0, 2.0]],
    }


def test_vector_from_table_fails_fast_for_missing_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="value_columns"):
        vector_from_table_v1(pd.DataFrame({"sequence": ["AAAA"]}), {"value_columns": []})

    with pytest.raises(ValueError, match="Missing required columns"):
        vector_from_table_v1(pd.DataFrame({"sequence": ["AAAA"], "a": [1.0]}), {"value_columns": ["a", "b"]})

    with pytest.raises(ValueError, match="finite"):
        vector_from_table_v1(
            pd.DataFrame({"sequence": ["AAAA"], "a": [float("nan")]}),
            {"value_columns": ["a"]},
        )


def test_vector_channel_objective_emits_selected_channel_and_mode() -> None:
    result = vector_channel_v1(
        y_pred=np.array([[0.2, 0.8], [0.7, 0.1]], dtype=float),
        params={"channel_index": 1, "channel_name": "tf_cpxR_presence", "mode": "minimize"},
        ctx=None,
        train_view=None,
        y_pred_std=None,
    )

    checked = validate_objective_result_v2(result=result, objective_name="vector_channel_v1", n_rows=2)
    assert checked.scores_by_name["tf_cpxR_presence"].tolist() == [0.8, 0.1]
    assert checked.modes_by_name == {"tf_cpxR_presence": "minimize"}
    assert checked.diagnostics["channel_index"] == 1


def test_vector_channel_objective_rejects_invalid_channel_index() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        vector_channel_v1(
            y_pred=np.array([[0.2, 0.8]], dtype=float),
            params={"channel_index": 3},
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )


def test_vector_target_similarity_objective_emits_negative_mse_score() -> None:
    result = vector_target_similarity_v1(
        y_pred=np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=float),
        params={"target_vector": [0.0, 0.0, 1.0, 1.0]},
        ctx=None,
        train_view=None,
        y_pred_std=None,
    )

    checked = validate_objective_result_v2(result=result, objective_name="vector_target_similarity_v1", n_rows=2)
    assert checked.scores_by_name["negative_mse"].tolist() == [-0.0, -0.5]
    assert checked.modes_by_name == {"negative_mse": "maximize"}
    assert checked.diagnostics["target_vector"] == [0.0, 0.0, 1.0, 1.0]


def test_vector_target_similarity_rejects_target_width_mismatch() -> None:
    with pytest.raises(ValueError, match="target_vector length"):
        vector_target_similarity_v1(
            y_pred=np.array([[0.2, 0.8]], dtype=float),
            params={"target_vector": [0.2]},
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )


def test_vector_target_plugin_schemas_validate_contract_parameters() -> None:
    assert validate_params(
        "transform_y",
        "vector_from_table_v1",
        {"id_column": "design_id", "value_columns": ["lexA", "cpxR"]},
    ) == {
        "id_column": "design_id",
        "sequence_column": "sequence",
        "value_columns": ["lexA", "cpxR"],
    }
    assert validate_params(
        "objective",
        "vector_channel_v1",
        {"channel_index": 2, "channel_name": "tf_baeR_count", "mode": "maximize"},
    ) == {"channel_index": 2, "channel_name": "tf_baeR_count", "mode": "maximize"}
    assert validate_params(
        "objective",
        "vector_target_similarity_v1",
        {"target_vector": [0, 0, 1, 1]},
    ) == {"target_vector": [0.0, 0.0, 1.0, 1.0]}


def test_plugin_param_schemas_reject_non_mapping_params() -> None:
    with pytest.raises(TypeError, match="params must be a mapping"):
        validate_params("transform_x", "identity", None)  # type: ignore[arg-type]
