"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_transform_sfxi_vec8_from_table_v1.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.transforms_y.sfxi_vec8_from_table_v1 import (
    sfxi_vec8_from_table_v1,
)


def _vec8_cols(n: int) -> dict[str, list[float]]:
    return {
        "v00": [0.0] * n,
        "v10": [1.0] * n,
        "v01": [0.0] * n,
        "v11": [1.0] * n,
        "y00_star": [0.0] * n,
        "y10_star": [0.0] * n,
        "y01_star": [0.0] * n,
        "y11_star": [0.0] * n,
        "intensity_log2_offset_delta": [0.0] * n,
    }


def test_vec8_from_table_sequence_only():
    df = pd.DataFrame({"sequence": ["AAA", "BBB"], **_vec8_cols(2)})
    out = sfxi_vec8_from_table_v1(df, params={}, ctx=None)
    assert list(out.columns) == ["sequence", "y"]
    assert out["sequence"].tolist() == ["AAA", "BBB"]


def test_vec8_from_table_id_only_default_id():
    df = pd.DataFrame({"id": ["id1", "id2"], **_vec8_cols(2)})
    out = sfxi_vec8_from_table_v1(df, params={}, ctx=None)
    assert list(out.columns) == ["id", "y"]
    assert out["id"].tolist() == ["id1", "id2"]


def test_vec8_from_table_id_only_custom_column():
    df = pd.DataFrame({"design_id": ["d1", "d2"], **_vec8_cols(2)})
    out = sfxi_vec8_from_table_v1(df, params={"id_column": "design_id"}, ctx=None)
    assert list(out.columns) == ["id", "y"]
    assert out["id"].tolist() == ["d1", "d2"]


def test_vec8_from_table_id_and_sequence():
    df = pd.DataFrame(
        {
            "id": ["id1", "id2"],
            "sequence": ["AAA", np.nan],
            **_vec8_cols(2),
        }
    )
    out = sfxi_vec8_from_table_v1(df, params={}, ctx=None)
    assert list(out.columns) == ["id", "sequence", "y"]
    assert out["sequence"].iloc[0] == "AAA"
    assert pd.isna(out["sequence"].iloc[1])


def test_vec8_from_table_rejects_blank_sequence_when_required():
    df = pd.DataFrame({"sequence": ["", "   "], **_vec8_cols(2)})
    with pytest.raises(ValueError, match="sequence"):
        sfxi_vec8_from_table_v1(df, params={}, ctx=None)


def test_vec8_from_table_rejects_blank_id_when_required():
    df = pd.DataFrame({"id": ["id1", "  "], **_vec8_cols(2)})
    with pytest.raises(ValueError, match="id column"):
        sfxi_vec8_from_table_v1(df, params={}, ctx=None)


def test_vec8_from_table_enforces_delta_match(tmp_path):
    df = pd.DataFrame({"sequence": ["AAA"], **_vec8_cols(1)})
    df["intensity_log2_offset_delta"] = [0.25]
    params_ok = {
        "expected_log2_offset_delta": 0.25,
        "enforce_log2_offset_match": True,
    }
    out = sfxi_vec8_from_table_v1(df, params=params_ok, ctx=None)
    assert list(out.columns) == ["sequence", "y"]

    params_bad = {
        "expected_log2_offset_delta": 0.5,
        "enforce_log2_offset_match": True,
    }
    with pytest.raises(ValueError, match="delta mismatch"):
        sfxi_vec8_from_table_v1(df, params=params_bad, ctx=None)
