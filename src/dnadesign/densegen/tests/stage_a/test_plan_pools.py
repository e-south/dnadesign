"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_plan_pools.py

Tests for plan-scoped pool assembly helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.densegen.src.core.artifacts.pool import POOL_MODE_SEQUENCE, POOL_MODE_TFBS, PoolData
from dnadesign.densegen.src.core.pipeline.plan_pools import build_plan_pool, plan_pool_label


def _tfbs_pool(name: str, df: pd.DataFrame) -> PoolData:
    return PoolData(
        name=name,
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=df,
        sequences=df["tfbs"].tolist(),
        pool_path=Path("."),
    )


def _sequence_pool(name: str, sequences: list[str]) -> PoolData:
    return PoolData(
        name=name,
        input_type="sequence_library",
        pool_mode=POOL_MODE_SEQUENCE,
        df=None,
        sequences=list(sequences),
        pool_path=Path("."),
    )


def test_plan_pool_label_is_safe() -> None:
    assert plan_pool_label("demo plan") == "plan_pool__demo_plan"


def test_build_plan_pool_merges_tfbs_pools() -> None:
    df_a = pd.DataFrame(
        {
            "input_name": ["a"],
            "tf": ["alpha"],
            "tfbs": ["AAAA"],
            "tfbs_core": ["AAAA"],
            "motif_id": ["motif_a"],
            "tfbs_id": ["id_a"],
        }
    )
    df_b = pd.DataFrame(
        {
            "input_name": ["b"],
            "tf": ["beta"],
            "tfbs": ["CCCC"],
            "tfbs_core": ["CCCC"],
            "motif_id": ["motif_b"],
            "tfbs_id": ["id_b"],
            "best_hit_score": [1.0],
        }
    )
    pool = build_plan_pool(
        plan_name="plan_one",
        include_inputs=["a", "b"],
        pool_data={"a": _tfbs_pool("a", df_a), "b": _tfbs_pool("b", df_b)},
    )

    assert pool.name == plan_pool_label("plan_one")
    assert pool.pool_mode == POOL_MODE_TFBS
    assert pool.df is not None
    assert set(pool.df["input_name"].unique()) == {pool.name}
    assert set(pool.df["input_source_name"].unique()) == {"a", "b"}
    assert set(pool.df["source"].dropna().unique()) == {"a", "b"}
    assert "best_hit_score" in pool.df.columns


def test_build_plan_pool_rejects_mixed_pool_modes() -> None:
    df = pd.DataFrame(
        {
            "input_name": ["a"],
            "tf": ["alpha"],
            "tfbs": ["AAAA"],
            "tfbs_core": ["AAAA"],
            "motif_id": ["motif_a"],
            "tfbs_id": ["id_a"],
        }
    )
    with pytest.raises(ValueError, match="pool_mode"):
        build_plan_pool(
            plan_name="mixed",
            include_inputs=["a", "b"],
            pool_data={"a": _tfbs_pool("a", df), "b": _sequence_pool("b", ["TTTT"])},
        )
