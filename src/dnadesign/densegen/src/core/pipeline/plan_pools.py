"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/plan_pools.py

Plan-scoped pool assembly helpers for Stage-B sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...adapters.sources.stage_a.stage_a_paths import safe_label
from ..artifacts.pool import POOL_MODE_SEQUENCE, POOL_MODE_TFBS, PoolData

PLAN_POOL_INPUT_TYPE = "plan_pool"


@dataclass(frozen=True)
class PlanPoolSpec:
    plan_name: str
    pool_name: str
    include_inputs: list[str]
    pool: PoolData


@dataclass(frozen=True)
class PlanPoolSource:
    name: str
    type: str = PLAN_POOL_INPUT_TYPE


def plan_pool_label(plan_name: str) -> str:
    return f"plan_pool__{safe_label(plan_name)}"


def _aligned_pool_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    columns: list[str] = []
    for df in frames:
        for col in df.columns:
            if col not in columns:
                columns.append(col)
    aligned: list[pd.DataFrame] = []
    for df in frames:
        missing = [col for col in columns if col not in df.columns]
        for col in missing:
            df[col] = None
        aligned.append(df[columns])
    return pd.concat(aligned, ignore_index=True)


def build_plan_pool(
    *,
    plan_name: str,
    include_inputs: list[str],
    pool_data: dict[str, PoolData],
) -> PoolData:
    if not include_inputs:
        raise ValueError("plan pools require at least one input name")
    missing = [name for name in include_inputs if name not in pool_data]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"plan pools reference unknown inputs: {preview}")
    pools = [pool_data[name] for name in include_inputs]
    pool_modes = {pool.pool_mode for pool in pools}
    if len(pool_modes) != 1:
        preview = ", ".join(sorted(pool_modes))
        raise ValueError(f"plan pools require a single pool_mode, got: {preview}")
    pool_mode = next(iter(pool_modes))
    pool_name = plan_pool_label(plan_name)

    if pool_mode == POOL_MODE_TFBS:
        frames: list[pd.DataFrame] = []
        for pool in pools:
            if pool.df is None:
                raise ValueError(f"plan pool input '{pool.name}' missing TFBS dataframe")
            df = pool.df.copy()
            if "input_name" not in df.columns:
                df.insert(0, "input_name", pool.name)
            df["input_source_name"] = df["input_name"]
            df["input_name"] = pool_name
            frames.append(df)
        combined = _aligned_pool_frames(frames)
        sequences = combined["tfbs"].tolist() if "tfbs" in combined.columns else []
        return PoolData(
            name=pool_name,
            input_type=PLAN_POOL_INPUT_TYPE,
            pool_mode=POOL_MODE_TFBS,
            df=combined,
            sequences=sequences,
            pool_path=Path("."),
        )

    if pool_mode == POOL_MODE_SEQUENCE:
        sequences: list[str] = []
        for pool in pools:
            sequences.extend(pool.sequences)
        return PoolData(
            name=pool_name,
            input_type=PLAN_POOL_INPUT_TYPE,
            pool_mode=POOL_MODE_SEQUENCE,
            df=None,
            sequences=sequences,
            pool_path=Path("."),
        )

    raise ValueError(f"Unsupported plan pool_mode: {pool_mode}")


def build_plan_pools(
    *,
    plan_items: list,
    pool_data: dict[str, PoolData],
) -> dict[str, PlanPoolSpec]:
    plan_pools: dict[str, PlanPoolSpec] = {}
    for plan in plan_items:
        if not hasattr(plan, "include_inputs"):
            raise ValueError(f"plan '{plan.name}' missing include_inputs for plan-scoped pools")
        include_inputs = list(getattr(plan, "include_inputs") or [])
        pool = build_plan_pool(plan_name=str(plan.name), include_inputs=include_inputs, pool_data=pool_data)
        plan_pools[str(plan.name)] = PlanPoolSpec(
            plan_name=str(plan.name),
            pool_name=pool.name,
            include_inputs=include_inputs,
            pool=pool,
        )
    return plan_pools
