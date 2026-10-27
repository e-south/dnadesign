"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run_inputs.py

Input-normalization helpers for DenseGen run-level plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .plot_run_helpers import _normalize_plan_name


def load_plan_quotas_from_effective_config(cfg: object) -> dict[str, int]:
    def _has_target_value(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        text = str(value).strip().lower()
        return text not in {"", "none", "nan"}

    def _plan_target(item: dict) -> object | None:
        quota_raw = item.get("quota")
        if _has_target_value(quota_raw):
            return quota_raw
        sequences_raw = item.get("sequences")
        if _has_target_value(sequences_raw):
            return sequences_raw
        return None

    if not cfg or not isinstance(cfg, dict):
        return {}
    candidate_paths = [
        ("densegen", "generation", "plan"),
        ("config", "densegen", "generation", "plan"),
        ("generation", "plan"),
        ("config", "generation", "plan"),
    ]
    plan_items: list[dict] = []
    for path in candidate_paths:
        node: object = cfg
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, list):
            valid = [
                item
                for item in node
                if isinstance(item, dict)
                and _normalize_plan_name(item.get("name")) is not None
                and _plan_target(item) is not None
            ]
            if valid:
                plan_items = valid
                break
    quotas: dict[str, int] = {}
    for item in plan_items:
        if not isinstance(item, dict):
            continue
        name = _normalize_plan_name(item.get("name"))
        quota_raw = _plan_target(item)
        if name is None:
            continue
        try:
            quota = int(quota_raw)
        except Exception:
            continue
        if quota > 0:
            quotas[name] = quota
    return quotas


def normalize_and_order_attempts(attempts_df: pd.DataFrame) -> pd.DataFrame:
    normalized = attempts_df.copy()
    normalized["status"] = normalized["status"].astype(str).str.strip().str.lower()
    allowed = {"ok", "rejected", "duplicate", "failed"}
    unknown = sorted({status for status in normalized["status"].tolist() if status not in allowed})
    if unknown:
        raise ValueError(
            f"Unknown attempt status values in attempts.parquet. Allowed statuses: {sorted(allowed)}. Found: {unknown}"
        )
    normalized["_row_order"] = np.arange(len(normalized), dtype=int)
    normalized["created_at"] = pd.to_datetime(normalized.get("created_at"), errors="coerce")
    if normalized["created_at"].notna().any():
        normalized = normalized.sort_values(["created_at", "_row_order"], kind="mergesort")
    else:
        normalized = normalized.sort_values(["_row_order"], kind="mergesort")
    normalized = normalized.reset_index(drop=True)
    normalized["plan_name"] = normalized["plan_name"].map(_normalize_plan_name).fillna("all plans")
    normalized["run_order"] = np.arange(1, len(normalized) + 1, dtype=int)
    return normalized
