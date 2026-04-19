"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_b_summary_data.py

Data-normalization helpers for Stage-B summary plots.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.artifacts.pool import TFBSPoolArtifact
from ..core.record_metadata_recovery import recover_densegen_metadata_from_source
from .plot_stage_a_common import (
    _stage_a_non_background_sampling_rows,
    _stage_a_pool_regulator_column,
    _stage_a_pool_tfbs_column,
)


def summary_output_dir(out_path: Path) -> Path:
    return out_path.parent / "stage_b_summary"


def coerce_used_tfbs_entries(value: object) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    entries: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if "regulator" not in normalized and "tf" in normalized:
            normalized["regulator"] = normalized.get("tf")
        if "sequence" not in normalized and "tfbs" in normalized:
            normalized["sequence"] = normalized.get("tfbs")
        entries.append(normalized)
    return entries


def normalize_output_records(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Stage-B summary plots require DenseGen output records.")
    normalized = recover_densegen_metadata_from_source(df.copy())
    required = {"densegen__plan", "densegen__used_tfbs_detail"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"DenseGen output records missing required columns: {sorted(missing)}")
    return normalized


def safe_int(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def deployed_tfbs_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_output_records(df)
    rows: list[dict[str, object]] = []
    for _, row in normalized.iterrows():
        plan_name = str(row.get("densegen__plan") or "unscoped").strip() or "unscoped"
        for item in coerce_used_tfbs_entries(row.get("densegen__used_tfbs_detail")):
            if str(item.get("part_kind") or "tfbs").strip().lower() != "tfbs":
                continue
            regulator = str(item.get("regulator") or "").strip()
            if not regulator:
                continue
            sequence = str(item.get("sequence") or item.get("tfbs") or "").strip().upper()
            if not sequence:
                continue
            length_value = safe_int(item.get("length"))
            if length_value is None:
                length_value = len(sequence)
            if int(length_value) <= 0:
                continue
            rows.append(
                {
                    "plan_name": plan_name,
                    "regulator": regulator,
                    "sequence": sequence,
                    "length": int(length_value),
                }
            )
    if not rows:
        raise ValueError("DenseGen output records do not contain deployed TFBS annotations.")
    return pd.DataFrame(rows)


def retained_pool_frame(pools: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    if not pools:
        raise ValueError("Stage-B bridge summary plots require Stage-A pools.")
    rows: list[dict[str, object]] = []
    for input_name, pool_df in pools.items():
        if pool_df is None or pool_df.empty:
            continue
        regulator_col = _stage_a_pool_regulator_column(pool_df, input_name=input_name)
        tfbs_col = _stage_a_pool_tfbs_column(pool_df, input_name=input_name)
        if regulator_col not in pool_df.columns or tfbs_col not in pool_df.columns:
            continue
        columns = [regulator_col, tfbs_col]
        has_tier = "tier" in pool_df.columns
        has_core = "tfbs_core" in pool_df.columns
        has_score = "best_hit_score" in pool_df.columns
        if has_tier:
            columns.append("tier")
        if has_core:
            columns.append("tfbs_core")
        if has_score:
            columns.append("best_hit_score")
        for values in pool_df[columns].itertuples(index=False, name=None):
            cursor = 0
            regulator = values[cursor]
            cursor += 1
            sequence = values[cursor]
            cursor += 1
            if has_tier:
                tier = values[cursor]
                cursor += 1
            else:
                tier = None
            if has_core:
                core = values[cursor]
                cursor += 1
            else:
                core = None
            if has_score:
                best_hit_score = values[cursor]
            else:
                best_hit_score = None
            regulator_text = str(regulator or "").strip()
            sequence_text = str(sequence or "").strip().upper()
            if not regulator_text or not sequence_text:
                continue
            tier_value = safe_int(tier) if has_tier else None
            core_text = str(core or "").strip().upper() if core is not None else ""
            try:
                score_value = float(best_hit_score) if best_hit_score is not None else np.nan
            except Exception:
                score_value = np.nan
            rows.append(
                {
                    "input_name": input_name,
                    "regulator": regulator_text,
                    "sequence": sequence_text,
                    "length": len(sequence_text),
                    "tier": tier_value,
                    "core_sequence": core_text or sequence_text,
                    "best_hit_score": score_value,
                }
            )
    if not rows:
        raise ValueError("Stage-A pools do not contain retained TFBS sequences.")
    return pd.DataFrame(rows)


def sampling_summary_frame(pool_manifest: TFBSPoolArtifact | None) -> pd.DataFrame:
    if pool_manifest is None:
        raise ValueError("Upstream evidence quality summary requires a Stage-A pool manifest.")
    rows: list[dict[str, object]] = []
    for input_name, entry in pool_manifest.inputs.items():
        sampling = entry.stage_a_sampling
        if sampling is None:
            continue
        for row in _stage_a_non_background_sampling_rows(input_name, sampling):
            theoretical_max = float(row.get("pwm_theoretical_max_score") or 0.0)
            consensus_score = float(row.get("pwm_consensus_score") or 0.0)
            rows.append(
                {
                    "input_name": input_name,
                    "regulator": str(row.get("regulator") or "").strip(),
                    "candidates_with_hit": int(row.get("candidates_with_hit") or 0),
                    "eligible_unique": int(row.get("eligible_unique") or 0),
                    "retained": int(row.get("retained") or 0),
                    "consensus_score": consensus_score,
                    "consensus_ratio": (consensus_score / theoretical_max) if theoretical_max > 0 else np.nan,
                }
            )
    if not rows:
        raise ValueError("Stage-A pool manifest does not contain non-background sampling summaries.")
    return pd.DataFrame(rows)


__all__ = [
    "coerce_used_tfbs_entries",
    "deployed_tfbs_frame",
    "normalize_output_records",
    "retained_pool_frame",
    "safe_int",
    "sampling_summary_frame",
    "summary_output_dir",
]
