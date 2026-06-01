"""Manifest-ready report and provenance metadata for TFBS nulls."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from ..schema import (
    TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
    TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES,
    TFBS_LEARNABILITY_ORACLE_VERSION,
)
from .contracts import TfbsNullConfig
from .strata import _group_indices, _SelectedStratum, _stratum_key_label


def _null_viability_report(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    null_version: str,
    seed: int,
    label_name: str,
    selected: _SelectedStratum,
    config: TfbsNullConfig,
    compare_columns: Sequence[str],
    label_joint_columns: Sequence[str],
    null_control_role: str,
    preserved_signal: str,
    disrupted_signal: str,
    negative_control_claim_status: str,
    warnings: Sequence[str],
) -> dict[str, Any]:
    unchanged_fraction = _unchanged_fraction(before, after, columns=compare_columns)
    leakage = _label_leakage_assessment(unchanged_fraction, config=config)
    report = {
        "schema_version": "stress_ethanol_cipro_growth.densegen_tfbs_null_viability.v1",
        "oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
        "null_version": null_version,
        "null_control_role": null_control_role,
        "preserved_signal": preserved_signal,
        "disrupted_signal": disrupted_signal,
        "negative_control_claim_status": negative_control_claim_status,
        "seed": int(seed),
        "row_count": int(len(before)),
        "label_name": label_name,
        "stratum_key": list(selected.stratum_columns),
        "stratum_count": selected.stratum_count,
        "min_rows_per_stratum": selected.min_rows_per_stratum,
        "median_rows_per_stratum": selected.median_rows_per_stratum,
        "max_rows_per_stratum": selected.max_rows_per_stratum,
        "fraction_rows_in_singleton_strata": selected.fraction_rows_in_singleton_strata,
        "fraction_rows_in_tiny_strata": selected.fraction_rows_in_tiny_strata,
        "configured_tiny_stratum_threshold": config.tiny_stratum_threshold,
        "unchanged_label_fraction_after_permutation": unchanged_fraction,
        "label_leakage_assessment": leakage,
        "label_marginal_before": _label_marginal(before, label_name),
        "label_marginal_after": _label_marginal(after, label_name),
        "label_joint_summary_before": _joint_summary(before, label_joint_columns),
        "label_joint_summary_after": _joint_summary(after, label_joint_columns),
        "permutation_entropy": _permutation_entropy(before, selected.stratum_columns),
        "estimated_effective_permutation_count": _effective_permutation_count(before, selected.stratum_columns),
        "coarsening_steps_applied": list(selected.coarsening_steps_applied),
        "viability_status": selected.viability_status,
        "warnings": [*warnings, *_unchanged_warnings(leakage)],
    }
    if report["label_marginal_before"] != report["label_marginal_after"]:
        report["viability_status"] = "FAIL_LABEL_DISTRIBUTION_CHANGED"
    if report["viability_status"] not in TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES:
        raise ValueError(f"unsupported TFBS null viability status: {report['viability_status']}")
    return report


def _with_null_metadata(
    labels: pd.DataFrame,
    *,
    null_version: str,
    null_control_role: str,
    negative_control_claim_status: str,
    seed: int,
    positive_labels: pd.DataFrame,
    selected: _SelectedStratum,
) -> pd.DataFrame:
    out = labels.copy()
    out["null_version"] = null_version
    out["null_control_role"] = null_control_role
    out["negative_control_claim_status"] = negative_control_claim_status
    out["null_seed"] = int(seed)
    out["positive_oracle_version"] = TFBS_LEARNABILITY_ORACLE_VERSION
    out["positive_label_table_hash"] = _dataframe_hash(positive_labels)
    out["null_recipe_hash"] = _null_recipe_hash(null_version=null_version, seed=seed, selected=selected)
    out["stratum_key"] = _stratum_key_label(selected.stratum_columns)
    out["coarsening_steps_applied"] = json.dumps(list(selected.coarsening_steps_applied), separators=(",", ":"))
    out["viability_status"] = selected.viability_status
    return out


def _content_warnings(selected: _SelectedStratum) -> list[str]:
    if selected.viability_status == "PASS_WITH_COARSENING":
        return ["family-content null required declared stratum coarsening"]
    return []


def _slot_warnings(selected: _SelectedStratum) -> list[str]:
    warnings = ["slot-geometry null preserves row-level LexA/CpxR/BaeR counts before permuting slot families"]
    if selected.viability_status == "PASS_WITH_COARSENING":
        warnings.append("slot-geometry null required declared stratum coarsening")
    return warnings


def _label_marginal(frame: pd.DataFrame, label_name: str) -> dict[str, int] | None:
    if label_name not in frame.columns:
        return None
    counts = frame[label_name].value_counts(dropna=False).sort_index().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _joint_summary(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    selected_columns = [column for column in columns if column in frame.columns]
    if not selected_columns:
        return {"columns": [], "unique_patterns": 0, "top_patterns": []}
    counts = frame.loc[:, selected_columns].astype(str).value_counts(dropna=False).sort_values(ascending=False).head(20)
    top_patterns = [
        {"pattern": list(key if isinstance(key, tuple) else (key,)), "count": int(value)}
        for key, value in counts.items()
    ]
    return {
        "columns": selected_columns,
        "unique_patterns": int(frame.loc[:, selected_columns].astype(str).drop_duplicates().shape[0]),
        "top_patterns": top_patterns,
        "summary_hash": _summary_hash(frame, selected_columns),
    }


def _unchanged_fraction(before: pd.DataFrame, after: pd.DataFrame, *, columns: Sequence[str]) -> float:
    selected_columns = [column for column in columns if column in before.columns and column in after.columns]
    if not selected_columns:
        return 0.0
    left = before.loc[:, selected_columns].astype(str).agg("\x1f".join, axis=1)
    right = after.loc[:, selected_columns].astype(str).agg("\x1f".join, axis=1)
    return float((left == right).mean())


def _permutation_entropy(frame: pd.DataFrame, stratum_columns: tuple[str, ...]) -> float:
    return float(sum(math.lgamma(len(indices) + 1) for indices in _group_indices(frame, stratum_columns)))


def _effective_permutation_count(frame: pd.DataFrame, stratum_columns: tuple[str, ...]) -> int | str:
    log_count = _permutation_entropy(frame, stratum_columns)
    if log_count > math.log(10) * 100:
        return f">1e{int(log_count / math.log(10))}"
    return int(round(math.exp(log_count)))


def _label_leakage_assessment(unchanged_fraction: float, *, config: TfbsNullConfig) -> dict[str, Any]:
    if unchanged_fraction >= config.fail_if_unchanged_label_fraction_ge:
        status = "FAIL_HIGH_LABEL_RETENTION"
    elif unchanged_fraction >= config.warn_if_unchanged_label_fraction_ge:
        status = "WARN_HIGH_LABEL_RETENTION"
    else:
        status = "PASS"
    return {
        "status": status,
        "unchanged_label_fraction": unchanged_fraction,
        "warn_if_unchanged_label_fraction_ge": float(config.warn_if_unchanged_label_fraction_ge),
        "fail_if_unchanged_label_fraction_ge": float(config.fail_if_unchanged_label_fraction_ge),
    }


def _unchanged_warnings(leakage: Mapping[str, Any]) -> list[str]:
    status = str(leakage.get("status") or "")
    unchanged_fraction = float(leakage.get("unchanged_label_fraction") or 0.0)
    if status == "FAIL_HIGH_LABEL_RETENTION":
        return [
            "null/control retained too much target-label identity to support a primary negative-control claim "
            f"(unchanged_fraction={unchanged_fraction:.3f})"
        ]
    if status == "WARN_HIGH_LABEL_RETENTION":
        return [
            "null/control retained substantial target-label identity; "
            "interpret as a conservative or confounded control "
            f"(unchanged_fraction={unchanged_fraction:.3f})"
        ]
    return []


def _null_recipe_hash(*, null_version: str, seed: int, selected: _SelectedStratum) -> str:
    payload = {
        "positive_recipe_hash": TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
        "null_version": null_version,
        "seed": int(seed),
        "stratum_key": list(selected.stratum_columns),
        "coarsening_steps_applied": list(selected.coarsening_steps_applied),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _dataframe_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame.astype(str), index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _summary_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = frame.loc[:, list(columns)].astype(str).sort_values(list(columns)).to_json(orient="split")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
