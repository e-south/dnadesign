"""Source, split, and plan-render helpers for the DenseGen OPAL probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .artifacts import ProbePlan
from .axis_oracle import build_train_ids
from .constants import AXIS_CLASS_TO_LOGIC4, ORACLE_ID, QUALITY_FLAGS, STATE_ORDER


def _quality_counts(labels: pd.DataFrame) -> dict[str, int]:
    counts = labels["quality_flag"].value_counts(dropna=False).to_dict()
    return {flag: int(counts.get(flag, 0)) for flag in QUALITY_FLAGS}


def _source_summary(labels: pd.DataFrame, *, run_root: Path, x_surface: Mapping[str, Any]) -> dict[str, Any]:
    counts = _quality_counts(labels)
    ok = counts.get("ok", 0)
    total = int(len(labels))
    class_counts = labels["axis_class"].value_counts(dropna=False).to_dict() if "axis_class" in labels.columns else {}
    return {
        "path_safety_pass": True,
        "forbidden_input_pass": True,
        "x_surface_pass": True,
        "x_surface": dict(x_surface),
        "quality_ok_fraction": float(ok / total) if total else 0.0,
        "quality_counts": counts,
        "axis_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "run_root": str(run_root),
        "oracle_id": ORACLE_ID,
        "state_order": list(STATE_ORDER),
    }


def _format_plan_text(
    *,
    plan: ProbePlan,
    safety: Mapping[str, Any],
    split_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        "opal_densegen_axis_probe_v0",
        f"mode: {'apply' if plan.apply else 'dry-run'}",
        f"run_root: {plan.run_root}",
        f"gate: {plan.gate or 'all'}",
        f"stop_after: {plan.stop_after}",
        f"rounds: {plan.rounds}",
        f"initial_label_count: {plan.initial_label_count}",
        f"selection_k: {plan.selection_k}",
        f"max_x_matrix_gib: {plan.max_x_matrix_gib or 'opal_default'}",
        f"score_batch_size: {plan.score_batch_size or 'opal_default'}",
        f"planned_runs: {len(plan.runs)}",
        f"quality_ok_fraction: {safety.get('quality_ok_fraction')}",
        f"x_surface: {safety.get('x_surface')}",
        "quality_flags:",
    ]
    for flag, count in dict(safety.get("quality_counts", {})).items():
        lines.append(f"  {flag}: {count}")
    lines.append("axis_class_counts:")
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        lines.append(f"  {axis_class}: {dict(safety.get('axis_class_counts', {})).get(axis_class, 0)}")
    lines.append("splits:")
    for split_id, metadata in split_metadata.items():
        extra = f", heldout_sigma35={metadata.get('heldout_sigma35')}" if metadata.get("heldout_sigma35") else ""
        lines.append(
            f"  {split_id}: train={len(metadata.get('train_ids', []))}, eval={len(metadata.get('eval_ids', []))}{extra}"
        )
    if plan.commands:
        lines.append("opal_commands:")
        lines.extend("  " + " ".join(map(str, command)) for command in plan.commands)
    elif not plan.apply:
        lines.append("next: add --apply to materialize source-gate labels/reports.")
    return "\n".join(lines) + "\n"


def _compact_split_metadata(split_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    compact: dict[str, dict[str, Any]] = {}
    for split_id, metadata in split_metadata.items():
        compact[split_id] = {
            "split_id": metadata.get("split_id", split_id),
            "initial_label_count": metadata.get("budget"),
            "per_class": metadata.get("per_class"),
            "class_budget": metadata.get("class_budget"),
            "seed": metadata.get("seed"),
            "heldout_sigma35": metadata.get("heldout_sigma35"),
            "train_count": len(metadata.get("train_ids", [])),
            "eval_count": len(metadata.get("eval_ids", [])),
        }
    return compact


def _persisted_split_metadata(split_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    compact = _compact_split_metadata(split_metadata)
    for split_id, metadata in compact.items():
        metadata["train_ids_path"] = f"{split_id}_train_ids.parquet"
        metadata["eval_ids_path"] = f"{split_id}_eval_ids.parquet"
    return compact


def _split_metadata_for_all(labels: pd.DataFrame, *, plan: ProbePlan) -> dict[str, dict[str, Any]]:
    metadata_by_split: dict[str, dict[str, Any]] = {}
    for split_id in tuple(dict.fromkeys(run.split_id for run in plan.runs)):
        train_ids, metadata = build_train_ids(
            labels,
            budget=plan.initial_label_count,
            seed=plan.seed,
            split_id=split_id,
            return_metadata=True,
        )
        metadata["train_ids"] = train_ids
        metadata_by_split[split_id] = metadata
    return metadata_by_split
