"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/configs/scopes.py

Candidate-scope materialization for Stage B TFBS config generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...candidate_scopes import build_count_fixed_slot_position_scope, filter_labels_to_scope
from ...profiles import is_count_fixed_slot_position_profile_id
from ...stage_a.manifests import file_sha256
from ..io import read_stage_b_label_table, write_stage_b_candidate_scope, write_stage_b_json, write_stage_b_parquet
from ..layout import TfbsStageBLayout
from ..seed import select_tfbs_stage_b_paired_initial_ids
from .contracts import TfbsStageBConfig


def materialize_label_scope_artifacts(
    *,
    layout: TfbsStageBLayout,
    label_name: str,
    target_profile_id: str,
    positive_label_table: pd.DataFrame,
    null_label_table: pd.DataFrame,
    pair_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Write label-specific scope artifacts when the target profile requires them."""

    if not uses_count_fixed_scope(target_profile_id):
        return {
            "candidate_scope_path": layout.candidate_scope_path,
            "positive_label_table_path": Path(str(pair_row["positive_label_table_path"])),
            "positive_label_table_hash": str(pair_row["positive_label_table_hash"]),
            "null_label_table_path": Path(str(pair_row["null_label_table_path"])),
            "null_label_table_hash": str(pair_row["null_label_table_hash"]),
            "candidate_scope_metadata": {},
        }

    scope = build_count_fixed_slot_position_scope(positive_label_table, label_name=label_name)
    candidate_scope_path = layout.label_candidate_scope_path(label_name)
    write_stage_b_candidate_scope(candidate_scope_path, scope.ids)
    positive_scoped = filter_labels_to_scope(positive_label_table, scope=scope)
    null_scoped = filter_labels_to_scope(null_label_table, scope=scope)
    _validate_count_fixed_label_table(
        positive_scoped,
        label_name=label_name,
        scope_manifest=scope.to_manifest(),
        surface="positive label table",
    )
    _validate_count_fixed_label_table(
        null_scoped,
        label_name=label_name,
        scope_manifest=scope.to_manifest(),
        surface="control label table",
    )
    _validate_count_fixed_control_distribution(positive_scoped, null_scoped, label_name=label_name)
    positive_path = layout.scoped_label_table_path(label_name, "positive")
    null_path = layout.scoped_label_table_path(label_name, "matched_null")
    write_stage_b_parquet(positive_path, positive_scoped)
    write_stage_b_parquet(null_path, null_scoped)
    scope_manifest_path = layout.label_candidate_scope_manifest_path(label_name)
    scope_manifest = {
        **scope.to_manifest(),
        "candidate_scope_path": str(candidate_scope_path),
        "candidate_scope_hash": file_sha256(candidate_scope_path),
        "positive_label_table_path": str(positive_path),
        "positive_label_table_hash": file_sha256(positive_path),
        "null_label_table_path": str(null_path),
        "null_label_table_hash": file_sha256(null_path),
        "null_control_role": str(pair_row.get("null_control_role") or ""),
        "negative_control_claim_status": str(pair_row.get("negative_control_claim_status") or ""),
    }
    write_stage_b_json(scope_manifest_path, scope_manifest)
    return {
        "candidate_scope_path": candidate_scope_path,
        "positive_label_table_path": positive_path,
        "positive_label_table_hash": file_sha256(positive_path),
        "null_label_table_path": null_path,
        "null_label_table_hash": file_sha256(null_path),
        "candidate_scope_metadata": {
            **scope.to_manifest(),
            "candidate_scope_manifest_path": str(scope_manifest_path),
            "candidate_scope_manifest_hash": file_sha256(scope_manifest_path),
        },
    }


def select_shared_initial_ids(
    *,
    cfg: TfbsStageBConfig,
    label_name: str,
    positive_label_table_path: Path,
    null_label_table_path: Path,
    target_profile_id: str,
    initial_seed_context: str,
) -> tuple[str, ...]:
    """Select shared positive/control initial IDs for a campaign pair."""

    positive_label_table = read_stage_b_label_table(positive_label_table_path)
    return select_tfbs_stage_b_paired_initial_ids(
        positive_label_table,
        read_stage_b_label_table(null_label_table_path),
        label_name=label_name,
        initial_label_count=cfg.initial_label_count,
        seed=cfg.seed,
        policy=cfg.initial_seed_policy,
        seed_context=initial_seed_context,
    )


def uses_count_fixed_scope(target_profile_id: str) -> bool:
    return is_count_fixed_slot_position_profile_id(target_profile_id)


def control_pair_label(*, target_profile_id: str) -> str:
    if uses_count_fixed_scope(target_profile_id):
        return "Sequence-matched metadata vs slot-shuffled control"
    return "Sequence-matched metadata vs row-shuffled control"


def control_role_display_label(*, target_profile_id: str) -> str:
    if uses_count_fixed_scope(target_profile_id):
        return "Slot-shuffled control"
    return "Row-shuffled control"


def _validate_count_fixed_label_table(
    frame: pd.DataFrame,
    *,
    label_name: str,
    scope_manifest: Mapping[str, Any],
    surface: str,
) -> None:
    count_column = str(scope_manifest["target_family_count_column"])
    required_count = int(scope_manifest["required_count_value"])
    missing = sorted({"id", label_name, count_column} - set(frame.columns))
    if missing:
        raise ValueError(f"count-fixed {surface} missing column(s): {missing}")
    observed = pd.to_numeric(frame[count_column], errors="raise")
    bad = frame.loc[observed != required_count, "id"].astype(str).head(10).tolist()
    if bad:
        raise ValueError(
            f"count-fixed {surface} contains out-of-scope row(s) for {label_name}; "
            f"expected {count_column} == {required_count}, sample={bad}"
        )
    if pd.to_numeric(frame[label_name], errors="raise").nunique(dropna=False) < 2:
        raise ValueError(f"count-fixed {surface} is degenerate for {label_name}")


def _validate_count_fixed_control_distribution(
    positive: pd.DataFrame,
    control: pd.DataFrame,
    *,
    label_name: str,
) -> None:
    positive_counts = pd.to_numeric(positive[label_name], errors="raise").value_counts(dropna=False).sort_index()
    control_counts = pd.to_numeric(control[label_name], errors="raise").value_counts(dropna=False).sort_index()
    if not positive_counts.equals(control_counts):
        raise ValueError(
            "count-fixed shuffled-slot control changed target-label marginal "
            f"for {label_name}: positive={positive_counts.to_dict()} control={control_counts.to_dict()}"
        )
    if set(positive["id"].astype(str)) != set(control["id"].astype(str)):
        raise ValueError(f"count-fixed positive/control candidate scopes differ for {label_name}")
