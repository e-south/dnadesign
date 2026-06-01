"""Notebook visual vocabulary for TFBS Stage B review surfaces."""

from __future__ import annotations

from pathlib import Path


def realized_visual_id(kind: str, *, label_name: str) -> str:
    slug = slug_token(label_name)
    if kind == "realized_label_lift_trajectory":
        return f"tfbs_stage_b_{slug}_realized_label_lift_trajectory"
    if kind == "positive_null_lift_summary":
        return f"tfbs_stage_b_{slug}_positive_null_lift_summary"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_visual_label(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return "Realized selected true-label lift trajectory"
    if kind == "positive_null_lift_summary":
        return "Realized positive-minus-null lift summary"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_group_key(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return "label_oracle_kind"
    if kind == "positive_null_lift_summary":
        return "peer_review_claim_status"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_metric_name(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return "selected_true_lift_ratio"
    if kind == "positive_null_lift_summary":
        return "positive_minus_null_lift_ratio"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_metric_label(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return "Selected true-label lift ratio"
    if kind == "positive_null_lift_summary":
        return "Positive-minus-null lift ratio"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_metric_expression(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return "selected_true_mean / pool_baseline"
    if kind == "positive_null_lift_summary":
        return "positive_lift_ratio - null_or_control_lift_ratio"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_summary_name(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return "per_round"
    if kind == "positive_null_lift_summary":
        return "final_and_normalized_auc"
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_tidy_csv_path(*, kind: str, trajectory_csv_path: Path, pair_summary_csv_path: Path) -> Path:
    if kind == "realized_label_lift_trajectory":
        return trajectory_csv_path
    if kind == "positive_null_lift_summary":
        return pair_summary_csv_path
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def realized_caption(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return (
            "Realized selected-label lift by round, computed by joining selected row IDs to the positive or "
            "null/control oracle label table. The square marker is the initial labeled seed batch; round 0 is the "
            "first model-selected acquisition batch after those labels are ingested. This is the learnability "
            "evidence surface; predicted score remains an acquisition trace."
        )
    if kind == "positive_null_lift_summary":
        return (
            "Final and normalized trajectory positive-minus-null/control lift for each sentinel TFBS label. "
            "Rows marked as confound controls should not be interpreted as clean negative-control separation."
        )
    raise ValueError(f"Unsupported Stage B realized review plot kind: {kind!r}")


def slot_visual_id(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return "tfbs_stage_b_slot_target_count_mean_trajectory"
    if kind == "slot_count_stratified_lift_trajectory":
        return "tfbs_stage_b_slot_count_stratified_lift_trajectory"
    if kind == "slot_count_stratified_lift_summary":
        return "tfbs_stage_b_slot_count_stratified_lift_summary"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_visual_label(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return "Slot selected target-family count trajectory"
    if kind == "slot_count_stratified_lift_trajectory":
        return "Slot count-stratified lift trajectory"
    if kind == "slot_count_stratified_lift_summary":
        return "Slot count-stratified positive-minus-null summary"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_group_key(kind: str) -> str:
    if kind in {"slot_target_count_mean_trajectory", "slot_count_stratified_lift_trajectory"}:
        return "label_oracle_kind"
    if kind == "slot_count_stratified_lift_summary":
        return "slot_diagnostic_status"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_metric_name(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return "selected_target_count_mean"
    if kind == "slot_count_stratified_lift_trajectory":
        return "count_stratified_lift_ratio"
    if kind == "slot_count_stratified_lift_summary":
        return "positive_minus_null_count_stratified_lift_ratio"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_metric_label(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return "Selected target-family count mean"
    if kind == "slot_count_stratified_lift_trajectory":
        return "Count-stratified slot-label lift ratio"
    if kind == "slot_count_stratified_lift_summary":
        return "Positive-minus-null count-stratified lift ratio"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_metric_expression(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return "mean(selected target-family count)"
    if kind == "slot_count_stratified_lift_trajectory":
        return "selected_nondeterministic_true_mean / selected_count_stratum_baseline"
    if kind == "slot_count_stratified_lift_summary":
        return "positive_count_stratified_lift_ratio - null_or_control_count_stratified_lift_ratio"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_summary_name(kind: str) -> str:
    if kind in {"slot_target_count_mean_trajectory", "slot_count_stratified_lift_trajectory"}:
        return "per_round"
    if kind == "slot_count_stratified_lift_summary":
        return "final_and_normalized_auc"
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_tidy_csv_path(
    *,
    kind: str,
    trajectory_csv_path: Path,
    pair_summary_csv_path: Path,
    count_distribution_csv_path: Path,
) -> Path:
    if kind in {"slot_target_count_mean_trajectory", "slot_count_stratified_lift_trajectory"}:
        return trajectory_csv_path
    if kind == "slot_count_stratified_lift_summary":
        return pair_summary_csv_path
    if kind == "slot_count_distribution":
        return count_distribution_csv_path
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slot_caption(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return (
            "Selected target-family count by round for slot-label campaigns. A null/control can look strong when "
            "OPAL selects rows with high target-family count rather than learning slot position."
        )
    if kind == "slot_count_stratified_lift_trajectory":
        return (
            "Slot-label lift after excluding deterministic count strata and comparing selected rows with the "
            "baseline for their own target-family count strata."
        )
    if kind == "slot_count_stratified_lift_summary":
        return (
            "Final and normalized trajectory positive-minus-null/control lift for slot labels after controlling "
            "for selected target-family count composition."
        )
    raise ValueError(f"Unsupported Stage B slot diagnostic plot kind: {kind!r}")


def slug_token(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "label"
