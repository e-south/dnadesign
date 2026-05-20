"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import ProbePlan, RunSpec
from .axis_oracle import _ok_labels, build_train_ids
from .constants import (
    AXIS_CLASS_TO_LOGIC4,
    DEFAULT_TOP_K,
    NULL_ORACLE_ID,
    ORACLE_ID,
    QUALITY_FLAGS,
    STATE_ORDER,
)
from .prediction_ledger import prediction_id_problems, read_probe_predictions
from .prediction_scoring import (
    label_lookup,
    macro_f1,
    predicted_axis_classes,
    top_ids_from_prediction_frame,
    validate_prediction_selection_contract,
)

PASS_CIPRO_RANDOM_GATE = "PASS_CIPRO_RANDOM_GATE"
PASS_RANDOM_ALL_GATE = "PASS_RANDOM_ALL_GATE"
PASS_LEAVE_SIGMA35_GATE = "PASS_LEAVE_SIGMA35_GATE"
PASS_FULL_MATRIX_GATE = "PASS_FULL_MATRIX_GATE"
PASS_SCOPED_GATE = "PASS_SCOPED_GATE"


def _evaluate_run(
    *,
    run: RunSpec,
    positive_labels: pd.DataFrame,
    run_labels: pd.DataFrame,
    split_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        predictions = read_probe_predictions(run.config_path)
    except Exception as exc:
        raise RuntimeError(f"failed to read OPAL predictions for scored run {run.run_key}: {exc}") from exc
    true_labels = label_lookup(positive_labels)
    oracle_labels = label_lookup(run_labels)
    train_ids = set(map(str, split_metadata["train_ids"]))
    if run.split_id == "leave_sigma35_variant":
        eval_ids = set(map(str, split_metadata["eval_ids"]))
    else:
        eval_ids = set(_ok_labels(positive_labels)["id"].astype(str).tolist()) - train_ids

    row: dict[str, Any] = {
        "run_key": run.run_key,
        "campaign": run.campaign_key,
        "oracle_id": run.oracle_id,
        "split_id": run.split_id,
        "target_class": run.target_class,
        "train_count": int(len(train_ids)),
        "eval_count": int(len(eval_ids)),
    }
    if predictions.empty:
        raise RuntimeError(f"missing OPAL prediction artifacts for scored run {run.run_key}: {run.workdir}")
    run_ids = sorted({str(value) for value in predictions["run_id"].dropna().tolist()})
    if len(run_ids) != 1:
        raise RuntimeError(f"OPAL prediction artifacts for scored run {run.run_key} must contain one run_id")
    rounds = sorted({int(value) for value in predictions["as_of_round"].dropna().tolist()})
    if len(rounds) != 1:
        raise RuntimeError(f"OPAL prediction artifacts for scored run {run.run_key} must contain one as_of_round")
    row["run_id"] = run_ids[0]
    row["as_of_round"] = rounds[0]
    required_prediction_columns = {
        "id",
        "pred__y_hat_model",
        "pred__score_selected",
        "sel__is_selected",
        "sel__rank_competition",
    }
    missing_prediction_columns = sorted(required_prediction_columns - set(predictions.columns))
    if missing_prediction_columns:
        raise RuntimeError(
            f"OPAL prediction artifacts for scored run {run.run_key} missing column(s): {missing_prediction_columns}"
        )
    prediction_problems = prediction_id_problems(predictions, eval_ids, run_key=run.run_key)
    if prediction_problems:
        raise RuntimeError(f"OPAL prediction artifacts invalid: {'; '.join(prediction_problems)}")
    validate_prediction_selection_contract(predictions)

    frame = predictions.loc[predictions["id"].astype(str).isin(eval_ids)].copy()
    frame["pred_axis_class"] = predicted_axis_classes(frame["pred__y_hat_model"].tolist())
    frame["true_axis_class"] = frame["id"].astype(str).map(true_labels)
    frame["oracle_axis_class"] = frame["id"].astype(str).map(oracle_labels)
    row["axis4_macro_f1_true"] = macro_f1(frame["true_axis_class"].astype(str), frame["pred_axis_class"].astype(str))
    row["axis4_macro_f1_oracle"] = macro_f1(
        frame["oracle_axis_class"].astype(str), frame["pred_axis_class"].astype(str)
    )

    selected_ids = top_ids_from_prediction_frame(frame, k=DEFAULT_TOP_K)
    selected_true = true_labels.reindex(selected_ids).astype(str).tolist()
    selected_oracle = oracle_labels.reindex(selected_ids).astype(str).tolist()
    true_eval_classes = frame["true_axis_class"].astype(str).tolist()
    oracle_eval_classes = frame["oracle_axis_class"].astype(str).tolist()

    def _lift(classes: Sequence[str], selected_classes: Sequence[str]) -> tuple[float, float, float]:
        if not classes or not selected_classes:
            return float("nan"), float("nan"), float("nan")
        prevalence = float(np.mean([axis_class == run.target_class for axis_class in classes]))
        precision = float(np.mean([axis_class == run.target_class for axis_class in selected_classes]))
        lift = float("nan") if prevalence == 0.0 else precision / prevalence
        return prevalence, precision, lift

    prev_true, prec_true, lift_true = _lift(true_eval_classes, selected_true)
    prev_oracle, prec_oracle, lift_oracle = _lift(oracle_eval_classes, selected_oracle)
    row.update(
        {
            "selected_ids": selected_ids,
            "target_class_prevalence_true": prev_true,
            "selected_target_precision_at_k_true": prec_true,
            "target_lift_at_k_true": lift_true,
            "target_class_prevalence_oracle": prev_oracle,
            "selected_target_precision_at_k_oracle": prec_oracle,
            "target_lift_at_k_oracle": lift_oracle,
            "off_target_class_distribution_true": {
                axis_class: int(selected_true.count(axis_class)) for axis_class in AXIS_CLASS_TO_LOGIC4
            },
        }
    )
    return row


def _decision_from_metrics(metrics: list[dict[str, Any]], safety: Mapping[str, Any]) -> str:
    if not safety.get("path_safety_pass") or not safety.get("forbidden_input_pass") or not safety.get("x_surface_pass"):
        return "STOP"
    if not metrics:
        return "PENDING"
    if any(row.get("status") == "missing_predictions" for row in metrics):
        raise ValueError("metrics contain missing_predictions; scored runs must fail before decision")
    evaluable = [row for row in metrics if row.get("status") != "missing_predictions"]
    if not evaluable:
        return "DEBUG"

    def _finite_metric(row: Mapping[str, Any], key: str) -> float | None:
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    null_lifts = [
        value
        for row in evaluable
        if row.get("oracle_id") == NULL_ORACLE_ID
        for value in [_finite_metric(row, "target_lift_at_k_true")]
        if value is not None
    ]
    if null_lifts and max(null_lifts) > 1.25:
        return "STOP"
    pair_keys = sorted({(str(row.get("campaign")), str(row.get("split_id"))) for row in evaluable})
    if not pair_keys:
        return "DEBUG"
    for campaign, split_id in pair_keys:
        positive = [
            value
            for row in evaluable
            if row.get("campaign") == campaign and row.get("split_id") == split_id and row.get("oracle_id") == ORACLE_ID
            for value in [_finite_metric(row, "target_lift_at_k_true")]
            if value is not None
        ]
        null = [
            value
            for row in evaluable
            if row.get("campaign") == campaign
            and row.get("split_id") == split_id
            and row.get("oracle_id") == NULL_ORACLE_ID
            for value in [_finite_metric(row, "target_lift_at_k_true")]
            if value is not None
        ]
        if not positive or not null:
            return "DEBUG"
        if max(positive) <= max(null):
            return "DEBUG"
    return _pass_decision_for_coverage(evaluable)


def _pass_decision_for_coverage(metrics: Sequence[Mapping[str, Any]]) -> str:
    campaigns = {str(row.get("campaign")) for row in metrics}
    splits = {str(row.get("split_id")) for row in metrics}
    pairs = {(str(row.get("campaign")), str(row.get("split_id"))) for row in metrics}
    all_campaigns = {"cipro", "ethanol", "dual"}
    if campaigns == {"cipro"} and splits == {"random_id"}:
        return PASS_CIPRO_RANDOM_GATE
    if all_campaigns.issubset(campaigns) and splits == {"random_id"}:
        return PASS_RANDOM_ALL_GATE
    if all_campaigns.issubset(campaigns) and splits == {"leave_sigma35_variant"}:
        return PASS_LEAVE_SIGMA35_GATE
    gate_splits = ("random_id", "leave_sigma35_variant")
    required_pairs = {(campaign, split_id) for campaign in all_campaigns for split_id in gate_splits}
    if required_pairs.issubset(pairs):
        return PASS_FULL_MATRIX_GATE
    return PASS_SCOPED_GATE


def _claim_statuses(metrics: list[dict[str, Any]], *, decision: str) -> dict[str, str]:
    if decision == "PENDING":
        deferred = "not evaluated until OPAL run metrics exist"
        return {
            "H-NULL": deferred,
            "H-CIPRO": deferred,
            "H-ETHANOL": deferred,
            "H-DUAL": deferred,
            "H-SIGMA35": deferred,
            "H-COLLAPSE": deferred,
        }
    evaluable = [
        row
        for row in metrics
        if row.get("status") != "missing_predictions" and row.get("target_lift_at_k_true") is not None
    ]
    campaigns = {str(row.get("campaign")) for row in evaluable}
    splits = {str(row.get("split_id")) for row in evaluable}
    has_null = any(row.get("oracle_id") == NULL_ORACLE_ID for row in evaluable)
    has_selection = any(row.get("selected_ids") for row in evaluable)
    return {
        "H-NULL": "evaluated" if has_null else "not evaluated in this run",
        "H-CIPRO": "evaluated" if "cipro" in campaigns else "not evaluated in this run",
        "H-ETHANOL": "evaluated" if "ethanol" in campaigns else "not evaluated in this run",
        "H-DUAL": "evaluated" if "dual" in campaigns else "not evaluated in this run",
        "H-SIGMA35": "evaluated" if "leave_sigma35_variant" in splits else "not evaluated in this run",
        "H-COLLAPSE": "evaluated" if has_selection else "not evaluated in this run",
    }


def _write_decision(
    *,
    path: Path,
    decision: str,
    safety: Mapping[str, Any],
    metrics: list[dict[str, Any]],
    quality_counts: Mapping[str, int],
) -> None:
    key_numbers = {
        "path_safety_pass": safety.get("path_safety_pass"),
        "forbidden_input_pass": safety.get("forbidden_input_pass"),
        "quality_ok_fraction": safety.get("quality_ok_fraction"),
    }
    for row in metrics:
        oracle_kind = "null" if row.get("oracle_id") == NULL_ORACLE_ID else "positive"
        key = f"{row.get('campaign')}_{oracle_kind}_{row.get('split_id')}_target_lift"
        key_numbers[key] = row.get("target_lift_at_k_true")
    claims_heading = "Claims tracked" if decision == "PENDING" else "Claims tested"
    claim_statuses = _claim_statuses(metrics, decision=decision)
    campaign_claims = [
        ("H-NULL", "permuted null did not enrich target classes"),
        ("H-CIPRO", "cipro campaign enriched cipro_only"),
        ("H-ETHANOL", "ethanol campaign enriched ethanol_only"),
        ("H-DUAL", "AND campaign enriched dual_axis_and"),
        ("H-SIGMA35", "signal survived held-out sigma35 variant or failed informatively"),
        ("H-COLLAPSE", "selection did not pathologically collapse into one sampling pocket"),
    ]
    lines = [
        "# opal_densegen_axis_probe_v0 decision",
        "",
        "## Decision",
        "",
        str(decision),
        "",
        f"## {claims_heading}",
        "",
        "- H-SAFE: synthetic labels stayed scratch-only.",
        "- H-SOURCE: oracle generation used DenseGen part metadata only.",
    ]
    for claim_id, description in campaign_claims:
        suffix = f" ({claim_statuses[claim_id]})."
        lines.append(f"- {claim_id}: {description}{suffix}")
    lines.extend(["", "## Key numbers", ""])
    lines.extend(f"- {key}: {value}" for key, value in key_numbers.items())
    lines.extend(["", "## Label quality flags", ""])
    lines.extend(f"- {key}: {value}" for key, value in quality_counts.items())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "No OPAL run metrics exist yet; this is a source/materialization status, "
                "not a scoped scored-gate decision."
                if decision == "PENDING"
                else "Generated from scratch probe metrics. Treat DEBUG/STOP conservatively; "
                "inspect metrics.json before expanding synthetic-oracle work."
            ),
            "",
            "## Next action",
            "",
            (
                "Run a campaign gate with OPAL execution when ready."
                if decision == "PENDING"
                else "Use this decision to choose OPAL/LatentDNA debugging, assay stratification, "
                "or a one-seed/budget follow-up."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
        for command in plan.commands:
            lines.append("  " + " ".join(map(str, command)))
    elif not plan.apply:
        lines.append("next: add --apply to materialize source-gate labels/reports.")
    return "\n".join(lines) + "\n"


def _compact_split_metadata(split_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    compact: dict[str, dict[str, Any]] = {}
    for split_id, metadata in split_metadata.items():
        compact[split_id] = {
            "split_id": metadata.get("split_id", split_id),
            "budget": metadata.get("budget"),
            "per_class": metadata.get("per_class"),
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
    required_splits = tuple(dict.fromkeys(run.split_id for run in plan.runs))
    for split_id in required_splits:
        train_ids, metadata = build_train_ids(
            labels, budget=plan.budget, seed=plan.seed, split_id=split_id, return_metadata=True
        )
        metadata["train_ids"] = train_ids
        metadata_by_split[split_id] = metadata
    return metadata_by_split
