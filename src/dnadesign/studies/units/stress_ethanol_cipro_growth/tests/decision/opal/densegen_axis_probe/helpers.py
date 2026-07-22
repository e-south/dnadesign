"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/helpers.py

Regression tests for helpers studies units stress ethanol cipro growth decision.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records

from .probe_modules import probe_module

_artifacts = probe_module("core.artifacts")
ProbeArtifactLayout = _artifacts.ProbeArtifactLayout
ProbePlan = _artifacts.ProbePlan
RunSpec = _artifacts.RunSpec

_constants = probe_module("core.constants")
AXIS_CLASS_TO_LOGIC4 = _constants.AXIS_CLASS_TO_LOGIC4
CANDIDATE_RECORDS = _constants.CANDIDATE_RECORDS
NULL_ORACLE_ID = _constants.NULL_ORACLE_ID
ORACLE_ID = _constants.ORACLE_ID
X_COLUMN = _constants.X_COLUMN

validate_run_root_policy = probe_module("core.paths").validate_run_root_policy
validate_candidate_x_surface = probe_module("core.source_contract").validate_candidate_x_surface

decision_evaluation = probe_module("evaluation.decision_evaluation")
_decision = probe_module("evaluation.decision")
_decision_from_metrics = _decision._decision_from_metrics
decision_reasons_from_metrics = _decision.decision_reasons_from_metrics
enrich_metric_rows = _decision.enrich_metric_rows
gate_results_from_metrics = _decision.gate_results_from_metrics
round_dynamics_summary = _decision.round_dynamics_summary
trajectory_qa_summary = _decision.trajectory_qa_summary

_decision_inputs = probe_module("evaluation.decision_inputs")
_compact_split_metadata = _decision_inputs._compact_split_metadata
_persisted_split_metadata = _decision_inputs._persisted_split_metadata
_split_metadata_for_all = _decision_inputs._split_metadata_for_all

_claim_statuses = probe_module("evaluation.decision_report")._claim_statuses
predicted_axis_classes = probe_module("evaluation.prediction_scoring").predicted_axis_classes

_trajectory_metrics = probe_module("evaluation.trajectory_metrics")
trajectory_gate_results_from_metrics = _trajectory_metrics.trajectory_gate_results_from_metrics
trajectory_metric_payload = _trajectory_metrics.trajectory_metric_payload

_axis_oracle = probe_module("plan_logic.axis_oracle")
build_axis_oracle = _axis_oracle.build_axis_oracle
build_train_ids = _axis_oracle.build_train_ids
class_from_logic4 = _axis_oracle.class_from_logic4
derive_axis_label = _axis_oracle.derive_axis_label
make_permuted_labels = _axis_oracle.make_permuted_labels
parse_sigma35_variant = _axis_oracle.parse_sigma35_variant

_label_families = probe_module("plan_logic.label_families")
label_family_manifest = _label_families.label_family_manifest
require_label_family_columns = _label_families.require_label_family_columns

null_provenance_payload = probe_module("plan_logic.nulls").null_provenance_payload
summarize_probe_progress = probe_module("reporting.progress").summarize_probe_progress
build_probe_review = probe_module("reporting.review").build_probe_review
audit_run_root = probe_module("reporting.status").audit_run_root
suite_manifest_payload = probe_module("reporting.suite_manifest").suite_manifest_payload

_execution = probe_module("runtime.execution")
materialize_probe_inputs = _execution.materialize_probe_inputs
run_opal_rounds_for_probe = _execution.run_opal_rounds_for_probe
selected_ids_from_round = _execution.selected_ids_from_round
write_followup_label_input = _execution.write_followup_label_input

_plan = probe_module("runtime.plan")
build_plan = _plan.build_plan
validate_scratch_paths = _plan.validate_scratch_paths

prepare_probe_run_root = probe_module("runtime.plan_fingerprint").prepare_probe_run_root

_scratch = probe_module("runtime.scratch")
_make_training_input = _scratch._make_training_input
_make_training_input_for_run = _scratch._make_training_input_for_run
_run_command = _scratch._run_command
_write_campaign_config = _scratch._write_campaign_config
write_campaign_plot_config = _scratch.write_campaign_plot_config

_sweep_contracts = probe_module("runtime.sweep_contracts")
build_sweep_execution_contract = _sweep_contracts.build_sweep_execution_contract
enforce_sweep_apply_contract = _sweep_contracts.enforce_sweep_apply_contract
_evaluate_run = decision_evaluation._evaluate_run
_evaluate_run_rounds = decision_evaluation._evaluate_run_rounds


def _detail(*regulators: str) -> list[dict[str, object]]:
    return [
        {"part_kind": "tfbs", "regulator": regulator, "sequence": f"{idx}{regulator}"}
        for idx, regulator in enumerate(regulators)
    ]


def _valid_metrics_payload(runs: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "safety": {
            "path_safety_pass": True,
            "forbidden_input_pass": True,
            "x_surface_pass": True,
            "quality_counts": {"ok": 1},
        },
        "runs": [] if runs is None else runs,
    }


def _write_probe_prediction_campaign(
    workdir: Path,
    predictions: pd.DataFrame,
    *,
    runs: list[tuple[str, int]] | None = None,
) -> Path:
    records_path = workdir / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(records_path)
    config_path = workdir / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)

    run_rows = runs or [("run-0", 0)]
    ledger_dir = workdir / "outputs" / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_id": [run_id for run_id, _ in run_rows],
            "as_of_round": [round_index for _, round_index in run_rows],
        }
    ).to_parquet(ledger_dir / "runs.parquet", index=False)

    pred = predictions.copy()
    if "run_id" not in pred.columns:
        pred["run_id"] = run_rows[0][0]
    if "as_of_round" not in pred.columns:
        pred["as_of_round"] = run_rows[0][1]
    if "view__is_selected" not in pred.columns:
        pred["view__is_selected"] = True
    if "view__rank_competition" not in pred.columns:
        pred["view__rank_competition"] = range(1, len(pred) + 1)
    if "view__selection_score" not in pred.columns:
        pred["view__selection_score"] = 0.0
    pred["pred__selection_views"] = [
        [
            {
                "selection_view_id": "primary",
                "objective_name": "test_objective",
                "selection_name": "top_n",
                "score": float(score),
                "score_ref": "score",
                "selection_score": float(score),
                "rank_competition": int(rank),
                "is_selected": selected,
                "top_k": len(pred),
                "uncertainty": None,
                "uncertainty_ref": None,
                "diagnostics": [],
            }
        ]
        for score, rank, selected in pred[
            ["view__selection_score", "view__rank_competition", "view__is_selected"]
        ].itertuples(index=False, name=None)
    ]
    pred = pred.drop(columns=["view__selection_score", "view__rank_competition", "view__is_selected"])
    predictions_dir = ledger_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(predictions_dir / "part.parquet", index=False)
    return config_path


def _write_stage_b_review_fixture(tmp_path: Path, *, include_missing_selection_id: bool = False) -> Path:
    campaigns = []
    pairs: dict[str, str] = {}
    for role in ("positive", "matched_null"):
        workdir = tmp_path / "campaigns" / f"lexA_present_{role}"
        config_path = workdir / "configs" / "campaign.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("campaign:\n  workdir: placeholder\n", encoding="utf-8")
        label_path = workdir / "labels.parquet"
        initial_label_path = workdir / "inputs" / "r0" / "labels-b0.parquet"
        values = [0, 0, 1, 1] if role == "positive" else [1, 0, 1, 0]
        frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "lexA_present": values})
        if role == "matched_null":
            frame["null_version"] = "densegen_tfbs_learnability_family_content_matched_null_v1"
        frame.to_parquet(label_path, index=False)
        initial_label_path.parent.mkdir(parents=True)
        frame.loc[frame["id"].isin(["a", "c"]), ["id", "lexA_present"]].to_parquet(initial_label_path, index=False)
        _write_stage_b_review_selection(workdir, 0, ["c", "a"] if role == "positive" else ["b", "d"], [0.1, 0.2])
        round_1_ids = ["c", "missing"] if include_missing_selection_id and role == "positive" else ["c", "d"]
        _write_stage_b_review_selection(workdir, 1, round_1_ids, [0.8, 0.7])
        campaign_key = f"lexA_present_{role}"
        pairs[role] = campaign_key
        campaigns.append(
            {
                "campaign_key": campaign_key,
                "label_name": "lexA_present",
                "label_family_id": "tf_family_presence",
                "oracle_role": role,
                "split_id": "random_id",
                "seed": 7,
                "selection_k": 2,
                "config_path": str(config_path),
                "label_table_path": str(label_path),
                "initial_label_input_path": str(initial_label_path),
            }
        )
    manifest_path = tmp_path / "stage_b_sentinel_config_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "fixture.stage_b",
                "status": "PASS",
                "stage": "B",
                "scope": "sentinel",
                "rounds": 2,
                "selection_k": 2,
                "campaign_count": 2,
                "campaigns": campaigns,
                "pairs": [
                    {
                        "label_name": "lexA_present",
                        "split_id": "random_id",
                        "seed": 7,
                        "positive_campaign_key": pairs["positive"],
                        "null_campaign_key": pairs["matched_null"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_stage_b_review_selection(workdir: Path, round_index: int, ids: list[str], scores: list[float]) -> None:
    path = workdir / "outputs" / "rounds" / f"round_{round_index}" / "selection" / "selections.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "selection_view_id": "primary", "score": scores}).to_parquet(path, index=False)


def _dark_edge_pixel_count(image: object, *, edge_width: int = 5) -> int:
    width, height = image.size
    edge_pixels = []
    edge_pixels.extend(image.crop((0, 0, width, edge_width)).get_flattened_data())
    edge_pixels.extend(image.crop((0, height - edge_width, width, height)).get_flattened_data())
    edge_pixels.extend(image.crop((0, 0, edge_width, height)).get_flattened_data())
    edge_pixels.extend(image.crop((width - edge_width, 0, width, height)).get_flattened_data())
    return sum(1 for red, green, blue in edge_pixels if min(red, green, blue) < 245)


__all__ = [
    "AXIS_CLASS_TO_LOGIC4",
    "CANDIDATE_RECORDS",
    "NULL_ORACLE_ID",
    "Namespace",
    "ORACLE_ID",
    "Path",
    "ProbeArtifactLayout",
    "ProbePlan",
    "RunSpec",
    "X_COLUMN",
    "_claim_statuses",
    "_compact_split_metadata",
    "_decision_from_metrics",
    "_detail",
    "_dark_edge_pixel_count",
    "_evaluate_run",
    "_evaluate_run_rounds",
    "_make_training_input",
    "_make_training_input_for_run",
    "_persisted_split_metadata",
    "_run_command",
    "_split_metadata_for_all",
    "_valid_metrics_payload",
    "_write_campaign_config",
    "_write_probe_prediction_campaign",
    "_write_stage_b_review_fixture",
    "annotations",
    "audit_run_root",
    "build_axis_oracle",
    "build_plan",
    "build_probe_review",
    "build_sweep_execution_contract",
    "build_train_ids",
    "class_from_logic4",
    "decision_reasons_from_metrics",
    "derive_axis_label",
    "enrich_metric_rows",
    "enforce_sweep_apply_contract",
    "gate_results_from_metrics",
    "importlib",
    "json",
    "label_family_manifest",
    "make_permuted_labels",
    "materialize_probe_inputs",
    "null_provenance_payload",
    "pa",
    "parse_sigma35_variant",
    "pd",
    "pq",
    "predicted_axis_classes",
    "prepare_probe_run_root",
    "pytest",
    "require_label_family_columns",
    "round_dynamics_summary",
    "run_opal_rounds_for_probe",
    "selected_ids_from_round",
    "subprocess",
    "summarize_probe_progress",
    "suite_manifest_payload",
    "sys",
    "trajectory_gate_results_from_metrics",
    "trajectory_metric_payload",
    "trajectory_qa_summary",
    "validate_candidate_x_surface",
    "validate_run_root_policy",
    "validate_scratch_paths",
    "write_campaign_plot_config",
    "write_campaign_yaml",
    "write_followup_label_input",
    "write_records",
    "yaml",
]
