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
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.artifacts import (
    ProbeArtifactLayout,
    ProbePlan,
    RunSpec,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.axis_oracle import (
    build_axis_oracle,
    build_train_ids,
    class_from_logic4,
    derive_axis_label,
    make_permuted_labels,
    parse_sigma35_variant,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.constants import (
    AXIS_CLASS_TO_LOGIC4,
    CANDIDATE_RECORDS,
    NULL_ORACLE_ID,
    ORACLE_ID,
    X_COLUMN,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.decision import (
    _decision_from_metrics,
    decision_reasons_from_metrics,
    enrich_metric_rows,
    gate_results_from_metrics,
    round_dynamics_summary,
    trajectory_qa_summary,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.decision_evaluation import (
    _evaluate_run,
    _evaluate_run_rounds,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.decision_inputs import (
    _compact_split_metadata,
    _persisted_split_metadata,
    _split_metadata_for_all,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.decision_report import (
    _claim_statuses,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.execution import (
    materialize_probe_inputs,
    run_opal_rounds_for_probe,
    selected_ids_from_round,
    write_followup_label_input,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.label_families import (
    label_family_manifest,
    require_label_family_columns,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.nulls import (
    null_provenance_payload,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.paths import (
    validate_run_root_policy,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.plan import (
    build_plan,
    validate_scratch_paths,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.plan_fingerprint import (
    prepare_probe_run_root,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.prediction_scoring import (
    predicted_axis_classes,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.progress import (
    summarize_probe_progress,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review import build_probe_review
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.scratch import (
    _make_training_input,
    _make_training_input_for_run,
    _run_command,
    _write_campaign_config,
    write_campaign_plot_config,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.source_contract import (
    validate_candidate_x_surface,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.status import (
    audit_run_root,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.suite_manifest import (
    suite_manifest_payload,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.sweep_contracts import (
    build_sweep_execution_contract,
    enforce_sweep_apply_contract,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.trajectory_metrics import (
    trajectory_gate_results_from_metrics,
    trajectory_metric_payload,
)


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
    if "sel__is_selected" not in pred.columns:
        pred["sel__is_selected"] = True
    if "sel__rank_competition" not in pred.columns:
        pred["sel__rank_competition"] = range(1, len(pred) + 1)
    predictions_dir = ledger_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(predictions_dir / "part.parquet", index=False)
    return config_path


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
