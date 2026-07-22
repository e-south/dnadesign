"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_status_source_contracts.py

Regression tests for status source studies units stress ethanol cipro growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import (
    CANDIDATE_RECORDS,
    NULL_ORACLE_ID,
    ORACLE_ID,
    X_COLUMN,
    Path,
    _valid_metrics_payload,
    audit_run_root,
    json,
    pa,
    pd,
    pq,
    pytest,
    validate_candidate_x_surface,
)
from .probe_modules import probe_module


def test_audit_run_root_reports_pending_source_gate(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    (run_root / "reports").mkdir(parents=True)
    label_frame = pd.DataFrame(
        {
            "oracle_id": [ORACLE_ID],
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "axis_class": ["background_only"],
            "quality_flag": ["ok"],
            "logic4": [[0, 0, 0, 0]],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [0.0],
        }
    )
    label_frame.to_parquet(run_root / "labels" / "densegen_plan_logic4.parquet", index=False)
    label_frame.assign(oracle_id=NULL_ORACLE_ID).to_parquet(
        run_root / "labels" / "permuted_densegen_plan_logic4.parquet",
        index=False,
    )
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "ok"
    assert audit.decision == "PENDING"
    assert audit.problems == ()


def test_audit_run_root_rejects_materialized_scored_plan_without_metrics(tmp_path: Path) -> None:
    probe_main = probe_module("cli").main

    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    label_frame = pd.DataFrame(
        {
            "oracle_id": [ORACLE_ID],
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "axis_class": ["background_only"],
            "quality_flag": ["ok"],
            "logic4": [[0, 0, 0, 0]],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [0.0],
        }
    )
    label_frame.to_parquet(run_root / "labels" / "densegen_plan_logic4.parquet", index=False)
    label_frame.assign(oracle_id=NULL_ORACLE_ID).to_parquet(
        run_root / "labels" / "permuted_densegen_plan_logic4.parquet",
        index=False,
    )
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "probe_plan.json").write_text(
        json.dumps({"plan": {"planned_runs": 2, "rounds": 12, "gate": "cipro-random", "stop_after": "status"}}),
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "metrics_missing_for_scored_plan" in audit.problems
    assert "decision_missing_for_scored_plan" in audit.problems
    assert probe_main(["status", "--run-root", str(run_root), "--json"]) == 1


def test_audit_run_root_rejects_corrupt_label_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    (run_root / "reports").mkdir(parents=True)
    (run_root / "labels" / "densegen_plan_logic4.parquet").write_bytes(b"placeholder")
    (run_root / "labels" / "permuted_densegen_plan_logic4.parquet").write_bytes(b"placeholder")
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "densegen_labels_parquet_unreadable" in audit.problems
    assert "null_labels_parquet_unreadable" in audit.problems


def test_audit_run_root_rejects_split_metadata_paths_outside_splits_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "splits").mkdir(parents=True)
    (run_root / "splits" / "split_metadata.json").write_text(
        json.dumps(
            {
                "random_id": {
                    "split_id": "random_id",
                    "train_ids_path": "../outside.parquet",
                    "eval_ids_path": "/tmp/outside.parquet",
                }
            }
        ),
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "split_metadata_random_id_train_ids_path_outside_splits_dir" in audit.problems
    assert "split_metadata_random_id_eval_ids_path_outside_splits_dir" in audit.problems


def test_audit_run_root_rejects_malformed_metrics_shape(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "reports").mkdir(parents=True)
    (run_root / "reports" / "metrics.json").write_text(
        json.dumps({"safety": {"path_safety_pass": True}, "runs": [42]}),
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "metrics_json_safety_missing_forbidden_input_pass" in audit.problems
    assert "metrics_json_safety_missing_x_surface_pass" in audit.problems
    assert "metrics_json_safety_missing_quality_counts" in audit.problems
    assert "metrics_json_runs_0_not_mapping" in audit.problems


def test_audit_run_root_rejects_invalid_decision_value(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "reports").mkdir(parents=True)
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nBOGUS\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "decision_value_invalid" in audit.problems


def test_audit_run_root_requires_candidate_scope_for_planned_campaigns(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    (run_root / "reports").mkdir(parents=True)
    (run_root / "scratch_campaigns" / "cipro_positive_random_id").mkdir(parents=True)
    label_frame = pd.DataFrame(
        {
            "oracle_id": [ORACLE_ID],
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "axis_class": ["background_only"],
            "quality_flag": ["ok"],
            "logic4": [[0, 0, 0, 0]],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [0.0],
        }
    )
    label_frame.to_parquet(run_root / "labels" / "densegen_plan_logic4.parquet", index=False)
    label_frame.assign(oracle_id=NULL_ORACLE_ID).to_parquet(
        run_root / "labels" / "permuted_densegen_plan_logic4.parquet",
        index=False,
    )
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "scratch_record_symlink_missing_for_planned_campaigns" in audit.problems
    assert "candidate_scope_missing_for_planned_campaigns" in audit.problems


def test_validate_candidate_x_surface_checks_schema_and_row_count(tmp_path: Path) -> None:
    records_path = tmp_path / CANDIDATE_RECORDS
    records_path.parent.mkdir(parents=True)
    values = pa.array([0.0] * (2 * 8192), type=pa.float32())
    table = pa.table(
        {
            "id": pa.array(["id-1", "id-2"]),
            X_COLUMN: pa.FixedSizeListArray.from_arrays(values, list_size=8192),
        }
    )
    pq.write_table(table, records_path)

    summary = validate_candidate_x_surface(tmp_path, expected_rows=2)

    assert summary["x_dim"] == 8192
    assert summary["x_value_type"] == "float"
    assert summary["row_count"] == 2
    assert summary["validation_level"] == "parquet_schema_and_row_count"


def test_validate_candidate_x_surface_rejects_wrong_x_dimension(tmp_path: Path) -> None:
    records_path = tmp_path / CANDIDATE_RECORDS
    records_path.parent.mkdir(parents=True)
    values = pa.array([0.0] * 4, type=pa.float32())
    table = pa.table(
        {
            "id": pa.array(["id-1"]),
            X_COLUMN: pa.FixedSizeListArray.from_arrays(values, list_size=4),
        }
    )
    pq.write_table(table, records_path)

    with pytest.raises(ValueError, match="dimension 4"):
        validate_candidate_x_surface(tmp_path, expected_rows=1)


def test_validate_candidate_x_surface_rejects_non_float32_x_values(tmp_path: Path) -> None:
    records_path = tmp_path / CANDIDATE_RECORDS
    records_path.parent.mkdir(parents=True)
    values = pa.array([0] * 8192, type=pa.int32())
    table = pa.table(
        {
            "id": pa.array(["id-1"]),
            X_COLUMN: pa.FixedSizeListArray.from_arrays(values, list_size=8192),
        }
    )
    pq.write_table(table, records_path)

    with pytest.raises(ValueError, match="float32"):
        validate_candidate_x_surface(tmp_path, expected_rows=1)
