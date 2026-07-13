"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/reporting/test_notebook.py

Regression tests for notebook OPAL reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.opal.src.analysis.notebook_components import (
    build_notebook_validity_rows,
    build_notebook_visual_surface_model,
)
from dnadesign.opal.src.reporting.notebook import build_notebook_view_model
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_round_log


def test_notebook_view_model_includes_artifact_garden_without_records_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workdir = tmp_path / ".var" / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)
    stale_plot = workdir / "outputs" / "plots" / "old.png"
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_bytes(b"stale")

    from dnadesign.opal.src.storage import records_io

    def fail_read_parquet_df(*args, **kwargs):
        raise AssertionError("notebook view model must not read records.parquet")

    monkeypatch.setattr(records_io, "read_parquet_df", fail_read_parquet_df)

    payload = build_notebook_view_model(config_path, round_selector="latest")

    audit = payload["artifact_garden"]
    assert audit["schema_version"] == "opal.artifact_garden.v1"
    assert audit["local_only"] is True
    assert audit["prune_plan"]["requires_apply"] is True
    assert any(row["path"] == str(stale_plot) for row in audit["stale_artifacts"])


def test_notebook_view_model_pins_requested_run_scope(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-1", round_index=0)
    write_round_log(workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl", run_id="run-1")

    payload = build_notebook_view_model(config_path, round_selector="latest", run_id="run-1")

    assert payload["status"]["round_selector"] == "0"
    assert payload["status"]["run_id_selector"] == "run-1"
    assert payload["progress"]["run_id"] == "run-1"
    run_scope = payload["progress"]["rounds"][0]["summary"]["run_scope"]
    assert run_scope["requested_run_id"] == "run-1"
    assert run_scope["resolved_run_id"] == "run-1"


def test_notebook_view_model_loads_run_scoped_selection_batch(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)
    write_ledger(workdir, run_id="run-1", round_index=0)
    write_round_log(workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl", run_id="run-1")

    import dnadesign.opal.src.reporting.notebook as notebook_reporting

    calls: list[tuple[str, str | None]] = []

    def fake_load_selection_batch(config: Path, *, round_selector: str, run_id: str | None):
        calls.append((round_selector, run_id))
        assert config == config_path.resolve()
        return {
            "schema_version": "opal.selection_batch.v1",
            "as_of_round": 0,
            "run_id": "run-1",
            "unique_count": 1,
            "rows": [{"id": "a", "selection_view_ids": ["primary"]}],
        }

    monkeypatch.setattr(notebook_reporting, "load_selection_batch", fake_load_selection_batch)

    payload = build_notebook_view_model(config_path, round_selector="latest", run_id="run-1")

    assert calls == [("0", "run-1")]
    assert payload["selection_batch"]["unique_count"] == 1


def test_notebook_view_model_includes_configured_plot_inventory(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(
        config_path,
        workdir=workdir,
        records_path=records_path,
        plots=[
            {
                "name": "score_vs_rank",
                "kind": "scatter_score_vs_rank",
                "round_selector": "latest",
            }
        ],
    )

    payload = build_notebook_view_model(config_path, round_selector="latest")

    assert payload["configured_plots"][0]["name"] == "score_vs_rank"
    visual_surface = build_notebook_visual_surface_model(payload)
    assert visual_surface["missing_outputs"] == ["score_vs_rank"]
    assert visual_surface["inventory_status_counts"] == {"configured_missing_output": 1}


def test_notebook_view_model_includes_reader_vec8_label_staging_inputs(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)
    label_input = workdir / "inputs" / "r0" / "reader_vec8_batch0.csv"
    label_input.parent.mkdir(parents=True, exist_ok=True)
    label_input.write_text(
        "id,sequence,v00,v10,v01,v11,y00_star,y10_star,y01_star,y11_star,intensity_log2_offset_delta\n"
        "candidate-1,ACGT,0,1,0,1,0.1,0.2,0.3,0.4,0.0\n",
        encoding="utf-8",
    )

    payload = build_notebook_view_model(config_path, round_selector="latest")

    assert payload["label_staging"] == [
        {
            "path": str(label_input),
            "path_label": "inputs/r0/reader_vec8_batch0.csv",
            "round": "r0",
            "status": "ready",
            "rows": 1,
            "distinct_ids": 1,
            "missing_columns": [],
        }
    ]


def test_notebook_view_model_includes_reader_evidence_manifests(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)
    reader_plot = tmp_path / "reader" / "experiments" / "2026" / "20260706_sfxi" / "outputs" / "plots" / "ts_ETH-01.pdf"
    reader_plot.parent.mkdir(parents=True, exist_ok=True)
    reader_plot.write_bytes(b"%PDF-1.4\n")
    evidence_manifest = workdir / "inputs" / "r0" / "reader_evidence_manifest.json"
    evidence_manifest.parent.mkdir(parents=True, exist_ok=True)
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.reader_evidence.v1",
                "campaign_slug": "secg_ethanol_rf_sfxi_topn",
                "round": "r0",
                "observed_round": 0,
                "summary": {
                    "rows": 1,
                    "distinct_ids": 1,
                    "reader_experiments": 1,
                    "artifact_count": 1,
                    "missing_artifact_rows": 0,
                },
                "rows": [
                    {
                        "id": "candidate-1",
                        "design_id": "pDual-10-SECG-B0-ETH-01",
                        "reader_experiment_id": "20260706_sfxi",
                        "time_selected_h": 12.04,
                        "artifacts": [
                            {
                                "semantic_kind": "raw_kinetics",
                                "path": str(reader_plot),
                                "exists": True,
                                "media_type": "application/pdf",
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_notebook_view_model(config_path, round_selector="latest")

    assert payload["reader_evidence"] == [
        {
            "path": str(evidence_manifest),
            "path_label": "inputs/r0/reader_evidence_manifest.json",
            "round": "r0",
            "status": "ready",
            "rows": 1,
            "distinct_ids": 1,
            "reader_experiments": 1,
            "artifact_count": 1,
            "missing_artifact_rows": 0,
        }
    ]


def test_notebook_view_model_marks_progress_contract_errors_blocking(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)

    import dnadesign.opal.src.reporting.notebook as notebook_reporting

    def fail_progress(*args, **kwargs):
        raise RuntimeError("progress contract exploded")

    monkeypatch.setattr(notebook_reporting, "build_campaign_progress", fail_progress)

    payload = build_notebook_view_model(config_path, round_selector="latest")

    assert {
        "category": "ProgressContractError",
        "severity": "error",
        "message": "progress contract exploded",
    } in payload["warnings"]
    validity = {row["field"]: row["value"] for row in build_notebook_validity_rows(payload)}
    assert validity["Blocking issues"] == 1
