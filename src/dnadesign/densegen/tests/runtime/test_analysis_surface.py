"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_analysis_surface.py

Coverage for the public DenseGen analysis-surface contract.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pandas as pd

from dnadesign.densegen import inspect_analysis_surface


def _write_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              out_dir: outputs/plots
              format: pdf
              default:
                [
                  source_cohort_concentration,
                  stage_a_sampling_yield,
                  stage_a_pool_diversity,
                  plan_regulator_deployment_heatmap,
                  placement_occupancy_map,
                  retained_pool_coverage_by_regulator,
                  attempt_outcome_timeline,
                  solve_pressure_and_progress
                ]
              options: {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_current_inventory(run_root: Path) -> None:
    plot_root = run_root / "outputs" / "plots"
    (plot_root / "dataset").mkdir(parents=True, exist_ok=True)
    (plot_root / "dataset" / "source_cohort_concentration.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
    (plot_root / "dataset" / "source_plan_input_heatmap.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
    payload = {
        "schema_version": "densegen.current_inventory.v2",
        "contract_version": "densegen.analysis_surface.v2",
        "generated_at": "2026-04-17T12:00:00+00:00",
        "plots": [
            {
                "name": "source_cohort_concentration",
                "plot_id": "source_cohort_concentration",
                "path": "dataset/source_cohort_concentration.pdf",
                "title": "Source cohort concentration",
                "caption": ("Bar chart showing how final DenseGen records are concentrated across source cohorts."),
                "alt_text": (
                    "Source cohort concentration. Bar chart showing how final "
                    "DenseGen records are concentrated across source cohorts."
                ),
                "generated_at": "2026-04-17T12:00:00+00:00",
            },
            {
                "name": "source_plan_input_heatmap",
                "plot_id": "source_plan_input_heatmap",
                "path": "dataset/source_plan_input_heatmap.pdf",
                "title": "Dataset provenance heatmap",
                "caption": (
                    "Heatmap showing source-to-plan and source-to-input provenance across the generated dataset."
                ),
                "alt_text": (
                    "Dataset provenance heatmap. Heatmap showing source-to-plan "
                    "and source-to-input provenance across the generated dataset."
                ),
                "generated_at": "2026-04-17T12:00:00+00:00",
            },
        ],
    }
    (plot_root / "current_inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifact_ledger(run_root: Path) -> None:
    plot_root = run_root / "outputs" / "plots"
    (plot_root / "stage_b_summary").mkdir(parents=True, exist_ok=True)
    (plot_root / "stage_b_summary" / "accepted_arrays_by_plan.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
    payload = {
        "schema_version": "densegen.artifact_ledger.v1",
        "contract_version": "densegen.analysis_surface.v2",
        "generated_at": "2026-04-17T12:00:00+00:00",
        "plots": [
            {
                "name": "accepted_arrays_by_plan",
                "plot_id": "accepted_arrays_by_plan",
                "path": "stage_b_summary/accepted_arrays_by_plan.pdf",
                "generated_at": "2026-04-17T12:00:00+00:00",
            }
        ],
    }
    (plot_root / "artifact_ledger.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_legacy_plot_manifest(run_root: Path) -> Path:
    plot_root = run_root / "outputs" / "plots"
    (plot_root / "dataset").mkdir(parents=True, exist_ok=True)
    legacy_plot = plot_root / "dataset" / "dataset_source_inventory.pdf"
    legacy_plot.write_text("%PDF-1.4\n", encoding="utf-8")
    payload = {
        "schema_version": "densegen.artifact_ledger.v1",
        "contract_version": "densegen.analysis_surface.v2",
        "generated_at": "2026-04-17T12:00:00+00:00",
        "plots": [
            {
                "name": "dataset_source_inventory",
                "plot_id": "dataset_source_inventory",
                "path": "dataset/dataset_source_inventory.pdf",
                "generated_at": "2026-04-17T12:00:00+00:00",
            }
        ],
    }
    manifest_path = plot_root / "plot_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def test_public_analysis_surface_is_exported_from_densegen_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n", encoding="utf-8")
    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "row-1",
                "sequence": "ACGTACGTAA",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "demo",
            }
        ]
    ).to_parquet(records_path, index=False)
    _write_current_inventory(run_root)
    _write_artifact_ledger(run_root)
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text("import marimo\n", encoding="utf-8")

    surface = inspect_analysis_surface(cfg_path)

    assert surface.contract_version == "densegen.analysis_surface.v2"
    assert surface.workspace_id == "demo"
    assert surface.runtime_summary.dataset_row_count == 1
    assert len(surface.current_inventory) == 2
    taxonomy = {entry.plot_id: entry for entry in surface.taxonomy}
    assert taxonomy["source_cohort_concentration"].generated_by_default is True
    assert taxonomy["source_cohort_concentration"].operator_visible_by_default is True
    assert taxonomy["source_cohort_concentration"].optional is False
    assert "accepted_arrays_by_plan" not in taxonomy
    assert taxonomy["source_plan_input_heatmap"].generated_by_default is False
    assert taxonomy["source_plan_input_heatmap"].operator_visible_by_default is False
    assert taxonomy["source_plan_input_heatmap"].notebook_visible_by_default is True
    assert taxonomy["source_plan_input_heatmap"].internal_hidden is False
    inventory = {entry.plot_id: entry for entry in surface.current_inventory}
    assert inventory["source_cohort_concentration"].visible is True
    assert inventory["source_plan_input_heatmap"].visible is True
    assert surface.notebook.gallery_visible_artifact_ids == [
        "source_cohort_concentration",
        "source_plan_input_heatmap",
    ]
    assert surface.notebook.hidden_artifact_ids == []
    assert surface.historical_ledger_surface == ["accepted_arrays_by_plan"]


def test_analysis_surface_does_not_fallback_to_legacy_manifest_when_current_inventory_is_missing(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n", encoding="utf-8")
    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "row-1",
                "sequence": "ACGTACGTAA",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "demo",
            }
        ]
    ).to_parquet(records_path, index=False)
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text("import marimo\n", encoding="utf-8")
    manifest_path = _write_legacy_plot_manifest(run_root)
    manifest_stat = manifest_path.stat()
    os.utime(notebook_path, (manifest_stat.st_atime - 5, manifest_stat.st_mtime - 5))

    surface = inspect_analysis_surface(cfg_path)

    assert surface.current_inventory == []
    assert surface.freshness.inventory_source == "missing"
    assert surface.freshness.manifest_freshness == "historical_only"
    assert surface.notebook.fresh is False
