"""
Promoter snapshot DenseGen seam tests.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dnadesign.densegen import CURRENT_INVENTORY_SCHEMA_VERSION, required_notebook_plot_ids
from dnadesign.studies.status_adapters.promoter_status.analysis_surfaces import inspect_promoter_exploratory_analysis
from dnadesign.studies.tests.test_promoter_snapshot import _make_study_context


def _fixture_plot_path(plot_id: str) -> str:
    if plot_id in {"source_cohort_concentration", "source_plan_input_heatmap"}:
        return f"dataset/{plot_id}.pdf"
    if plot_id in {
        "background_sequence_logo",
        "stage_a_pool_diversity",
        "stage_a_pool_score_strata",
        "stage_a_sampling_yield",
    }:
        return f"stage_a/{plot_id}.pdf"
    if plot_id in {"placement_occupancy_map", "tfbs_concentration_profile"}:
        return f"stage_b/demo_plan/{plot_id}.pdf"
    if plot_id in {
        "plan_regulator_deployment_heatmap",
        "score_strata_and_deployed_length_bridge",
        "retained_pool_coverage_by_regulator",
        "retained_vs_deployed_length_mix_by_regulator",
        "retained_vs_deployed_tier_mix_by_regulator",
        "upstream_motif_supply_and_pwm_strength",
    }:
        return f"stage_b/all_plans/{plot_id}.pdf"
    if plot_id in {"attempt_outcome_timeline", "compression_ratio_by_plan", "solve_pressure_and_progress"}:
        return f"run_health/{plot_id}.pdf"
    if plot_id == "dense_array_showcase_video":
        return "stage_b/all_plans/showcase.mp4"
    raise AssertionError(f"unexpected plot id in DenseGen seam fixture: {plot_id}")


def _write_densegen_workspace_fixture(repo_root: Path) -> Path:
    workspace_dir = repo_root / "src" / "dnadesign" / "densegen" / "workspaces" / "demo_surface"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "README.md").write_text("# demo surface\n", encoding="utf-8")
    (workspace_dir / "inputs.csv").write_text("tf,tfbs\n", encoding="utf-8")
    (workspace_dir / "config.yaml").write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo_surface
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
                [source_cohort_concentration, background_sequence_logo]
              options: {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    records_path = workspace_dir / "outputs" / "tables" / "records.parquet"
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
    plot_root = workspace_dir / "outputs" / "plots"
    plot_entries: list[dict[str, str]] = []
    for plot_id in required_notebook_plot_ids():
        relative_path = _fixture_plot_path(plot_id)
        absolute_path = plot_root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text("%PDF-1.4\n", encoding="utf-8")
        plot_entries.append(
            {
                "name": plot_id,
                "plot_id": plot_id,
                "path": relative_path,
                "generated_at": "2026-04-17T12:00:00+00:00",
            }
        )
    (plot_root / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": CURRENT_INVENTORY_SCHEMA_VERSION,
                "contract_version": "densegen.analysis_surface.v2",
                "generated_at": "2026-04-17T12:00:00+00:00",
                "plots": plot_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    notebook_path = workspace_dir / "outputs" / "notebooks" / "densegen_run_overview.py"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text("import marimo\n", encoding="utf-8")
    return workspace_dir


def test_inspect_promoter_exploratory_analysis_uses_public_densegen_surface_contract(tmp_path: Path) -> None:
    workspace_dir = _write_densegen_workspace_fixture(tmp_path)
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_pipeline={
            **base_context.study_pipeline,
            "densegen": {
                "workspace": workspace_dir.relative_to(tmp_path).as_posix(),
                "doc": (workspace_dir / "README.md").relative_to(tmp_path).as_posix(),
                "analysis_surface": {
                    "contract_ref": "densegen.analysis_surface.v2",
                },
            },
        },
    )

    surfaces = inspect_promoter_exploratory_analysis(
        study_context=study_context,
        latentdna_state=None,
        downstream_surfaces={},
    )

    densegen = surfaces["densegen"]
    assert densegen["configured"] is True
    assert densegen["state"] == "ok"
    assert densegen["surface_status"] == "ok"
    assert densegen["contract_ref"] == "densegen.analysis_surface.v2"
    assert "source_cohort_concentration" in densegen["operator_visible_surface"]
    assert "background_sequence_logo" in densegen["optional_surface"]
    assert densegen["internal_or_hidden_surface"] == []
    assert densegen["rendered_plot_count"] == len(required_notebook_plot_ids())
    assert densegen["artifact_paths"]["current_inventory"] == str(
        workspace_dir / "outputs" / "plots" / "current_inventory.json"
    )


def test_promoter_densegen_surface_reports_explicit_degraded_state_when_contract_load_fails(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "src" / "dnadesign" / "densegen" / "workspaces" / "broken_surface"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "README.md").write_text("# broken surface\n", encoding="utf-8")

    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_pipeline={
            **base_context.study_pipeline,
            "densegen": {
                "workspace": workspace_dir.relative_to(tmp_path).as_posix(),
                "doc": (workspace_dir / "README.md").relative_to(tmp_path).as_posix(),
                "analysis_surface": {
                    "contract_ref": "densegen.analysis_surface.v2",
                },
            },
        },
    )

    surfaces = inspect_promoter_exploratory_analysis(
        study_context=study_context,
        latentdna_state=None,
        downstream_surfaces={},
    )

    densegen = surfaces["densegen"]
    assert densegen["configured"] is True
    assert densegen["state"] == "degraded"
    assert densegen["surface_status"] == "degraded"
    assert densegen["reason_code"] == "densegen_analysis_surface_unavailable"
    assert "config" in str(densegen["reason_message"]).lower()
    assert densegen["blocking"] is True


def test_promoter_analysis_surface_module_does_not_import_densegen_internals() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source = (repo_root / "src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py").read_text(
        encoding="utf-8"
    )

    assert ".".join(["dnadesign", "densegen", "src"]) not in source
