"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_solver_integration_usr_resume.py

Runs a real solver workflow with USR output and verifies resume safety.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pyarrow.parquet as pq

import dnadesign.usr as usr_pkg
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.pipeline.orchestrator import run_pipeline


def _write_usr_registry(path: Path) -> None:
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo_tfbs_baseline
                root: "."

              inputs:
                - name: basic_sites
                  type: binding_sites
                  path: inputs/sites.csv

              output:
                targets: [parquet, usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
                  deduplicate: true
                  chunk_size: 64
                usr:
                  dataset: demo_workspace
                  root: outputs/usr
                  chunk_size: 1
                  allow_overwrite: false

              generation:
                sequence_length: 3
                sampling:
                  pool_strategy: full
                  library_size: 1
                  unique_binding_sites: true
                  unique_binding_cores: true
                  relax_on_exhaustion: false
                plan:
                  - name: baseline
                    quota: 1
                    sampling:
                      include_inputs: [basic_sites]
                    regulator_constraints:
                      groups: []

              solver:
                backend: CBC
                strategy: iterate

              runtime:
                round_robin: false
                arrays_generated_before_resample: 1
                min_count_per_tf: 0
                max_duplicate_solutions: 3
                stall_seconds_before_resample: 5
                stall_warning_every_seconds: 5
                max_consecutive_failures: 10
                max_seconds_per_plan: 20
                max_failed_solutions: 0
                random_seed: 1

              postprocess:
                pad:
                  mode: off

              logging:
                log_dir: outputs/logs
                level: INFO
                progress_style: summary

            plots:
              source: parquet
              out_dir: outputs/plots
            """
        ).strip()
        + "\n"
    )


def _write_config_usr_only(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo_tfbs_baseline
                root: "."

              inputs:
                - name: basic_sites
                  type: binding_sites
                  path: inputs/sites.csv

              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  dataset: demo_workspace
                  root: outputs/usr
                  chunk_size: 1
                  allow_overwrite: false

              generation:
                sequence_length: 3
                sampling:
                  pool_strategy: full
                  library_size: 1
                  unique_binding_sites: true
                  unique_binding_cores: true
                  relax_on_exhaustion: false
                plan:
                  - name: baseline
                    quota: 1
                    sampling:
                      include_inputs: [basic_sites]
                    regulator_constraints:
                      groups: []

              solver:
                backend: CBC
                strategy: iterate

              runtime:
                round_robin: false
                arrays_generated_before_resample: 1
                min_count_per_tf: 0
                max_duplicate_solutions: 3
                stall_seconds_before_resample: 5
                stall_warning_every_seconds: 5
                max_consecutive_failures: 10
                max_seconds_per_plan: 20
                max_failed_solutions: 0
                random_seed: 1

              postprocess:
                pad:
                  mode: off

              logging:
                log_dir: outputs/logs
                level: INFO
                progress_style: summary
            """
        ).strip()
        + "\n"
    )


def test_real_solver_run_writes_usr_and_resume_is_safe(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace / "inputs" / "sites.csv").write_text("tf,tfbs\nTF1,AAA\n")
    _write_usr_registry(workspace / "outputs" / "usr" / "registry.yaml")

    cfg_path = workspace / "config.yaml"
    _write_config(cfg_path)
    loaded = load_config(cfg_path)

    # A registry-only USR root does not count as existing run outputs.
    first = run_pipeline(loaded, resume=False, build_stage_a=True)
    assert first.total_generated >= 1

    records_path = workspace / "outputs" / "usr" / "demo_workspace" / "records.parquet"
    assert records_path.exists()
    row_count_before_resume = pq.read_table(records_path).num_rows
    assert row_count_before_resume >= 1

    overlay_dir = workspace / "outputs" / "usr" / "demo_workspace" / "_derived" / "densegen"
    assert list(overlay_dir.glob("part-*.parquet"))

    run_pipeline(loaded, resume=True, build_stage_a=False)
    row_count_after_resume = pq.read_table(records_path).num_rows
    assert row_count_after_resume == row_count_before_resume


def test_real_solver_run_usr_only_resume_is_safe(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace_usr_only"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace / "inputs" / "sites.csv").write_text("tf,tfbs\nTF1,AAA\n")
    _write_usr_registry(workspace / "outputs" / "usr" / "registry.yaml")

    cfg_path = workspace / "config.yaml"
    _write_config_usr_only(cfg_path)
    loaded = load_config(cfg_path)

    first = run_pipeline(loaded, resume=False, build_stage_a=True)
    assert first.total_generated >= 1

    records_path = workspace / "outputs" / "usr" / "demo_workspace" / "records.parquet"
    assert records_path.exists()
    row_count_before_resume = pq.read_table(records_path).num_rows
    assert row_count_before_resume >= 1

    run_pipeline(loaded, resume=True, build_stage_a=False)
    row_count_after_resume = pq.read_table(records_path).num_rows
    assert row_count_after_resume == row_count_before_resume
