"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/docs/test_eco1_bu_scc_qsub_contracts.py

Contract checks for Eco1 BU SCC job templates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
BU_SCC_JOBS = REPO_ROOT / "docs" / "bu-scc" / "jobs"
ECO1_COMMAND_GROUPS = REPO_ROOT / "docs" / "studies" / "eco1_rt_repack" / "operations" / "runtime" / "command-groups"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text(encoding="utf-8")


def test_eco1_proteinmpnn_design_class_qsub_is_submit_ready() -> None:
    qsub_script = _read(BU_SCC_JOBS / "eco1-proteinmpnn-design-class.qsub")

    assert qsub_script.startswith("#!/bin/bash -l")
    assert "#$ -P dunlop" in qsub_script
    assert "#$ -t 1-5" in qsub_script
    assert "#$ -now y" not in qsub_script
    assert 'export OMP_NUM_THREADS="${NSLOTS:-4}"' in qsub_script
    assert "SGE_TASK_ID" in qsub_script
    assert "DESIGN_CLASS_ID" in qsub_script
    assert "ECO1_DESIGN_CLASS_IDS" in qsub_script
    assert 'proteinmpnn_root="${PROTEINMPNN_ROOT:-.var/tools/proteinmpnn}"' in qsub_script
    assert "protein_mpnn_run.py" in qsub_script
    assert "materialization.design_classes" in qsub_script
    assert "materialization.proteinmpnn_sample_ingest" in qsub_script
    assert "materialization.candidate_table" in qsub_script
    assert '--output-root "$class_root"' in qsub_script
    assert '--proteinmpnn-root "$proteinmpnn_root"' in qsub_script
    assert "candidate_table.parquet" in qsub_script
    assert "/project/dunlop/esouth/proteinmpnn/eco1_rt_design_classes/sge_logs" in qsub_script


def test_bu_scc_jobs_readme_documents_design_class_smoke_and_array_submit() -> None:
    jobs_readme = _read(BU_SCC_JOBS / "README.md")

    assert "eco1-proteinmpnn-design-class.qsub" in jobs_readme
    assert "Eco1 ProteinMPNN design-class submissions" in jobs_readme
    assert "PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn" in jobs_readme
    assert "qsub -t 1-1" in jobs_readme
    assert "qsub -t 1-5" in jobs_readme
    assert "candidate_pool_manifest.yaml" in jobs_readme


def test_eco1_command_groups_include_scc_design_class_execution_lanes() -> None:
    command_groups = _read(ECO1_COMMAND_GROUPS / "README.md")
    pipeline = yaml.safe_load(_read(ECO1_COMMAND_GROUPS / "pipeline.yaml"))
    by_id = {lane["id"]: lane for lane in pipeline["lanes"]}

    assert "docs/bu-scc/jobs/eco1-proteinmpnn-design-class.qsub" in command_groups
    assert "design_class_proteinmpnn_scc_smoke" in by_id
    assert "design_class_proteinmpnn_scc_array" in by_id
    assert by_id["design_class_proteinmpnn_scc_smoke"]["owner"] == "bu_scc_runtime"
    assert by_id["design_class_proteinmpnn_scc_array"]["owner"] == "bu_scc_runtime"

    smoke_argv = by_id["design_class_proteinmpnn_scc_smoke"]["command"]["argv"]
    array_argv = by_id["design_class_proteinmpnn_scc_array"]["command"]["argv"]
    assert smoke_argv[:3] == ["qsub", "-t", "1-1"]
    assert array_argv[:3] == ["qsub", "-t", "1-5"]
    assert any("PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn" in arg for arg in smoke_argv)
    assert "docs/bu-scc/jobs/eco1-proteinmpnn-design-class.qsub" in array_argv
