"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/docs/test_eco1_bu_scc_qsub_contracts.py

Contract checks for Eco1 BU SCC job templates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
BU_SCC_JOBS = REPO_ROOT / "docs" / "bu-scc" / "jobs"
ECO1_COMMAND_GROUPS = REPO_ROOT / "docs" / "studies" / "eco1_rt_repack" / "operations" / "runtime" / "command-groups"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text(encoding="utf-8")


def _qsub_commands(bash_blocks: list[str]) -> list[str]:
    commands: list[str] = []
    current: list[str] = []
    for line in "\n".join(bash_blocks).splitlines():
        if not current:
            if not line.startswith("qsub"):
                continue
            current.append(line)
        else:
            current.append(line)
        if not line.rstrip().endswith("\\"):
            commands.append("\n".join(current))
            current = []
    assert not current, "Documented qsub command has an unterminated line continuation."
    return commands


def test_eco1_proteinmpnn_generation_policy_qsub_is_submit_ready() -> None:
    qsub_script = _read(BU_SCC_JOBS / "eco1-proteinmpnn-generation-policy.qsub")

    assert qsub_script.startswith("#!/bin/bash -l")
    assert "#$ -P dunlop" in qsub_script
    assert "#$ -t" not in qsub_script
    assert "#$ -now y" not in qsub_script
    assert "#$ -pe omp 1" in qsub_script
    assert 'export OMP_NUM_THREADS="${NSLOTS:-1}"' in qsub_script
    assert "SGE_TASK_ID" in qsub_script
    assert "GENERATION_POLICY_ID" in qsub_script
    assert "ECO1_GENERATION_POLICY_IDS" in qsub_script
    assert 'proteinmpnn_root="${PROTEINMPNN_ROOT:-.var/tools/proteinmpnn}"' in qsub_script
    assert "protein_mpnn_run.py" in qsub_script
    assert "requests_lock=" in qsub_script
    assert "flock 9" in qsub_script
    assert "materialization.generation_policies" in qsub_script
    assert "materialization.proteinmpnn_sample_ingest" in qsub_script
    assert "materialization.candidate_table" in qsub_script
    assert '--output-root "$policy_root"' in qsub_script
    assert '--proteinmpnn-root "$proteinmpnn_root"' in qsub_script
    assert "candidate_table.parquet" in qsub_script
    assert "/project/dunlop/esouth" not in qsub_script
    assert "#$ -o" not in qsub_script


def test_bu_scc_jobs_readme_documents_generation_policy_smoke_and_array_submit() -> None:
    jobs_readme = _read(BU_SCC_JOBS / "README.md")

    assert "eco1-proteinmpnn-generation-policy.qsub" in jobs_readme
    assert "Eco1 ProteinMPNN generation-policy submissions" in jobs_readme
    assert "eco1-proteinmpnn-design-class.qsub" not in jobs_readme
    assert "PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn" in jobs_readme
    assert "qsub -t 1" in jobs_readme
    assert "qsub -t 1-3" in jobs_readme
    assert "candidate-pool" in jobs_readme


def test_eco1_submission_and_log_pull_share_one_scc_log_root() -> None:
    jobs_readme = _read(BU_SCC_JOBS / "README.md")

    assert "${SCC_LOG_ROOT}/eco1-rt-repack/proteinmpnn.\\$JOB_ID.\\$TASK_ID.out" in jobs_readme
    assert "${SCC_USER}@scc1.bu.edu:${SCC_LOG_ROOT}/eco1-rt-repack/" in jobs_readme
    assert "${SCC_PROJECT_ROOT}/proteinmpnn/eco1_rt_generation_policies/sge_logs/" not in jobs_readme


def test_bu_scc_readme_routes_every_qsub_to_a_parameterized_log_path() -> None:
    jobs_readme = _read(BU_SCC_JOBS / "README.md")
    bash_blocks = re.findall(r"```bash\n(.*?)```", jobs_readme, flags=re.DOTALL)
    qsub_commands = _qsub_commands(bash_blocks)

    assert qsub_commands
    assert all(command.endswith(".qsub") for command in qsub_commands)
    assert all(" -o " in command or "\n  -o " in command for command in qsub_commands)
    assert all("${SCC_LOG_ROOT}/" in command and r"\$JOB_ID" in command for command in qsub_commands)


def test_eco1_colabfold_qsub_requires_modern_gpu_capability() -> None:
    qsub_script = _read(BU_SCC_JOBS / "eco1-colabfold-foldcheck.qsub")

    assert "#$ -l gpus=1" in qsub_script
    assert "#$ -l gpu_compute_capability=6.0" in qsub_script
    assert "/project/dunlop/esouth" not in qsub_script
    assert "#$ -o" not in qsub_script
    assert "foldcheck_scc_runs" in qsub_script


def test_eco1_command_groups_include_scc_generation_policy_execution_lanes() -> None:
    command_groups = _read(ECO1_COMMAND_GROUPS / "README.md")
    pipeline = yaml.safe_load(_read(ECO1_COMMAND_GROUPS / "pipeline.yaml"))
    by_id = {lane["id"]: lane for lane in pipeline["lanes"]}

    assert "docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub" in command_groups
    command_group_blocks = re.findall(r"```bash\n(.*?)```", command_groups, flags=re.DOTALL)
    documented_qsub = _qsub_commands(command_group_blocks)
    assert documented_qsub
    assert all(command.endswith(".qsub") for command in documented_qsub)
    assert all("${SCC_LOG_ROOT}/" in command and r"\$JOB_ID" in command for command in documented_qsub)
    assert "/project/dunlop/esouth" not in command_groups
    assert "/projectnb/dunlop/esouth" not in command_groups
    assert "generation_policy_proteinmpnn_scc_smoke" in by_id
    assert "generation_policy_proteinmpnn_scc_array" in by_id
    assert by_id["generation_policy_proteinmpnn_scc_smoke"]["owner"] == "bu_scc_runtime"
    assert by_id["generation_policy_proteinmpnn_scc_array"]["owner"] == "bu_scc_runtime"

    smoke_argv = by_id["generation_policy_proteinmpnn_scc_smoke"]["command"]["argv"]
    array_argv = by_id["generation_policy_proteinmpnn_scc_array"]["command"]["argv"]
    assert smoke_argv[:3] == ["qsub", "-t", "1"]
    assert array_argv[:3] == ["qsub", "-t", "1-3"]
    assert any("PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn" in arg for arg in smoke_argv)
    assert "docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub" in array_argv

    qsub_lanes = [lane for lane in pipeline["lanes"] if (lane.get("command") or {}).get("argv", [None])[0] == "qsub"]
    assert qsub_lanes
    for lane in qsub_lanes:
        argv = lane["command"]["argv"]
        assert "-o" in argv
        assert argv[argv.index("-o") + 1].startswith("<scc_log_root>/")

    serialized = yaml.safe_dump(pipeline)
    assert "/project/dunlop/esouth" not in serialized
    assert "/projectnb/dunlop/esouth" not in serialized
