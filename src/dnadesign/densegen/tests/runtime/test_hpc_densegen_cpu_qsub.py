"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_hpc_densegen_cpu_qsub.py

Portable qsub-path pressure tests for DenseGen CPU batch script behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import dnadesign.usr as usr_pkg


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _write_fake_uv(bin_dir: Path) -> tuple[Path, Path, Path]:
    capture_path = bin_dir / "uv.calls"
    actor_capture_path = bin_dir / "uv.actors"
    run_count_path = bin_dir / "uv.run_count"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "$*" >> "$UV_CAPTURE"',
                'if [[ "$1" == "run" && "$2" == "dense" && "$3" == "run" ]]; then',
                '  if [[ -n "${UV_ACTOR_CAPTURE:-}" ]]; then',
                '    echo "${USR_ACTOR_TOOL:-}:${USR_ACTOR_RUN_ID:-}" >> "$UV_ACTOR_CAPTURE"',
                "  fi",
                '  if [[ -n "${UV_RUN_COUNT_FILE:-}" ]]; then',
                "    count=0",
                '    if [[ -f "$UV_RUN_COUNT_FILE" ]]; then',
                '      count="$(cat "$UV_RUN_COUNT_FILE")"',
                "    fi",
                "    count=$((count + 1))",
                '    echo "$count" > "$UV_RUN_COUNT_FILE"',
                '    fail_on="${UV_FAIL_ON_RUN_COUNT:-0}"',
                '    if [[ "$fail_on" != "0" && "$count" -eq "$fail_on" ]]; then',
                '      echo "simulated run interruption" >&2',
                "      exit 75",
                "    fi",
                "  fi",
                "fi",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    return capture_path, actor_capture_path, run_count_path


def _write_usr_registry(path: Path) -> None:
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_workspace_config(path: Path) -> None:
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
                    sequences: 1
                    sampling:
                      include_inputs: [basic_sites]
                    regulator_constraints:
                      groups: []

              solver:
                backend: CBC
                strategy: iterate

              runtime:
                round_robin: false
                max_accepted_per_library: 1
                min_count_per_tf: 0
                max_duplicate_solutions: 3
                no_progress_seconds_before_resample: 5
                max_consecutive_no_progress_resamples: 10
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
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_densegen_cpu_qsub_requires_existing_config(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/densegen-cpu.qsub"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "Missing DenseGen config:" in combined
    assert "Set DENSEGEN_CONFIG=/abs/path/to/config.yaml" in combined


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_densegen_cpu_qsub_passes_configured_validate_and_run_args(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/densegen-cpu.qsub"

    config_path = tmp_path / "config.yaml"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path, actor_capture_path, run_count_path = _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["UV_ACTOR_CAPTURE"] = str(actor_capture_path)
    env["UV_RUN_COUNT_FILE"] = str(run_count_path)
    env["DENSEGEN_CONFIG"] = str(config_path)
    env["DENSEGEN_VALIDATE_ARGS"] = "--probe-solver --dry-run"
    env["DENSEGEN_RUN_ARGS"] = "--resume --extend-quota 4 --no-plot"
    env["JOB_ID"] = "511"
    env["SGE_TASK_ID"] = "9"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "run dense validate-config --probe-solver --dry-run -c" in calls[0]
    assert str(config_path) in calls[0]
    assert "run dense run --resume --extend-quota 4 --no-plot -c" in calls[1]
    assert str(config_path) in calls[1]
    assert actor_capture_path.read_text(encoding="utf-8").splitlines() == ["densegen:511.9"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_densegen_cpu_qsub_supports_retry_after_injected_run_failure(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/densegen-cpu.qsub"

    config_path = tmp_path / "config.yaml"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path, actor_capture_path, run_count_path = _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["UV_ACTOR_CAPTURE"] = str(actor_capture_path)
    env["UV_RUN_COUNT_FILE"] = str(run_count_path)
    env["UV_FAIL_ON_RUN_COUNT"] = "1"
    env["DENSEGEN_CONFIG"] = str(config_path)
    env["DENSEGEN_RUN_ARGS"] = "--resume --extend-quota 8 --no-plot"
    env["JOB_ID"] = "700"
    env["SGE_TASK_ID"] = "2"

    first = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode != 0
    assert "simulated run interruption" in first.stderr

    second = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr

    calls = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    run_calls = [line for line in calls if "run dense run " in line]
    assert len(run_calls) == 2
    assert all("--resume --extend-quota 8 --no-plot -c" in line for line in run_calls)
    assert actor_capture_path.read_text(encoding="utf-8").splitlines() == ["densegen:700.2", "densegen:700.2"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_densegen_cpu_qsub_supports_multiple_interrupted_submissions_before_success(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/densegen-cpu.qsub"

    config_path = tmp_path / "config.yaml"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path, actor_capture_path, run_count_path = _write_fake_uv(bin_dir)

    base_env = dict(os.environ)
    base_env["PATH"] = f"{bin_dir}:{base_env.get('PATH', '')}"
    base_env["UV_CAPTURE"] = str(capture_path)
    base_env["UV_ACTOR_CAPTURE"] = str(actor_capture_path)
    base_env["UV_RUN_COUNT_FILE"] = str(run_count_path)
    base_env["DENSEGEN_CONFIG"] = str(config_path)
    base_env["DENSEGEN_RUN_ARGS"] = "--resume --extend-quota 8 --no-plot"
    base_env["JOB_ID"] = "811"
    base_env["SGE_TASK_ID"] = "4"

    first_env = dict(base_env)
    first_env["UV_FAIL_ON_RUN_COUNT"] = "1"
    first = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=first_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode != 0
    assert "simulated run interruption" in first.stderr

    second_env = dict(base_env)
    second_env["UV_FAIL_ON_RUN_COUNT"] = "2"
    second = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=second_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "simulated run interruption" in second.stderr

    third_env = dict(base_env)
    third_env["UV_FAIL_ON_RUN_COUNT"] = "0"
    third = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=third_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert third.returncode == 0, third.stderr

    calls = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    run_calls = [line for line in calls if "run dense run " in line]
    assert len(run_calls) == 3
    assert all("--resume --extend-quota 8 --no-plot -c" in line for line in run_calls)
    assert actor_capture_path.read_text(encoding="utf-8").splitlines() == [
        "densegen:811.4",
        "densegen:811.4",
        "densegen:811.4",
    ]


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("uv") is None,
    reason="bash and uv are required",
)
def test_densegen_cpu_qsub_real_local_resume_extend_flow(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/densegen-cpu.qsub"

    workspace = tmp_path / "workspace"
    (workspace / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace / "inputs" / "sites.csv").write_text("tf,tfbs\nTF1,AAA\n", encoding="utf-8")
    _write_usr_registry(workspace / "outputs" / "usr" / "registry.yaml")
    cfg_path = workspace / "config.yaml"
    _write_workspace_config(cfg_path)

    env = dict(os.environ)
    env["DENSEGEN_CONFIG"] = str(cfg_path)
    env["JOB_ID"] = "9001"
    env["SGE_TASK_ID"] = "3"

    first = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    records_path = workspace / "outputs" / "usr" / "demo_workspace" / "records.parquet"
    rows_after_first = int(pq.read_table(records_path).num_rows)
    assert rows_after_first >= 1

    second_env = dict(env)
    second_env["DENSEGEN_RUN_ARGS"] = "--resume --extend-quota 1 --no-plot"
    second = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(repo_root),
        env=second_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    rows_after_second = int(pq.read_table(records_path).num_rows)
    assert rows_after_second == rows_after_first + 1

    third_env = dict(env)
    third_env["DENSEGEN_RUN_ARGS"] = "--resume --no-plot"
    third = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(repo_root),
        env=third_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert third.returncode == 0, third.stderr
    rows_after_third = int(pq.read_table(records_path).num_rows)
    assert rows_after_third == rows_after_second

    state_path = workspace / "outputs" / "meta" / "run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert int(payload.get("total_generated", -1)) == rows_after_third
