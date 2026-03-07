"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_runbook_orchestrator.py

Contract tests for ops runbook loading, mode selection, and plan rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import inspect
import os
import shlex
from pathlib import Path
from typing import get_args

import pytest
import yaml
from typer.testing import CliRunner

import dnadesign.ops.orchestrator.state as orchestrator_state
import dnadesign.ops.runbooks.schema as runbook_schema
from dnadesign.ops.cli import app
from dnadesign.ops.orchestrator.execute import execute_batch_plan
from dnadesign.ops.orchestrator.plan import (
    BatchPlan,
    CommandSpec,
    OrchestrationNotifySpec,
    build_batch_plan,
)
from dnadesign.ops.orchestrator.state import discover_active_job_ids_for_runbook, resolve_mode_decision
from dnadesign.ops.runbooks.schema import load_orchestration_runbook


def test_workflow_helpers_classify_all_schema_workflow_ids() -> None:
    workflow_ids = get_args(runbook_schema.OrchestrationRunbookV1.model_fields["workflow_id"].annotation)
    assert workflow_ids
    for workflow_id in workflow_ids:
        is_densegen = runbook_schema.is_densegen_workflow_id(workflow_id)
        is_infer = runbook_schema.is_infer_workflow_id(workflow_id)
        assert is_densegen != is_infer


def test_ops_plan_avoids_infer_internal_module_imports() -> None:
    import dnadesign.ops.orchestrator.plan as plan_module

    plan_source = inspect.getsource(plan_module)
    assert "dnadesign.infer.src." not in plan_source


def test_ops_plan_import_does_not_eagerly_load_gpu_runtime_modules() -> None:
    import subprocess
    import sys

    script = """
import sys
import dnadesign.ops.orchestrator.plan
print(f"torch_loaded={'torch' in sys.modules}")
print(f"evo2_loaded={'evo2' in sys.modules}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    lines = {line.strip() for line in (result.stdout or "").splitlines() if line.strip()}
    assert "torch_loaded=False" in lines
    assert "evo2_loaded=False" in lines


def _render_block(commands: list[CommandSpec]) -> str:
    return "\n".join(command.render_shell() for command in commands)


def _write_runbook(
    tmp_path: Path,
    *,
    include_smoke: bool = True,
    include_notify: bool = True,
) -> Path:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    notify_root = workspace_root / "outputs" / "notify" / "densegen"
    if include_notify:
        notify_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "config.yaml").write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    runbook_payload: dict[str, object] = {
        "schema_version": 1,
        "id": "study_stress_ethanol_cipro",
        "workflow_id": ("densegen_batch_with_notify_slack" if include_notify else "densegen_batch_submit"),
        "project": "dunlop",
        "workspace_root": str(workspace_root),
        "logging": {
            "stdout_dir": str(workspace_root / "outputs" / "logs" / "ops" / "sge" / "study_stress_ethanol_cipro"),
        },
        "densegen": {
            "config": str(workspace_root / "config.yaml"),
            "qsub_template": "docs/bu-scc/jobs/densegen-cpu.qsub",
            "run_args": {
                "fresh": "--fresh --no-plot",
                "resume": "--resume --no-plot",
            },
        },
        "resources": {
            "pe_omp": 16,
            "h_rt": "08:00:00",
            "mem_per_core": "8G",
        },
        "mode_policy": {
            "default": "auto",
            "on_active_job": "hold_jid",
        },
    }
    if include_notify:
        notify_block: dict[str, object] = {
            "tool": "densegen",
            "policy": "densegen",
            "profile": str(notify_root / "profile.json"),
            "cursor": str(notify_root / "cursor"),
            "spool_dir": str(notify_root / "spool"),
            "webhook_env": "NOTIFY_WEBHOOK",
            "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
        }
        if include_smoke:
            notify_block["smoke"] = "dry"
        runbook_payload["notify"] = notify_block

    payload = {
        "runbook": {
            **runbook_payload,
        }
    }
    runbook_path = tmp_path / "runbook.yaml"
    runbook_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return runbook_path


def _infer_runbook_payload(
    workspace_root: Path,
    *,
    runbook_id: str = "infer_evo2_batch",
    mode_default: str = "auto",
    usr_root: Path | None = None,
    usr_dataset: str = "demo",
) -> dict[str, object]:
    workspace_root.mkdir(parents=True, exist_ok=True)
    selected_usr_root = usr_root or (workspace_root / "outputs" / "usr_datasets")
    config_path = workspace_root / "config.yaml"
    if not config_path.exists():
        config_path.write_text(
            """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: job_a
    operation: extract
    ingest:
      source: usr
      root: "__USR_ROOT__"
      dataset: "__USR_DATASET__"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
""".strip()
            .replace("__USR_ROOT__", str(selected_usr_root))
            .replace("__USR_DATASET__", usr_dataset)
            + "\n",
            encoding="utf-8",
        )

    return {
        "runbook": {
            "schema_version": 1,
            "id": runbook_id,
            "workflow_id": "infer_batch_submit",
            "project": "dunlop",
            "workspace_root": str(workspace_root),
            "logging": {
                "stdout_dir": str(workspace_root / "outputs" / "logs" / "ops" / "sge" / runbook_id),
            },
            "infer": {
                "config": str(config_path),
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
            "mode_policy": {
                "default": mode_default,
                "on_active_job": "hold_jid",
            },
        }
    }


@pytest.fixture(autouse=True)
def _set_notify_webhook_file_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook_file = tmp_path / "notify_webhook.secret"
    webhook_file.write_text("https://hooks.slack.com/services/T000/B000/TEST\n", encoding="utf-8")
    monkeypatch.setenv("NOTIFY_WEBHOOK_FILE", str(webhook_file.resolve()))
    monkeypatch.delenv("NOTIFY_WEBHOOK", raising=False)


def test_runbook_notify_smoke_defaults_to_dry(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path, include_smoke=False)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.notify.smoke == "dry"


def test_runbook_relative_paths_resolve_against_runbook_parent(tmp_path: Path) -> None:
    runbook_dir = tmp_path / "contracts"
    workspace_dir = runbook_dir / "workspace"
    notify_dir = workspace_dir / "outputs" / "notify" / "densegen"
    notify_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "config.yaml").write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "study_stress_ethanol_cipro",
            "workflow_id": "densegen_batch_with_notify_slack",
            "project": "dunlop",
            "workspace_root": "workspace",
            "logging": {
                "stdout_dir": "workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
            },
            "densegen": {
                "config": "workspace/config.yaml",
                "qsub_template": "docs/bu-scc/jobs/densegen-cpu.qsub",
                "run_args": {
                    "fresh": "--fresh --no-plot",
                    "resume": "--resume --no-plot",
                },
            },
            "notify": {
                "tool": "densegen",
                "policy": "densegen",
                "profile": "workspace/outputs/notify/densegen/profile.json",
                "cursor": "workspace/outputs/notify/densegen/cursor",
                "spool_dir": "workspace/outputs/notify/densegen/spool",
                "webhook_env": "NOTIFY_WEBHOOK",
                "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
                "smoke": "dry",
            },
            "resources": {
                "pe_omp": 16,
                "h_rt": "08:00:00",
                "mem_per_core": "8G",
            },
            "mode_policy": {
                "default": "auto",
                "on_active_job": "hold_jid",
            },
        }
    }
    runbook_path = runbook_dir / "runbook.yaml"
    runbook_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    runbook = load_orchestration_runbook(runbook_path)

    assert runbook.workspace_root == workspace_dir.resolve()
    assert runbook.densegen is not None
    assert runbook.densegen.config == (workspace_dir / "config.yaml").resolve()
    assert runbook.notify.profile == (notify_dir / "profile.json").resolve()


def test_runbook_default_post_run_template_resolves_to_repo_jobs_template(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path, include_notify=False)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.densegen is not None
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    expected_template = (repo_root / "docs" / "bu-scc" / "jobs" / "densegen-analysis.qsub").resolve()
    assert runbook.densegen.post_run.qsub_template == expected_template


def test_runbook_rejects_stdout_dir_outside_workspace_ops_logs(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["logging"]["stdout_dir"] = str(tmp_path / "outside" / "logs")

    with pytest.raises(ValueError, match="logging.stdout_dir must be under"):
        load_orchestration_runbook(runbook_path, raw=payload)


def test_runbook_rejects_stdout_dir_not_scoped_to_runbook_id(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    workspace_root = Path(payload["runbook"]["workspace_root"])
    payload["runbook"]["logging"]["stdout_dir"] = str(
        workspace_root / "outputs" / "logs" / "ops" / "sge" / "different_runbook_id"
    )

    with pytest.raises(ValueError, match="logging.stdout_dir must be exactly"):
        load_orchestration_runbook(runbook_path, raw=payload)


def test_runbook_rejects_invalid_log_retention_values(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["logging"]["retention"] = {
        "keep_last": 0,
        "max_age_days": 7,
    }

    with pytest.raises(ValueError, match="keep_last"):
        load_orchestration_runbook(runbook_path, raw=payload)


def test_runbook_rejects_notify_profile_outside_workspace_notify_namespace(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["notify"]["profile"] = str(tmp_path / "outside" / "notify" / "profile.json")

    with pytest.raises(ValueError, match="notify.profile must be"):
        load_orchestration_runbook(runbook_path, raw=payload)


def test_runbook_rejects_legacy_overlay_guard_namespace_key_with_migration_hint(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["densegen"]["overlay_guard"] = {  # type: ignore[index]
        "max_projected_overlay_parts": 20000,
        "max_existing_overlay_parts": 5000,
        "auto_compact_existing_overlay_parts": True,
        "namespace": "densegen",
    }

    with pytest.raises(
        ValueError,
        match="overlay_guard\\.namespace is not supported; use overlay_guard\\.overlay_namespace",
    ):
        load_orchestration_runbook(runbook_path, raw=payload)


def test_runbook_rejects_invalid_overlay_guard_namespace_pattern(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["densegen"]["overlay_guard"] = {  # type: ignore[index]
        "max_projected_overlay_parts": 20000,
        "max_existing_overlay_parts": 5000,
        "auto_compact_existing_overlay_parts": True,
        "overlay_namespace": "DenseGen-Invalid",
    }

    with pytest.raises(
        ValueError,
        match="densegen\\.overlay_guard\\.overlay_namespace must match",
    ):
        load_orchestration_runbook(runbook_path, raw=payload)


def test_mode_auto_selects_fresh_without_artifacts(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())
    assert decision.selected_mode == "fresh"
    assert decision.run_args == "--fresh --no-plot"


def test_mode_auto_selects_resume_with_artifacts(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "meta" / "run_manifest.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")

    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())
    assert decision.selected_mode == "resume"
    assert decision.run_args == "--resume --no-plot"


def test_mode_auto_treats_registry_only_state_as_fresh(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "usr_datasets" / "registry.yaml"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("version: 1\n", encoding="utf-8")

    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())
    assert decision.selected_mode == "fresh"
    assert decision.run_args == "--fresh --no-plot"


def test_mode_auto_raises_when_partial_artifacts_exist(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "tables" / "records.parquet"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("not-a-parquet-file\n", encoding="utf-8")

    with pytest.raises(ValueError, match="auto mode blocked"):
        resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())


def test_mode_auto_raises_when_orphan_densegen_artifacts_exist(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "pools" / "pool_manifest.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="auto mode blocked"):
        resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())


def test_mode_resume_raises_when_resume_not_ready(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "usr_datasets" / "registry.yaml"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume mode blocked"):
        resolve_mode_decision(runbook=runbook, requested_mode="resume", active_job_ids=())


def test_mode_resume_raises_when_resume_records_missing_densegen_columns(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    records_path = runbook.workspace_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.table(
        {
            "id": ["r1"],
            "sequence": ["ATGCATGC"],
        }
    )
    pyarrow_parquet.write_table(table, records_path)

    with pytest.raises(ValueError, match="resume mode blocked"):
        resolve_mode_decision(runbook=runbook, requested_mode="resume", active_job_ids=())


def test_mode_resume_accepts_nested_densegen_used_tfbs_detail_column(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    records_path = runbook.workspace_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    used_tfbs_type = pyarrow.list_(
        pyarrow.struct(
            [
                pyarrow.field("part_kind", pyarrow.string()),
            ]
        )
    )
    used_tfbs_detail = pyarrow.array([[{"part_kind": "tfbs"}]], type=used_tfbs_type)
    table = pyarrow.table(
        {
            "id": ["r1"],
            "sequence": ["ATGCATGC"],
            "densegen__run_id": ["study_stress_ethanol_cipro"],
            "densegen__input_name": ["plan_pool__ethanol__sig35_f"],
            "densegen__plan": ["ethanol__sig35=f"],
            "densegen__used_tfbs_detail": used_tfbs_detail,
        }
    )
    pyarrow_parquet.write_table(table, records_path)

    decision = resolve_mode_decision(runbook=runbook, requested_mode="resume", active_job_ids=())
    assert decision.selected_mode == "resume"
    assert decision.run_args == "--resume --no-plot"


def test_mode_auto_selects_resume_with_record_part_artifacts_only(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    records_part_path = runbook.workspace_root / "outputs" / "tables" / "records__part-test.parquet"
    records_part_path.parent.mkdir(parents=True, exist_ok=True)
    used_tfbs_type = pyarrow.list_(
        pyarrow.struct(
            [
                pyarrow.field("part_kind", pyarrow.string()),
            ]
        )
    )
    used_tfbs_detail = pyarrow.array([[{"part_kind": "tfbs"}]], type=used_tfbs_type)
    records_part_table = pyarrow.table(
        {
            "id": ["r1"],
            "sequence": ["ATGCATGC"],
            "densegen__run_id": ["study_stress_ethanol_cipro"],
            "densegen__input_name": ["plan_pool__ethanol__sig35_f"],
            "densegen__plan": ["ethanol__sig35=f"],
            "densegen__used_tfbs_detail": used_tfbs_detail,
        }
    )
    pyarrow_parquet.write_table(records_part_table, records_part_path)

    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())
    assert decision.selected_mode == "resume"
    assert decision.run_args == "--resume --no-plot"


def test_mode_auto_selects_resume_when_attempt_artifacts_exist_with_usr_base_records(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    usr_records_path = (
        runbook.workspace_root
        / "outputs"
        / "usr_datasets"
        / "densegen"
        / "study_stress_ethanol_cipro"
        / "records.parquet"
    )
    usr_records_path.parent.mkdir(parents=True, exist_ok=True)
    base_records_table = pyarrow.table({"id": ["r1"], "sequence": ["ATGCATGC"]})
    pyarrow_parquet.write_table(base_records_table, usr_records_path)

    attempts_path = runbook.workspace_root / "outputs" / "tables" / "attempts_part-test.parquet"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_table = pyarrow.table({"attempt_id": ["a1"]})
    pyarrow_parquet.write_table(attempts_table, attempts_path)

    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())
    assert decision.selected_mode == "resume"
    assert decision.run_args == "--resume --no-plot"


def test_mode_fresh_raises_when_resume_artifacts_exist_without_reset_ack(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "meta" / "run_manifest.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fresh mode blocked"):
        resolve_mode_decision(runbook=runbook, requested_mode="fresh", active_job_ids=())


def test_mode_fresh_allows_explicit_reset_ack_when_resume_artifacts_exist(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    marker = runbook.workspace_root / "outputs" / "meta" / "run_manifest.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")

    decision = resolve_mode_decision(
        runbook=runbook,
        requested_mode="fresh",
        active_job_ids=(),
        allow_fresh_reset=True,
    )
    assert decision.selected_mode == "fresh"
    assert decision.run_args == "--fresh --no-plot"


def test_mode_auto_with_active_jobs_returns_hold_jid(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=("81001", "81002"))
    assert decision.submit_behavior == "hold_jid"
    assert decision.hold_jid == "81001,81002"


def test_mode_auto_with_active_jobs_normalizes_hold_jid_list(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    decision = resolve_mode_decision(
        runbook=runbook,
        requested_mode=None,
        active_job_ids=("81002", "81001", "81002", "  ", ""),
    )
    assert decision.submit_behavior == "hold_jid"
    assert decision.hold_jid == "81001,81002"


def test_mode_auto_with_active_jobs_normalizes_comma_delimited_ids(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    decision = resolve_mode_decision(
        runbook=runbook,
        requested_mode=None,
        active_job_ids=("81002,81001", "81002"),
    )
    assert decision.submit_behavior == "hold_jid"
    assert decision.hold_jid == "81001,81002"


def test_build_batch_plan_forwards_allow_fresh_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    captured: dict[str, object] = {}

    def _fake_resolve_mode_decision(*, runbook, requested_mode, active_job_ids, allow_fresh_reset=False):
        captured["requested_mode"] = requested_mode
        captured["active_job_ids"] = tuple(active_job_ids)
        captured["allow_fresh_reset"] = allow_fresh_reset
        return orchestrator_state.ModeDecision(
            requested_mode="fresh",
            selected_mode="fresh",
            run_args="--fresh --no-plot",
            resume_artifacts_found=True,
            submit_behavior="submit",
            hold_jid=None,
            reason="selected_mode=fresh; resume_ready=true; fresh_reset_ack=true",
        )

    monkeypatch.setattr("dnadesign.ops.orchestrator.plan.resolve_mode_decision", _fake_resolve_mode_decision)

    plan = build_batch_plan(
        runbook=runbook,
        requested_mode="fresh",
        requested_smoke=None,
        active_job_ids=(),
        allow_fresh_reset=True,
    )

    assert captured["requested_mode"] == "fresh"
    assert captured["active_job_ids"] == ()
    assert captured["allow_fresh_reset"] is True
    assert plan.selected_mode == "fresh"


def test_discover_active_job_ids_matches_densegen_config_and_notify_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    assert runbook.densegen is not None
    assert runbook.notify is not None
    qstat_table = """
job-ID prior name user state submit/start at queue slots ja-task-ID
--------------------------------------------------------------------------------
81001 0.555 a b r 03/01/2026 queueA 16
81002 0.555 a b qw 03/01/2026 queueA 1
81003 0.555 a b qw 03/01/2026 queueA 1
"""
    job_details = {
        "81001": f"env_list: DENSEGEN_CONFIG={runbook.densegen.config}",
        "81002": f"env_list: NOTIFY_PROFILE={runbook.notify.profile}",
        "81003": "env_list: DENSEGEN_CONFIG=/tmp/other/config.yaml",
    }

    def _probe(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[:2] == ("qstat", "-u"):
            return 0, qstat_table, ""
        if argv[:2] == ("qstat", "-j"):
            return 0, job_details.get(argv[2], ""), ""
        raise AssertionError(f"Unexpected probe argv: {argv}")

    monkeypatch.setattr(orchestrator_state, "_run_probe", _probe)
    discovered = discover_active_job_ids_for_runbook(runbook, max_jobs=12)

    assert discovered == ("81001", "81002")


def test_discover_active_job_ids_raises_when_qstat_snapshot_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    monkeypatch.setattr(orchestrator_state, "_run_probe", lambda argv: (1, "", "qstat unavailable"))

    with pytest.raises(RuntimeError, match="qstat unavailable"):
        discover_active_job_ids_for_runbook(runbook, max_jobs=12)


def test_batch_plan_uses_dry_smoke_by_default(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    smoke_block = _render_block(plan.notify_smoke_commands)
    assert "--dry-run" in smoke_block
    assert "--no-advance-cursor-on-dry-run" in smoke_block
    assert "notify send" not in smoke_block


def test_batch_plan_enables_orchestration_notifications_by_default(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    assert plan.orchestration_notify is not None
    assert plan.orchestration_notify.tool == "densegen"
    assert plan.orchestration_notify.webhook_env == "NOTIFY_WEBHOOK"
    assert plan.orchestration_notify.secret_ref == Path(os.environ["NOTIFY_WEBHOOK_FILE"]).resolve().as_uri()


def test_batch_plan_uses_secret_ref_for_orchestration_notifications_when_webhook_file_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    webhook_file = (tmp_path / "notify_webhook.secret").resolve()
    webhook_file.write_text("https://example.invalid/webhook\n", encoding="utf-8")
    monkeypatch.setenv("NOTIFY_WEBHOOK_FILE", str(webhook_file))

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    assert plan.orchestration_notify is not None
    assert plan.orchestration_notify.webhook_env == "NOTIFY_WEBHOOK"
    assert plan.orchestration_notify.secret_ref == webhook_file.as_uri()


def test_batch_plan_allows_orchestration_notification_opt_out(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["notify"]["orchestration_events"] = False
    runbook = load_orchestration_runbook(runbook_path, raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    assert plan.orchestration_notify is None


def test_batch_plan_requires_tls_ca_bundle_for_notify_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr("dnadesign.ops.orchestrator.plan.DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES", ())

    with pytest.raises(ValueError, match="notify TLS CA bundle is not configured"):
        build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())


def test_batch_plan_rejects_unreadable_ssl_cert_file_for_notify_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    monkeypatch.setenv("SSL_CERT_FILE", str((tmp_path / "missing-ca-bundle.pem").resolve()))

    with pytest.raises(ValueError, match="SSL_CERT_FILE"):
        build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())


def test_densegen_batch_submit_plan_skips_notify_smoke_and_watcher_submit(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path, include_notify=False)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    submit_block = _render_block(plan.submit_commands)

    assert plan.workflow_id == "densegen_batch_submit"
    assert plan.notify_smoke_commands == []
    assert "NOTIFY_PROFILE" not in submit_block
    assert "DENSEGEN_CONFIG" in submit_block
    assert "DENSEGEN_RUN_ARGS=--fresh --no-plot" in submit_block
    assert f"DENSEGEN_TRACE_DIR={runbook.workspace_root}/outputs/logs/ops/runtime" in submit_block
    assert "docs/bu-scc/jobs/densegen-analysis.qsub" in submit_block
    assert "-hold_jid study_stress_ethanol_cipro_densegen_cpu" in submit_block
    assert "DENSEGEN_NOTEBOOK_FORCE" not in submit_block


def test_densegen_preflight_verifies_post_run_analysis_template(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "docs/bu-scc/jobs/densegen-analysis.qsub" in preflight_block
    assert "qa-submit-preflight --template" in preflight_block
    assert "densegen-analysis.qsub" in preflight_block


def test_densegen_post_run_can_use_dedicated_resources(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["densegen"]["post_run"] = {
        "qsub_template": "docs/bu-scc/jobs/densegen-analysis.qsub",
        "resources": {
            "pe_omp": 1,
            "h_rt": "00:20:00",
            "mem_per_core": "2G",
        },
    }
    runbook = load_orchestration_runbook(runbook_path, raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    post_run_verify = next(
        command
        for command in plan.preflight_commands
        if command.argv is not None
        and command.argv[:2] == ("qsub", "-verify")
        and command.argv[-1].endswith("densegen-analysis.qsub")
    )
    post_run_submit = next(
        command
        for command in plan.submit_commands
        if command.argv is not None
        and command.argv[0] == "qsub"
        and command.argv[-1].endswith("densegen-analysis.qsub")
    )

    post_run_verify_shell = post_run_verify.render_shell()
    post_run_submit_shell = post_run_submit.render_shell()
    assert "-pe omp 1" in post_run_verify_shell
    assert "-l h_rt=00:20:00" in post_run_verify_shell
    assert "-l mem_per_core=2G" in post_run_verify_shell
    assert "-pe omp 1" in post_run_submit_shell
    assert "-l h_rt=00:20:00" in post_run_submit_shell
    assert "-l mem_per_core=2G" in post_run_submit_shell
    assert "-hold_jid study_stress_ethanol_cipro_densegen_cpu" in post_run_submit_shell


def test_densegen_post_run_defaults_to_small_analysis_resources(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    post_run_submit = next(
        command
        for command in plan.submit_commands
        if command.argv is not None
        and command.argv[0] == "qsub"
        and command.argv[-1].endswith("densegen-analysis.qsub")
    )

    post_run_submit_shell = post_run_submit.render_shell()
    assert "-pe omp 4" in post_run_submit_shell
    assert "-l h_rt=01:00:00" in post_run_submit_shell
    assert "-l mem_per_core=4G" in post_run_submit_shell
    assert "-hold_jid study_stress_ethanol_cipro_densegen_cpu" in post_run_submit_shell


def test_notify_submit_uses_webhook_file_without_embedding_secret(
    tmp_path: Path,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    submit_block = _render_block(plan.submit_commands)

    assert "NOTIFY_PROFILE=" in submit_block
    assert "WEBHOOK_ENV=NOTIFY_WEBHOOK" in submit_block
    assert f"WEBHOOK_FILE={Path(os.environ['NOTIFY_WEBHOOK_FILE']).resolve()}" in submit_block
    assert "https://hooks.slack.com/services/" not in submit_block


def test_notify_submit_includes_webhook_file_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    webhook_file = tmp_path / "notify_webhook.secret"
    webhook_file.write_text("https://example.invalid/webhook\n", encoding="utf-8")
    monkeypatch.setenv("NOTIFY_WEBHOOK_FILE", str(webhook_file))

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    submit_block = _render_block(plan.submit_commands)

    assert f"WEBHOOK_FILE={webhook_file}" in submit_block
    assert "WEBHOOK_ENV=NOTIFY_WEBHOOK" in submit_block


def test_notify_submit_aligns_runtime_and_idle_timeout_with_runbook(
    tmp_path: Path,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    notify_submit = plan.submit_commands[0].render_shell()
    assert "-l h_rt=08:00:00" in notify_submit
    assert "NOTIFY_IDLE_TIMEOUT_SECONDS=28800" in notify_submit
    assert "NOTIFY_ENFORCE_TERMINAL_ON_IDLE=1" in notify_submit


def test_notify_submit_includes_tls_ca_bundle_for_watcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    ca_bundle = tmp_path / "ca-bundle.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_bundle))
    monkeypatch.setenv("NOTIFY_WEBHOOK", "https://hooks.slack.com/services/T000/B000/TEST")

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    notify_submit = plan.submit_commands[0].render_shell()

    assert f"NOTIFY_TLS_CA_BUNDLE={ca_bundle}" in notify_submit


def test_batch_plan_uses_single_notify_profile_smoke_command(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)
    smoke_block = _render_block(plan.notify_smoke_commands)

    assert "notify profile doctor" not in preflight_block
    assert "notify setup slack" not in smoke_block
    assert "notify usr-events watch" not in smoke_block
    assert "notify profile smoke --profile" in smoke_block
    assert "--dry-run" in smoke_block
    assert "--no-advance-cursor-on-dry-run" in smoke_block


def test_batch_plan_requires_webhook_file_for_notify_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    monkeypatch.delenv("NOTIFY_WEBHOOK_FILE", raising=False)

    with pytest.raises(ValueError, match="NOTIFY_WEBHOOK_FILE"):
        build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())


def test_batch_plan_uses_profile_secret_ref_when_webhook_file_env_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    monkeypatch.delenv("NOTIFY_WEBHOOK_FILE", raising=False)
    webhook_file = (tmp_path / "persisted_notify_webhook.secret").resolve()
    webhook_file.write_text("https://example.invalid/persisted\n", encoding="utf-8")
    runbook.notify.profile.write_text(
        json.dumps({"webhook": {"source": "secret_ref", "ref": webhook_file.as_uri()}}),
        encoding="utf-8",
    )

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    submit_block = _render_block(plan.submit_commands)

    assert plan.orchestration_notify is not None
    assert plan.orchestration_notify.secret_ref == webhook_file.as_uri()
    assert f"WEBHOOK_FILE={webhook_file}" in submit_block


def test_batch_plan_rejects_non_file_profile_secret_ref_when_webhook_file_env_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    monkeypatch.delenv("NOTIFY_WEBHOOK_FILE", raising=False)
    runbook.notify.profile.write_text(
        json.dumps({"webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/default"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="file://"):
        build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())


def test_batch_plan_notify_setup_uses_file_secret_contract_when_webhook_file_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    webhook_file = (tmp_path / "notify_webhook.secret").resolve()
    webhook_file.write_text("https://example.invalid/webhook\n", encoding="utf-8")
    monkeypatch.setenv("NOTIFY_WEBHOOK_FILE", str(webhook_file))

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    smoke_block = _render_block(plan.notify_smoke_commands)

    assert "--secret-source file" in smoke_block
    assert f"--secret-ref {webhook_file.as_uri()}" in smoke_block
    assert "--no-store-webhook" in smoke_block
    assert "--url-env NOTIFY_WEBHOOK" not in smoke_block


def test_batch_plan_notify_smoke_uses_profile_smoke_cli(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    smoke_block = _render_block(plan.notify_smoke_commands)
    assert "notify profile smoke --profile" in smoke_block
    assert "--tool densegen" in smoke_block
    assert "--config" in smoke_block


def test_batch_plan_includes_preflight_gate_commands(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "dnadesign.ops.orchestrator.gates qa-submit-preflight" in preflight_block
    assert "dnadesign.ops.orchestrator.gates submit-shape-advisor" in preflight_block
    assert "dnadesign.ops.orchestrator.gates operator-brief" in preflight_block


def test_batch_plan_includes_log_retention_prune_gate(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["logging"]["retention"] = {
        "keep_last": 3,
        "max_age_days": 5,
    }
    runbook = load_orchestration_runbook(runbook_path, raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "dnadesign.ops.orchestrator.gates prune-ops-logs" in preflight_block
    assert f"--stdout-dir {shlex.quote(str(runbook.logging.stdout_dir))}" in preflight_block
    assert "--runbook-id study_stress_ethanol_cipro" in preflight_block
    assert "--keep-last 3" in preflight_block
    assert "--max-age-days 5" in preflight_block
    assert "--json" in preflight_block
    assert "--log-kind sge" in preflight_block


def test_batch_plan_includes_runtime_log_retention_prune_gate(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    runtime_dir = runbook.workspace_root / "outputs" / "logs" / "ops" / "runtime"
    expected_manifest = runtime_dir / "retention-manifest.json"
    assert "dnadesign.ops.orchestrator.gates ensure-dir-writable" in preflight_block
    assert f"--path {shlex.quote(str(runtime_dir))}" in preflight_block
    assert f"--stdout-dir {shlex.quote(str(runtime_dir))}" in preflight_block
    assert "--log-kind runtime" in preflight_block
    assert f"--manifest-path {shlex.quote(str(expected_manifest))}" in preflight_block


def test_batch_plan_includes_session_counts_gate_instead_of_qstat_shell_awk(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "dnadesign.ops.orchestrator.gates session-counts" in preflight_block
    assert 'qstat -u "$USER" | awk' not in preflight_block


def test_densegen_notify_preflight_requires_usr_events_path_contract(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "uv run dense inspect run --usr-events-path -c" in preflight_block


def test_densegen_batch_only_preflight_skips_usr_events_path_contract(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path, include_notify=False)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "uv run dense inspect run --usr-events-path -c" not in preflight_block


def test_densegen_preflight_solver_probe_includes_gurobi_runtime_env(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    probe_command = next(
        command
        for command in plan.preflight_commands
        if command.argv is not None and command.argv[:4] == ("uv", "run", "dense", "validate-config")
    )

    assert probe_command.env["GUROBI_HOME"].endswith("gurobi/10.0.1/install")
    assert probe_command.env["GRB_LICENSE_FILE"] == "/usr/local/gurobi/gurobi.lic"
    assert probe_command.env["TOKENSERVER"] == "sccsvc.bu.edu"
    assert probe_command.env["LD_LIBRARY_PATH"].startswith("/share/pkg.7/gurobi/10.0.1/install/lib")


def test_densegen_preflight_includes_overlay_sprawl_guard_command(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.densegen is not None

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "dnadesign.ops.orchestrator.gates usr-overlay-guard" in preflight_block
    assert "--tool densegen" in preflight_block
    assert f"--config {shlex.quote(str(runbook.densegen.config))}" in preflight_block
    assert f"--workspace-root {shlex.quote(str(runbook.workspace_root))}" in preflight_block
    assert "--mode fresh" in preflight_block
    assert "--run-args '--fresh --no-plot'" in preflight_block


def test_densegen_preflight_includes_records_part_guard_command(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.densegen is not None

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "dnadesign.ops.orchestrator.gates usr-records-part-guard" in preflight_block
    assert "--tool densegen" in preflight_block
    assert f"--config {shlex.quote(str(runbook.densegen.config))}" in preflight_block
    assert f"--workspace-root {shlex.quote(str(runbook.workspace_root))}" in preflight_block
    assert "--max-projected-records-parts" in preflight_block
    assert "--max-existing-records-parts" in preflight_block
    assert "--max-existing-records-part-age-days" in preflight_block


def test_densegen_preflight_includes_archived_overlay_guard_command(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.densegen is not None

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "dnadesign.ops.orchestrator.gates usr-archived-overlay-guard" in preflight_block
    assert "--tool densegen" in preflight_block
    assert f"--config {shlex.quote(str(runbook.densegen.config))}" in preflight_block
    assert f"--workspace-root {shlex.quote(str(runbook.workspace_root))}" in preflight_block
    assert "--max-archived-entries" in preflight_block
    assert "--max-archived-bytes" in preflight_block


def test_batch_plan_includes_live_canary_when_overridden(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke="live", active_job_ids=())
    smoke_block = _render_block(plan.notify_smoke_commands)
    assert "--dry-run" in smoke_block
    assert "notify send" in smoke_block


def test_batch_plan_uses_structured_specs_and_safe_shell_rendering(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.densegen is not None

    config_with_spaces = tmp_path / "workspace with spaces" / "config with spaces.yaml"
    profile_with_spaces = tmp_path / "workspace with spaces" / "outputs" / "notify" / "densegen" / "profile.json"
    runbook = runbook.model_copy(
        update={
            "densegen": runbook.densegen.model_copy(update={"config": config_with_spaces}),
            "notify": runbook.notify.model_copy(update={"profile": profile_with_spaces}),
        }
    )
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    assert plan.preflight_commands
    assert isinstance(plan.preflight_commands[0], CommandSpec)

    rendered = [spec.render_shell() for spec in plan.preflight_commands + plan.notify_smoke_commands]
    expected_config = shlex.quote(str(config_with_spaces))
    expected_profile = shlex.quote(str(profile_with_spaces))
    assert any(expected_config in command for command in rendered)
    assert any(expected_profile in command for command in rendered)


def test_batch_plan_enforces_workspace_scoped_stdout_dir_for_verify_and_submit(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)
    submit_block = _render_block(plan.submit_commands)

    expected_stdout_file = f"{runbook.workspace_root}/outputs/logs/ops/sge/{runbook.id}/$JOB_NAME.$JOB_ID.out"
    assert "dnadesign.ops.orchestrator.gates ensure-dir-writable" in preflight_block
    assert expected_stdout_file in preflight_block
    assert expected_stdout_file in submit_block
    assert "qsub -verify -P dunlop -o" in preflight_block
    assert "qsub -terse -P dunlop -o" in submit_block


def test_densegen_qsub_template_requires_explicit_mode_and_failure_messages() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    template_path = repo_root / "docs" / "bu-scc" / "jobs" / "densegen-cpu.qsub"
    template_text = template_path.read_text(encoding="utf-8")

    assert "DENSEGEN_RUN_ARGS must include exactly one of --fresh or --resume" in template_text
    assert "dense validate-config failed" in template_text
    assert "dense run failed" in template_text


def test_infer_runbook_uses_gpu_submit_template_and_filters(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    config_path.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_evo2_demo",
            "workflow_id": "infer_batch_with_notify_slack",
            "project": "dunlop",
            "workspace_root": str(workspace_root),
            "logging": {
                "stdout_dir": str(workspace_root / "outputs" / "logs" / "ops" / "sge" / "infer_evo2_demo"),
            },
            "infer": {
                "config": str(config_path),
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "notify": {
                "tool": "infer",
                "policy": "infer",
                "profile": str(workspace_root / "outputs/notify/infer/profile.json"),
                "cursor": str(workspace_root / "outputs/notify/infer/cursor"),
                "spool_dir": str(workspace_root / "outputs/notify/infer/spool"),
                "webhook_env": "NOTIFY_WEBHOOK",
                "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
            "mode_policy": {
                "default": "fresh",
                "on_active_job": "hold_jid",
            },
        }
    }
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)
    submit_block = _render_block(plan.submit_commands)
    smoke_block = _render_block(plan.notify_smoke_commands)

    assert "dnadesign.ops.orchestrator.gates usr-overlay-guard" in preflight_block
    assert "--tool infer " in preflight_block
    assert "--tool infer_evo2" not in preflight_block
    assert "evo2-gpu-infer.qsub" in submit_block
    assert "gpus=1" in submit_block
    assert "gpu_c=8.9" in submit_block
    assert "NOTIFY_PROFILE" in submit_block
    assert "notify profile smoke --profile" in smoke_block
    assert "--tool infer " in smoke_block
    assert "setup resolve-events --tool infer --config" not in smoke_block
    assert "--only-tools infer" in smoke_block


def test_infer_workflow_rejects_notify_tool_mismatch() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_evo2_demo",
            "workflow_id": "infer_batch_with_notify_slack",
            "project": "dunlop",
            "workspace_root": "/tmp/workspace",
            "logging": {
                "stdout_dir": "/tmp/workspace/outputs/logs/ops/sge/infer_evo2_demo",
            },
            "infer": {
                "config": "/tmp/workspace/config.yaml",
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "notify": {
                "tool": "densegen",
                "policy": "infer",
                "profile": "/tmp/workspace/outputs/notify/infer/profile.json",
                "cursor": "/tmp/workspace/outputs/notify/infer/cursor",
                "spool_dir": "/tmp/workspace/outputs/notify/infer/spool",
                "webhook_env": "NOTIFY_WEBHOOK",
                "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
        }
    }
    with pytest.raises(ValueError, match="infer workflow requires notify.tool=infer"):
        load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)


def test_infer_batch_submit_without_notify_skips_notify_phase(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_batch_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    config_path.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_evo2_batch",
            "workflow_id": "infer_batch_submit",
            "project": "dunlop",
            "workspace_root": str(workspace_root),
            "logging": {
                "stdout_dir": str(workspace_root / "outputs" / "logs" / "ops" / "sge" / "infer_evo2_batch"),
            },
            "infer": {
                "config": str(config_path),
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
            "mode_policy": {
                "default": "fresh",
                "on_active_job": "hold_jid",
            },
        }
    }
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    submit_block = _render_block(plan.submit_commands)

    assert runbook.notify is None
    assert plan.notify_smoke_commands == []
    assert "NOTIFY_PROFILE" not in submit_block
    assert "INFER_CONFIG=" in submit_block


def test_infer_mode_auto_selects_fresh_when_only_usr_registry_exists(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_mode_registry_only"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_registry_only",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    marker = runbook.workspace_root / "outputs" / "usr_datasets" / "registry.yaml"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("version: 1\n", encoding="utf-8")

    decision = resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())
    assert decision.selected_mode == "fresh"
    assert decision.run_args == ""


def test_infer_mode_auto_selects_resume_when_infer_overlay_exists(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "infer_mode_overlay_exists"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_overlay_exists",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    overlay_path = runbook.workspace_root / "outputs" / "usr_datasets" / "demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(
        pyarrow.table(
            {
                "id": ["id-1"],
                "infer__evo2_7b__job_a__ll_mean": [1.0],
            }
        ),
        overlay_path,
    )

    decision = resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())
    assert decision.selected_mode == "resume"
    assert decision.run_args == ""


def test_infer_mode_auto_selects_resume_when_external_usr_overlay_exists(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "infer_mode_external_overlay"
    external_usr_root = tmp_path / "external_usr_root"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_external_overlay",
        mode_default="auto",
        usr_root=external_usr_root,
        usr_dataset="external_demo",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    overlay_path = external_usr_root / "external_demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(
        pyarrow.table(
            {
                "id": ["id-1"],
                "infer__evo2_7b__job_a__ll_mean": [1.0],
            }
        ),
        overlay_path,
    )

    decision = resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())
    assert decision.selected_mode == "resume"
    assert decision.run_args == ""


def test_infer_mode_resume_raises_without_resume_artifacts(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_mode_resume_missing_artifacts"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_resume_missing_artifacts",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    with pytest.raises(ValueError, match="resume mode blocked: workspace has no resume artifacts"):
        resolve_mode_decision(runbook=runbook, requested_mode="resume", active_job_ids=())


def test_infer_mode_auto_raises_when_infer_usr_destination_is_ambiguous(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_mode_ambiguous_destination"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_ambiguous_destination",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    runbook.infer.config.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: job_a
    operation: extract
    ingest:
      source: usr
      root: "__USR_ROOT_A__"
      dataset: "dataset_a"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
  - id: job_b
    operation: extract
    ingest:
      source: usr
      root: "__USR_ROOT_B__"
      dataset: "dataset_b"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
""".strip()
        .replace("__USR_ROOT_A__", str(tmp_path / "external_usr_a"))
        .replace("__USR_ROOT_B__", str(tmp_path / "external_usr_b"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="infer mode probe requires a single resolvable USR destination"):
        resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())


def test_infer_mode_fresh_requires_reset_ack_when_resume_artifacts_exist(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "infer_mode_fresh_reset_ack"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_fresh_reset_ack",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    overlay_path = runbook.workspace_root / "outputs" / "usr_datasets" / "demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(
        pyarrow.table(
            {
                "id": ["id-1"],
                "infer__evo2_7b__job_a__ll_mean": [1.0],
            }
        ),
        overlay_path,
    )

    with pytest.raises(ValueError, match="fresh mode blocked"):
        resolve_mode_decision(runbook=runbook, requested_mode="fresh", active_job_ids=())

    decision = resolve_mode_decision(
        runbook=runbook,
        requested_mode="fresh",
        active_job_ids=(),
        allow_fresh_reset=True,
    )
    assert decision.selected_mode == "fresh"


def test_mode_auto_blocks_for_infer_when_resume_policy_marks_workspace_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "infer_workspace"
    tables_root = workspace_root / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    records_path = tables_root / "records.parquet"
    pyarrow_parquet.write_table(pyarrow.table({"id": ["r1"], "sequence": ["ATGC"]}), records_path)

    config_path = workspace_root / "config.yaml"
    config_path.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_resume_policy_demo",
            "workflow_id": "infer_batch_submit",
            "project": "dunlop",
            "workspace_root": str(workspace_root),
            "logging": {
                "stdout_dir": str(workspace_root / "outputs" / "logs" / "ops" / "sge" / "infer_resume_policy_demo"),
            },
            "infer": {
                "config": str(config_path),
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
            "mode_policy": {
                "default": "auto",
                "on_active_job": "hold_jid",
            },
        }
    }
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    class _Policy:
        def __init__(self) -> None:
            self.tool = "infer"
            self.required_record_columns = ("infer__score",)
            self.orphan_artifact_markers = ()

    monkeypatch.setattr(
        orchestrator_state,
        "resolve_resume_readiness_policy",
        lambda tool: _Policy() if tool == "infer" else None,
        raising=False,
    )

    with pytest.raises(ValueError, match="auto mode blocked: resume artifacts exist but workspace is not resume-ready"):
        resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())


def test_infer_runbook_resource_contract_fails_for_40b_single_gpu(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_resource_guard"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    config_path.write_text(
        """
model:
  id: evo2_40b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_resource_guard",
            "workflow_id": "infer_batch_submit",
            "project": "dunlop",
            "workspace_root": str(workspace_root),
            "logging": {
                "stdout_dir": str(workspace_root / "outputs" / "logs" / "ops" / "sge" / "infer_resource_guard"),
            },
            "infer": {
                "config": str(config_path),
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
        }
    }
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    with pytest.raises(ValueError, match="infer runbook resources are incompatible with infer model contract"):
        build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())


def test_densegen_workflow_rejects_gpu_fields() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "study_stress_ethanol_cipro",
            "workflow_id": "densegen_batch_with_notify_slack",
            "project": "dunlop",
            "workspace_root": "/tmp/workspace",
            "logging": {
                "stdout_dir": "/tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
            },
            "densegen": {
                "config": "/tmp/workspace/config.yaml",
                "qsub_template": "docs/bu-scc/jobs/densegen-cpu.qsub",
                "run_args": {
                    "fresh": "--fresh --no-plot",
                    "resume": "--resume --no-plot",
                },
            },
            "notify": {
                "tool": "densegen",
                "policy": "densegen",
                "profile": "/tmp/workspace/outputs/notify/densegen/profile.json",
                "cursor": "/tmp/workspace/outputs/notify/densegen/cursor",
                "spool_dir": "/tmp/workspace/outputs/notify/densegen/spool",
                "webhook_env": "NOTIFY_WEBHOOK",
                "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
            },
            "resources": {
                "pe_omp": 16,
                "h_rt": "08:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
        }
    }
    with pytest.raises(
        ValueError,
        match="densegen workflow does not accept resources.gpus, resources.gpu_capability, or resources.gpu_memory_gib",
    ):
        load_orchestration_runbook(Path("densegen-runbook.yaml"), raw=payload)


def test_densegen_run_args_rejects_fresh_mode_without_fresh_flag() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "study_stress_ethanol_cipro",
            "workflow_id": "densegen_batch_submit",
            "project": "dunlop",
            "workspace_root": "/tmp/workspace",
            "logging": {
                "stdout_dir": "/tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
            },
            "densegen": {
                "config": "/tmp/workspace/config.yaml",
                "qsub_template": "docs/bu-scc/jobs/densegen-cpu.qsub",
                "run_args": {
                    "fresh": "--no-plot",
                    "resume": "--resume --no-plot",
                },
            },
            "resources": {
                "pe_omp": 16,
                "h_rt": "08:00:00",
                "mem_per_core": "8G",
            },
            "mode_policy": {
                "default": "fresh",
                "on_active_job": "hold_jid",
            },
        }
    }

    with pytest.raises(ValueError, match="run args for fresh mode must include --fresh"):
        load_orchestration_runbook(Path("densegen-runbook.yaml"), raw=payload)


def test_execute_batch_plan_writes_audit_json(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    audit_path = tmp_path / "audit" / "result.json"

    seen_commands: list[str] = []

    def _runner(command: CommandSpec) -> tuple[int, str, str]:
        seen_commands.append(command.render_shell())
        return 0, "ok", ""

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=False,
        command_runner=_runner,
    )

    assert result.ok is True
    assert audit_path.exists()
    assert seen_commands
    assert all("qsub -terse" not in cmd for cmd in seen_commands)


def test_execute_batch_plan_emits_orchestration_started_and_success_notifications(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode="fresh", requested_smoke=None, active_job_ids=())
    audit_path = tmp_path / "audit" / "notify-success.json"
    seen_commands: list[str] = []

    def _runner(command: CommandSpec) -> tuple[int, str, str]:
        rendered = command.render_shell()
        seen_commands.append(rendered)
        if "qsub -terse" in rendered:
            return 0, "3442001\n", ""
        return 0, "ok", ""

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=True,
        command_runner=_runner,
    )

    assert result.ok is True
    notify_commands = [command for command in seen_commands if "uv run notify send" in command]
    assert any("--status started" in command for command in notify_commands)
    assert any("--status success" in command for command in notify_commands)
    assert all("--tls-ca-bundle" in command for command in notify_commands)


def test_execute_batch_plan_uses_orchestration_secret_ref_when_webhook_file_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    webhook_file = tmp_path / "notify_webhook.secret"
    webhook_file.write_text("https://example.invalid/webhook\n", encoding="utf-8")
    monkeypatch.setenv("NOTIFY_WEBHOOK_FILE", str(webhook_file.resolve()))

    plan = build_batch_plan(runbook=runbook, requested_mode="fresh", requested_smoke=None, active_job_ids=())
    assert plan.orchestration_notify is not None
    assert plan.orchestration_notify.secret_ref == webhook_file.resolve().as_uri()
    audit_path = tmp_path / "audit" / "notify-secret-ref-success.json"
    seen_commands: list[str] = []

    def _runner(command: CommandSpec) -> tuple[int, str, str]:
        rendered = command.render_shell()
        seen_commands.append(rendered)
        if "qsub -terse" in rendered:
            return 0, "3442001\n", ""
        return 0, "ok", ""

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=True,
        command_runner=_runner,
    )

    assert result.ok is True
    notify_commands = [command for command in seen_commands if "uv run notify send" in command]
    assert notify_commands
    assert all("--secret-ref" in command for command in notify_commands)
    assert all("--url-env" not in command for command in notify_commands)


def test_execute_batch_plan_emits_orchestration_failure_notification(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode="fresh", requested_smoke=None, active_job_ids=())
    audit_path = tmp_path / "audit" / "notify-failure.json"
    seen_commands: list[str] = []

    def _runner(command: CommandSpec) -> tuple[int, str, str]:
        rendered = command.render_shell()
        seen_commands.append(rendered)
        if "notify profile smoke" in rendered:
            return 2, "", "doctor failed"
        return 0, "ok", ""

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=True,
        command_runner=_runner,
    )

    assert result.ok is False
    notify_commands = [command for command in seen_commands if "uv run notify send" in command]
    assert any("--status failure" in command for command in notify_commands)


def test_execute_batch_plan_fails_fast_before_submits(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    audit_path = tmp_path / "audit" / "failed.json"
    seen_commands: list[str] = []

    def _runner(command: CommandSpec) -> tuple[int, str, str]:
        rendered = command.render_shell()
        seen_commands.append(rendered)
        if "notify profile smoke" in rendered:
            return 2, "", "doctor failed"
        return 0, "ok", ""

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=True,
        command_runner=_runner,
    )

    assert result.ok is False
    assert result.failed_phase == "notify_smoke"
    assert audit_path.exists()
    assert all("qsub -terse" not in cmd for cmd in seen_commands)


def test_cli_plan_invalid_runbook_shows_contract_error_without_traceback(tmp_path: Path) -> None:
    runbook_path = tmp_path / "invalid-runbook.yaml"
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_bad_notify",
            "workflow_id": "infer_batch_with_notify_slack",
            "project": "dunlop",
            "workspace_root": "/tmp/workspace",
            "logging": {
                "stdout_dir": "/tmp/workspace/outputs/logs/ops/sge/infer_bad_notify",
            },
            "infer": {
                "config": "/tmp/workspace/config.yaml",
                "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
                "cuda_module": "cuda/12.4",
                "gcc_module": "gcc/13.2.0",
            },
            "notify": {
                "tool": "densegen",
                "policy": "infer",
                "profile": "/tmp/workspace/outputs/notify/infer/profile.json",
                "cursor": "/tmp/workspace/outputs/notify/infer/cursor",
                "spool_dir": "/tmp/workspace/outputs/notify/infer/spool",
                "webhook_env": "NOTIFY_WEBHOOK",
                "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
            },
            "resources": {
                "pe_omp": 4,
                "h_rt": "04:00:00",
                "mem_per_core": "8G",
                "gpus": 1,
                "gpu_capability": "8.9",
            },
        }
    }
    runbook_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["runbook", "plan", "--runbook", str(runbook_path)])

    assert result.exit_code == 2
    assert "infer workflow requires notify.tool=infer" in result.output
    assert "Traceback" not in result.output


def test_execute_batch_plan_fails_when_command_times_out(tmp_path: Path) -> None:
    plan = BatchPlan(
        workflow_id="densegen_batch_with_notify_slack",
        project="dunlop",
        selected_mode="fresh",
        selected_smoke="dry",
        submit_behavior="submit",
        hold_jid=None,
        preflight_commands=[CommandSpec(shell="sleep 1")],
        notify_smoke_commands=[],
        submit_commands=[],
        orchestration_notify=None,
        decision_reason="timeout-test",
    )
    audit_path = tmp_path / "audit" / "timeout.json"

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=False,
        command_timeout_seconds=0.01,
    )

    assert result.ok is False
    assert result.failed_phase == "preflight"
    assert result.commands
    assert result.commands[0].returncode == 124
    assert "timed out" in result.commands[0].stderr


def test_execute_batch_plan_requires_secret_ref_for_orchestration_notify(tmp_path: Path) -> None:
    plan = BatchPlan(
        workflow_id="densegen_batch_with_notify_slack",
        project="dunlop",
        selected_mode="fresh",
        selected_smoke="dry",
        submit_behavior="submit",
        hold_jid=None,
        preflight_commands=[],
        notify_smoke_commands=[],
        submit_commands=[
            CommandSpec(argv=("qsub", "-terse", "docs/bu-scc/jobs/densegen-cpu.qsub")),
        ],
        orchestration_notify=OrchestrationNotifySpec(
            tool="densegen",
            provider="slack",
            webhook_env="NOTIFY_WEBHOOK",
            secret_ref="",
            run_id="study_stress_ethanol_cipro",
            tls_ca_bundle="/etc/ssl/certs/ca-certificates.crt",
        ),
        decision_reason="secret-ref-required-test",
    )
    audit_path = tmp_path / "audit" / "orchestration-secret-ref-required.json"

    with pytest.raises(ValueError, match="secret_ref is required"):
        execute_batch_plan(
            plan=plan,
            audit_json_path=audit_path,
            submit=True,
            command_runner=lambda command: (0, "ok", ""),
        )


def test_cli_execute_help_includes_command_timeout_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["runbook", "execute", "--help"])

    assert result.exit_code == 0
    assert "Per-command" in result.output
    assert "execute phases." in result.output


def test_cli_plan_accepts_repo_root_option(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runbook",
            "plan",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(Path.cwd()),
            "--max-discovery-jobs",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert "--max-discovery-jobs must be > 0" in result.output
    assert "No such option: --repo-root" not in result.output


def test_cli_active_jobs_accepts_repo_root_option(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runbook",
            "active-jobs",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(Path.cwd()),
            "--max-discovery-jobs",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert "--max-discovery-jobs must be > 0" in result.output
    assert "No such option: --repo-root" not in result.output


def test_cli_execute_accepts_repo_root_option(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    audit_path = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "audit" / "result.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runbook",
            "execute",
            "--runbook",
            str(runbook_path),
            "--audit-json",
            str(audit_path),
            "--repo-root",
            str(Path.cwd()),
            "--max-discovery-jobs",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert "--max-discovery-jobs must be > 0" in result.output
    assert "No such option: --repo-root" not in result.output


def test_cli_execute_defaults_timeout_to_300_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path)
    audit_path = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "audit" / "result.json"
    captured: dict[str, object] = {}

    class _Result:
        ok = True

        @staticmethod
        def as_dict() -> dict[str, object]:
            return {
                "ok": True,
                "failed_phase": None,
                "audit_json_path": str(audit_path),
                "commands": [],
            }

    def _fake_execute_batch_plan(*, plan, audit_json_path, submit, command_timeout_seconds):
        captured["workflow_id"] = plan.workflow_id
        captured["audit_json_path"] = audit_json_path
        captured["submit"] = submit
        captured["command_timeout_seconds"] = command_timeout_seconds
        return _Result()

    monkeypatch.setattr("dnadesign.ops.cli.execute_batch_plan", _fake_execute_batch_plan)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runbook",
            "execute",
            "--runbook",
            str(runbook_path),
            "--audit-json",
            str(audit_path),
            "--no-submit",
            "--no-discover-active-jobs",
        ],
    )

    assert result.exit_code == 0
    assert captured["workflow_id"] == "densegen_batch_with_notify_slack"
    assert captured["submit"] is False
    assert captured["command_timeout_seconds"] == 300.0


def test_cli_execute_rejects_audit_json_outside_workspace_ops_audit(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    outside_audit_path = tmp_path / "audit" / "outside.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runbook",
            "execute",
            "--runbook",
            str(runbook_path),
            "--audit-json",
            str(outside_audit_path),
            "--no-submit",
        ],
    )

    assert result.exit_code == 2
    assert "audit-json path must be exactly" in result.output
    assert "<workspace-root>/outputs/logs/ops/audit/<file>.json" in result.output


def test_cli_runbook_init_creates_valid_densegen_contract(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "densegen-runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--project",
            "dunlop",
            "--id",
            "densegen_demo",
        ],
    )

    assert result.exit_code == 0
    assert runbook_path.exists()
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.workflow_id == "densegen_batch_with_notify_slack"
    assert loaded.project == "dunlop"
    assert loaded.id == "densegen_demo"
    assert loaded.notify.smoke == "dry"
    assert loaded.densegen is not None
    assert loaded.resources.pe_omp == 12
    assert loaded.densegen.config == (workspace_root / "config.yaml").resolve()
    assert (
        loaded.logging.stdout_dir == (workspace_root / "outputs" / "logs" / "ops" / "sge" / "densegen_demo").resolve()
    )
    raw_payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    assert raw_payload["runbook"]["logging"]["retention"]["keep_last"] == 20
    assert raw_payload["runbook"]["logging"]["retention"]["max_age_days"] == 14
    assert raw_payload["runbook"]["densegen"]["overlay_guard"]["max_projected_overlay_parts"] == 10000
    assert raw_payload["runbook"]["densegen"]["overlay_guard"]["max_existing_overlay_parts"] == 1000
    assert raw_payload["runbook"]["densegen"]["overlay_guard"]["auto_compact_existing_overlay_parts"] is True
    assert raw_payload["runbook"]["densegen"]["overlay_guard"]["overlay_namespace"] == "densegen"
    assert raw_payload["runbook"]["densegen"]["records_part_guard"]["max_projected_records_parts"] == 10000
    assert raw_payload["runbook"]["densegen"]["records_part_guard"]["max_existing_records_parts"] == 1000
    assert raw_payload["runbook"]["densegen"]["records_part_guard"]["max_existing_records_part_age_days"] == 14
    assert raw_payload["runbook"]["densegen"]["records_part_guard"]["auto_compact_existing_records_parts"] is True
    assert raw_payload["runbook"]["densegen"]["archived_overlay_guard"]["max_archived_entries"] == 1000
    assert raw_payload["runbook"]["densegen"]["archived_overlay_guard"]["max_archived_bytes"] == 2147483648


def test_cli_runbook_init_supports_densegen_without_notify(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "densegen-runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--no-notify",
        ],
    )

    assert result.exit_code == 0
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.workflow_id == "densegen_batch_submit"
    assert loaded.notify is None


def test_cli_runbook_init_applies_resource_overrides(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "densegen-runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--h-rt",
            "02:00:00",
            "--pe-omp",
            "12",
            "--mem-per-core",
            "6G",
        ],
    )

    assert result.exit_code == 0
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.resources.h_rt == "02:00:00"
    assert loaded.resources.pe_omp == 12
    assert loaded.resources.mem_per_core == "6G"


def test_cli_runbook_init_uses_repo_root_for_template_contracts(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "densegen-runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    repo_root = tmp_path / "repo"
    (repo_root / "docs" / "bu-scc" / "jobs").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "bu-scc" / "jobs" / "notify-watch.qsub").write_text("#!/bin/bash -l\n", encoding="utf-8")
    (repo_root / "docs" / "bu-scc" / "jobs" / "densegen-cpu.qsub").write_text("#!/bin/bash -l\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--repo-root",
            str(repo_root),
        ],
    )

    assert result.exit_code == 0
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.notify.qsub_template == (repo_root / "docs" / "bu-scc" / "jobs" / "notify-watch.qsub").resolve()
    assert loaded.densegen is not None
    assert loaded.densegen.qsub_template == (repo_root / "docs" / "bu-scc" / "jobs" / "densegen-cpu.qsub").resolve()


def test_cli_runbook_init_resolves_relative_workspace_root_against_repo_root(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "densegen-runbook.yaml"
    repo_root = tmp_path / "repo"
    workspace_relative = Path("src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_relative),
            "--repo-root",
            str(repo_root),
        ],
    )

    assert result.exit_code == 0
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.workspace_root == (repo_root / workspace_relative).resolve()


def test_cli_runbook_init_rejects_repo_root_runbook_path(tmp_path: Path) -> None:
    runbook_path = tmp_path / "runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not be at repository root" in result.output


def test_cli_runbook_init_rejects_tmp_ops_runbook_path(tmp_path: Path) -> None:
    runbook_path = tmp_path / ".tmp_ops" / "runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use '.tmp_ops'" in result.output


def test_cli_runbook_init_rejects_tmp_ops_unhidden_runbook_path(tmp_path: Path) -> None:
    runbook_path = tmp_path / "tmp_ops" / "runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use 'tmp_ops'" in result.output


def test_cli_runbook_init_rejects_codex_tmp_runbook_path(tmp_path: Path) -> None:
    runbook_path = tmp_path / ".codex_tmp" / "runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "densegen",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_plan_rejects_codex_tmp_runbook_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path / ".codex_tmp")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "plan", "--runbook", str(runbook_path)])

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_active_jobs_rejects_codex_tmp_runbook_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path / ".codex_tmp")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "active-jobs", "--runbook", str(runbook_path)])

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_execute_rejects_codex_tmp_runbook_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path / ".codex_tmp")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "execute",
            "--runbook",
            str(runbook_path),
            "--audit-json",
            str(tmp_path / "audit.json"),
            "--no-submit",
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_plan_rejects_codex_tmp_runbook_path_when_cwd_is_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\nversion='0.0.0'\n", encoding="utf-8")
    runbook_path = _write_runbook(repo_root / ".codex_tmp")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_dir)
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "plan", "--runbook", str(runbook_path)])

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_plan_repo_root_override_enforces_path_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo_without_markers"
    runbook_path = _write_runbook(repo_root / ".codex_tmp")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_dir)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "plan",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(repo_root),
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_active_jobs_repo_root_override_enforces_path_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo_without_markers"
    runbook_path = _write_runbook(repo_root / ".codex_tmp")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_dir)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "active-jobs",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(repo_root),
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_runbook_execute_repo_root_override_enforces_path_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo_without_markers"
    runbook_path = _write_runbook(repo_root / ".codex_tmp")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_dir)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "execute",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(repo_root),
            "--audit-json",
            str(tmp_path / "audit.json"),
            "--no-submit",
        ],
    )

    assert result.exit_code == 2
    assert "runbook path must not use '.codex_tmp'" in result.output


def test_cli_plan_uses_discovered_active_job_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path)
    monkeypatch.setattr(
        "dnadesign.ops.cli.discover_active_job_ids_for_runbook",
        lambda runbook, max_jobs: ("93331",),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "plan", "--runbook", str(runbook_path), "--discover-active-jobs"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submit_behavior"] == "hold_jid"
    assert payload["hold_jid"] == "93331"


def test_cli_plan_chains_all_discovered_active_job_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path)
    monkeypatch.setattr(
        "dnadesign.ops.cli.discover_active_job_ids_for_runbook",
        lambda runbook, max_jobs: ("93332", "93331", "93332"),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "plan", "--runbook", str(runbook_path), "--discover-active-jobs"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submit_behavior"] == "hold_jid"
    assert payload["hold_jid"] == "93331,93332"


def test_cli_plan_accepts_comma_delimited_active_job_ids(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "plan",
            "--runbook",
            str(runbook_path),
            "--no-discover-active-jobs",
            "--active-job-id",
            "93332,93331",
            "--active-job-id",
            "93332",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submit_behavior"] == "hold_jid"
    assert payload["hold_jid"] == "93331,93332"


def test_cli_active_jobs_emits_discovered_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path)
    monkeypatch.setattr(
        "dnadesign.ops.cli.discover_active_job_ids_for_runbook",
        lambda runbook, max_jobs: ("95001", "95002"),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "active-jobs", "--runbook", str(runbook_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["active_job_ids"] == ["95001", "95002"]
    assert payload["active_job_count"] == 2
    assert payload["active_job_ids_csv"] == "95001,95002"
    assert payload["active_job_id_args"] == "--active-job-id 95001 --active-job-id 95002"
    assert "--no-discover-active-jobs --active-job-id 95001 --active-job-id 95002" in payload["plan_command_hint"]


def test_packaged_runbook_precedents_exist_and_load() -> None:
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    preset_dir = repo_root / "src" / "dnadesign" / "ops" / "runbooks" / "presets"

    preset_files = sorted(preset_dir.glob("*.yaml"))
    assert preset_files

    for preset_path in preset_files:
        loaded = load_orchestration_runbook(preset_path)
        assert loaded.workflow_id in {
            "densegen_batch_submit",
            "densegen_batch_with_notify_slack",
            "infer_batch_submit",
            "infer_batch_with_notify_slack",
        }


def test_cli_precedents_lists_packaged_runbooks() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["runbook", "precedents"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["precedents"]
    assert all(entry["path"].endswith(".yaml") for entry in payload["precedents"])
