"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_runbook_orchestrator.py

Contract tests for ops runbook loading, mode selection, and plan rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
import yaml
from typer.testing import CliRunner

import dnadesign.ops.orchestrator.state as orchestrator_state
import dnadesign.ops.runbooks.schema as runbook_schema
from dnadesign.ops.cli import app
from dnadesign.ops.orchestrator.execute import execute_batch_plan
from dnadesign.ops.orchestrator.infer_fill import build_infer_fill_plan, execute_infer_fill_plan
from dnadesign.ops.orchestrator.mode_tools import resolve_mode_tool_adapter_for_workflow_id
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


def test_list_workflow_tools_matches_schema_workflow_ids() -> None:
    workflow_ids = get_args(runbook_schema.OrchestrationRunbookV1.model_fields["workflow_id"].annotation)
    resolved_tools = tuple(sorted({runbook_schema.resolve_workflow_tool(workflow_id) for workflow_id in workflow_ids}))
    assert runbook_schema.list_workflow_tools() == resolved_tools


def test_mode_tool_adapters_cover_all_schema_workflow_ids() -> None:
    workflow_ids = get_args(runbook_schema.OrchestrationRunbookV1.model_fields["workflow_id"].annotation)
    assert workflow_ids
    for workflow_id in workflow_ids:
        adapter = resolve_mode_tool_adapter_for_workflow_id(workflow_id)
        assert adapter.tool in {"densegen", "infer"}


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


def _exported_env_names(command: CommandSpec) -> tuple[str, ...]:
    assert command.argv is not None
    return tuple(command.argv[command.argv.index("-v") + 1].split(","))


def _write_runbook(
    tmp_path: Path,
    *,
    include_smoke: bool = True,
    include_notify: bool = True,
    usr_root: Path | None = None,
    usr_dataset: str = "densegen_prom_eth_cip_source",
) -> Path:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    notify_root = workspace_root / "outputs" / "notify" / "densegen"
    if include_notify:
        notify_root.mkdir(parents=True, exist_ok=True)
    selected_usr_root = usr_root or (workspace_root / "outputs" / "usr_datasets")
    (workspace_root / "config.yaml").write_text(
        f"""
densegen:
  run:
    id: demo
    root: .
  output:
    targets: [usr]
    usr:
      root: "{selected_usr_root}"
      dataset: "{usr_dataset}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runbook_payload: dict[str, object] = {
        "schema_version": 1,
        "id": "study_stress_ethanol_cipro",
        "workflow_id": ("densegen_batch_with_notify" if include_notify else "densegen_batch_submit"),
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


def _write_sequence_view_infer_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: sequence_view_job
    operation: extract
    ingest:
      source: records
      field: sequence
    feature_bundle:
      intermediate_block: 26
      collect_log_likelihood: true
      collect_output_layer_mean: true
      collect_intermediate_embedding: true
      sequence_view_inputs:
        - dataset: demo_sequence_views
          root: ../../usr/datasets
          view_selector:
            product_kind: source_record
          pooling:
            operation: seq_mean
    io:
      write_back: false
      overwrite: false
""".strip()
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _set_notify_webhook_file_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook_file = tmp_path / "notify_webhook.secret"
    webhook_file.write_text("https://hooks.slack.com/services/T000/B000/TEST\n", encoding="utf-8")
    monkeypatch.setenv("NOTIFY_WEBHOOK_FILE", str(webhook_file.resolve()))
    ca_bundle = tmp_path / "ca-bundle.pem"
    ca_bundle.write_text("test-ca\n", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_bundle.resolve()))
    monkeypatch.delenv("NOTIFY_WEBHOOK", raising=False)


def test_runbook_notify_smoke_defaults_to_dry(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path, include_smoke=False)
    runbook = load_orchestration_runbook(runbook_path)
    assert runbook.notify.smoke == "dry"


def test_runbook_path_resolution_module_is_available() -> None:
    from dnadesign.ops.runbooks import runbook_paths

    assert callable(runbook_paths.resolve_runbook_paths)


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
            "workflow_id": "densegen_batch_with_notify",
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


def test_runbook_default_densegen_and_notify_templates_resolve_to_repo_jobs_templates(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    del payload["runbook"]["densegen"]["qsub_template"]
    del payload["runbook"]["notify"]["qsub_template"]

    runbook = load_orchestration_runbook(runbook_path, raw=payload)
    assert runbook.densegen is not None

    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    assert runbook.densegen.qsub_template == (repo_root / "docs" / "bu-scc" / "jobs" / "densegen-cpu.qsub").resolve()
    assert runbook.notify.qsub_template == (repo_root / "docs" / "bu-scc" / "jobs" / "notify-watch.qsub").resolve()


def test_runbook_default_infer_template_resolves_to_repo_jobs_template(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_workspace"
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_default_template")
    del payload["runbook"]["infer"]["qsub_template"]

    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    assert runbook.infer is not None
    assert runbook.infer.qsub_template == (repo_root / "docs" / "bu-scc" / "jobs" / "evo2-gpu-infer.qsub").resolve()


def test_runbook_default_templates_fall_back_to_packaged_qsub_templates_when_repo_root_missing(tmp_path: Path) -> None:
    from dnadesign.ops.runbooks import runbook_paths

    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    del payload["runbook"]["densegen"]["qsub_template"]
    del payload["runbook"]["notify"]["qsub_template"]

    infer_workspace = tmp_path / "infer_workspace"
    infer_payload = _infer_runbook_payload(infer_workspace, runbook_id="infer_packaged_template")
    del infer_payload["runbook"]["infer"]["qsub_template"]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runbook_paths, "_resolve_repo_root_from_module", lambda: None)
    try:
        densegen_runbook = load_orchestration_runbook(runbook_path, raw=payload)
        infer_runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=infer_payload)
    finally:
        monkeypatch.undo()

    assert densegen_runbook.densegen is not None
    assert densegen_runbook.notify is not None
    assert densegen_runbook.densegen.qsub_template.name == "densegen-cpu.qsub"
    assert densegen_runbook.notify.qsub_template.name == "notify-watch.qsub"
    assert "runbooks/templates" in densegen_runbook.densegen.qsub_template.as_posix()
    assert "runbooks/templates" in densegen_runbook.notify.qsub_template.as_posix()
    assert infer_runbook.infer is not None
    assert infer_runbook.infer.qsub_template.name == "evo2-gpu-infer.qsub"
    assert "runbooks/templates" in infer_runbook.infer.qsub_template.as_posix()


def test_packaged_qsub_templates_match_repo_docs_templates() -> None:
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break

    template_names = (
        "densegen-cpu.qsub",
        "densegen-analysis.qsub",
        "evo2-gpu-infer.qsub",
        "notify-watch.qsub",
    )
    for template_name in template_names:
        docs_template = repo_root / "docs" / "bu-scc" / "jobs" / template_name
        packaged_template = repo_root / "src" / "dnadesign" / "ops" / "runbooks" / "templates" / template_name
        assert packaged_template.read_text(encoding="utf-8") == docs_template.read_text(encoding="utf-8")


def test_runbook_notify_policy_defaults_to_generic_when_omitted(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    del payload["runbook"]["notify"]["policy"]

    runbook = load_orchestration_runbook(runbook_path, raw=payload)

    assert runbook.notify is not None
    assert runbook.notify.policy == "generic"


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


def test_runbook_rejects_retired_workflow_id_with_replacement_hint(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    payload["runbook"]["workflow_id"] = "densegen_batch_with_notify_slack"

    with pytest.raises(
        ValueError,
        match=(
            "unsupported orchestration workflow id: densegen_batch_with_notify_slack "
            r"\(retired; use densegen_batch_with_notify; supported: "
            r"densegen_batch_submit, densegen_batch_with_notify, infer_batch_submit, "
            r"infer_batch_with_notify\)"
        ),
    ):
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


def test_infer_runbook_rejects_non_infer_overlay_namespace(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    infer_config = tmp_path / "infer_config.yaml"
    infer_config.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    payload["runbook"]["workflow_id"] = "infer_batch_submit"
    payload["runbook"]["densegen"] = None
    payload["runbook"]["infer"] = {
        "config": str(infer_config),
        "qsub_template": "docs/bu-scc/jobs/evo2-gpu-infer.qsub",
        "cuda_module": "cuda/12.4",
        "gcc_module": "gcc/13.2.0",
        "overlay_guard": {
            "max_projected_overlay_parts": 10000,
            "max_existing_overlay_parts": 1000,
            "auto_compact_existing_overlay_parts": True,
            "overlay_namespace": "custom_infer",
        },
    }
    payload["runbook"]["notify"] = None
    payload["runbook"]["resources"] = {
        "pe_omp": 4,
        "h_rt": "04:00:00",
        "mem_per_core": "8G",
        "gpus": 1,
        "gpu_capability": "8.9",
    }

    with pytest.raises(
        ValueError,
        match='infer\\.overlay_guard\\.overlay_namespace must be exactly "infer"',
    ):
        load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)


def test_mode_auto_selects_fresh_without_artifacts(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())
    assert decision.selected_mode == "fresh"
    assert decision.run_args == "--fresh --no-plot"


def test_mode_decision_raises_when_runbook_has_no_workload_blocks(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    invalid_runbook = runbook.model_copy(update={"densegen": None, "infer": None})

    with pytest.raises(ValueError, match="runbook workload contract must define exactly one tool block"):
        resolve_mode_decision(runbook=invalid_runbook, requested_mode=None, active_job_ids=())


def test_mode_decision_raises_when_runbook_has_multiple_workload_blocks(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    infer_config = tmp_path / "infer_config.yaml"
    infer_config.write_text(
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
    infer_workload = runbook_schema.InferWorkloadContract(
        config=infer_config,
        qsub_template=Path("docs/bu-scc/jobs/evo2-gpu-infer.qsub"),
        cuda_module="cuda/12.4",
        gcc_module="gcc/13.2.0",
    )
    invalid_runbook = runbook.model_copy(update={"infer": infer_workload})

    with pytest.raises(ValueError, match="runbook workload contract must define exactly one tool block"):
        resolve_mode_decision(runbook=invalid_runbook, requested_mode=None, active_job_ids=())


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


def test_mode_auto_selects_resume_when_attempt_artifacts_exist_with_external_usr_base_records(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    external_usr_root = tmp_path / "external_usr_root"
    runbook_path = _write_runbook(
        tmp_path,
        usr_root=external_usr_root,
        usr_dataset="densegen_prom_eth_cip_source",
    )
    runbook = load_orchestration_runbook(runbook_path)

    usr_records_path = external_usr_root / "densegen" / "study_stress_ethanol_cipro" / "records.parquet"
    usr_records_path.parent.mkdir(parents=True, exist_ok=True)
    used_tfbs_type = pyarrow.list_(
        pyarrow.struct(
            [
                pyarrow.field("part_kind", pyarrow.string()),
            ]
        )
    )
    used_tfbs_detail = pyarrow.array([[{"part_kind": "tfbs"}]], type=used_tfbs_type)
    base_records_table = pyarrow.table(
        {
            "id": ["r1"],
            "sequence": ["ATGCATGC"],
            "densegen__run_id": ["study_stress_ethanol_cipro"],
            "densegen__input_name": ["plan_pool__ethanol__sig35_f"],
            "densegen__plan": ["ethanol__sig35=f"],
            "densegen__used_tfbs_detail": used_tfbs_detail,
        }
    )
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


def test_mode_auto_blocks_submit_when_current_host_is_not_submit_host_even_with_override(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    decision = resolve_mode_decision(
        runbook=runbook,
        requested_mode=None,
        active_job_ids=(),
        runtime_visibility=orchestrator_state.RuntimeVisibility(
            scheduler_probe_state=orchestrator_state.SchedulerProbeState.HOST_DENIED,
            active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.UNKNOWN,
            degraded=True,
            degraded_reasons=('denied: host "scc1.bu.edu" is no submit host',),
        ),
        allow_unknown_active_jobs=True,
    )

    assert decision.submit_behavior == "blocked"
    assert decision.hold_jid is None
    assert "current_host_not_submit_host" in decision.reason
    assert "submission_override_allow_unknown_active_jobs=true" not in decision.reason


def test_build_batch_plan_forwards_allow_fresh_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    captured: dict[str, object] = {}

    def _fake_resolve_mode_decision(
        *,
        runbook,
        requested_mode,
        active_job_ids,
        runtime_visibility=None,
        allow_fresh_reset=False,
        allow_unknown_active_jobs=False,
    ):
        captured["requested_mode"] = requested_mode
        captured["active_job_ids"] = tuple(active_job_ids)
        captured["runtime_visibility"] = runtime_visibility
        captured["allow_fresh_reset"] = allow_fresh_reset
        captured["allow_unknown_active_jobs"] = allow_unknown_active_jobs
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


def test_infer_sequence_view_auto_mode_does_not_force_overwrite(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_sequence_view")
    runbook_path = tmp_path / "runbook.yaml"
    runbook_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    runbook = load_orchestration_runbook(runbook_path)

    decision = resolve_mode_decision(runbook=runbook, requested_mode=None, active_job_ids=())

    assert decision.selected_mode == "fresh"
    assert decision.run_args == ""


def test_infer_sequence_view_plan_uses_sidecar_guard_not_overlay_guard(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_sequence_view")
    runbook = load_orchestration_runbook(tmp_path / "runbook.yaml", raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)
    submit_block = _render_block(plan.submit_commands)

    assert "infer validate sequence-view-completion" in preflight_block
    assert "dnadesign.ops.orchestrator.gates usr-overlay-guard" not in preflight_block
    assert "INFER_RUN_ARGS" not in preflight_block
    assert "INFER_RUN_ARGS" not in submit_block


def test_infer_fill_discovers_study_runbooks_and_plans_missing_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dnadesign.ops.orchestrator.infer_fill as infer_fill

    repo_root = tmp_path
    study_dir = repo_root / "docs" / "studies" / "demo"
    (study_dir / "operations").mkdir(parents=True)
    workspace_root = repo_root / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    runbook_path = repo_root / "runbooks" / "infer.yaml"
    runbook_path.parent.mkdir(parents=True)
    runbook_path.write_text(
        yaml.safe_dump(_infer_runbook_payload(workspace_root, runbook_id="infer_sequence_view")),
        encoding="utf-8",
    )
    (study_dir / "operations" / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "execution_surfaces": {
                    "infer_sequence_views": {
                        "surface_type": "runbook",
                        "runbook_ref": "repo:runbooks/infer.yaml",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        infer_fill,
        "plan_sequence_view_feature_inventory_completion_from_config",
        lambda _config: (
            {
                "required_views": 2,
                "required_vectors": 4,
                "required_scalars": 4,
                "missing_products": 0,
                "missing_vectors": 3,
                "missing_scalars": 2,
                "stale_vectors": 0,
                "stale_scalars": 0,
                "shard_plan": {
                    "schema_version": "infer_feature_shard_ledger_v1",
                    "shard_size_views": 50000,
                    "shard_count": 1,
                    "pending_view_estimate": 2,
                    "pending_vector_keys": 3,
                    "pending_scalar_keys": 2,
                    "runtime_fingerprint_key": "fingerprint-test",
                    "ledger_relative_path": "_derived/infer/checkpoints/infer_sequence_view/ledger.json",
                    "commit_policy": "temp_validate_promote",
                    "resume_policy": "skip_committed_retry_failed",
                },
            },
        ),
    )
    monkeypatch.setattr(
        infer_fill,
        "resolve_active_job_resolution",
        lambda **_kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.SKIPPED,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.NO_MATCH,
                degraded=False,
            ),
        ),
    )
    monkeypatch.setattr(
        infer_fill,
        "build_batch_plan",
        lambda **_kwargs: SimpleNamespace(submit_commands=("notify", "infer"), as_dict=lambda: {"fake": True}),
    )

    fill_plan = build_infer_fill_plan(repo_root=repo_root, study_dir=study_dir)

    assert fill_plan.aggregate_submit_commands == 2
    assert len(fill_plan.lanes) == 1
    lane = fill_plan.lanes[0]
    assert lane.action == "run"
    assert lane.missing_vectors == 3
    assert lane.missing_scalars == 2
    assert lane.as_dict()["completion"][0]["shard_plan"]["shard_count"] == 1
    assert (
        lane.as_dict()["completion"][0]["shard_plan"]["ledger_relative_path"]
        == "_derived/infer/checkpoints/infer_sequence_view/ledger.json"
    )
    assert lane.audit_json_path == workspace_root / "outputs" / "logs" / "ops" / "audit" / (
        "infer_sequence_view.fill-infer.json"
    )


def test_infer_fill_blocks_missing_sequence_products_before_batch_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dnadesign.ops.orchestrator.infer_fill as infer_fill

    workspace_root = tmp_path / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    runbook_path = tmp_path / "infer.yaml"
    runbook_path.write_text(
        yaml.safe_dump(_infer_runbook_payload(workspace_root, runbook_id="infer_missing_products")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        infer_fill,
        "plan_sequence_view_feature_inventory_completion_from_config",
        lambda _config: (
            {
                "required_views": 2,
                "required_vectors": 4,
                "required_scalars": 4,
                "missing_products": 1,
                "missing_vectors": 3,
                "missing_scalars": 2,
                "stale_vectors": 0,
                "stale_scalars": 0,
            },
        ),
    )

    def _unexpected_batch_plan(**_kwargs):
        raise AssertionError("missing sequence products must block before batch planning")

    monkeypatch.setattr(infer_fill, "build_batch_plan", _unexpected_batch_plan)

    fill_plan = build_infer_fill_plan(repo_root=tmp_path, runbook_paths=(runbook_path,))

    assert fill_plan.aggregate_submit_commands == 0
    assert fill_plan.lanes[0].action == "blocked"
    assert "missing sequence products block submit" in fill_plan.lanes[0].reasons

    executed = execute_infer_fill_plan(fill_plan=fill_plan, submit=True)

    assert executed.ok is False
    assert executed.executed is False
    assert "infer_missing_products: lane blocked: missing sequence products block submit" in executed.errors


def test_infer_fill_plans_multi_shard_lanes_when_durable_shard_plan_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dnadesign.ops.orchestrator.infer_fill as infer_fill

    workspace_root = tmp_path / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    runbook_path = tmp_path / "infer.yaml"
    runbook_path.write_text(
        yaml.safe_dump(_infer_runbook_payload(workspace_root, runbook_id="infer_multi_shard")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        infer_fill,
        "plan_sequence_view_feature_inventory_completion_from_config",
        lambda _config: (
            {
                "required_views": 314558,
                "required_vectors": 629116,
                "required_scalars": 314558,
                "missing_products": 0,
                "missing_vectors": 629116,
                "missing_scalars": 314558,
                "stale_vectors": 0,
                "stale_scalars": 0,
                "shard_plan": {
                    "schema_version": "infer_feature_shard_ledger_v1",
                    "shard_size_views": 50000,
                    "shard_count": 7,
                    "pending_view_estimate": 314558,
                    "pending_vector_keys": 629116,
                    "pending_scalar_keys": 314558,
                    "runtime_fingerprint_key": "fingerprint-test",
                    "ledger_relative_path": "_derived/infer/checkpoints/context_forward/ledger.json",
                    "commit_policy": "temp_validate_promote",
                    "resume_policy": "skip_committed_retry_failed",
                },
            },
        ),
    )
    monkeypatch.setattr(
        infer_fill,
        "resolve_active_job_resolution",
        lambda **_kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.SKIPPED,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.NO_MATCH,
                degraded=False,
            ),
        ),
    )
    monkeypatch.setattr(
        infer_fill,
        "build_batch_plan",
        lambda **_kwargs: SimpleNamespace(submit_commands=("notify", "infer"), as_dict=lambda: {"fake": True}),
    )

    fill_plan = build_infer_fill_plan(repo_root=tmp_path, runbook_paths=(runbook_path,))

    assert fill_plan.aggregate_submit_commands == 2
    assert fill_plan.lanes[0].action == "run"
    assert "missing or stale vectors/scalars remain" in fill_plan.lanes[0].reasons


def test_infer_fill_plans_stale_sidecar_repair_when_durable_shard_plan_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dnadesign.ops.orchestrator.infer_fill as infer_fill

    workspace_root = tmp_path / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    runbook_path = tmp_path / "infer.yaml"
    runbook_path.write_text(
        yaml.safe_dump(_infer_runbook_payload(workspace_root, runbook_id="infer_stale_repair")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        infer_fill,
        "plan_sequence_view_feature_inventory_completion_from_config",
        lambda _config: (
            {
                "required_views": 10,
                "required_vectors": 20,
                "required_scalars": 20,
                "missing_products": 0,
                "missing_vectors": 0,
                "missing_scalars": 0,
                "stale_vectors": 20,
                "stale_scalars": 20,
                "shard_plan": {
                    "schema_version": "infer_feature_shard_ledger_v1",
                    "shard_size_views": 50000,
                    "shard_count": 1,
                    "pending_view_estimate": 10,
                    "pending_vector_keys": 20,
                    "pending_scalar_keys": 20,
                    "runtime_fingerprint_key": "fingerprint-test",
                    "ledger_relative_path": "_derived/infer/checkpoints/infer_stale_repair/ledger.json",
                    "commit_policy": "temp_validate_promote",
                    "resume_policy": "skip_committed_retry_failed",
                },
            },
        ),
    )
    monkeypatch.setattr(
        infer_fill,
        "resolve_active_job_resolution",
        lambda **_kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.SKIPPED,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.NO_MATCH,
                degraded=False,
            ),
        ),
    )
    monkeypatch.setattr(
        infer_fill,
        "build_batch_plan",
        lambda **_kwargs: SimpleNamespace(submit_commands=("notify", "infer"), as_dict=lambda: {"fake": True}),
    )

    fill_plan = build_infer_fill_plan(repo_root=tmp_path, runbook_paths=(runbook_path,))

    assert fill_plan.aggregate_submit_commands == 2
    assert fill_plan.lanes[0].action == "run"
    assert fill_plan.lanes[0].stale_vectors == 20
    assert fill_plan.lanes[0].stale_scalars == 20


def test_infer_fill_blocks_infer_runbooks_without_sequence_view_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dnadesign.ops.orchestrator.infer_fill as infer_fill

    workspace_root = tmp_path / "workspace"
    _write_sequence_view_infer_config(workspace_root / "config.yaml")
    runbook_path = tmp_path / "infer.yaml"
    runbook_path.write_text(
        yaml.safe_dump(_infer_runbook_payload(workspace_root, runbook_id="infer_legacy_features")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        infer_fill,
        "plan_sequence_view_feature_inventory_completion_from_config",
        lambda _config: (_ for _ in ()).throw(ValueError("No selected jobs use feature_bundle.sequence_view_inputs.")),
    )

    def _unexpected_batch_plan(**_kwargs):
        raise AssertionError("unsupported Infer runbooks should not reach batch planning")

    monkeypatch.setattr(infer_fill, "build_batch_plan", _unexpected_batch_plan)

    fill_plan = build_infer_fill_plan(repo_root=tmp_path, runbook_paths=(runbook_path,))

    assert fill_plan.aggregate_submit_commands == 0
    assert fill_plan.lanes[0].action == "blocked"
    assert fill_plan.lanes[0].reasons == (
        "unsupported Infer config: selected jobs must define feature_bundle.sequence_view_inputs",
    )


def test_discover_active_job_ids_matches_explicit_identity_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    identity = orchestrator_state.resolve_ops_job_identity(runbook)

    qstat_table = """
job-ID prior name user state submit/start at queue slots ja-task-ID
--------------------------------------------------------------------------------
81001 0.555 a b r 03/01/2026 queueA 16
81002 0.555 a b qw 03/01/2026 queueA 1
81003 0.555 a b qw 03/01/2026 queueA 1
81004 0.555 a b qw 03/01/2026 queueA 1
"""
    job_details = {
        "81001": (
            f"job_name: ops.{identity.job_name_slug}.densegen_cpu\n"
            f"context: ops_job_role=densegen_cpu,ops_run_group_id={identity.run_group_id},"
            f"ops_workspace_id={identity.workspace_id},ops_workflow_id={identity.workflow_id}\n"
        ),
        "81002": (
            f"job_name: ops.{identity.job_name_slug}.notify\n"
            f"env_list: OPS_JOB_ROLE=notify,OPS_RUN_GROUP_ID={identity.run_group_id},"
            f"OPS_WORKSPACE_ID={identity.workspace_id},OPS_WORKFLOW_ID={identity.workflow_id}\n"
        ),
        "81003": (
            f"job_name: ops.{identity.job_name_slug}.densegen_cpu\n"
            f"context: ops_job_role=densegen_cpu,ops_run_group_id=foreign1234,"
            f"ops_workspace_id={identity.workspace_id},ops_workflow_id={identity.workflow_id}\n"
        ),
        "81004": (
            "job_name: ops.looks_similar.densegen_cpu\n"
            "env_list: DENSEGEN_CONFIG=/tmp/other/config.yaml,NOTIFY_PROFILE=/tmp/other/profile.json\n"
        ),
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


def test_discover_active_job_ids_scans_full_qstat_listing_before_capping_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    identity = orchestrator_state.resolve_ops_job_identity(runbook)

    qstat_lines = [
        "job-ID prior name user state submit/start at queue slots ja-task-ID",
        "--------------------------------------------------------------------------------",
    ]
    job_details = {
        str(81000 + idx): "context: ops_run_group_id=foreign1234,ops_workspace_id=foreign5678,ops_workflow_id=other"
        for idx in range(1, 27)
    }
    job_details["81025"] = (
        f"job_name: ops.{identity.job_name_slug}.densegen_cpu\n"
        f"context: ops_job_role=densegen_cpu,ops_run_group_id={identity.run_group_id},"
        f"ops_workspace_id={identity.workspace_id},ops_workflow_id={identity.workflow_id}\n"
    )
    for idx in range(1, 27):
        qstat_lines.append(f"{81000 + idx} 0.555 a b qw 03/01/2026 queueA 1")
    qstat_table = "\n".join(qstat_lines)
    seen_job_probes: list[str] = []

    def _probe(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[:2] == ("qstat", "-u"):
            return 0, qstat_table, ""
        if argv[:2] == ("qstat", "-j"):
            seen_job_probes.append(str(argv[2]))
            return 0, job_details.get(argv[2], ""), ""
        raise AssertionError(f"Unexpected probe argv: {argv}")

    monkeypatch.setattr(orchestrator_state, "_run_probe", _probe)

    discovered = discover_active_job_ids_for_runbook(runbook, max_jobs=1)

    assert discovered == ("81025",)
    assert seen_job_probes[-1] == "81025"
    assert "81026" not in seen_job_probes


def test_batch_plan_submit_commands_include_explicit_job_identity_tags(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())
    submit_block = _render_block(plan.submit_commands)
    identity = plan.job_identity
    payload = plan.as_dict()

    assert plan.runbook_id == runbook.id
    assert plan.workspace_root == str(runbook.workspace_root)
    assert payload["job_identity"]["run_group_id"] == identity.run_group_id
    assert payload["job_identity"]["workspace_id"] == identity.workspace_id
    assert payload["job_identity"]["workflow_id"] == runbook.workflow_id
    assert f"OPS_RUN_GROUP_ID={identity.run_group_id}" in submit_block
    assert f"ops_run_group_id={identity.run_group_id}" in submit_block
    assert f"ops.{identity.job_name_slug}.notify" in submit_block
    assert f"ops.{identity.job_name_slug}.densegen_cpu" in submit_block


def test_discover_active_job_ids_raises_when_qstat_snapshot_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    monkeypatch.setattr(orchestrator_state, "_run_probe", lambda argv: (1, "", "qstat unavailable"))

    with pytest.raises(RuntimeError, match="qstat unavailable"):
        discover_active_job_ids_for_runbook(runbook, max_jobs=12)


def test_discover_active_job_ids_raises_clean_error_when_qstat_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    def _missing_qstat(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "qstat")

    monkeypatch.setattr(orchestrator_state.subprocess, "run", _missing_qstat)

    with pytest.raises(RuntimeError, match="qstat unavailable"):
        discover_active_job_ids_for_runbook(runbook, max_jobs=12)


def test_discover_active_job_ids_raises_clean_error_when_qstat_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    def _timeout_qstat(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0] if args else ["qstat"]), timeout=10.0)

    monkeypatch.setattr(orchestrator_state.subprocess, "run", _timeout_qstat)

    with pytest.raises(RuntimeError, match=r"qstat unavailable: timed out after 10 seconds"):
        discover_active_job_ids_for_runbook(runbook, max_jobs=12)


def test_probe_active_jobs_for_runbook_classifies_submit_host_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    monkeypatch.setattr(
        orchestrator_state,
        "_run_probe",
        lambda argv: (1, "", 'error: denied: host "scc1.bu.edu" is neither submit nor admin host'),
    )

    resolution = orchestrator_state.probe_active_jobs_for_runbook(runbook, max_jobs=12)

    assert resolution.discovered_job_ids == ()
    assert resolution.runtime_visibility.scheduler_probe_state == orchestrator_state.SchedulerProbeState.HOST_DENIED
    assert (
        resolution.runtime_visibility.active_job_resolution_state == orchestrator_state.ActiveJobResolutionState.UNKNOWN
    )
    assert resolution.runtime_visibility.degraded is True
    assert resolution.runtime_visibility.degraded_reasons == (
        'error: denied: host "scc1.bu.edu" is neither submit nor admin host',
    )


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
    monkeypatch.setattr("dnadesign.ops.orchestrator.orchestration_notify.DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES", ())

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
    densegen_job_name = orchestrator_state.render_sge_job_name(plan.job_identity, role="densegen_cpu")

    assert plan.workflow_id == "densegen_batch_submit"
    assert plan.notify_smoke_commands == []
    assert "NOTIFY_PROFILE" not in submit_block
    assert "DENSEGEN_CONFIG" in submit_block
    assert "DENSEGEN_RUN_ARGS='--fresh --no-plot'" in submit_block
    assert f"DENSEGEN_TRACE_DIR={runbook.workspace_root}/outputs/logs/ops/runtime" in submit_block
    assert "docs/bu-scc/jobs/densegen-analysis.qsub" in submit_block
    assert f"-hold_jid {densegen_job_name}" in submit_block
    assert "-v DENSEGEN_CONFIG,DENSEGEN_RUN_ARGS,DENSEGEN_TRACE_DIR" in submit_block
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
    densegen_job_name = orchestrator_state.render_sge_job_name(plan.job_identity, role="densegen_cpu")
    assert "-pe omp 1" in post_run_verify_shell
    assert "-l h_rt=00:20:00" in post_run_verify_shell
    assert "-l mem_per_core=2G" in post_run_verify_shell
    assert "-pe omp 1" in post_run_submit_shell
    assert "-l h_rt=00:20:00" in post_run_submit_shell
    assert "-l mem_per_core=2G" in post_run_submit_shell
    assert f"-hold_jid {densegen_job_name}" in post_run_submit_shell


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
    densegen_job_name = orchestrator_state.render_sge_job_name(plan.job_identity, role="densegen_cpu")
    assert "-pe omp 4" in post_run_submit_shell
    assert "-l h_rt=01:00:00" in post_run_submit_shell
    assert "-l mem_per_core=4G" in post_run_submit_shell
    assert f"-hold_jid {densegen_job_name}" in post_run_submit_shell


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


def test_densegen_and_notify_qsub_commands_export_comma_bearing_values_via_env(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path / "study,2026")
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    densegen_submit = next(
        command
        for command in plan.submit_commands
        if command.argv is not None and command.argv[-1].endswith("densegen-cpu.qsub")
    )
    notify_submit = next(
        command
        for command in plan.submit_commands
        if command.argv is not None and command.argv[-1].endswith("notify-watch.qsub")
    )

    assert densegen_submit.argv is not None
    densegen_export_names = _exported_env_names(densegen_submit)
    assert densegen_export_names == tuple(densegen_submit.env)
    assert densegen_export_names[:3] == ("DENSEGEN_CONFIG", "DENSEGEN_RUN_ARGS", "DENSEGEN_TRACE_DIR")
    assert "OPS_RUN_GROUP_ID" in densegen_export_names
    assert "OPS_WORKSPACE_ID" in densegen_export_names
    assert "," in densegen_submit.env["DENSEGEN_CONFIG"]
    assert f"-v {','.join(densegen_export_names)}" in densegen_submit.render_shell()

    assert notify_submit.argv is not None
    notify_export_names = _exported_env_names(notify_submit)
    assert notify_export_names == tuple(notify_submit.env)
    assert notify_export_names[:6] == (
        "NOTIFY_PROFILE",
        "WEBHOOK_ENV",
        "NOTIFY_IDLE_TIMEOUT_SECONDS",
        "NOTIFY_ENFORCE_TERMINAL_ON_IDLE",
        "NOTIFY_TLS_CA_BUNDLE",
        "WEBHOOK_FILE",
    )
    assert "OPS_RUN_GROUP_ID" in notify_export_names
    assert "OPS_WORKSPACE_ID" in notify_export_names
    assert "," in notify_submit.env["NOTIFY_PROFILE"]
    assert f"-v {','.join(notify_export_names)}" in notify_submit.render_shell()


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


def test_notify_submit_inherits_hold_jid_when_active_jobs_are_detected(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(
        runbook=runbook,
        requested_mode=None,
        requested_smoke=None,
        active_job_ids=("81234",),
    )

    notify_submit = plan.submit_commands[0].render_shell()
    assert "-hold_jid 81234" in notify_submit


def test_preflight_operator_brief_inherits_requires_order_when_active_jobs_are_detected(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)

    plan = build_batch_plan(
        runbook=runbook,
        requested_mode=None,
        requested_smoke=None,
        active_job_ids=("81234",),
    )

    preflight_block = _render_block(plan.preflight_commands)
    assert "ops runbook diagnostics submit-shape-advisor" in preflight_block
    assert "ops runbook diagnostics operator-brief" in preflight_block
    assert "--requires-order" in preflight_block


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
    assert "ops runbook diagnostics submit-shape-advisor" in preflight_block
    assert "ops runbook diagnostics operator-brief" in preflight_block


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

    assert "ops runbook diagnostics session-counts" in preflight_block
    assert 'qstat -u "$USER" | awk' not in preflight_block


def test_batch_plan_can_render_explicit_degraded_queue_probe_flags(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(
        runbook=runbook,
        requested_mode=None,
        requested_smoke=None,
        active_job_ids=(),
        allow_missing_qstat=True,
    )
    preflight_block = _render_block(plan.preflight_commands)

    assert "ops runbook diagnostics session-counts --allow-missing-qstat" in preflight_block
    assert "ops runbook diagnostics submit-shape-advisor --planned-submits" in preflight_block
    assert "--allow-missing-qstat" in preflight_block
    assert "ops runbook diagnostics operator-brief --planned-submits" in preflight_block


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
    assert "--tls-ca-bundle" in smoke_block


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
    assert "qsub -terse -P dunlop " in submit_block
    assert " -o " in submit_block


def test_densegen_qsub_template_requires_explicit_mode_and_failure_messages() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    template_path = repo_root / "docs" / "bu-scc" / "jobs" / "densegen-cpu.qsub"
    template_text = template_path.read_text(encoding="utf-8")

    assert "DENSEGEN_RUN_ARGS must include exactly one of --fresh or --resume" in template_text
    assert "dense validate-config failed" in template_text
    assert "dense run failed" in template_text


def test_infer_qsub_template_exports_usr_actor_tags() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    template_text = (repo_root / "docs" / "bu-scc" / "jobs" / "evo2-gpu-infer.qsub").read_text(encoding="utf-8")

    assert 'export USR_ACTOR_TOOL="${USR_ACTOR_TOOL:-infer}"' in template_text
    assert 'DEFAULT_RUN_ID="${OPS_JOB_NAME_SLUG:-$JOB_ID_VALUE}"' in template_text
    assert (
        'if [[ -n "$TASK_ID_VALUE" && "$TASK_ID_VALUE" != "undefined" && "$TASK_ID_VALUE" != "NONE" ]]; then'
        in template_text
    )
    assert 'export USR_ACTOR_RUN_ID="${USR_ACTOR_RUN_ID:-$DEFAULT_RUN_ID}"' in template_text


def test_infer_qsub_template_preserves_infer_exit_codes() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    template_text = (repo_root / "docs" / "bu-scc" / "jobs" / "evo2-gpu-infer.qsub").read_text(encoding="utf-8")

    assert 'echo "infer validate config failed (exit=$validate_rc)" >&2' in template_text
    assert 'exit "$validate_rc"' in template_text
    assert 'echo "infer run failed (exit=$infer_rc)" >&2' in template_text
    assert 'exit "$infer_rc"' in template_text


def test_densegen_analysis_template_requires_records_for_placement_map() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    template_text = (repo_root / "docs" / "bu-scc" / "jobs" / "densegen-analysis.qsub").read_text(encoding="utf-8")

    assert 'RECORDS_PARQUET="$TABLES_DIR/records.parquet"' in template_text
    assert "needs_records_artifact=0" in template_text
    assert "Missing records.parquet and records__part-*.parquet under: $TABLES_DIR" in template_text


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
            "workflow_id": "infer_batch_with_notify",
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
    assert "INFER_RUN_ARGS=--overwrite" in preflight_block
    assert "INFER_RUN_ARGS=--overwrite" in submit_block
    assert "NOTIFY_PROFILE" in submit_block
    assert "notify profile smoke --profile" in smoke_block
    assert "--tool infer " in smoke_block
    assert "setup resolve-events --tool infer --config" not in smoke_block
    assert "--only-tools infer" in smoke_block


def test_infer_preflight_uses_run_dry_run_contract(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_workspace"
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_preflight_dry_run")
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode="fresh", requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "uv run infer run --config" in preflight_block
    assert "--dry-run" in preflight_block
    assert "uv run infer validate config --config" not in preflight_block


def test_sequence_view_infer_runbook_preflight_gates_missing_products_not_missing_vectors(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.sequence_views.yaml"
    config_path.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: context_reverse_complement_anchor_mean_7b
    operation: extract
    ingest:
      source: records
      field: sequence
    feature_bundle:
      collect_log_likelihood: false
      collect_output_layer_mean: false
      collect_intermediate_embedding: true
      sequence_view_inputs:
        - dataset: construct_prom_eth_cip_context
          root: ../../../usr/datasets
          view_selector:
            product_kind: realized_context
            orientation: reverse_complement
          pooling:
            operation: anchor_mean
            bounds_from: sequence_view
""".strip()
        + "\n",
        encoding="utf-8",
    )
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_sequence_view_preflight")
    payload["runbook"]["infer"]["config"] = str(config_path)
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode="fresh", requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)

    assert "uv run infer validate sequence-view-completion --config" in preflight_block
    assert "--max-missing-products 0" in preflight_block
    assert "--max-stale-vectors 0" in preflight_block
    assert "--max-stale-scalars 0" in preflight_block
    assert "--max-missing-vectors" not in preflight_block
    assert "--max-missing-scalars" not in preflight_block
    assert "uv run infer run --config" in preflight_block
    assert "--dry-run" in preflight_block


def test_infer_runbook_plan_emits_exact_gpu_type_when_declared(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_workspace"
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_blackwell_pin")
    payload["runbook"]["resources"]["gpu_capability"] = "12.0"
    payload["runbook"]["resources"]["gpu_type"] = "RTXP6000"
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode="fresh", requested_smoke=None, active_job_ids=())
    preflight_block = _render_block(plan.preflight_commands)
    submit_block = _render_block(plan.submit_commands)

    assert "gpu_c=12.0" in preflight_block
    assert "gpu_t=RTXP6000" in preflight_block
    assert "gpu_c=12.0" in submit_block
    assert "gpu_t=RTXP6000" in submit_block


def test_infer_workflow_rejects_notify_tool_mismatch() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_evo2_demo",
            "workflow_id": "infer_batch_with_notify",
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
    assert "INFER_RUN_ARGS=--overwrite" in submit_block


def test_infer_qsub_commands_export_comma_bearing_values_via_env(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer,workspace"
    payload = _infer_runbook_payload(workspace_root, runbook_id="infer_qsub_comma_paths")
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    plan = build_batch_plan(runbook=runbook, requested_mode=None, requested_smoke=None, active_job_ids=())

    infer_verify = next(
        command
        for command in plan.preflight_commands
        if command.argv is not None
        and command.argv[:2] == ("qsub", "-verify")
        and command.argv[-1].endswith("evo2-gpu-infer.qsub")
    )
    infer_submit = next(
        command
        for command in plan.submit_commands
        if command.argv is not None and command.argv[-1].endswith("evo2-gpu-infer.qsub")
    )

    for command in (infer_verify, infer_submit):
        assert command.argv is not None
        export_names = _exported_env_names(command)
        assert export_names == tuple(command.env)
        assert export_names[0] == "INFER_CONFIG"
        assert export_names[-2:] == ("CUDA_MODULE", "GCC_MODULE")
        if command is infer_submit:
            assert "OPS_RUN_GROUP_ID" in export_names
            assert "OPS_WORKSPACE_ID" in export_names
        else:
            assert "OPS_RUN_GROUP_ID" not in export_names
            assert "OPS_WORKSPACE_ID" not in export_names
        assert "," in command.env["INFER_CONFIG"]
        assert f"-v {','.join(export_names)}" in command.render_shell()


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
    assert decision.run_args == "--overwrite"


def test_infer_mode_auto_selects_fresh_when_only_run_manifest_exists(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_mode_manifest_only"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_manifest_only",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    marker = runbook.workspace_root / "outputs" / "meta" / "run_manifest.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")

    decision = resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())
    assert decision.selected_mode == "fresh"
    assert decision.run_args == "--overwrite"


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

    with pytest.raises(ValueError, match="auto mode blocked: infer resume destination is ambiguous or incomplete"):
        resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())


def test_infer_mode_auto_blocks_when_usr_destination_is_ambiguous_even_with_stale_workspace_overlay(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "infer_mode_ambiguous_destination_stale_overlay"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_ambiguous_destination_stale_overlay",
        mode_default="auto",
    )
    runbook = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)
    stale_overlay = workspace_root / "outputs" / "usr_datasets" / "stale" / "_derived" / "infer.parquet"
    stale_overlay.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(
        pyarrow.table(
            {
                "id": ["id-1"],
                "infer__evo2_7b__job_a__ll_mean": [1.0],
            }
        ),
        stale_overlay,
    )
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

    with pytest.raises(ValueError, match="auto mode blocked: infer resume destination is ambiguous or incomplete"):
        resolve_mode_decision(runbook=runbook, requested_mode="auto", active_job_ids=())


def test_infer_mode_fresh_allows_explicit_multi_job_overwrite(tmp_path: Path) -> None:
    workspace_root = tmp_path / "infer_mode_explicit_fresh_multi_job"
    payload = _infer_runbook_payload(
        workspace_root,
        runbook_id="infer_mode_explicit_fresh_multi_job",
        mode_default="fresh",
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

    decision = resolve_mode_decision(runbook=runbook, requested_mode="fresh", active_job_ids=())
    assert decision.selected_mode == "fresh"
    assert decision.run_args == "--overwrite"


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
    assert decision.run_args == "--overwrite"


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
            "workflow_id": "densegen_batch_with_notify",
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
                "gpu_type": "RTXP6000",
            },
        }
    }
    with pytest.raises(
        ValueError,
        match=(
            "densegen workflow does not accept resources.gpus, "
            "resources.gpu_capability, resources.gpu_type, or resources.gpu_memory_gib"
        ),
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
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["plan"]["runbook_id"] == runbook.id
    assert payload["plan"]["workspace_root"] == str(runbook.workspace_root)
    assert payload["plan"]["job_identity"]["run_group_id"] == plan.job_identity.run_group_id


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


def test_execute_batch_plan_blocks_submit_when_plan_uses_allow_missing_qstat(tmp_path: Path) -> None:
    runbook_path = _write_runbook(tmp_path)
    runbook = load_orchestration_runbook(runbook_path)
    plan = build_batch_plan(
        runbook=runbook,
        requested_mode=None,
        requested_smoke=None,
        active_job_ids=(),
        allow_missing_qstat=True,
    )
    audit_path = tmp_path / "audit" / "allow-missing-qstat-blocked.json"
    seen_commands: list[str] = []

    def _runner(command: CommandSpec) -> tuple[int, str, str]:
        seen_commands.append(command.render_shell())
        return 0, "ok", ""

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=True,
        command_runner=_runner,
    )

    assert result.ok is False
    assert result.failed_phase == "preflight"
    assert any("--allow-missing-qstat" in command for command in seen_commands)
    assert all("qsub -terse" not in command for command in seen_commands)


def test_execute_batch_plan_captures_gate_stderr_for_nonzero_native_gate_command(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "gate-stderr.json"
    plan = BatchPlan(
        runbook_id="gate-stderr-test",
        workflow_id="densegen_batch_submit",
        project="dunlop",
        workspace_root=str(tmp_path / "workspace"),
        job_identity=orchestrator_state.OpsJobIdentity(
            workflow_id="densegen_batch_submit",
            run_group_id="gatestderr123456",
            workspace_id="workspace9876",
            job_name_slug="gate-stderr-test",
            runbook_id="gate-stderr-test",
        ),
        selected_mode="fresh",
        selected_smoke=None,
        submit_behavior="submit",
        hold_jid=None,
        preflight_commands=[
            CommandSpec(
                argv=(
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "dnadesign.ops.orchestrator.gates",
                    "qa-submit-preflight",
                    "--template",
                    "/tmp/does-not-exist.qsub",
                )
            )
        ],
        notify_smoke_commands=[],
        submit_commands=[],
        orchestration_notify=None,
        decision_reason="capture gate stderr",
    )

    result = execute_batch_plan(
        plan=plan,
        audit_json_path=audit_path,
        submit=False,
    )

    assert result.ok is False
    assert result.failed_phase == "preflight"
    assert result.commands
    assert result.commands[0].returncode == 2
    assert "template_missing=/tmp/does-not-exist.qsub" in result.commands[0].stderr


def test_cli_plan_invalid_runbook_shows_contract_error_without_traceback(tmp_path: Path) -> None:
    runbook_path = tmp_path / "invalid-runbook.yaml"
    payload = {
        "runbook": {
            "schema_version": 1,
            "id": "infer_bad_notify",
            "workflow_id": "infer_batch_with_notify",
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
        runbook_id="timeout-test",
        workflow_id="densegen_batch_with_notify",
        project="dunlop",
        workspace_root=str(tmp_path / "workspace"),
        job_identity=orchestrator_state.OpsJobIdentity(
            workflow_id="densegen_batch_with_notify",
            run_group_id="timeout1234567890",
            workspace_id="workspace1234",
            job_name_slug="timeout-test",
            runbook_id="timeout-test",
        ),
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
        runbook_id="secret-ref-required-test",
        workflow_id="densegen_batch_with_notify",
        project="dunlop",
        workspace_root=str(tmp_path / "workspace"),
        job_identity=orchestrator_state.OpsJobIdentity(
            workflow_id="densegen_batch_with_notify",
            run_group_id="secretref1234567",
            workspace_id="workspace5678",
            job_name_slug="secret-ref-test",
            runbook_id="secret-ref-required-test",
        ),
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

    monkeypatch.setattr("dnadesign.ops.api.execute_batch_plan", _fake_execute_batch_plan)

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
    assert captured["workflow_id"] == "densegen_batch_with_notify"
    assert captured["submit"] is False
    assert captured["command_timeout_seconds"] == 300.0


def test_cli_execute_forwards_allow_missing_qstat_to_plan_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    audit_path = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "audit" / "result.json"
    captured: dict[str, object] = {}

    class _Plan:
        workflow_id = "densegen_batch_with_notify"
        project = "demo"

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

    def _fake_build_batch_plan(
        *,
        runbook,
        requested_mode,
        requested_smoke,
        active_job_ids,
        runtime_visibility=None,
        allow_fresh_reset=False,
        allow_missing_qstat=False,
        allow_unknown_active_jobs=False,
    ):
        captured["allow_missing_qstat"] = allow_missing_qstat
        captured["runtime_visibility"] = runtime_visibility
        captured["allow_unknown_active_jobs"] = allow_unknown_active_jobs
        return _Plan()

    def _fake_execute_batch_plan(*, plan, audit_json_path, submit, command_timeout_seconds):
        return _Result()

    monkeypatch.setattr("dnadesign.ops.api.build_batch_plan", _fake_build_batch_plan)
    monkeypatch.setattr("dnadesign.ops.api.execute_batch_plan", _fake_execute_batch_plan)

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
            "--allow-missing-qstat",
        ],
    )

    assert result.exit_code == 0
    assert captured["allow_missing_qstat"] is True


def test_cli_execute_rejects_submit_with_allow_missing_qstat(tmp_path: Path) -> None:
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
            "--submit",
            "--no-discover-active-jobs",
            "--allow-missing-qstat",
        ],
    )

    assert result.exit_code == 2
    assert "Runbook contract error:" in result.output
    assert "--allow-missing-qstat" in result.output
    assert "--no-submit" in result.output


def test_cli_plan_blocks_submit_decision_when_active_job_visibility_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    monkeypatch.setattr(
        "dnadesign.ops.api.resolve_active_job_resolution",
        lambda **kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.UNAVAILABLE,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=("qstat unavailable",),
            ),
        ),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "plan", "--runbook", str(runbook_path), "--discover-active-jobs"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submit_behavior"] == "blocked"
    assert payload["runtime_visibility"]["scheduler_probe_state"] == "unavailable"
    assert payload["runtime_visibility"]["active_job_resolution_state"] == "unknown"
    assert payload["runtime_visibility"]["degraded"] is True


def test_cli_execute_blocks_submit_when_active_job_visibility_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    audit_path = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "audit" / "result.json"
    execute_called = False

    monkeypatch.setattr(
        "dnadesign.ops.api.resolve_active_job_resolution",
        lambda **kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.UNAVAILABLE,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=("qstat unavailable",),
            ),
        ),
    )

    def _fake_execute_batch_plan(*, plan, audit_json_path, submit, command_timeout_seconds):
        nonlocal execute_called
        execute_called = True
        raise AssertionError("execute_batch_plan should not be called when active-job visibility is unknown")

    monkeypatch.setattr("dnadesign.ops.api.execute_batch_plan", _fake_execute_batch_plan)
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
            "--submit",
        ],
    )

    assert result.exit_code == 2
    assert "active-job visibility is unavailable" in result.output
    assert execute_called is False


def test_cli_execute_blocks_submit_when_current_host_is_not_submit_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runbook_path = _write_runbook(tmp_path)
    audit_path = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "audit" / "result.json"
    execute_called = False

    monkeypatch.setattr(
        "dnadesign.ops.api.resolve_active_job_resolution",
        lambda **kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.HOST_DENIED,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=('error: denied: host "scc1.bu.edu" is neither submit nor admin host',),
            ),
        ),
    )

    def _fake_execute_batch_plan(*, plan, audit_json_path, submit, command_timeout_seconds):
        nonlocal execute_called
        execute_called = True
        raise AssertionError("execute_batch_plan should not be called when current host is not a submit host")

    monkeypatch.setattr("dnadesign.ops.api.execute_batch_plan", _fake_execute_batch_plan)
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
            "--submit",
            "--allow-unknown-active-jobs",
        ],
    )

    assert result.exit_code == 2
    assert "current host is not a submit host for SCC batch submission" in result.output
    assert execute_called is False


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
    assert loaded.workflow_id == "densegen_batch_with_notify"
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
    assert raw_payload["runbook"]["notify"]["policy"] == "generic"
    assert "Notify contract required before planning" in result.stderr
    assert "NOTIFY_WEBHOOK_FILE" in result.stderr
    assert str(workspace_root / "outputs" / "notify" / "densegen" / "profile.json") in result.stderr


def test_cli_runbook_init_accepts_named_preset_for_project_defaults(tmp_path: Path) -> None:
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
            "--preset",
            "bu-scc-dunlop",
            "--id",
            "densegen_demo",
        ],
    )

    assert result.exit_code == 0
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.project == "dunlop"


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
            "--project",
            "dunlop",
            "--no-notify",
        ],
    )

    assert result.exit_code == 0
    loaded = load_orchestration_runbook(runbook_path)
    assert loaded.workflow_id == "densegen_batch_submit"
    assert loaded.notify is None


def test_cli_runbook_init_generates_infer_notify_scaffold_with_infer_policy(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "infer-runbook.yaml"
    workspace_root = tmp_path / "workspace_infer"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "infer",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--project",
            "dunlop",
            "--id",
            "infer_demo",
        ],
    )

    assert result.exit_code == 0
    raw_payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    assert raw_payload["runbook"]["notify"]["tool"] == "infer"
    assert raw_payload["runbook"]["notify"]["policy"] == "infer"
    assert "Notify contract required before planning" in result.stderr
    assert "NOTIFY_WEBHOOK_FILE" in result.stderr
    assert str(workspace_root / "outputs" / "notify" / "infer" / "profile.json") in result.stderr


def test_cli_runbook_init_without_notify_emits_no_notify_contract_warning(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "infer-runbook.yaml"
    workspace_root = tmp_path / "workspace_infer"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "runbook",
            "init",
            "--workflow",
            "infer",
            "--runbook",
            str(runbook_path),
            "--workspace-root",
            str(workspace_root),
            "--project",
            "dunlop",
            "--id",
            "infer_demo",
            "--no-notify",
        ],
    )

    assert result.exit_code == 0
    assert "Notify contract required before planning" not in result.stderr


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
            "--project",
            "dunlop",
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
            "--project",
            "dunlop",
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
            "--project",
            "dunlop",
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
            "--project",
            "dunlop",
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
            "--project",
            "dunlop",
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
            "--project",
            "dunlop",
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
            "--project",
            "dunlop",
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
        "dnadesign.ops.api.resolve_active_job_resolution",
        lambda **kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=("93331",),
            effective_job_ids=("93331",),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.OK,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.MATCHED,
                degraded=False,
            ),
        ),
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
        "dnadesign.ops.api.resolve_active_job_resolution",
        lambda **kwargs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=("93332", "93331", "93332"),
            effective_job_ids=("93332", "93331", "93332"),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.OK,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.MULTIPLE_MATCHES,
                degraded=False,
            ),
        ),
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
        "dnadesign.ops.api.probe_active_jobs_for_runbook",
        lambda runbook, max_jobs: orchestrator_state.ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=("95001", "95002"),
            effective_job_ids=("95001", "95002"),
            runtime_visibility=orchestrator_state.RuntimeVisibility(
                scheduler_probe_state=orchestrator_state.SchedulerProbeState.OK,
                active_job_resolution_state=orchestrator_state.ActiveJobResolutionState.MULTIPLE_MATCHES,
                degraded=False,
            ),
        ),
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


def test_packaged_runbook_presets_exist_and_load() -> None:
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
            "densegen_batch_with_notify",
            "infer_batch_submit",
            "infer_batch_with_notify",
        }


def test_infer_runbook_allows_workspace_local_config_variants() -> None:
    workspace_root = Path("/tmp/infer_layout_alt_config")
    payload = _infer_runbook_payload(workspace_root)
    payload["runbook"]["infer"]["config"] = str(workspace_root / "config.anchor_only.evo2_7b.yaml")

    loaded = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    assert loaded.infer is not None
    assert loaded.infer.config.name == "config.anchor_only.evo2_7b.yaml"


def test_infer_runbook_allows_lane_scoped_notify_state() -> None:
    workspace_root = Path("/tmp/infer_layout_lane_notify")
    payload = _infer_runbook_payload(workspace_root)
    payload["runbook"]["workflow_id"] = "infer_batch_with_notify"
    payload["runbook"]["notify"] = {
        "tool": "infer",
        "policy": "infer",
        "profile": str(workspace_root / "outputs" / "notify" / "infer" / "anchor_only_7b" / "profile.json"),
        "cursor": str(workspace_root / "outputs" / "notify" / "infer" / "anchor_only_7b" / "cursor"),
        "spool_dir": str(workspace_root / "outputs" / "notify" / "infer" / "anchor_only_7b" / "spool"),
        "webhook_env": "NOTIFY_WEBHOOK",
        "qsub_template": "docs/bu-scc/jobs/notify-watch.qsub",
        "smoke": "dry",
    }

    loaded = load_orchestration_runbook(Path("infer-runbook.yaml"), raw=payload)

    assert loaded.notify is not None
    assert loaded.notify.profile.parent.name == "anchor_only_7b"


def test_densegen_packaged_presets_use_repo_default_qsub_tokens() -> None:
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    preset_dir = repo_root / "src" / "dnadesign" / "ops" / "runbooks" / "presets"

    expected_densegen = "docs/bu-scc/jobs/densegen-cpu.qsub"
    expected_post_run = "docs/bu-scc/jobs/densegen-analysis.qsub"
    expected_notify = "docs/bu-scc/jobs/notify-watch.qsub"

    batch_payload = yaml.safe_load(
        (preset_dir / "densegen_stress_ethanol_cipro_batch.yaml").read_text(encoding="utf-8")
    )
    assert batch_payload["runbook"]["densegen"]["qsub_template"] == expected_densegen
    assert batch_payload["runbook"]["densegen"]["post_run"]["qsub_template"] == expected_post_run

    notify_payload = yaml.safe_load(
        (preset_dir / "densegen_stress_ethanol_cipro_batch_with_notify.yaml").read_text(encoding="utf-8")
    )
    assert notify_payload["runbook"]["densegen"]["qsub_template"] == expected_densegen
    assert notify_payload["runbook"]["densegen"]["post_run"]["qsub_template"] == expected_post_run
    assert notify_payload["runbook"]["notify"]["qsub_template"] == expected_notify


def test_stress_ethanol_cipro_infer_presets_are_blackwell_pinned() -> None:
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    preset_dir = repo_root / "src" / "dnadesign" / "ops" / "runbooks" / "presets"

    for preset_name in (
        "infer_stress_ethanol_cipro_anchor_only_20b_batch_with_notify.yaml",
        "infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml",
        "infer_stress_ethanol_cipro_sequence_views_anchor_construct_insert_7b_batch_with_notify.yaml",
        "infer_stress_ethanol_cipro_sequence_views_context_forward_seq_and_anchor_mean_7b_batch_with_notify.yaml",
        (
            "infer_stress_ethanol_cipro_sequence_views_context_reverse_complement_seq_and_anchor_mean_7b"
            "_batch_with_notify.yaml"
        ),
        ("infer_stress_ethanol_cipro_sequence_views_reference_analysis_window_core60_7b_batch_with_notify.yaml"),
        (
            "infer_stress_ethanol_cipro_sequence_views_reference_context_forward_seq_and_anchor_mean_7b"
            "_batch_with_notify.yaml"
        ),
        (
            "infer_stress_ethanol_cipro_sequence_views_reference_context_reverse_complement_seq_and_anchor_mean_7b"
            "_batch_with_notify.yaml"
        ),
    ):
        payload = yaml.safe_load((preset_dir / preset_name).read_text(encoding="utf-8"))
        resources = payload["runbook"]["resources"]
        assert resources["gpu_capability"] == "12.0"
        assert resources["gpu_type"] == "RTXP6000"
        assert resources["gpu_memory_gib"] == 80.0


def test_stress_ethanol_cipro_infer_configs_match_pressure_test_matrix() -> None:
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    config_dir = repo_root / "src" / "dnadesign" / "infer" / "workspaces" / "study_stress_ethanol_cipro"
    expected = {
        "config.sequence_views.anchor_construct_insert.evo2_7b.yaml": 128,
        "config.sequence_views.context_forward_seq_and_anchor_mean.evo2_7b.yaml": 128,
        "config.sequence_views.context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml": 128,
        "config.sequence_views.reference_analysis_window_core60.evo2_7b.yaml": 128,
        "config.sequence_views.reference_context_forward_seq_and_anchor_mean.evo2_7b.yaml": 128,
        "config.sequence_views.reference_context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml": 128,
        "config.anchor_only.evo2_20b.yaml": 256,
        "config.anchor_plus_template.evo2_20b.yaml": 48,
    }

    for config_name, batch_size in expected.items():
        payload = yaml.safe_load((config_dir / config_name).read_text(encoding="utf-8"))
        assert payload["model"]["batch_size"] == batch_size


def test_stress_ethanol_cipro_anchor_plus_template_20b_preset_uses_24h_walltime() -> None:
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    preset_path = (
        repo_root
        / "src"
        / "dnadesign"
        / "ops"
        / "runbooks"
        / "presets"
        / "infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml"
    )
    payload = yaml.safe_load(preset_path.read_text(encoding="utf-8"))

    assert payload["runbook"]["resources"]["h_rt"] == "24:00:00"


def test_packaged_runbook_preset_path_is_rejected_without_repo_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset_path = tmp_path / "site-packages" / "dnadesign" / "ops" / "runbooks" / "presets" / "demo.yaml"
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text("runbook: {}\n", encoding="utf-8")
    monkeypatch.setattr(runbook_schema, "_resolve_repo_root_from_module", lambda: None)

    with pytest.raises(ValueError, match="starter assets only"):
        load_orchestration_runbook(preset_path)


def test_cli_presets_lists_packaged_runbooks() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["runbook", "presets"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["presets"]
    assert all(entry["path"].endswith(".yaml") for entry in payload["presets"])


def test_cli_precedents_command_is_not_supported() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["runbook", "precedents"])

    assert result.exit_code != 0
