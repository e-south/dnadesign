"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/plan_tools.py

Plan-tool adapter contracts for workflow-specific preflight, submit, and notify
planning behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import yaml

from dnadesign.infer import validate_runbook_gpu_resources

from ..runbooks.path_policy import WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR
from ..runbooks.schema import OrchestrationRunbookV1
from .state import (
    ModeDecision,
    OpsJobIdentity,
    build_ops_job_context,
    build_ops_job_env,
    render_sge_context_value,
    render_sge_job_name,
)
from .workflow_tools import (
    build_workflow_tool_registry,
    freeze_workflow_tool_registry,
    list_registered_workflow_tools,
    register_workflow_tool_adapter,
    resolve_workflow_tool_adapter_for_runbook,
    resolve_workflow_tool_adapter_for_workflow_id,
)

ToolCommandKind = Literal["argv", "ops_gate"]


@dataclass(frozen=True)
class ToolCommandSpec:
    kind: ToolCommandKind
    parts: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanToolAdapter:
    tool: str
    validate_resources: Callable[[OrchestrationRunbookV1], None]
    notify_config_path: Callable[[OrchestrationRunbookV1], Path]
    build_preflight_commands: Callable[[OrchestrationRunbookV1, ModeDecision, str], tuple[ToolCommandSpec, ...]]
    build_submit_commands: Callable[
        [OrchestrationRunbookV1, ModeDecision, OpsJobIdentity, str, tuple[str, ...]],
        tuple[ToolCommandSpec, ...],
    ]


def _tool_argv(*parts: object, env: dict[str, str] | None = None) -> ToolCommandSpec:
    return ToolCommandSpec(kind="argv", parts=tuple(str(part) for part in parts), env=env or {})


def _tool_ops_gate(*parts: object) -> ToolCommandSpec:
    return ToolCommandSpec(kind="ops_gate", parts=tuple(str(part) for part in parts))


def _qsub_export_names(env_vars: dict[str, str]) -> str:
    names = [str(name).strip() for name in env_vars.keys() if str(name).strip()]
    if not names:
        raise ValueError("qsub export list requires at least one environment variable")
    return ",".join(names)


def _infer_scheduler_resource_parts(runbook: OrchestrationRunbookV1) -> tuple[str, ...]:
    gpu_count = runbook.resources.gpus
    if gpu_count is None:
        raise ValueError("infer workflow requires resources.gpus")
    parts: list[str] = ["-l", f"gpus={gpu_count}"]
    if runbook.resources.gpu_capability is not None:
        parts.extend(("-l", f"gpu_c={runbook.resources.gpu_capability}"))
    if runbook.resources.gpu_type is not None:
        parts.extend(("-l", f"gpu_t={runbook.resources.gpu_type}"))
    return tuple(parts)


def _densegen_post_run_resource_values(runbook: OrchestrationRunbookV1) -> tuple[str, str, str]:
    if runbook.densegen is None:
        raise ValueError("densegen plan adapter requires runbook.densegen")
    post_run_resources = runbook.densegen.post_run.resources
    return (
        str(post_run_resources.pe_omp),
        post_run_resources.h_rt,
        post_run_resources.mem_per_core,
    )


def _validate_densegen_resources(_runbook: OrchestrationRunbookV1) -> None:
    return None


def _validate_infer_resources(runbook: OrchestrationRunbookV1) -> None:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    if runbook.resources.gpus is None:
        raise ValueError("infer workflow requires resources.gpus")
    try:
        validate_runbook_gpu_resources(
            config_path=Path(runbook.infer.config),
            declared_gpus=int(runbook.resources.gpus),
            gpu_capability=runbook.resources.gpu_capability,
            gpu_memory_gib=runbook.resources.gpu_memory_gib,
        )
    except ValueError as exc:
        raise ValueError(f"infer runbook resources are incompatible with infer model contract: {exc}") from exc


def _densegen_notify_config_path(runbook: OrchestrationRunbookV1) -> Path:
    if runbook.densegen is None:
        raise ValueError("densegen plan adapter requires runbook.densegen")
    return Path(runbook.densegen.config)


def _infer_notify_config_path(runbook: OrchestrationRunbookV1) -> Path:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    return Path(runbook.infer.config)


def _densegen_preflight_commands(
    runbook: OrchestrationRunbookV1,
    mode_decision: ModeDecision,
    stdout_file: str,
) -> tuple[ToolCommandSpec, ...]:
    if runbook.densegen is None:
        raise ValueError("densegen plan adapter requires runbook.densegen")
    config = str(runbook.densegen.config)
    densegen_template = str(runbook.densegen.qsub_template)
    densegen_post_run_template = str(runbook.densegen.post_run.qsub_template)
    post_run_pe_omp, post_run_h_rt, post_run_mem_per_core = _densegen_post_run_resource_values(runbook)

    commands: list[ToolCommandSpec] = []
    overlay_guard = runbook.densegen.overlay_guard
    overlay_guard_parts: list[str] = [
        "usr-overlay-guard",
        "--tool",
        "densegen",
        "--config",
        config,
        "--workspace-root",
        str(runbook.workspace_root),
        "--mode",
        mode_decision.selected_mode,
        "--run-args",
        mode_decision.run_args,
        "--max-projected-overlay-parts",
        str(overlay_guard.max_projected_overlay_parts),
        "--max-existing-overlay-parts",
        str(overlay_guard.max_existing_overlay_parts),
        "--overlay-namespace",
        overlay_guard.overlay_namespace,
        "--json",
    ]
    if overlay_guard.auto_compact_existing_overlay_parts:
        overlay_guard_parts.append("--auto-compact-existing-overlay-parts")
    commands.append(_tool_ops_gate(*overlay_guard_parts))

    records_part_guard = runbook.densegen.records_part_guard
    records_part_guard_parts: list[str] = [
        "usr-records-part-guard",
        "--tool",
        "densegen",
        "--config",
        config,
        "--workspace-root",
        str(runbook.workspace_root),
        "--mode",
        mode_decision.selected_mode,
        "--run-args",
        mode_decision.run_args,
        "--max-projected-records-parts",
        str(records_part_guard.max_projected_records_parts),
        "--max-existing-records-parts",
        str(records_part_guard.max_existing_records_parts),
        "--max-existing-records-part-age-days",
        str(records_part_guard.max_existing_records_part_age_days),
        "--json",
    ]
    if records_part_guard.auto_compact_existing_records_parts:
        records_part_guard_parts.append("--auto-compact-existing-records-parts")
    commands.append(_tool_ops_gate(*records_part_guard_parts))

    archived_overlay_guard = runbook.densegen.archived_overlay_guard
    commands.append(
        _tool_ops_gate(
            "usr-archived-overlay-guard",
            "--tool",
            "densegen",
            "--config",
            config,
            "--workspace-root",
            str(runbook.workspace_root),
            "--max-archived-entries",
            str(archived_overlay_guard.max_archived_entries),
            "--max-archived-bytes",
            str(archived_overlay_guard.max_archived_bytes),
            "--json",
        )
    )

    if runbook.notify is not None:
        commands.append(
            _tool_argv(
                "uv",
                "run",
                "dense",
                "inspect",
                "run",
                "--usr-events-path",
                "-c",
                config,
            )
        )

    gurobi_home = os.environ.get("GUROBI_HOME", "/share/pkg.7/gurobi/10.0.1/install")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    gurobi_ld_library_path = f"{gurobi_home}/lib"
    if ld_library_path:
        gurobi_ld_library_path = f"{gurobi_ld_library_path}:{ld_library_path}"
    solver_probe_env = {
        "GUROBI_HOME": gurobi_home,
        "GRB_LICENSE_FILE": os.environ.get("GRB_LICENSE_FILE", "/usr/local/gurobi/gurobi.lic"),
        "TOKENSERVER": os.environ.get("TOKENSERVER", "sccsvc.bu.edu"),
        "LD_LIBRARY_PATH": gurobi_ld_library_path,
    }
    commands.extend(
        [
            _tool_argv(
                "uv",
                "run",
                "dense",
                "validate-config",
                "--probe-solver",
                "-c",
                config,
                env=solver_probe_env,
            ),
            _tool_argv(
                "qsub",
                "-verify",
                "-P",
                runbook.project,
                "-o",
                stdout_file,
                "-pe",
                "omp",
                str(runbook.resources.pe_omp),
                "-l",
                f"h_rt={runbook.resources.h_rt}",
                "-l",
                f"mem_per_core={runbook.resources.mem_per_core}",
                "-v",
                f"DENSEGEN_CONFIG={config}",
                densegen_template,
            ),
            _tool_ops_gate("qa-submit-preflight", "--template", densegen_template),
            _tool_argv(
                "qsub",
                "-verify",
                "-P",
                runbook.project,
                "-o",
                stdout_file,
                "-pe",
                "omp",
                post_run_pe_omp,
                "-l",
                f"h_rt={post_run_h_rt}",
                "-l",
                f"mem_per_core={post_run_mem_per_core}",
                "-v",
                f"DENSEGEN_CONFIG={config}",
                densegen_post_run_template,
            ),
            _tool_ops_gate("qa-submit-preflight", "--template", densegen_post_run_template),
        ]
    )
    return tuple(commands)


def _infer_preflight_commands(
    runbook: OrchestrationRunbookV1,
    mode_decision: ModeDecision,
    stdout_file: str,
) -> tuple[ToolCommandSpec, ...]:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    config = str(runbook.infer.config)
    infer_template = str(runbook.infer.qsub_template)
    infer_overlay_guard = runbook.infer.overlay_guard
    infer_overlay_guard_parts: list[str] = [
        "usr-overlay-guard",
        "--tool",
        "infer",
        "--config",
        config,
        "--workspace-root",
        str(runbook.workspace_root),
        "--mode",
        mode_decision.selected_mode,
        "--run-args",
        mode_decision.run_args,
        "--max-projected-overlay-parts",
        str(infer_overlay_guard.max_projected_overlay_parts),
        "--max-existing-overlay-parts",
        str(infer_overlay_guard.max_existing_overlay_parts),
        "--overlay-namespace",
        infer_overlay_guard.overlay_namespace,
        "--json",
    ]
    if infer_overlay_guard.auto_compact_existing_overlay_parts:
        infer_overlay_guard_parts.append("--auto-compact-existing-overlay-parts")
    infer_env: dict[str, str] = {"INFER_CONFIG": config}
    if mode_decision.run_args:
        infer_env["INFER_RUN_ARGS"] = mode_decision.run_args
    infer_env["CUDA_MODULE"] = runbook.infer.cuda_module
    infer_env["GCC_MODULE"] = runbook.infer.gcc_module

    commands = [
        _tool_ops_gate(*infer_overlay_guard_parts),
    ]
    if _infer_config_uses_sequence_view_inputs(Path(runbook.infer.config)):
        commands.append(
            _tool_argv(
                "uv",
                "run",
                "infer",
                "validate",
                "sequence-view-completion",
                "--config",
                config,
                "--format",
                "json",
                "--max-missing-products",
                "0",
                "--max-stale-vectors",
                "0",
            )
        )
    commands.extend(
        (
            _tool_argv("uv", "run", "infer", "run", "--config", config, "--dry-run"),
            _tool_argv(
                "qsub",
                "-verify",
                "-P",
                runbook.project,
                "-o",
                stdout_file,
                "-pe",
                "omp",
                str(runbook.resources.pe_omp),
                "-l",
                f"h_rt={runbook.resources.h_rt}",
                "-l",
                f"mem_per_core={runbook.resources.mem_per_core}",
                *_infer_scheduler_resource_parts(runbook),
                "-v",
                _qsub_export_names(infer_env),
                infer_template,
                env=infer_env,
            ),
            _tool_ops_gate("qa-submit-preflight", "--template", infer_template),
        )
    )
    return tuple(commands)


def _infer_config_uses_sequence_view_inputs(config_path: Path) -> bool:
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"infer config is not readable: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"infer config is not valid yaml: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"infer config root must be a mapping: {config_path}")
    jobs = payload.get("jobs") or ()
    if not isinstance(jobs, list):
        return False
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        feature_bundle = job.get("feature_bundle")
        if not isinstance(feature_bundle, Mapping):
            continue
        sequence_view_inputs = feature_bundle.get("sequence_view_inputs")
        if isinstance(sequence_view_inputs, list) and sequence_view_inputs:
            return True
    return False


def _densegen_submit_commands(
    runbook: OrchestrationRunbookV1,
    mode_decision: ModeDecision,
    job_identity: OpsJobIdentity,
    stdout_file: str,
    hold_fragment: tuple[str, ...],
) -> tuple[ToolCommandSpec, ...]:
    if runbook.densegen is None:
        raise ValueError("densegen plan adapter requires runbook.densegen")
    post_run_pe_omp, post_run_h_rt, post_run_mem_per_core = _densegen_post_run_resource_values(runbook)
    runtime_trace_dir = (runbook.workspace_root / WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR).resolve()
    densegen_job_name = render_sge_job_name(job_identity, role="densegen_cpu")
    densegen_post_run_job_name = render_sge_job_name(job_identity, role="densegen_postrun")
    densegen_env = {
        "DENSEGEN_CONFIG": str(runbook.densegen.config),
        "DENSEGEN_RUN_ARGS": mode_decision.run_args,
        "DENSEGEN_TRACE_DIR": str(runtime_trace_dir),
        **build_ops_job_env(job_identity, role="densegen_cpu"),
    }
    densegen_post_run_env = {
        "DENSEGEN_CONFIG": str(runbook.densegen.config),
        **build_ops_job_env(job_identity, role="densegen_postrun"),
    }
    return (
        _tool_argv(
            "qsub",
            "-terse",
            "-P",
            runbook.project,
            *hold_fragment,
            "-N",
            densegen_job_name,
            "-ac",
            render_sge_context_value(build_ops_job_context(job_identity, role="densegen_cpu")),
            "-o",
            stdout_file,
            "-pe",
            "omp",
            str(runbook.resources.pe_omp),
            "-l",
            f"h_rt={runbook.resources.h_rt}",
            "-l",
            f"mem_per_core={runbook.resources.mem_per_core}",
            "-v",
            _qsub_export_names(densegen_env),
            str(runbook.densegen.qsub_template),
            env=densegen_env,
        ),
        _tool_argv(
            "qsub",
            "-terse",
            "-P",
            runbook.project,
            "-hold_jid",
            densegen_job_name,
            "-N",
            densegen_post_run_job_name,
            "-ac",
            render_sge_context_value(build_ops_job_context(job_identity, role="densegen_postrun")),
            "-o",
            stdout_file,
            "-pe",
            "omp",
            post_run_pe_omp,
            "-l",
            f"h_rt={post_run_h_rt}",
            "-l",
            f"mem_per_core={post_run_mem_per_core}",
            "-v",
            _qsub_export_names(densegen_post_run_env),
            str(runbook.densegen.post_run.qsub_template),
            env=densegen_post_run_env,
        ),
    )


def _infer_submit_commands(
    runbook: OrchestrationRunbookV1,
    mode_decision: ModeDecision,
    job_identity: OpsJobIdentity,
    stdout_file: str,
    hold_fragment: tuple[str, ...],
) -> tuple[ToolCommandSpec, ...]:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    infer_env: dict[str, str] = {
        "INFER_CONFIG": str(runbook.infer.config),
        **build_ops_job_env(job_identity, role="infer_gpu"),
    }
    if mode_decision.run_args:
        infer_env["INFER_RUN_ARGS"] = mode_decision.run_args
    infer_env["CUDA_MODULE"] = runbook.infer.cuda_module
    infer_env["GCC_MODULE"] = runbook.infer.gcc_module
    return (
        _tool_argv(
            "qsub",
            "-terse",
            "-P",
            runbook.project,
            *hold_fragment,
            "-N",
            render_sge_job_name(job_identity, role="infer_gpu"),
            "-ac",
            render_sge_context_value(build_ops_job_context(job_identity, role="infer_gpu")),
            "-o",
            stdout_file,
            "-pe",
            "omp",
            str(runbook.resources.pe_omp),
            "-l",
            f"h_rt={runbook.resources.h_rt}",
            "-l",
            f"mem_per_core={runbook.resources.mem_per_core}",
            *_infer_scheduler_resource_parts(runbook),
            "-v",
            _qsub_export_names(infer_env),
            str(runbook.infer.qsub_template),
            env=infer_env,
        ),
    )


_PLAN_TOOL_ADAPTERS = build_workflow_tool_registry(
    contract_name="plan tool adapter",
    adapters=(
        PlanToolAdapter(
            tool="densegen",
            validate_resources=_validate_densegen_resources,
            notify_config_path=_densegen_notify_config_path,
            build_preflight_commands=_densegen_preflight_commands,
            build_submit_commands=_densegen_submit_commands,
        ),
        PlanToolAdapter(
            tool="infer",
            validate_resources=_validate_infer_resources,
            notify_config_path=_infer_notify_config_path,
            build_preflight_commands=_infer_preflight_commands,
            build_submit_commands=_infer_submit_commands,
        ),
    ),
)


def register_plan_tool_adapter(tool: str, adapter: PlanToolAdapter) -> None:
    global _PLAN_TOOL_ADAPTERS
    updated = dict(_PLAN_TOOL_ADAPTERS)
    register_workflow_tool_adapter(
        updated,
        contract_name="plan tool adapter",
        tool=tool,
        adapter=adapter,
    )
    _PLAN_TOOL_ADAPTERS = freeze_workflow_tool_registry(updated, contract_name="plan tool adapter")


def list_registered_plan_tools() -> tuple[str, ...]:
    return list_registered_workflow_tools(_PLAN_TOOL_ADAPTERS)


def resolve_plan_tool_adapter_for_workflow_id(workflow_id: str) -> PlanToolAdapter:
    return resolve_workflow_tool_adapter_for_workflow_id(
        _PLAN_TOOL_ADAPTERS,
        contract_name="plan tool adapter",
        workflow_id=workflow_id,
    )


def resolve_plan_tool_adapter(runbook: OrchestrationRunbookV1) -> PlanToolAdapter:
    return resolve_workflow_tool_adapter_for_runbook(
        _PLAN_TOOL_ADAPTERS,
        contract_name="plan tool adapter",
        runbook=runbook,
    )
