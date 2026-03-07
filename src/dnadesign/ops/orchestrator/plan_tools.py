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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from dnadesign.infer import validate_runbook_gpu_resources

from ..runbooks.path_policy import WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR
from ..runbooks.schema import OrchestrationRunbookV1
from .state import ModeDecision
from .workflow_tools import (
    list_registered_workflow_tools,
    register_workflow_tool_adapter,
    resolve_workflow_tool_adapter_for_runbook,
    resolve_workflow_tool_adapter_for_workflow_id,
    validate_workflow_tool_registry,
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
        [OrchestrationRunbookV1, ModeDecision, str, tuple[str, ...]],
        tuple[ToolCommandSpec, ...],
    ]


def _tool_argv(*parts: object, env: dict[str, str] | None = None) -> ToolCommandSpec:
    return ToolCommandSpec(kind="argv", parts=tuple(str(part) for part in parts), env=env or {})


def _tool_ops_gate(*parts: object) -> ToolCommandSpec:
    return ToolCommandSpec(kind="ops_gate", parts=tuple(str(part) for part in parts))


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
        raise ValueError(
            "infer runbook resources are incompatible with infer model contract: "
            f"{exc}"
        ) from exc


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
    return (
        _tool_ops_gate(*infer_overlay_guard_parts),
        _tool_argv("uv", "run", "infer", "validate", "config", "--config", config),
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
            "-l",
            f"gpus={runbook.resources.gpus}",
            "-l",
            f"gpu_c={runbook.resources.gpu_capability}",
            "-v",
            f"CUDA_MODULE={runbook.infer.cuda_module},GCC_MODULE={runbook.infer.gcc_module}",
            infer_template,
        ),
        _tool_ops_gate("qa-submit-preflight", "--template", infer_template),
    )


def _densegen_submit_commands(
    runbook: OrchestrationRunbookV1,
    mode_decision: ModeDecision,
    stdout_file: str,
    hold_fragment: tuple[str, ...],
) -> tuple[ToolCommandSpec, ...]:
    if runbook.densegen is None:
        raise ValueError("densegen plan adapter requires runbook.densegen")
    post_run_pe_omp, post_run_h_rt, post_run_mem_per_core = _densegen_post_run_resource_values(runbook)
    runtime_trace_dir = (runbook.workspace_root / WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR).resolve()
    densegen_job_name = _sge_job_name(runbook_id=runbook.id, suffix="densegen_cpu")
    densegen_post_run_job_name = _sge_job_name(runbook_id=runbook.id, suffix="densegen_postrun")
    return (
        _tool_argv(
            "qsub",
            "-terse",
            "-P",
            runbook.project,
            *hold_fragment,
            "-N",
            densegen_job_name,
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
            "DENSEGEN_CONFIG="
            f"{runbook.densegen.config},DENSEGEN_RUN_ARGS={mode_decision.run_args},DENSEGEN_TRACE_DIR={runtime_trace_dir}",
            str(runbook.densegen.qsub_template),
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
            f"DENSEGEN_CONFIG={runbook.densegen.config}",
            str(runbook.densegen.post_run.qsub_template),
        ),
    )


def _infer_submit_commands(
    runbook: OrchestrationRunbookV1,
    _mode_decision: ModeDecision,
    stdout_file: str,
    hold_fragment: tuple[str, ...],
) -> tuple[ToolCommandSpec, ...]:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    return (
        _tool_argv(
            "qsub",
            "-terse",
            "-P",
            runbook.project,
            *hold_fragment,
            "-o",
            stdout_file,
            "-pe",
            "omp",
            str(runbook.resources.pe_omp),
            "-l",
            f"h_rt={runbook.resources.h_rt}",
            "-l",
            f"mem_per_core={runbook.resources.mem_per_core}",
            "-l",
            f"gpus={runbook.resources.gpus}",
            "-l",
            f"gpu_c={runbook.resources.gpu_capability}",
            "-v",
            "INFER_CONFIG="
            f"{runbook.infer.config},CUDA_MODULE={runbook.infer.cuda_module},GCC_MODULE={runbook.infer.gcc_module}",
            str(runbook.infer.qsub_template),
        ),
    )


def _sge_job_name(*, runbook_id: str, suffix: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(runbook_id or "").strip()).strip("_.-")
    suffix_token = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(suffix or "").strip()).strip("_.-")
    if not token:
        token = "runbook"
    if not suffix_token:
        suffix_token = "job"
    return f"{token}_{suffix_token}"[:128]


_PLAN_TOOL_ADAPTERS: dict[str, PlanToolAdapter] = {}


def register_plan_tool_adapter(tool: str, adapter: PlanToolAdapter) -> None:
    register_workflow_tool_adapter(
        _PLAN_TOOL_ADAPTERS,
        contract_name="plan tool adapter",
        tool=tool,
        adapter=adapter,
    )


def list_registered_plan_tools() -> tuple[str, ...]:
    return list_registered_workflow_tools(_PLAN_TOOL_ADAPTERS)


def _validate_plan_tool_registry() -> None:
    validate_workflow_tool_registry(_PLAN_TOOL_ADAPTERS, contract_name="plan tool adapter")


register_plan_tool_adapter(
    "densegen",
    PlanToolAdapter(
        tool="densegen",
        validate_resources=_validate_densegen_resources,
        notify_config_path=_densegen_notify_config_path,
        build_preflight_commands=_densegen_preflight_commands,
        build_submit_commands=_densegen_submit_commands,
    ),
)
register_plan_tool_adapter(
    "infer",
    PlanToolAdapter(
        tool="infer",
        validate_resources=_validate_infer_resources,
        notify_config_path=_infer_notify_config_path,
        build_preflight_commands=_infer_preflight_commands,
        build_submit_commands=_infer_submit_commands,
    ),
)
_validate_plan_tool_registry()


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
