"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/plan_tools.py

Plan-tool adapter contracts for workflow-specific preflight, submit, and notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import yaml

from dnadesign.infer import validate_runbook_gpu_resources

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


def _infer_notify_config_path(runbook: OrchestrationRunbookV1) -> Path:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    return Path(runbook.infer.config)


def _infer_preflight_commands(
    runbook: OrchestrationRunbookV1,
    mode_decision: ModeDecision,
    stdout_file: str,
) -> tuple[ToolCommandSpec, ...]:
    if runbook.infer is None:
        raise ValueError("infer plan adapter requires runbook.infer")
    config = str(runbook.infer.config)
    infer_template = str(runbook.infer.qsub_template)
    uses_sequence_views = _infer_config_uses_sequence_view_inputs(Path(runbook.infer.config))
    infer_env: dict[str, str] = {"INFER_CONFIG": config}
    if mode_decision.run_args:
        infer_env["INFER_RUN_ARGS"] = mode_decision.run_args
    infer_env["CUDA_MODULE"] = runbook.infer.cuda_module
    infer_env["GCC_MODULE"] = runbook.infer.gcc_module

    commands: list[ToolCommandSpec] = []
    if uses_sequence_views:
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
                "--max-stale-scalars",
                "0",
            )
        )
    else:
        infer_overlay_guard = runbook.infer.overlay_guard
        infer_overlay_guard_parts: list[str] = [
            "seqledger-overlay-guard",
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
        commands.append(_tool_ops_gate(*infer_overlay_guard_parts))
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
