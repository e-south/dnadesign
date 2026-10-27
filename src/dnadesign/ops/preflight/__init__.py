"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/__init__.py

Lazy public exports for generic Ops preflight contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "CommandCheckDependencies",
    "CommandCheckTarget",
    "CommandExecution",
    "ContractPreflightCheckDependencies",
    "EnvironmentCheckTarget",
    "PreflightCheck",
    "PreflightCheckEvaluation",
    "RunbookPlanCheckDependencies",
    "RunbookPlanCheckTarget",
    "SchedulerQueueCheckDependencies",
    "SchedulerQueueCheckTarget",
    "build_command_check",
    "build_command_checks",
    "build_contract_preflight_checks",
    "build_environment_checks",
    "build_infer_notify_setup_command",
    "build_runbook_plan_checks",
    "build_scheduler_queue_checks",
    "build_state_check",
    "check_matches_scope",
    "choose_command_summary",
    "contract_environment_flag_state",
    "execute_runbook_plan",
    "evaluate_preflight_checks",
    "infer_usr_dataset_requirements",
    "load_orchestration_runbook_payload",
    "render_argv",
    "run_preflight_command",
    "safe_json_loads",
    "supported_preflight_check_kinds",
]

_EXPORT_MODULES = {
    "CommandCheckDependencies": "dnadesign.ops.preflight.checks",
    "CommandCheckTarget": "dnadesign.ops.preflight.check_protocols",
    "CommandExecution": "dnadesign.ops.preflight.models",
    "ContractPreflightCheckDependencies": "dnadesign.ops.preflight.contract_checks",
    "EnvironmentCheckTarget": "dnadesign.ops.preflight.check_protocols",
    "PreflightCheck": "dnadesign.ops.preflight.models",
    "PreflightCheckEvaluation": "dnadesign.ops.preflight.coordinator",
    "RunbookPlanCheckDependencies": "dnadesign.ops.preflight.checks",
    "RunbookPlanCheckTarget": "dnadesign.ops.preflight.check_protocols",
    "SchedulerQueueCheckDependencies": "dnadesign.ops.preflight.checks",
    "SchedulerQueueCheckTarget": "dnadesign.ops.preflight.check_protocols",
    "build_command_check": "dnadesign.ops.preflight.models",
    "build_command_checks": "dnadesign.ops.preflight.checks",
    "build_contract_preflight_checks": "dnadesign.ops.preflight.contract_checks",
    "build_environment_checks": "dnadesign.ops.preflight.checks",
    "build_infer_notify_setup_command": "dnadesign.ops.preflight.support",
    "build_runbook_plan_checks": "dnadesign.ops.preflight.checks",
    "build_scheduler_queue_checks": "dnadesign.ops.preflight.checks",
    "build_state_check": "dnadesign.ops.preflight.models",
    "check_matches_scope": "dnadesign.ops.preflight.coordinator",
    "choose_command_summary": "dnadesign.ops.preflight.support",
    "contract_environment_flag_state": "dnadesign.ops.preflight.contract_checks",
    "execute_runbook_plan": "dnadesign.ops.preflight.support",
    "evaluate_preflight_checks": "dnadesign.ops.preflight.coordinator",
    "infer_usr_dataset_requirements": "dnadesign.ops.preflight.support",
    "load_orchestration_runbook_payload": "dnadesign.ops.preflight.support",
    "render_argv": "dnadesign.ops.preflight.models",
    "run_preflight_command": "dnadesign.ops.preflight.support",
    "safe_json_loads": "dnadesign.ops.preflight.support",
    "supported_preflight_check_kinds": "dnadesign.ops.preflight.models",
}

if TYPE_CHECKING:
    from dnadesign.ops.preflight.check_protocols import (
        CommandCheckTarget,
        EnvironmentCheckTarget,
        RunbookPlanCheckTarget,
        SchedulerQueueCheckTarget,
    )
    from dnadesign.ops.preflight.checks import (
        CommandCheckDependencies,
        RunbookPlanCheckDependencies,
        SchedulerQueueCheckDependencies,
        build_command_checks,
        build_environment_checks,
        build_runbook_plan_checks,
        build_scheduler_queue_checks,
    )
    from dnadesign.ops.preflight.contract_checks import (
        ContractPreflightCheckDependencies,
        build_contract_preflight_checks,
        contract_environment_flag_state,
    )
    from dnadesign.ops.preflight.coordinator import (
        PreflightCheckEvaluation,
        check_matches_scope,
        evaluate_preflight_checks,
    )
    from dnadesign.ops.preflight.models import (
        CommandExecution,
        PreflightCheck,
        build_command_check,
        build_state_check,
        render_argv,
        supported_preflight_check_kinds,
    )
    from dnadesign.ops.preflight.support import (
        build_infer_notify_setup_command,
        choose_command_summary,
        execute_runbook_plan,
        infer_usr_dataset_requirements,
        load_orchestration_runbook_payload,
        run_preflight_command,
        safe_json_loads,
    )


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
