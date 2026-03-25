"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/stress_promoter_ethanol_cipro/preflight_orchestration.py

Study-owned preflight builders for orchestration and notify environment
surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.preflight import CommandExecution, PreflightCheck, build_command_check, build_state_check

_NOTIFY_WEBHOOK_ENV_KEYS = ("NOTIFY_WEBHOOK", "NOTIFY_WEBHOOK_FILE")
_NOTIFY_TLS_ENV_KEY = "SSL_CERT_FILE"


@dataclass(frozen=True)
class PromoterPreflightNotifyEnvironmentDependencies:
    pass


@dataclass(frozen=True)
class PromoterPreflightRunbookPlanDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]


@dataclass(frozen=True)
class PromoterPreflightRunbookPlanTarget:
    check_id: str
    check_group: str
    phase: str
    phase_id: str | None
    runbook_path: Path
    fallback_summary: str
    details: Mapping[str, object] | None = None


def resolve_notify_environment_state(
    *,
    environ: Mapping[str, object | None],
) -> dict[str, bool]:
    return {
        env_var: bool(str(environ.get(env_var) or "").strip())
        for env_var in (*_NOTIFY_WEBHOOK_ENV_KEYS, _NOTIFY_TLS_ENV_KEY)
    }


def build_promoter_preflight_notify_environment_checks(
    *,
    notify_env_state: Mapping[str, bool],
    notify_environment_phase_id: str,
    enabled_groups: Collection[str],
    dependencies: PromoterPreflightNotifyEnvironmentDependencies,
) -> tuple[PreflightCheck, ...]:
    del dependencies
    if "notify_environment" not in enabled_groups:
        return ()
    webhook_ready = any(bool(notify_env_state.get(env_var)) for env_var in _NOTIFY_WEBHOOK_ENV_KEYS)
    return (
        build_state_check(
            check_id="notify.environment.webhook",
            check_group="notify_environment",
            phase="notify",
            phase_id=notify_environment_phase_id,
            state="ok" if webhook_ready else "attention",
            summary=(
                "batch notify secret is configured in the environment"
                if webhook_ready
                else "batch notify secret is not configured; export NOTIFY_WEBHOOK_FILE or NOTIFY_WEBHOOK"
            ),
            details=dict(notify_env_state),
        ),
        build_state_check(
            check_id="notify.environment.tls",
            check_group="notify_environment",
            phase="notify",
            phase_id=notify_environment_phase_id,
            state="ok" if bool(notify_env_state.get(_NOTIFY_TLS_ENV_KEY)) else "attention",
            summary=(
                "SSL_CERT_FILE is configured for notify profile doctor and live delivery"
                if bool(notify_env_state.get(_NOTIFY_TLS_ENV_KEY))
                else "SSL_CERT_FILE is not configured for notify profile doctor and live delivery"
            ),
            details=dict(notify_env_state),
        ),
    )


def build_promoter_preflight_runbook_plan_checks(
    *,
    study_repo_root: Path,
    targets: Sequence[PromoterPreflightRunbookPlanTarget],
    dependencies: PromoterPreflightRunbookPlanDependencies,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    for target in targets:
        runbook_plan = dependencies.run_preflight_command(
            (
                "uv",
                "run",
                "ops",
                "runbook",
                "plan",
                "--runbook",
                str(target.runbook_path),
                "--repo-root",
                str(study_repo_root),
            ),
            cwd=study_repo_root,
        )
        runbook_plan_payload = (
            dependencies.safe_json_loads(getattr(runbook_plan, "stdout", None))
            if getattr(runbook_plan, "returncode", 1) == 0
            else None
        )
        runbook_details: dict[str, object] = {
            "runbook": str(target.runbook_path),
            **dict(target.details or {}),
        }
        if isinstance(runbook_plan_payload, dict):
            runbook_details.update(
                {
                    "selected_mode": runbook_plan_payload.get("selected_mode"),
                    "workflow_id": runbook_plan_payload.get("workflow_id"),
                    "notify_secret_ref": dict(runbook_plan_payload.get("orchestration_notify") or {}).get("secret_ref"),
                }
            )
        checks.append(
            build_command_check(
                check_id=target.check_id,
                check_group=target.check_group,
                phase=target.phase,
                phase_id=target.phase_id,
                summary=dependencies.choose_command_summary(
                    runbook_plan,
                    fallback=target.fallback_summary,
                ),
                execution=runbook_plan,
                details=runbook_details,
            )
        )
    return tuple(checks)


__all__ = [
    "PromoterPreflightNotifyEnvironmentDependencies",
    "PromoterPreflightRunbookPlanDependencies",
    "PromoterPreflightRunbookPlanTarget",
    "build_promoter_preflight_notify_environment_checks",
    "build_promoter_preflight_runbook_plan_checks",
    "resolve_notify_environment_state",
]
