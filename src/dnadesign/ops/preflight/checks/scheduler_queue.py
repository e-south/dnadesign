"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/checks/scheduler_queue.py

Generic scheduler-queue preflight check executors.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.preflight.check_protocols import SchedulerQueueCheckTarget

from ..models import CommandExecution, PreflightCheck, build_command_check


@dataclass(frozen=True)
class SchedulerQueueCheckDependencies:
    run_preflight_command: Callable[..., CommandExecution]


def build_scheduler_queue_checks(
    *,
    repo_root: Path,
    targets: Sequence[SchedulerQueueCheckTarget],
    dependencies: SchedulerQueueCheckDependencies,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    for target in targets:
        backend = str(target.backend or "").strip().lower()
        if backend != "sge":
            raise ValueError(f"scheduler queue check {target.check_id!r} has unsupported backend {backend!r}")
        execution = dependencies.run_preflight_command(
            (
                "uv",
                "run",
                "python",
                "-m",
                "dnadesign.ops.orchestrator.gates",
                "session-counts",
                "--allow-missing-qstat",
            ),
            cwd=repo_root,
        )
        details = {
            "backend": backend,
            "max_running_jobs": target.max_running_jobs,
            "max_queued_jobs": target.max_queued_jobs,
            **dict(target.details or {}),
        }
        record = _parse_record(execution.stdout)
        queue_probe = str(record.get("queue_probe") or "").strip().lower() or "unknown"
        running_jobs = _int_or_none(record.get("running_jobs"))
        queued_jobs = _int_or_none(record.get("queued_jobs"))
        eqw_jobs = _int_or_none(record.get("eqw_jobs"))
        details.update(
            {
                "queue_probe": queue_probe,
                "running_jobs": running_jobs,
                "queued_jobs": queued_jobs,
                "eqw_jobs": eqw_jobs,
            }
        )

        if execution.returncode != 0:
            summary = f"scheduler queue probe failed for backend {backend}"
            override_state = "attention"
        elif (
            queue_probe != "ok" or running_jobs is None or (target.max_queued_jobs is not None and queued_jobs is None)
        ):
            summary = f"scheduler queue probe is unavailable for backend {backend}"
            override_state = "attention"
        else:
            over_running = running_jobs > target.max_running_jobs
            over_queued = target.max_queued_jobs is not None and (queued_jobs or 0) > target.max_queued_jobs
            if over_running or over_queued:
                summary = (
                    "scheduler queue exceeds declared thresholds "
                    f"(running={running_jobs}, queued={queued_jobs}, eqw={eqw_jobs})"
                )
                override_state = "attention"
            else:
                summary = (
                    "scheduler queue is below declared thresholds "
                    f"(running={running_jobs}, queued={queued_jobs}, eqw={eqw_jobs})"
                )
                override_state = "ok"

        checks.append(
            build_command_check(
                check_id=target.check_id,
                kind="scheduler_queue",
                required=target.required,
                check_group=target.check_group,
                phase=target.phase,
                phase_id=target.phase_id,
                summary=summary,
                execution=execution,
                surface_id=target.surface_id,
                details=details,
                override_state=override_state,
            )
        )
    return tuple(checks)


def _parse_record(text: str | None) -> dict[str, str]:
    payload: dict[str, str] = {}
    for token in str(text or "").split():
        key, sep, value = token.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key:
            payload[key] = value
    return payload


def _int_or_none(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or text == "unknown":
        return None
    return int(text)


__all__ = [
    "SchedulerQueueCheckDependencies",
    "build_scheduler_queue_checks",
]
