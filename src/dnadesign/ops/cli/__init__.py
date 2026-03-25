from .app import app, main


def build_batch_plan(*args, **kwargs):
    from dnadesign.ops.orchestrator.plan import build_batch_plan as _build_batch_plan

    return _build_batch_plan(*args, **kwargs)


def discover_active_job_ids_for_runbook(*args, **kwargs):
    from dnadesign.ops.orchestrator.state import (
        discover_active_job_ids_for_runbook as _discover_active_job_ids_for_runbook,
    )

    return _discover_active_job_ids_for_runbook(*args, **kwargs)


def execute_batch_plan(*args, **kwargs):
    from dnadesign.ops.orchestrator.execute import execute_batch_plan as _execute_batch_plan

    return _execute_batch_plan(*args, **kwargs)


__all__ = [
    "app",
    "build_batch_plan",
    "discover_active_job_ids_for_runbook",
    "execute_batch_plan",
    "main",
]
