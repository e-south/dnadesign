"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/__init__.py

Lazy public exports for mode resolution and deterministic batch-plan construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "BatchExecutionResult": ("dnadesign.ops.orchestrator.execute", "BatchExecutionResult"),
    "BatchPlan": ("dnadesign.ops.orchestrator.plan", "BatchPlan"),
    "ModeDecision": ("dnadesign.ops.orchestrator.state", "ModeDecision"),
    "build_batch_plan": ("dnadesign.ops.orchestrator.plan", "build_batch_plan"),
    "execute_batch_plan": ("dnadesign.ops.orchestrator.execute", "execute_batch_plan"),
    "resolve_mode_decision": ("dnadesign.ops.orchestrator.state", "resolve_mode_decision"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'dnadesign.ops.orchestrator' has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BatchExecutionResult",
    "BatchPlan",
    "ModeDecision",
    "build_batch_plan",
    "execute_batch_plan",
    "resolve_mode_decision",
]
