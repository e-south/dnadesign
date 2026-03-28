"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/__init__.py

Public package exports for cross-tool batch orchestration contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib

from .contracts import (
    ResumeReadinessPolicy,
    USRProducerContract,
    resolve_resume_readiness_policy,
    resolve_usr_producer_contract,
)


def __getattr__(name: str):
    if name == "api":
        return importlib.import_module("dnadesign.ops.api")
    raise AttributeError(f"module 'dnadesign.ops' has no attribute {name!r}")


__all__ = [
    "ResumeReadinessPolicy",
    "USRProducerContract",
    "api",
    "resolve_resume_readiness_policy",
    "resolve_usr_producer_contract",
]
