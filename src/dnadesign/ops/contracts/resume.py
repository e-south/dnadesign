"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/contracts/resume.py

Resume-readiness policy resolution for Ops runbook planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .models import ResumeReadinessPolicy

_RESUME_READINESS_POLICIES: dict[str, ResumeReadinessPolicy] = {}

_TOOLS_WITHOUT_RESUME_POLICY = frozenset({"infer"})


def resolve_resume_readiness_policy(tool: str) -> ResumeReadinessPolicy | None:
    tool_name = str(tool or "").strip().lower()
    if not tool_name:
        raise ValueError("resume readiness policy tool must be non-empty")
    policy = _RESUME_READINESS_POLICIES.get(tool_name)
    if policy is not None:
        return policy
    if tool_name in _TOOLS_WITHOUT_RESUME_POLICY:
        return None
    supported = ", ".join(sorted(set(_RESUME_READINESS_POLICIES) | set(_TOOLS_WITHOUT_RESUME_POLICY)))
    raise ValueError(f"unsupported resume readiness policy tool: {tool_name} (supported: {supported})")
