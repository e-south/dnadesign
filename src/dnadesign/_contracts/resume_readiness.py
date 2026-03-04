"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/resume_readiness.py

Shared producer resume-readiness policy contracts for ops mode resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResumeReadinessPolicy:
    tool: str
    required_record_columns: tuple[str, ...]
    orphan_artifact_markers: tuple[str, ...]


_DENSEGEN_RESUME_POLICY = ResumeReadinessPolicy(
    tool="densegen",
    required_record_columns=(
        "densegen__run_id",
        "densegen__input_name",
        "densegen__plan",
        "densegen__used_tfbs_detail",
    ),
    orphan_artifact_markers=(
        "outputs/pools/pool_manifest.json",
        "outputs/tables/attempts.parquet",
        "outputs/tables/solutions.parquet",
        "outputs/tables/run_metrics.parquet",
        "outputs/meta/effective_config.json",
    ),
)

_RESUME_READINESS_POLICIES: dict[str, ResumeReadinessPolicy] = {
    "densegen": _DENSEGEN_RESUME_POLICY,
}

_TOOLS_WITHOUT_RESUME_POLICY = frozenset(
    {
        "infer",
        "infer_evo2",
        "infer-evo2",
    }
)


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
