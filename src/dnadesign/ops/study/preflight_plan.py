"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/study/preflight_plan.py

Generic study-preflight scope planning and blocker evaluation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .models import StudyPreflightContract


@dataclass(frozen=True)
class StudyPreflightPlan:
    scope: str
    included_groups: tuple[str, ...]

    def includes_group(self, group: str) -> bool:
        return str(group).strip() in self.included_groups


@dataclass(frozen=True)
class CompiledStudyPreflightCheck:
    check_id: str
    kind: str
    check_group: str
    check_set_id: str
    category: str
    summary: str
    required: bool
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledStudyPreflightExecutionPlan:
    checks: tuple[CompiledStudyPreflightCheck, ...]


def normalize_study_preflight_scope(scope: str | None, *, default_scope: str = "full") -> str:
    normalized = str(scope or default_scope).strip().lower()
    if normalized not in {"full", "next"}:
        raise ValueError("study preflight scope must be one of: full, next")
    return normalized


def build_study_preflight_plan(
    *,
    scope: str | None,
    contract: StudyPreflightContract,
) -> StudyPreflightPlan:
    scope_norm = normalize_study_preflight_scope(scope, default_scope=contract.default_scope)
    included_groups = contract.known_groups if scope_norm == "full" else contract.scope_groups.get("next")
    if included_groups is None:
        raise ValueError("study preflight next scope requires explicit preflight.scopes.next.include_groups")
    return StudyPreflightPlan(scope=scope_norm, included_groups=included_groups)


def compile_study_preflight_execution_plan(
    *,
    contract: StudyPreflightContract,
    enabled_groups: Sequence[str],
) -> CompiledStudyPreflightExecutionPlan:
    enabled_group_set = {str(group).strip() for group in enabled_groups if str(group).strip()}
    compiled_checks: list[CompiledStudyPreflightCheck] = []
    for check_set_id, specs in contract.check_specs.items():
        for spec in specs:
            check_group = str(spec.get("check_group") or "").strip()
            if not check_group or check_group not in enabled_group_set:
                continue
            kind = str(spec.get("kind") or "").strip()
            category = _check_category(check_group=check_group, kind=kind)
            compiled_checks.append(
                CompiledStudyPreflightCheck(
                    check_id=str(spec.get("check_id") or "").strip(),
                    kind=kind,
                    check_group=check_group,
                    check_set_id=check_set_id,
                    category=category,
                    summary=str(spec.get("summary") or "").strip(),
                    required=bool(spec.get("required", True)),
                    payload={
                        key: value
                        for key, value in dict(spec).items()
                        if key not in {"check_id", "kind", "check_group", "summary", "required"}
                    },
                )
            )
    return CompiledStudyPreflightExecutionPlan(checks=tuple(compiled_checks))


def _check_category(*, check_group: str, kind: str) -> str:
    if kind == "runbook_plan" or check_group.endswith("_plan"):
        return "ops"
    if kind == "scheduler_queue":
        return "scheduler"
    if check_group.startswith("notify"):
        return "notify"
    return check_group or kind


__all__ = [
    "CompiledStudyPreflightCheck",
    "CompiledStudyPreflightExecutionPlan",
    "StudyPreflightPlan",
    "build_study_preflight_plan",
    "compile_study_preflight_execution_plan",
    "normalize_study_preflight_scope",
]
