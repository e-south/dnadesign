"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/stress_promoter_ethanol_cipro/preflight_scope.py

Study-owned preflight scope planning and blocker classification helpers for the
stress_promoter_ethanol_cipro family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dnadesign.ops.contracts import InferRuntimePhaseTarget

_STATE_PRIORITY = {"missing": 0, "attention": 1}


@dataclass(frozen=True)
class PromoterPreflightScopePlan:
    scope: str
    target_phase_id: str | None
    included_groups: tuple[str, ...]
    include_densegen_checks: bool
    include_construct_checks: bool
    include_infer_checks: bool
    include_notify_checks: bool
    include_infer_batch_plan_checks: bool


@dataclass(frozen=True)
class PromoterPreflightCheckEvaluation:
    scoped_checks: tuple[Mapping[str, object], ...]
    blocker_checks: tuple[Mapping[str, object], ...]
    deferred_blockers: tuple[Mapping[str, object], ...]
    nonblocking_attention_checks: tuple[Mapping[str, object], ...]
    scoped_counts: dict[str, int]
    effective_counts: dict[str, int]

    @property
    def blocked_by_ids(self) -> tuple[str, ...]:
        return tuple(str(check["id"]) for check in self.blocker_checks)

    @property
    def deferred_check_ids(self) -> tuple[str, ...]:
        return tuple(str(check["id"]) for check in self.deferred_blockers)

    @property
    def nonblocking_attention_ids(self) -> tuple[str, ...]:
        return tuple(str(check["id"]) for check in self.nonblocking_attention_checks)


def normalize_promoter_preflight_scope(scope: str | None, *, default_scope: str = "full") -> str:
    normalized = str(scope or default_scope).strip().lower()
    if normalized not in {"full", "next"}:
        raise ValueError("promoter-study-preflight scope must be one of: full, next")
    return normalized


def build_promoter_preflight_scope_plan(
    *,
    current_phase: str | None,
    next_ready_phase: Mapping[str, object] | None,
    scope: str | None,
    default_scope: str = "full",
    phase_group_overrides: Mapping[str, Sequence[str]] | None = None,
    infer_lane_groups: Sequence[str] | None = None,
    infer_phase_targets: Mapping[str, InferRuntimePhaseTarget] | None = None,
) -> PromoterPreflightScopePlan:
    scope_norm = normalize_promoter_preflight_scope(scope, default_scope=default_scope)
    target_phase_id = _promoter_preflight_target_phase_id(
        current_phase=current_phase,
        next_ready_phase=next_ready_phase,
        scope=scope_norm,
    )
    included_groups = _resolve_included_groups(
        target_phase_id=target_phase_id,
        scope=scope_norm,
        phase_group_overrides=phase_group_overrides,
        infer_lane_groups=infer_lane_groups,
        infer_phase_targets=infer_phase_targets,
    )
    return PromoterPreflightScopePlan(
        scope=scope_norm,
        target_phase_id=target_phase_id,
        included_groups=included_groups,
        include_densegen_checks=("densegen" in included_groups),
        include_construct_checks=("construct" in included_groups),
        include_infer_checks=("infer" in included_groups),
        include_notify_checks=("notify" in included_groups),
        include_infer_batch_plan_checks=("infer_batch_plan" in included_groups),
    )


def evaluate_promoter_preflight_checks(
    checks: Sequence[Mapping[str, object]],
    *,
    phase_states: Sequence[Mapping[str, object]] | None,
    scope_plan: PromoterPreflightScopePlan,
    infer_phase_targets: Mapping[str, InferRuntimePhaseTarget] | None = None,
) -> PromoterPreflightCheckEvaluation:
    scoped_checks = tuple(
        check
        for check in checks
        if promoter_preflight_check_matches_scope(
            str(check["id"]),
            scope_plan=scope_plan,
            infer_phase_targets=infer_phase_targets,
        )
    )
    scoped_counts = _counts_by_state(scoped_checks)
    all_counts = _counts_by_state(checks)
    blocker_checks = _ordered_preflight_blockers(
        scoped_checks,
        phase_states=phase_states,
        scope=scope_plan.scope,
    )
    deferred_blockers = _ordered_preflight_blockers(
        [check for check in checks if check not in scoped_checks],
        phase_states=phase_states,
        scope=scope_plan.scope,
    )
    nonblocking_attention_checks = _ordered_nonblocking_preflight_attention(
        scoped_checks,
        phase_states=phase_states,
        scope=scope_plan.scope,
    )
    effective_counts = scoped_counts if scope_plan.scope == "next" else all_counts
    return PromoterPreflightCheckEvaluation(
        scoped_checks=scoped_checks,
        blocker_checks=blocker_checks,
        deferred_blockers=deferred_blockers,
        nonblocking_attention_checks=nonblocking_attention_checks,
        scoped_counts=scoped_counts,
        effective_counts=effective_counts,
    )


def promoter_preflight_check_matches_scope(
    check_id: str,
    *,
    scope_plan: PromoterPreflightScopePlan,
    infer_phase_targets: Mapping[str, InferRuntimePhaseTarget] | None = None,
) -> bool:
    target_phase_id = scope_plan.target_phase_id
    if scope_plan.scope != "next" or target_phase_id is None:
        return True
    lane_target = (infer_phase_targets or {}).get(target_phase_id)
    if lane_target is not None:
        return _lane_check_matches_scope(
            check_id,
            scope_plan=scope_plan,
            lane_target=lane_target,
        )
    return _group_for_check_id(check_id) in scope_plan.included_groups


def _counts_by_state(checks: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(check["state"]) for check in checks)
    return {state: int(counts.get(state, 0)) for state in ("ok", "attention", "missing")}


def _promoter_preflight_target_phase_id(
    *,
    current_phase: str | None,
    next_ready_phase: Mapping[str, object] | None,
    scope: str,
) -> str | None:
    if scope != "next":
        return None
    if next_ready_phase is not None:
        phase_id = str(next_ready_phase.get("id") or "").strip()
        return phase_id or None
    return current_phase


def _resolve_included_groups(
    *,
    target_phase_id: str | None,
    scope: str,
    phase_group_overrides: Mapping[str, Sequence[str]] | None,
    infer_lane_groups: Sequence[str] | None,
    infer_phase_targets: Mapping[str, InferRuntimePhaseTarget] | None,
) -> tuple[str, ...]:
    all_groups = ("densegen", "construct", "infer", "notify", "infer_batch_plan")
    if scope != "next" or target_phase_id is None:
        return all_groups
    if phase_group_overrides and target_phase_id in phase_group_overrides:
        return tuple(str(group).strip() for group in phase_group_overrides[target_phase_id] if str(group).strip())
    if infer_phase_targets and target_phase_id in infer_phase_targets:
        default_infer_groups = ("infer", "notify", "infer_batch_plan")
        resolved_infer_lane_groups = tuple(
            str(group).strip() for group in (infer_lane_groups or default_infer_groups) if str(group).strip()
        )
        return resolved_infer_lane_groups
    return all_groups


def _group_for_check_id(check_id: str) -> str:
    if check_id.startswith("densegen."):
        return "densegen"
    if check_id.startswith("construct."):
        return "construct"
    if check_id.startswith("ops.runbook_plan."):
        return "infer_batch_plan"
    if check_id.startswith("notify."):
        return "notify"
    if check_id.startswith("infer."):
        return "infer"
    return "other"


def _lane_check_matches_scope(
    check_id: str,
    *,
    scope_plan: PromoterPreflightScopePlan,
    lane_target: InferRuntimePhaseTarget,
) -> bool:
    group = _group_for_check_id(check_id)
    if group not in scope_plan.included_groups:
        return False
    if check_id.startswith("notify.environment."):
        return True
    if group == "infer_batch_plan":
        return check_id == f"ops.runbook_plan.{lane_target.runbook_surface_label}"
    if group in {"infer", "notify"}:
        return lane_target.runtime_label in check_id or lane_target.config_label in check_id
    return False


def _ordered_preflight_blockers(
    checks: Sequence[Mapping[str, object]],
    *,
    phase_states: Sequence[Mapping[str, object]] | None,
    scope: str,
) -> tuple[Mapping[str, object], ...]:
    failing = [check for check in checks if check["state"] != "ok"]
    if scope == "full":
        phase_status_index = _promoter_phase_status_index(phase_states)
        failing = [
            check for check in failing if _preflight_check_is_blocking(check, phase_status_index=phase_status_index)
        ]
    return tuple(
        sorted(
            failing,
            key=lambda check: (
                _STATE_PRIORITY.get(str(check["state"]), 99),
                str(check["id"]),
            ),
        )
    )


def _ordered_nonblocking_preflight_attention(
    checks: Sequence[Mapping[str, object]],
    *,
    phase_states: Sequence[Mapping[str, object]] | None,
    scope: str,
) -> tuple[Mapping[str, object], ...]:
    if scope != "full":
        return ()
    phase_status_index = _promoter_phase_status_index(phase_states)
    nonblocking = [
        check
        for check in checks
        if check["state"] != "ok" and not _preflight_check_is_blocking(check, phase_status_index=phase_status_index)
    ]
    return tuple(
        sorted(
            nonblocking,
            key=lambda check: (
                _STATE_PRIORITY.get(str(check["state"]), 99),
                str(check["id"]),
            ),
        )
    )


def _preflight_check_is_blocking(
    check: Mapping[str, object],
    *,
    phase_status_index: Mapping[str, str],
) -> bool:
    phase_id = str(check.get("phase_id") or "").strip()
    if not phase_id:
        return True
    phase_status = phase_status_index.get(phase_id)
    return phase_status not in {"complete", "parallel_optional"}


def _promoter_phase_status_index(phase_states: Sequence[Mapping[str, object]] | None) -> dict[str, str]:
    if phase_states is None:
        return {}
    result: dict[str, str] = {}
    for phase in phase_states:
        phase_id = str(phase.get("id") or "").strip()
        phase_status = str(phase.get("status") or "").strip()
        if phase_id and phase_status:
            result[phase_id] = phase_status
    return result


__all__ = [
    "PromoterPreflightCheckEvaluation",
    "PromoterPreflightScopePlan",
    "build_promoter_preflight_scope_plan",
    "evaluate_promoter_preflight_checks",
    "normalize_promoter_preflight_scope",
    "promoter_preflight_check_matches_scope",
]
