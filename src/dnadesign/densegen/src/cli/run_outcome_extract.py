"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/run_outcome_extract.py

Outcome extraction helpers for DenseGen run intros.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .run_intro import PlanOutcome, RunOutcomeSummary, _safe_float, _safe_int


def _aggregate_per_plan(items: object, *, plan_order: Sequence[str]) -> tuple[tuple[PlanOutcome, ...], int, int, int]:
    if not isinstance(items, list):
        return (tuple(), 0, 0, 0)

    by_plan: dict[str, dict[str, int]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("plan_name") or "").strip()
        if not name:
            continue
        state = by_plan.setdefault(
            name,
            {
                "generated": 0,
                "quota": 0,
                "stall_events": 0,
                "total_resamples": 0,
                "failed_solutions": 0,
            },
        )
        state["generated"] += _safe_int(item.get("generated")) or 0
        state["quota"] += _safe_int(item.get("quota")) or 0
        state["stall_events"] += _safe_int(item.get("stall_events")) or 0
        state["total_resamples"] += _safe_int(item.get("total_resamples")) or 0
        state["failed_solutions"] += _safe_int(item.get("failed_solutions")) or 0

    ordered_names: list[str] = []
    seen: set[str] = set()
    for name in plan_order:
        if name in by_plan and name not in seen:
            ordered_names.append(name)
            seen.add(name)
    for name in by_plan:
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    per_plan: list[PlanOutcome] = []
    stall_events = 0
    total_resamples = 0
    failed_solutions = 0
    for name in ordered_names:
        state = by_plan[name]
        per_plan.append(
            PlanOutcome(
                name=name,
                generated=int(state["generated"]),
                quota=int(state["quota"]),
                stall_events=int(state["stall_events"]),
                total_resamples=int(state["total_resamples"]),
                failed_solutions=int(state["failed_solutions"]),
            )
        )
        stall_events += int(state["stall_events"])
        total_resamples += int(state["total_resamples"])
        failed_solutions += int(state["failed_solutions"])
    return (tuple(per_plan), int(stall_events), int(total_resamples), int(failed_solutions))


def extract_outcome(
    manifest_payload: Mapping[str, object] | None,
    *,
    plan_order: Sequence[str],
    error_message: str | None = None,
    outcome_source: str = "manifest",
    pressure_source: str | None = None,
    notes: Sequence[str] = (),
    pressure_available: bool | None = None,
    pressure_message: str = "",
) -> RunOutcomeSummary:
    resolved_pressure_source = str(pressure_source or outcome_source)
    note_items = tuple(str(item).strip() for item in notes if str(item).strip())
    if error_message is not None:
        return RunOutcomeSummary(
            available=False,
            message=error_message,
            generated_total=None,
            quota_total=None,
            progress_pct=None,
            per_plan=tuple(),
            stall_events=0,
            total_resamples=0,
            failed_solutions=0,
            created_at=None,
            outcome_source=str(outcome_source),
            pressure_source=resolved_pressure_source,
            notes=note_items,
            pressure_available=False if pressure_available is None else bool(pressure_available),
            pressure_message=str(pressure_message or "").strip(),
        )

    if not isinstance(manifest_payload, Mapping):
        return RunOutcomeSummary(
            available=False,
            message="manifest not found",
            generated_total=None,
            quota_total=None,
            progress_pct=None,
            per_plan=tuple(),
            stall_events=0,
            total_resamples=0,
            failed_solutions=0,
            created_at=None,
            outcome_source=str(outcome_source),
            pressure_source=resolved_pressure_source,
            notes=note_items,
            pressure_available=False if pressure_available is None else bool(pressure_available),
            pressure_message=str(pressure_message or "").strip(),
        )

    generated_total = _safe_int(manifest_payload.get("total_generated"))
    if generated_total is None:
        return RunOutcomeSummary(
            available=False,
            message="manifest not found",
            generated_total=None,
            quota_total=None,
            progress_pct=None,
            per_plan=tuple(),
            stall_events=0,
            total_resamples=0,
            failed_solutions=0,
            created_at=None,
            outcome_source=str(outcome_source),
            pressure_source=resolved_pressure_source,
            notes=note_items,
            pressure_available=False if pressure_available is None else bool(pressure_available),
            pressure_message=str(pressure_message or "").strip(),
        )

    per_plan, stall_events, total_resamples, failed_solutions = _aggregate_per_plan(
        manifest_payload.get("items"),
        plan_order=plan_order,
    )
    quota_total = _safe_int(manifest_payload.get("total_quota"))
    if quota_total is None and per_plan:
        quota_total = int(sum(item.quota for item in per_plan))

    progress_pct = _safe_float(manifest_payload.get("quota_progress_pct"))
    if progress_pct is None and quota_total is not None and quota_total > 0:
        progress_pct = float(generated_total) * 100.0 / float(quota_total)

    created_at = str(manifest_payload.get("created_at") or "").strip() or None
    return RunOutcomeSummary(
        available=True,
        message="",
        generated_total=int(generated_total),
        quota_total=int(quota_total) if quota_total is not None else None,
        progress_pct=progress_pct,
        per_plan=per_plan,
        stall_events=int(stall_events),
        total_resamples=int(total_resamples),
        failed_solutions=int(failed_solutions),
        created_at=created_at,
        outcome_source=str(outcome_source),
        pressure_source=resolved_pressure_source,
        notes=note_items,
        pressure_available=True if pressure_available is None else bool(pressure_available),
        pressure_message=str(pressure_message or "").strip(),
    )
