"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/plan_execution.py

Stage-B plan scheduling for pipeline execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from ...config import ResolvedPlanItem
    from ..artifacts.library import LibraryRecord
    from ..artifacts.pool import PoolData
    from .plan_pools import PlanPoolSource, PlanPoolSpec

PlanKey = tuple[str, str]


@dataclass(frozen=True)
class PlanEntry:
    item: ResolvedPlanItem
    spec: PlanPoolSpec
    source_cfg: PlanPoolSource
    key: PlanKey
    quota: int


@dataclass
class PlanExecutionResult:
    total: int
    per_plan: dict[PlanKey, int]
    plan_leaderboards: dict[PlanKey, dict]


def run_plan_schedule(
    *,
    plan_items: Iterable[ResolvedPlanItem],
    plan_pools: dict[str, PlanPoolSpec],
    plan_pool_sources: dict[str, PlanPoolSource],
    existing_counts: dict[PlanKey, int],
    round_robin: bool,
    process_plan: Callable[..., tuple[int, dict]],
    process_plan_args: dict,
    accumulate_stats: Callable[[PlanKey, dict], None],
    plan_pool_input_meta: Callable[[PlanPoolSpec], dict],
    inputs_manifest_entries: dict[str, dict],
    existing_usage_by_plan: dict[PlanKey, dict] | None,
    state_counts: dict[PlanKey, int],
    checkpoint_every: int,
    write_state: Callable[[], None],
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None,
    source_cache: dict[str, PoolData],
    attempt_counters: dict[PlanKey, int],
    library_records: dict[PlanKey, list[LibraryRecord]],
    library_cursor: dict[PlanKey, int],
    library_source: str | None,
    library_build_rows: list[dict],
    library_member_rows: list[dict],
    solution_rows: list[dict],
    composition_rows: list[dict],
    events_path: Path,
    display_map_by_input: dict[str, dict[str, str]],
) -> PlanExecutionResult:
    plan_entries: list[PlanEntry] = []
    for item in plan_items:
        spec = plan_pools[item.name]
        plan_entries.append(
            PlanEntry(
                item=item,
                spec=spec,
                source_cfg=plan_pool_sources[item.name],
                key=(spec.pool_name, item.name),
                quota=int(item.quota),
            )
        )

    def _process_entry(
        entry: PlanEntry,
        *,
        one_subsample_only: bool,
        already_generated: int,
    ) -> tuple[int, dict]:
        usage_counts = existing_usage_by_plan.get(entry.key) if existing_usage_by_plan else None
        return process_plan(
            entry.source_cfg,
            entry.item,
            **process_plan_args,
            one_subsample_only=one_subsample_only,
            already_generated=already_generated,
            inputs_manifest=inputs_manifest_entries,
            existing_usage_counts=usage_counts,
            state_counts=state_counts,
            checkpoint_every=checkpoint_every,
            write_state=write_state,
            site_failure_counts=site_failure_counts,
            source_cache=source_cache,
            pool_override=entry.spec.pool,
            input_meta_override=plan_pool_input_meta(entry.spec),
            attempt_counters=attempt_counters,
            library_records=library_records,
            library_cursor=library_cursor,
            library_source=library_source,
            library_build_rows=library_build_rows,
            library_member_rows=library_member_rows,
            solution_rows=solution_rows,
            composition_rows=composition_rows,
            events_path=events_path,
            display_map_by_input=display_map_by_input,
        )

    plan_leaderboards: dict[PlanKey, dict] = {}
    if not round_robin:
        per_plan: dict[PlanKey, int] = dict(existing_counts)
        total = sum(per_plan.values())
        for entry in plan_entries:
            produced, stats = _process_entry(
                entry,
                one_subsample_only=False,
                already_generated=int(existing_counts.get(entry.key, 0)),
            )
            per_plan[entry.key] = per_plan.get(entry.key, 0) + produced
            total += produced
            leaderboard_latest = stats.get("leaderboard_latest")
            if leaderboard_latest is not None:
                plan_leaderboards[entry.key] = leaderboard_latest
            accumulate_stats(entry.key, stats)
        return PlanExecutionResult(total=total, per_plan=per_plan, plan_leaderboards=plan_leaderboards)

    produced_counts: dict[PlanKey, int] = dict(existing_counts)
    done = False
    while not done:
        done = True
        for entry in plan_entries:
            current = produced_counts.get(entry.key, 0)
            if current >= entry.quota:
                continue
            done = False
            produced, stats = _process_entry(
                entry,
                one_subsample_only=True,
                already_generated=current,
            )
            produced_counts[entry.key] = current + produced
            leaderboard_latest = stats.get("leaderboard_latest")
            if leaderboard_latest is not None:
                plan_leaderboards[entry.key] = leaderboard_latest
            accumulate_stats(entry.key, stats)

    per_plan = produced_counts
    total = sum(per_plan.values())
    return PlanExecutionResult(total=total, per_plan=per_plan, plan_leaderboards=plan_leaderboards)
