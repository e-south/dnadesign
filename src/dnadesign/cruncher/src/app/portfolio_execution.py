"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/portfolio_execution.py

Execution helpers for Portfolio preparation and aggregation flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from dnadesign.cruncher.app.portfolio_preflight import (
    _render_prepare_runbook_command,
    _render_prepare_runbook_path,
    _requires_full_runbook_prepare,
)
from dnadesign.cruncher.app.portfolio_studies import (
    _ensure_required_source_studies_for_sources,
    _load_source_sequence_length_rows,
)
from dnadesign.cruncher.portfolio.manifest import PortfolioPreparedSource, PortfolioSourceRun
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource, PortfolioSpec
from dnadesign.cruncher.workspaces.runbook import run_workspace_runbook

PrepareReadyPolicy = Literal["rerun", "skip"]
PortfolioEventCallback = Callable[[str, dict[str, object]], None]
RunStudyFn = Callable[..., Path]
LoadSourceRowsFn = Callable[
    ...,
    tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, object],
        PortfolioSourceRun,
        dict[str, object] | None,
    ],
]
MaterializePortfolioOutputsFn = Callable[..., tuple[list[Path], list[Path], pd.DataFrame]]
EmitPortfolioEventFn = Callable[..., None]


@dataclass
class _PortfolioAggregationState:
    ensured_study_runs: dict[tuple[str, str], Path]
    all_window_rows: list[dict[str, object]]
    all_elite_rows: list[dict[str, object]]
    all_consensus_rows: list[dict[str, object]]
    source_summary_rows: list[dict[str, object]]
    study_summary_rows: list[dict[str, object]]
    sequence_length_rows: list[dict[str, object]]
    table_paths: list[Path]
    plot_paths: list[Path]
    source_runs: list[PortfolioSourceRun]
    prepared_sources: list[PortfolioPreparedSource]
    elite_summary_df: pd.DataFrame


def _new_portfolio_aggregation_state(
    *,
    ensured_study_runs: dict[tuple[str, str], Path],
) -> _PortfolioAggregationState:
    return _PortfolioAggregationState(
        ensured_study_runs=ensured_study_runs,
        all_window_rows=[],
        all_elite_rows=[],
        all_consensus_rows=[],
        source_summary_rows=[],
        study_summary_rows=[],
        sequence_length_rows=[],
        table_paths=[],
        plot_paths=[],
        source_runs=[],
        prepared_sources=[],
        elite_summary_df=pd.DataFrame(),
    )


def _prepare_source_log_path(run_dir: Path, source_id: str) -> Path:
    from dnadesign.cruncher.portfolio.layout import portfolio_logs_dir

    return portfolio_logs_dir(run_dir) / f"prepare__{source_id}.log"


def _prepare_source(
    source: PortfolioSource,
    *,
    readiness: dict[str, dict[str, object]],
    prepare_ready_policy: PrepareReadyPolicy,
    prepare_log_path: Path | None = None,
) -> PortfolioPreparedSource:
    if source.prepare is None:
        raise ValueError(
            "portfolio.execution.mode=prepare_then_aggregate requires prepare for every source: "
            f"missing source={source.id!r}"
        )
    source_id = str(source.id)
    source_readiness = readiness.get(source_id)
    is_ready = bool(source_readiness and source_readiness.get("ready"))
    if is_ready and prepare_ready_policy == "skip":
        return PortfolioPreparedSource(
            source_id=source_id,
            runbook_path=str(source.prepare.runbook),
            step_ids=[],
        )
    try:
        result = run_workspace_runbook(
            source.prepare.runbook,
            step_ids=source.prepare.step_ids,
            dry_run=False,
            output_log_path=prepare_log_path,
        )
    except RuntimeError as exc:
        nudge_cmd = _render_prepare_runbook_command(source, include_steps=True)
        lines = [
            "Portfolio source preparation failed.",
            f"source={source_id} workspace={source.workspace.name}",
            f"runbook={_render_prepare_runbook_path(source)}",
            f"step_ids={list(source.prepare.step_ids)}",
        ]
        readiness_issues = list(source_readiness.get("issues", [])) if source_readiness else []
        if readiness_issues:
            lines.append("preflight issues:")
            for issue in readiness_issues:
                lines.append(f"  - {issue}")
        lines.append(
            "nudge: include all steps needed for source readiness "
            "(usually sample_run, analyze_summary, export_sequences_latest)."
        )
        lines.append(f"nudge: {nudge_cmd}")
        if _requires_full_runbook_prepare(readiness_issues) and source.prepare.step_ids:
            lines.append(
                f"nudge: full runbook required: {_render_prepare_runbook_command(source, include_steps=False)}"
            )
        if prepare_log_path is not None:
            lines.append(f"log: {prepare_log_path}")
        lines.append(f"cause: {exc}")
        raise ValueError("\n".join(lines)) from exc
    return PortfolioPreparedSource(
        source_id=source_id,
        runbook_path=str(result.runbook_path),
        step_ids=list(result.executed_step_ids),
    )


def _aggregate_source_into_state(
    *,
    source: PortfolioSource,
    spec: PortfolioSpec,
    run_dir: Path,
    state: _PortfolioAggregationState,
    on_event: PortfolioEventCallback | None,
    emit_event_fn: EmitPortfolioEventFn,
    load_source_rows_fn: LoadSourceRowsFn,
    materialize_portfolio_outputs_fn: MaterializePortfolioOutputsFn,
    run_study_fn: RunStudyFn,
) -> None:
    source_id = str(source.id)
    emit_event_fn(on_event, "aggregate_source_started", source_id=source_id)
    (
        source_windows_rows,
        source_elite_rows,
        source_consensus_rows,
        source_summary_row,
        source_run,
        source_study_summary,
    ) = load_source_rows_fn(source, studies_enabled=spec.studies.enabled, on_event=on_event)
    state.all_window_rows.extend(source_windows_rows)
    state.all_elite_rows.extend(source_elite_rows)
    state.all_consensus_rows.extend(source_consensus_rows)
    state.source_summary_rows.append(source_summary_row)
    state.source_runs.append(source_run)
    emit_event_fn(
        on_event,
        "aggregate_source_completed",
        source_id=source_id,
        selected_elites=int(source_summary_row["n_selected_elites"]),
    )
    if source_study_summary is not None:
        state.study_summary_rows.append(source_study_summary)
    if spec.studies.enabled and spec.studies.sequence_length_table.enabled:
        state.sequence_length_rows.extend(
            _load_source_sequence_length_rows(
                source,
                study_spec=spec.studies.sequence_length_table.study_spec,
                top_n_lengths=int(spec.studies.sequence_length_table.top_n_lengths),
                ensured_study_runs=state.ensured_study_runs,
                run_study_fn=run_study_fn,
                on_event=on_event,
            )
        )
    state.table_paths, state.plot_paths, state.elite_summary_df = materialize_portfolio_outputs_fn(
        run_dir=run_dir,
        spec=spec,
        all_window_rows=state.all_window_rows,
        all_elite_rows=state.all_elite_rows,
        all_consensus_rows=state.all_consensus_rows,
        source_summary_rows=state.source_summary_rows,
        study_summary_rows=state.study_summary_rows,
        sequence_length_rows=state.sequence_length_rows,
    )
    emit_event_fn(
        on_event,
        "aggregate_source_outputs_updated",
        source_id=source_id,
        completed_sources=len(state.source_runs),
        table_count=len(state.table_paths),
        plot_count=len(state.plot_paths),
    )


def _run_prepare_then_aggregate(
    *,
    spec: PortfolioSpec,
    run_dir: Path,
    state: _PortfolioAggregationState,
    readiness: dict[str, dict[str, object]],
    prepare_ready_policy: PrepareReadyPolicy,
    on_event: PortfolioEventCallback | None,
    emit_event_fn: EmitPortfolioEventFn,
    load_source_rows_fn: LoadSourceRowsFn,
    materialize_portfolio_outputs_fn: MaterializePortfolioOutputsFn,
    run_study_fn: RunStudyFn,
) -> None:
    emit_event_fn(on_event, "prepare_phase_started", source_count=len(spec.sources))
    max_workers = min(int(spec.execution.max_parallel_sources), len(spec.sources))
    prepared_by_id: dict[str, PortfolioPreparedSource] = {}
    pending_by_id: dict[str, tuple[Future[PortfolioPreparedSource], Path]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for source in spec.sources:
            source_id = str(source.id)
            source_readiness = readiness.get(source_id)
            is_ready = bool(source_readiness and source_readiness.get("ready"))
            if is_ready and prepare_ready_policy == "skip":
                emit_event_fn(
                    on_event,
                    "prepare_source_skipped",
                    source_id=source_id,
                    reason="source already ready",
                )
                if source.prepare is None:
                    raise ValueError(
                        "portfolio.execution.mode=prepare_then_aggregate requires prepare for every source: "
                        f"missing source={source.id!r}"
                    )
                prepared_by_id[source_id] = PortfolioPreparedSource(
                    source_id=source_id,
                    runbook_path=str(source.prepare.runbook),
                    step_ids=[],
                )
                continue
            log_path = _prepare_source_log_path(run_dir, source_id)
            emit_event_fn(
                on_event,
                "prepare_source_started",
                source_id=source_id,
                runbook=str(source.prepare.runbook if source.prepare is not None else ""),
                log_path=str(log_path),
            )
            future = executor.submit(
                _prepare_source,
                source,
                readiness=readiness,
                prepare_ready_policy=prepare_ready_policy,
                prepare_log_path=log_path,
            )
            pending_by_id[source_id] = (future, log_path)

        emit_event_fn(on_event, "aggregate_phase_started", source_count=len(spec.sources))
        for source in spec.sources:
            source_id = str(source.id)
            prepared = prepared_by_id.get(source_id)
            if prepared is None:
                future, log_path = pending_by_id[source_id]
                try:
                    prepared = future.result()
                except Exception:
                    for queued, _ in pending_by_id.values():
                        queued.cancel()
                    raise
                emit_event_fn(
                    on_event,
                    "prepare_source_completed",
                    source_id=source_id,
                    executed_steps=list(prepared.step_ids),
                    log_path=str(log_path),
                )
                prepared_by_id[source_id] = prepared
            state.prepared_sources.append(prepared)
            if spec.studies.enabled:
                state.ensured_study_runs.update(
                    _ensure_required_source_studies_for_sources(
                        spec,
                        [source],
                        run_study_fn=run_study_fn,
                        on_event=on_event,
                    )
                )
            _aggregate_source_into_state(
                source=source,
                spec=spec,
                run_dir=run_dir,
                state=state,
                on_event=on_event,
                emit_event_fn=emit_event_fn,
                load_source_rows_fn=load_source_rows_fn,
                materialize_portfolio_outputs_fn=materialize_portfolio_outputs_fn,
                run_study_fn=run_study_fn,
            )
    emit_event_fn(on_event, "prepare_phase_completed", prepared_count=len(state.prepared_sources))
    emit_event_fn(on_event, "aggregate_phase_completed", source_count=len(spec.sources))


def _run_aggregate_only(
    *,
    spec: PortfolioSpec,
    run_dir: Path,
    state: _PortfolioAggregationState,
    on_event: PortfolioEventCallback | None,
    emit_event_fn: EmitPortfolioEventFn,
    load_source_rows_fn: LoadSourceRowsFn,
    materialize_portfolio_outputs_fn: MaterializePortfolioOutputsFn,
    run_study_fn: RunStudyFn,
) -> None:
    emit_event_fn(on_event, "aggregate_phase_started", source_count=len(spec.sources))
    for source in spec.sources:
        _aggregate_source_into_state(
            source=source,
            spec=spec,
            run_dir=run_dir,
            state=state,
            on_event=on_event,
            emit_event_fn=emit_event_fn,
            load_source_rows_fn=load_source_rows_fn,
            materialize_portfolio_outputs_fn=materialize_portfolio_outputs_fn,
            run_study_fn=run_study_fn,
        )
    emit_event_fn(on_event, "aggregate_phase_completed", source_count=len(spec.sources))
