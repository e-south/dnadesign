"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/portfolio_workflow.py

Orchestrate cross-workspace Portfolio aggregation outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Literal

from dnadesign.cruncher.app.portfolio_execution import (
    _aggregate_source_into_state as _aggregate_source_into_state_helper,
)
from dnadesign.cruncher.app.portfolio_execution import (
    _new_portfolio_aggregation_state as _new_portfolio_aggregation_state_helper,
)
from dnadesign.cruncher.app.portfolio_execution import (
    _PortfolioAggregationState as _PortfolioAggregationState_helper,
)
from dnadesign.cruncher.app.portfolio_execution import (
    _prepare_source as _prepare_source_helper,
)
from dnadesign.cruncher.app.portfolio_execution import (
    _prepare_source_log_path as _prepare_source_log_path_helper,
)
from dnadesign.cruncher.app.portfolio_execution import (
    _run_aggregate_only as _run_aggregate_only_helper,
)
from dnadesign.cruncher.app.portfolio_execution import (
    _run_prepare_then_aggregate as _run_prepare_then_aggregate_helper,
)
from dnadesign.cruncher.app.portfolio_materialization import (
    _materialize_portfolio_outputs as _materialize_portfolio_outputs_helper,
)
from dnadesign.cruncher.app.portfolio_materialization import (
    _select_portfolio_showcase_elites as _select_portfolio_showcase_elites_helper,
)
from dnadesign.cruncher.app.portfolio_materialization import (
    _write_tradeoff_plot as _write_tradeoff_plot_helper,
)
from dnadesign.cruncher.app.portfolio_preflight import (
    _collect_source_readiness,
    _raise_aggregate_only_preflight,
)
from dnadesign.cruncher.app.portfolio_preflight import (
    _preflight_source_readiness as _preflight_source_readiness_helper,
)
from dnadesign.cruncher.app.portfolio_preflight import (
    _render_prepare_runbook_command as _render_prepare_runbook_command_helper,
)
from dnadesign.cruncher.app.portfolio_preflight import (
    _resolve_source_label as _resolve_source_label_helper,
)
from dnadesign.cruncher.app.portfolio_source_load import (
    _load_analysis_summary as _load_analysis_summary_helper,
)
from dnadesign.cruncher.app.portfolio_source_load import (
    _load_export_elites_windows_and_consensus as _load_export_elites_windows_and_consensus_helper,
)
from dnadesign.cruncher.app.portfolio_source_load import (
    _load_source_rows_with_study_runner,
)
from dnadesign.cruncher.app.portfolio_source_load import (
    _mean_pairwise_hamming_bp as _mean_pairwise_hamming_bp_helper,
)
from dnadesign.cruncher.app.portfolio_studies import (
    _ensure_required_source_studies,
)
from dnadesign.cruncher.portfolio.layout import (
    portfolio_logs_dir,
    portfolio_manifest_path,
    portfolio_meta_dir,
    portfolio_plot_glob,
    portfolio_plots_dir,
    portfolio_status_path,
    portfolio_tables_dir,
    resolve_portfolio_run_dir,
)
from dnadesign.cruncher.portfolio.layout import (
    portfolio_plot_path as _portfolio_plot_path,
)
from dnadesign.cruncher.portfolio.load import load_portfolio_spec
from dnadesign.cruncher.portfolio.manifest import (
    PortfolioManifestV1,
    PortfolioPreparedSource,
    PortfolioSourceRun,
    PortfolioStatusV1,
    load_portfolio_manifest,
    load_portfolio_status,
    utc_now_iso,
    write_portfolio_manifest,
    write_portfolio_status,
)
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource, PortfolioSpec
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

PrepareReadyPolicy = Literal["rerun", "skip"]
PortfolioEventCallback = Callable[[str, dict[str, object]], None]
_preflight_source_readiness = _preflight_source_readiness_helper
_render_prepare_runbook_command = _render_prepare_runbook_command_helper
_resolve_source_label = _resolve_source_label_helper
_PortfolioAggregationState = _PortfolioAggregationState_helper


def _portfolio_id(spec: PortfolioSpec) -> str:
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


def _apply_studies_override(spec: PortfolioSpec, studies_enabled: bool | None) -> PortfolioSpec:
    if studies_enabled is None:
        return spec
    payload = spec.model_dump(mode="python")
    studies_payload = dict(payload.get("studies", {}))
    studies_payload["enabled"] = bool(studies_enabled)
    if not studies_enabled:
        sequence_length_payload = studies_payload.get("sequence_length_table")
        if isinstance(sequence_length_payload, dict):
            updated_sequence_length_payload = dict(sequence_length_payload)
            updated_sequence_length_payload["enabled"] = False
            studies_payload["sequence_length_table"] = updated_sequence_length_payload
    payload["studies"] = studies_payload
    return PortfolioSpec.model_validate(payload)


def run_study(*args, **kwargs):
    from dnadesign.cruncher.app.study_workflow import run_study as _run_study

    return _run_study(*args, **kwargs)


def _emit_event(on_event: PortfolioEventCallback | None, name: str, **payload: object) -> None:
    if on_event is None:
        return
    on_event(name, dict(payload))


def portfolio_preflight_payload(spec_path: Path) -> dict[str, object]:
    resolved_spec = spec_path.expanduser().resolve()
    spec = load_portfolio_spec(resolved_spec)
    readiness = _collect_source_readiness(spec)
    ready_ids = [source_id for source_id, record in readiness.items() if bool(record.get("ready"))]
    unready_ids = [source_id for source_id, record in readiness.items() if not bool(record.get("ready"))]
    return {
        "spec_path": str(resolved_spec),
        "execution_mode": spec.execution.mode,
        "source_count": len(spec.sources),
        "ready_source_ids": ready_ids,
        "unready_source_ids": unready_ids,
        "sources": list(readiness.values()),
    }


_load_analysis_summary = _load_analysis_summary_helper
_load_export_elites_windows_and_consensus = _load_export_elites_windows_and_consensus_helper
_mean_pairwise_hamming_bp = _mean_pairwise_hamming_bp_helper


def _load_source_rows(
    source: PortfolioSource,
    *,
    studies_enabled: bool,
    on_event: PortfolioEventCallback | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    PortfolioSourceRun,
    dict[str, object] | None,
]:
    return _load_source_rows_with_study_runner(
        source,
        studies_enabled=studies_enabled,
        run_study_fn=run_study,
        on_event=on_event,
    )


_materialize_portfolio_outputs = _materialize_portfolio_outputs_helper
_select_portfolio_showcase_elites = _select_portfolio_showcase_elites_helper
_write_tradeoff_plot = _write_tradeoff_plot_helper
portfolio_plot_path = _portfolio_plot_path


def _prepare_source_log_path(run_dir: Path, source_id: str) -> Path:
    return _prepare_source_log_path_helper(run_dir, source_id)


def _prepare_source(
    source: PortfolioSource,
    *,
    readiness: dict[str, dict[str, object]],
    prepare_ready_policy: PrepareReadyPolicy,
    prepare_log_path: Path | None = None,
) -> PortfolioPreparedSource:
    return _prepare_source_helper(
        source,
        readiness=readiness,
        prepare_ready_policy=prepare_ready_policy,
        prepare_log_path=prepare_log_path,
    )


def _new_portfolio_aggregation_state(
    *,
    ensured_study_runs: dict[tuple[str, str], Path],
) -> _PortfolioAggregationState:
    return _new_portfolio_aggregation_state_helper(ensured_study_runs=ensured_study_runs)


def _resolve_workspace_root(spec_path: Path) -> Path:
    workspace_root = spec_path.parent
    if spec_path.parent.name == "configs":
        workspace_root = spec_path.parent.parent
    return workspace_root


def _ensure_portfolio_run_dirs(
    *,
    run_dir: Path,
    workspace_root: Path,
    force_overwrite: bool,
) -> None:
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"Portfolio run path already exists and is not a directory: {run_dir}")
        if force_overwrite:
            shutil.rmtree(run_dir)
        else:
            raise ValueError(f"Portfolio run directory already exists: {run_dir}. Use --force-overwrite.")
    portfolio_meta_dir(run_dir).mkdir(parents=True, exist_ok=True)
    portfolio_logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    portfolio_tables_dir(run_dir).mkdir(parents=True, exist_ok=True)
    portfolio_plots_dir(run_dir).mkdir(parents=True, exist_ok=True)
    ensure_mpl_cache(workspace_root / ".cruncher")


def _create_running_status(*, spec: PortfolioSpec, portfolio_id: str) -> PortfolioStatusV1:
    return PortfolioStatusV1(
        portfolio_name=spec.name,
        portfolio_id=portfolio_id,
        status="running",
        n_sources=len(spec.sources),
        n_selected_elites=0,
        warnings=[],
        started_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )


def _initial_ensured_study_runs(
    *,
    spec: PortfolioSpec,
    on_event: PortfolioEventCallback | None,
) -> dict[tuple[str, str], Path]:
    if spec.execution.mode == "prepare_then_aggregate":
        return {}
    if not spec.studies.enabled:
        return {}
    return _ensure_required_source_studies(spec, run_study_fn=run_study, on_event=on_event)


def _aggregate_source_into_state(
    *,
    source: PortfolioSource,
    spec: PortfolioSpec,
    run_dir: Path,
    state: _PortfolioAggregationState,
    on_event: PortfolioEventCallback | None,
) -> None:
    _aggregate_source_into_state_helper(
        source=source,
        spec=spec,
        run_dir=run_dir,
        state=state,
        on_event=on_event,
        emit_event_fn=_emit_event,
        load_source_rows_fn=_load_source_rows,
        materialize_portfolio_outputs_fn=_materialize_portfolio_outputs,
        run_study_fn=run_study,
    )


def _run_prepare_then_aggregate(
    *,
    spec: PortfolioSpec,
    run_dir: Path,
    state: _PortfolioAggregationState,
    readiness: dict[str, dict[str, object]],
    prepare_ready_policy: PrepareReadyPolicy,
    on_event: PortfolioEventCallback | None,
) -> None:
    _run_prepare_then_aggregate_helper(
        spec=spec,
        run_dir=run_dir,
        state=state,
        readiness=readiness,
        prepare_ready_policy=prepare_ready_policy,
        on_event=on_event,
        emit_event_fn=_emit_event,
        load_source_rows_fn=_load_source_rows,
        materialize_portfolio_outputs_fn=_materialize_portfolio_outputs,
        run_study_fn=run_study,
    )


def _run_aggregate_only(
    *,
    spec: PortfolioSpec,
    run_dir: Path,
    state: _PortfolioAggregationState,
    on_event: PortfolioEventCallback | None,
) -> None:
    _run_aggregate_only_helper(
        spec=spec,
        run_dir=run_dir,
        state=state,
        on_event=on_event,
        emit_event_fn=_emit_event,
        load_source_rows_fn=_load_source_rows,
        materialize_portfolio_outputs_fn=_materialize_portfolio_outputs,
        run_study_fn=run_study,
    )


def _write_portfolio_manifest_and_status(
    *,
    run_dir: Path,
    resolved_spec: Path,
    spec: PortfolioSpec,
    portfolio_id: str,
    state: _PortfolioAggregationState,
    status: PortfolioStatusV1,
) -> None:
    manifest = PortfolioManifestV1(
        portfolio_name=spec.name,
        portfolio_id=portfolio_id,
        spec_path=str(resolved_spec),
        spec_sha256=sha256_path(resolved_spec),
        created_at=utc_now_iso(),
        execution_mode=spec.execution.mode,
        source_runs=state.source_runs,
        prepared_sources=state.prepared_sources,
        table_paths=[str(path.resolve()) for path in state.table_paths],
        plot_paths=[str(path.resolve()) for path in state.plot_paths],
    )
    write_portfolio_manifest(portfolio_manifest_path(run_dir), manifest)
    status.status = "completed"
    status.n_sources = len(state.source_runs)
    status.n_selected_elites = int(len(state.elite_summary_df))
    status.updated_at = utc_now_iso()
    status.finished_at = utc_now_iso()
    write_portfolio_status(portfolio_status_path(run_dir), status)


def _mark_portfolio_failed(*, run_dir: Path, status: PortfolioStatusV1) -> None:
    status.status = "failed"
    status.updated_at = utc_now_iso()
    status.finished_at = utc_now_iso()
    write_portfolio_status(portfolio_status_path(run_dir), status)


def run_portfolio(
    spec_path: Path,
    *,
    force_overwrite: bool = False,
    prepare_ready_policy: PrepareReadyPolicy = "rerun",
    studies_enabled: bool | None = None,
    on_event: PortfolioEventCallback | None = None,
) -> Path:
    if prepare_ready_policy not in {"rerun", "skip"}:
        raise ValueError(f"Invalid prepare_ready_policy: {prepare_ready_policy!r}")
    resolved_spec = spec_path.expanduser().resolve()
    _emit_event(on_event, "portfolio_started", spec_path=str(resolved_spec))
    spec = load_portfolio_spec(resolved_spec)
    spec = _apply_studies_override(spec, studies_enabled)
    readiness = _collect_source_readiness(spec)
    _emit_event(
        on_event,
        "preflight_completed",
        ready_source_ids=[key for key, value in readiness.items() if bool(value.get("ready"))],
        unready_source_ids=[key for key, value in readiness.items() if not bool(value.get("ready"))],
    )
    if spec.execution.mode == "aggregate_only":
        _raise_aggregate_only_preflight(spec, readiness)

    workspace_root = _resolve_workspace_root(resolved_spec)
    portfolio_id = _portfolio_id(spec)
    run_dir = resolve_portfolio_run_dir(workspace_root, spec.name, portfolio_id)
    _ensure_portfolio_run_dirs(run_dir=run_dir, workspace_root=workspace_root, force_overwrite=force_overwrite)

    status = _create_running_status(spec=spec, portfolio_id=portfolio_id)
    write_portfolio_status(portfolio_status_path(run_dir), status)

    try:
        ensured_study_runs = _initial_ensured_study_runs(spec=spec, on_event=on_event)
        state = _new_portfolio_aggregation_state(ensured_study_runs=ensured_study_runs)
        if spec.execution.mode == "prepare_then_aggregate":
            _run_prepare_then_aggregate(
                spec=spec,
                run_dir=run_dir,
                state=state,
                readiness=readiness,
                prepare_ready_policy=prepare_ready_policy,
                on_event=on_event,
            )
        else:
            _run_aggregate_only(
                spec=spec,
                run_dir=run_dir,
                state=state,
                on_event=on_event,
            )

        _write_portfolio_manifest_and_status(
            run_dir=run_dir,
            resolved_spec=resolved_spec,
            spec=spec,
            portfolio_id=portfolio_id,
            state=state,
            status=status,
        )
        _emit_event(on_event, "portfolio_completed", run_dir=str(run_dir))
        return run_dir
    except Exception:
        _mark_portfolio_failed(run_dir=run_dir, status=status)
        _emit_event(on_event, "portfolio_failed")
        raise


def portfolio_show_payload(portfolio_run_dir: Path) -> dict[str, object]:
    run_dir = portfolio_run_dir.expanduser().resolve()
    manifest = load_portfolio_manifest(portfolio_manifest_path(run_dir))
    status = load_portfolio_status(portfolio_status_path(run_dir))
    table_paths = sorted(portfolio_tables_dir(run_dir).glob("table__*"))
    plot_paths = sorted(portfolio_plots_dir(run_dir).glob(portfolio_plot_glob(run_dir)))
    source_runs = [
        {
            "source_id": item.source_id,
            "source_label": item.source_label,
            "source_top_k": int(item.source_top_k),
            "selected_elites": int(item.selected_elites),
            "workspace_name": item.workspace_name,
            "run_name": item.run_name,
        }
        for item in manifest.source_runs
    ]
    return {
        "portfolio_name": manifest.portfolio_name,
        "portfolio_id": manifest.portfolio_id,
        "status": status.status,
        "n_sources": status.n_sources,
        "n_selected_elites": status.n_selected_elites,
        "source_runs": source_runs,
        "manifest_path": str(portfolio_manifest_path(run_dir)),
        "status_path": str(portfolio_status_path(run_dir)),
        "table_paths": [str(path) for path in table_paths],
        "plot_paths": [str(path) for path in plot_paths],
    }
