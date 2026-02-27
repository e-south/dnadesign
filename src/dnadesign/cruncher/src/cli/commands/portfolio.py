"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/portfolio.py

CLI entrypoints for cross-workspace Portfolio aggregation workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from dnadesign.cruncher.app.progress import progress_output_enabled
from dnadesign.cruncher.cli.config_resolver import resolve_invocation_cwd
from dnadesign.cruncher.cli.paths import render_path

app = typer.Typer(no_args_is_help=True, help="Aggregate selected workspace runs into one handoff portfolio.")
console = Console()
PrepareReadyOption = Literal["prompt", "rerun", "skip"]


def run_portfolio(*args, **kwargs):
    from dnadesign.cruncher.app.portfolio_workflow import run_portfolio as _run_portfolio

    return _run_portfolio(*args, **kwargs)


def portfolio_preflight_payload(*args, **kwargs):
    from dnadesign.cruncher.app.portfolio_workflow import portfolio_preflight_payload as _portfolio_preflight_payload

    return _portfolio_preflight_payload(*args, **kwargs)


def portfolio_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.portfolio_workflow import portfolio_show_payload as _portfolio_show_payload

    return _portfolio_show_payload(*args, **kwargs)


def _progress_enabled() -> bool:
    return bool(progress_output_enabled() and console.is_terminal)


def _resolve_cli_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (resolve_invocation_cwd() / expanded).resolve()


def _run_with_noninteractive_env(fn):
    existing = os.environ.get("CRUNCHER_NONINTERACTIVE")
    os.environ["CRUNCHER_NONINTERACTIVE"] = "1"
    try:
        return fn()
    finally:
        if existing is None:
            os.environ.pop("CRUNCHER_NONINTERACTIVE", None)
        else:
            os.environ["CRUNCHER_NONINTERACTIVE"] = existing


def _print_source_run_counts(payload: dict[str, object]) -> None:
    source_runs = payload.get("source_runs")
    if not isinstance(source_runs, list):
        return
    for row in source_runs:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id", "")).strip()
        selected = row.get("selected_elites")
        top_k = row.get("source_top_k")
        if source_id and isinstance(selected, int) and isinstance(top_k, int):
            console.print(f"  source: {source_id} elites={selected}/{top_k}")


def _as_int(payload: dict[str, object], key: str, *, default: int = 0) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _study_name_from_payload(payload: dict[str, object]) -> str:
    explicit = str(payload.get("study_name", "")).strip()
    if explicit:
        return explicit
    spec_text = str(payload.get("study_spec", "")).strip()
    if spec_text:
        stem = Path(spec_text).stem
        if stem.endswith(".study"):
            return stem[: -len(".study")]
        return stem
    return "study"


def _study_source_from_payload(payload: dict[str, object]) -> str:
    source_id = str(payload.get("source_id", "")).strip()
    return source_id or "unknown_source"


def _active_trial_ids(payload: dict[str, object]) -> list[str]:
    raw = payload.get("active_trial_ids")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _format_active_trials(active_trial_ids: list[str], *, max_items: int = 3) -> str:
    if not active_trial_ids:
        return "-"
    preview = active_trial_ids[:max_items]
    suffix = ",..." if len(active_trial_ids) > max_items else ""
    return ",".join(preview) + suffix


@app.command("run", help="Execute a Portfolio spec and write aggregate handoff outputs.")
def run_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/<name>.portfolio.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Delete and recreate an existing deterministic portfolio run directory.",
    ),
    prepare_ready: PrepareReadyOption = typer.Option(
        "prompt",
        "--prepare-ready",
        help=("When execution.mode=prepare_then_aggregate and some sources are already ready: prompt, rerun, or skip."),
    ),
) -> None:
    resolved_spec = _resolve_cli_path(spec)
    try:
        preflight = portfolio_preflight_payload(resolved_spec)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    execution_mode = str(preflight.get("execution_mode", "aggregate_only"))
    ready_source_ids = [str(item) for item in preflight.get("ready_source_ids", [])]
    source_count = int(preflight.get("source_count", 0))
    prepare_ready_policy: Literal["rerun", "skip"] = "rerun"
    if execution_mode == "prepare_then_aggregate" and ready_source_ids:
        if prepare_ready == "prompt":
            if not sys.stdin.isatty():
                console.print(
                    "Error: ready sources detected but interactive prompt is unavailable. "
                    "Re-run with --prepare-ready skip|rerun."
                )
                raise typer.Exit(code=1)
            preview = ", ".join(ready_source_ids[:8])
            if len(ready_source_ids) > 8:
                preview = f"{preview}, ..."
            console.print(
                f"Ready sources detected ({len(ready_source_ids)}): {preview}. "
                "Skip these and prepare only missing sources?"
            )
            skip_ready = typer.confirm("Choose skip for ready sources", default=True)
            prepare_ready_policy = "skip" if skip_ready else "rerun"
        else:
            prepare_ready_policy = "skip" if prepare_ready == "skip" else "rerun"

    study_progress_snapshots: dict[tuple[str, str], tuple[int, int, int, int, int, int, tuple[str, ...]]] = {}

    def _render_study_progress_line(payload: dict[str, object]) -> str | None:
        source_id = _study_source_from_payload(payload)
        study_name = _study_name_from_payload(payload)
        worker_count = max(_as_int(payload, "worker_count", default=1), 1)
        running_runs = max(_as_int(payload, "running_runs", default=0), 0)
        completed_runs = max(_as_int(payload, "completed_runs", default=0), 0)
        total_runs = max(_as_int(payload, "total_runs", default=0), 0)
        error_runs = max(_as_int(payload, "error_runs", default=0), 0)
        queued_runs = max(_as_int(payload, "queued_runs", default=0), 0)
        active_trials = _active_trial_ids(payload)
        snapshot = (
            running_runs,
            worker_count,
            completed_runs,
            total_runs,
            error_runs,
            queued_runs,
            tuple(active_trials),
        )
        key = (source_id, study_name)
        if study_progress_snapshots.get(key) == snapshot:
            return None
        study_progress_snapshots[key] = snapshot
        return (
            "Study progress: "
            f"source={source_id} "
            f"study={study_name} "
            f"workers={running_runs}/{worker_count} "
            f"done={completed_runs}/{total_runs} "
            f"error={error_runs} "
            f"queued={queued_runs} "
            f"active={_format_active_trials(active_trials)}"
        )

    def _render_study_ensure_line(name: str, payload: dict[str, object]) -> str | None:
        source_id = _study_source_from_payload(payload)
        study_name = _study_name_from_payload(payload)
        if name == "study_ensure_started":
            mode = "resume" if bool(payload.get("resume")) else "fresh"
            return f"Study ensure: source={source_id} study={study_name} mode={mode}"
        if name == "study_ensure_completed":
            status = str(payload.get("study_status", "completed"))
            return f"Study ensure complete: source={source_id} study={study_name} status={status}"
        if name == "study_ensure_ready":
            status = str(payload.get("study_status", "completed"))
            return f"Study ready: source={source_id} study={study_name} status={status}"
        return None

    def _run_with_progress(force_flag: bool) -> Path:
        def _render_source_label(source_id: str) -> str:
            source_id = source_id.strip()
            if len(source_id) <= 56:
                return source_id
            return source_id[:53] + "..."

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        )
        prepare_total = source_count
        prepare_task = None
        prepare_done = 0
        if execution_mode == "prepare_then_aggregate":
            prepare_task = progress.add_task(f"Prepare sources (0/{prepare_total})", total=max(prepare_total, 1))
        aggregate_done = 0
        aggregate_total = source_count
        aggregate_task = progress.add_task(
            f"Aggregate sources (0/{aggregate_total})",
            total=max(source_count, 1),
            visible=execution_mode != "prepare_then_aggregate",
        )

        def _on_event(name: str, payload: dict[str, object]) -> None:
            nonlocal prepare_done, aggregate_done
            source_id = str(payload.get("source_id", "")).strip()
            source_label = _render_source_label(source_id)
            if name == "prepare_source_started" and prepare_task is not None and source_label:
                progress.update(
                    prepare_task,
                    description=f"Prepare source {prepare_done + 1}/{prepare_total}: {source_label}",
                )
            elif name == "prepare_source_skipped" and prepare_task is not None:
                prepare_done += 1
                progress.advance(prepare_task, 1)
                if source_label:
                    progress.update(
                        prepare_task,
                        description=f"Prepare sources ({prepare_done}/{prepare_total}) [skip {source_label}]",
                    )
                else:
                    progress.update(prepare_task, description=f"Prepare sources ({prepare_done}/{prepare_total})")
            elif name == "aggregate_source_started" and source_label:
                progress.update(
                    aggregate_task,
                    description=f"Aggregate source {aggregate_done + 1}/{aggregate_total}: {source_label}",
                )
            elif name == "aggregate_phase_started":
                progress.update(
                    aggregate_task,
                    visible=True,
                    description=f"Aggregate sources ({aggregate_done}/{aggregate_total})",
                )
            elif name == "prepare_phase_completed" and prepare_task is not None:
                progress.update(prepare_task, visible=False)
            if name == "prepare_source_completed" and prepare_task is not None:
                prepare_done += 1
                progress.advance(prepare_task, 1)
                progress.update(prepare_task, description=f"Prepare sources ({prepare_done}/{prepare_total})")
            elif name == "aggregate_source_completed":
                aggregate_done += 1
                progress.advance(aggregate_task, 1)
                progress.update(aggregate_task, description=f"Aggregate sources ({aggregate_done}/{aggregate_total})")
            elif name in {"study_ensure_started", "study_ensure_completed", "study_ensure_ready"}:
                line = _render_study_ensure_line(name, payload)
                if line:
                    progress.console.print(line, soft_wrap=True)
            elif name == "study_trial_progress":
                line = _render_study_progress_line(payload)
                if line:
                    progress.console.print(line, soft_wrap=True)

        with progress:
            return _run_with_noninteractive_env(
                lambda: run_portfolio(
                    resolved_spec,
                    force_overwrite=force_flag,
                    prepare_ready_policy=prepare_ready_policy,
                    on_event=_on_event,
                )
            )

    def _run_without_progress(force_flag: bool) -> Path:
        def _on_event(name: str, payload: dict[str, object]) -> None:
            if name == "prepare_phase_started":
                total = int(payload.get("source_count", 0))
                console.print(f"Prepare phase: {total} source(s).", soft_wrap=True)
            elif name == "prepare_phase_completed":
                total = int(payload.get("prepared_count", 0))
                console.print(f"Prepare phase complete: {total} source(s).", soft_wrap=True)
            elif name == "aggregate_phase_started":
                total = int(payload.get("source_count", 0))
                console.print(f"Aggregate phase: {total} source(s).", soft_wrap=True)
            elif name == "aggregate_phase_completed":
                total = int(payload.get("source_count", 0))
                console.print(f"Aggregate phase complete: {total} source(s).", soft_wrap=True)
            elif name == "aggregate_source_outputs_updated":
                source_id = str(payload.get("source_id", ""))
                completed = int(payload.get("completed_sources", 0))
                table_count = int(payload.get("table_count", 0))
                plot_count = int(payload.get("plot_count", 0))
                if source_id:
                    console.print(
                        "Aggregate source complete: "
                        f"{source_id} ({completed} done, tables={table_count}, plots={plot_count}).",
                        soft_wrap=True,
                    )
            elif name in {"study_ensure_started", "study_ensure_completed", "study_ensure_ready"}:
                line = _render_study_ensure_line(name, payload)
                if line:
                    console.print(line, soft_wrap=True)
            elif name == "study_trial_progress":
                line = _render_study_progress_line(payload)
                if line:
                    console.print(line, soft_wrap=True)

        return run_portfolio(
            resolved_spec,
            force_overwrite=force_flag,
            prepare_ready_policy=prepare_ready_policy,
            on_event=_on_event,
        )

    run_portfolio_call = _run_with_progress if _progress_enabled() else _run_without_progress

    try:
        run_dir = run_portfolio_call(force_overwrite)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        message = str(exc)
        if (
            not force_overwrite
            and "already exists" in message
            and sys.stdin.isatty()
            and typer.confirm("Portfolio aggregate already exists. Overwrite and rerun?", default=False)
        ):
            run_dir = run_portfolio_call(True)
        else:
            console.print(f"Error: {message}")
            raise typer.Exit(code=1) from exc

    payload = portfolio_show_payload(run_dir)
    console.print(f"Portfolio outputs -> {render_path(run_dir)}", soft_wrap=True)
    console.print(f"  status: {payload['status']}")
    console.print(f"  sources: {payload['n_sources']}")
    console.print(f"  selected elites: {payload['n_selected_elites']}")
    _print_source_run_counts(payload)
    console.print(f"  manifest: {render_path(Path(str(payload['manifest_path'])))}", soft_wrap=True)


@app.command("show", help="Print Portfolio run status and key artifact paths.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a Portfolio run directory."),
) -> None:
    resolved_run = _resolve_cli_path(run)
    try:
        payload = portfolio_show_payload(resolved_run)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Portfolio -> {payload['portfolio_name']} ({payload['portfolio_id']})")
    console.print(f"  status: {payload['status']}")
    console.print(f"  sources: {payload['n_sources']}")
    console.print(f"  selected elites: {payload['n_selected_elites']}")
    _print_source_run_counts(payload)
    console.print(f"  manifest: {render_path(Path(str(payload['manifest_path'])))}", soft_wrap=True)
    console.print(f"  status file: {render_path(Path(str(payload['status_path'])))}", soft_wrap=True)
    for path in payload["table_paths"]:
        console.print(f"  table: {render_path(Path(str(path)))}", soft_wrap=True)
    for path in payload["plot_paths"]:
        console.print(f"  plot: {render_path(Path(str(path)))}", soft_wrap=True)
