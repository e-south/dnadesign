"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/runs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.run_service import (
    get_run,
    list_runs,
    load_run_status,
    rebuild_run_index,
)
from dnadesign.cruncher.utils.analysis_layout import (
    current_analysis_id,
    list_analysis_entries,
)
from dnadesign.cruncher.utils.artifacts import normalize_artifacts
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from rich.console import Console
from rich.live import Live
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="List, inspect, or watch past run artifacts.")
console = Console()


def _analysis_ids(run_dir: Path) -> list[str]:
    entries = list_analysis_entries(run_dir)
    return sorted({entry["id"] for entry in entries if entry.get("id")})


def _latest_analysis_id(run_dir: Path) -> str | None:
    try:
        return current_analysis_id(run_dir)
    except (FileNotFoundError, ValueError):
        return None


def _artifact_counts(artifacts: list[dict]) -> list[dict[str, str | int]]:
    counts: dict[tuple[str, str], int] = {}
    for item in artifacts:
        stage = str(item.get("stage") or "unknown")
        kind = str(item.get("type") or "unknown")
        key = (stage, kind)
        counts[key] = counts.get(key, 0) + 1
    payload = [{"stage": stage, "type": kind, "count": count} for (stage, kind), count in sorted(counts.items())]
    return payload


def _tail_lines(path: Path, *, max_lines: int, max_bytes: int = 65536) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        block = b""
        while pos > 0 and block.count(b"\n") <= max_lines:
            read_size = min(max_bytes, pos)
            pos -= read_size
            fh.seek(pos)
            block = fh.read(read_size) + block
            if pos == 0:
                break
    lines = block.splitlines()[-max_lines:]
    return [line.decode("utf-8", errors="ignore") for line in lines if line.strip()]


def _load_live_metrics(run_dir: Path, *, max_points: int) -> list[dict]:
    metrics_path = run_dir / "live_metrics.jsonl"
    if not metrics_path.exists():
        return []
    lines = _tail_lines(metrics_path, max_lines=max_points)
    payloads: list[dict] = []
    for line in lines:
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return payloads


def _sparkline(values: list[float], *, width: int = 32) -> str:
    if not values:
        return "-"
    charset = " .:-=+*#%@"
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return "-"
    if len(cleaned) > width:
        step = max(len(cleaned) / width, 1.0)
        cleaned = [cleaned[int(i * step)] for i in range(width)]
    vmin = min(cleaned)
    vmax = max(cleaned)
    if vmax == vmin:
        return charset[-1] * len(cleaned)
    bins = len(charset) - 1
    chars = []
    for value in cleaned:
        idx = int((value - vmin) / (vmax - vmin) * bins)
        idx = max(0, min(bins, idx))
        chars.append(charset[idx])
    return "".join(chars)


def _plot_live_metrics(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    best_vals = [item.get("best_score") for item in history if item.get("best_score") is not None]
    current_vals = [item.get("current_score") for item in history if item.get("current_score") is not None]
    if not best_vals and not current_vals:
        return
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    if best_vals:
        ax.plot(best_vals, label="best_score", color="steelblue")
    if current_vals:
        ax.plot(current_vals, label="current_score", color="darkorange", alpha=0.7)
    ax.set_xlabel("update")
    ax.set_ylabel("score")
    ax.set_title("Live score trend")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@app.command("list", help="List run artifacts found in the results directory.")
def list_runs_cmd(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage (parse, sample, analyze, report)."),
    full: bool = typer.Option(False, "--full", help="Show full run names without truncation."),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    runs = list_runs(cfg, config_path, stage=stage)
    if not runs:
        console.print("No runs found.")
        console.print("Hint: run cruncher sample <config> or cruncher parse <config> to create a run.")
        raise typer.Exit(code=1)
    if json_output:
        payload = [
            {
                "name": run.name,
                "stage": run.stage,
                "status": run.status,
                "created_at": run.created_at,
                "motif_count": run.motif_count,
                "pwm_source": run.pwm_source,
                "regulator_set": run.regulator_set,
                "run_dir": str(run.run_dir),
                "artifacts": run.artifacts,
            }
            for run in runs
        ]
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title="Runs", header_style="bold")
    table.add_column("Name", overflow="fold" if full else "ellipsis")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Motifs")
    table.add_column("Regulator set")
    table.add_column("PWM source")
    for run in runs:
        reg_label = "-"
        if run.regulator_set:
            idx = run.regulator_set.get("index")
            tfs = run.regulator_set.get("tfs") or []
            reg_label = f"set{idx}:" + ",".join(tfs) if idx else ",".join(tfs)
        table.add_row(
            run.name,
            run.stage,
            run.status or "-",
            run.created_at or "-",
            str(run.motif_count),
            reg_label,
            run.pwm_source or "-",
        )
    console.print(table)


@app.command("show", help="Show metadata and artifacts for a specific run.")
def show_run_cmd(
    args: list[str] = typer.Argument(
        None,
        help="Run name (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path, run_name = parse_config_and_value(
            args,
            config_option,
            value_label="RUN",
            command_hint="cruncher runs show <run_name>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        run = get_run(cfg, config_path, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list (or add --config <path>) to see available runs.")
        raise typer.Exit(code=1)
    if json_output:
        artifacts = normalize_artifacts(run.artifacts)
        analysis_ids = _analysis_ids(run.run_dir)
        latest_analysis = _latest_analysis_id(run.run_dir)
        artifact_counts = _artifact_counts(artifacts)
        typer.echo(
            json.dumps(
                {
                    "name": run.name,
                    "stage": run.stage,
                    "status": run.status,
                    "created_at": run.created_at,
                    "motif_count": run.motif_count,
                    "pwm_source": run.pwm_source,
                    "regulator_set": run.regulator_set,
                    "run_dir": str(run.run_dir),
                    "artifacts": run.artifacts,
                    "analysis_ids": analysis_ids,
                    "latest_analysis": latest_analysis,
                    "artifact_counts": artifact_counts,
                },
                indent=2,
            )
        )
        return
    console.print(f"run: {run.name}")
    console.print(f"stage: {run.stage}")
    console.print(f"status: {run.status or '-'}")
    console.print(f"created_at: {run.created_at or '-'}")
    console.print(f"motif_count: {run.motif_count}")
    if run.regulator_set:
        console.print(f"regulator_set: {run.regulator_set}")
    console.print(f"pwm_source: {run.pwm_source or '-'}")
    console.print(f"run_dir: {run.run_dir}")
    artifacts = normalize_artifacts(run.artifacts)
    analysis_ids = _analysis_ids(run.run_dir)
    latest_analysis = _latest_analysis_id(run.run_dir)
    if analysis_ids:
        console.print(f"analysis_ids: {', '.join(analysis_ids)}")
    if latest_analysis:
        console.print(f"latest_analysis: {latest_analysis}")
        notebook_path = run.run_dir / "analysis" / "notebooks" / "run_overview.py"
        if notebook_path.exists():
            console.print(f"notebook: {notebook_path}")
    if artifacts:
        console.print("artifacts:")
        table = Table(header_style="bold")
        table.add_column("Stage")
        table.add_column("Type")
        table.add_column("Label")
        table.add_column("Path")
        for item in artifacts:
            table.add_row(
                str(item.get("stage") or "-"),
                str(item.get("type") or "-"),
                str(item.get("label") or "-"),
                str(item.get("path") or "-"),
            )
        console.print(table)
        counts = _artifact_counts(artifacts)
        if counts:
            count_table = Table(title="Artifact counts", header_style="bold")
            count_table.add_column("Stage")
            count_table.add_column("Type")
            count_table.add_column("Count")
            for item in counts:
                count_table.add_row(str(item["stage"]), str(item["type"]), str(item["count"]))
            console.print(count_table)


@app.command("latest", help="Print the most recent run name (optionally filtered by stage).")
def latest_run_cmd(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage (parse, sample, analyze, report)."),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    runs = list_runs(cfg, config_path, stage=stage)
    if not runs:
        console.print("No runs found.")
        console.print("Hint: run cruncher sample <config> or cruncher parse <config> to create a run.")
        raise typer.Exit(code=1)
    run = runs[0]
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "name": run.name,
                    "stage": run.stage,
                    "status": run.status,
                    "created_at": run.created_at,
                    "motif_count": run.motif_count,
                    "pwm_source": run.pwm_source,
                    "regulator_set": run.regulator_set,
                    "run_dir": str(run.run_dir),
                    "artifacts": run.artifacts,
                },
                indent=2,
            )
        )
        return
    console.print(run.name)


@app.command("rebuild-index", help="Rebuild the run index from run_manifest.json files.")
def rebuild_index_cmd(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    index_path = rebuild_run_index(cfg, config_path)
    console.print(f"Rebuilt run index → {index_path}")


@app.command("watch", help="Tail run_status.json for a live progress snapshot.")
def watch_run_cmd(
    args: list[str] = typer.Argument(
        None,
        help="Run name (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    interval: float = typer.Option(1.0, "--interval", help="Polling interval in seconds."),
    metrics: bool = typer.Option(True, "--metrics/--no-metrics", help="Show live metrics trends if available."),
    metric_points: int = typer.Option(40, "--metric-points", help="Number of recent metric points to render."),
    metric_width: int = typer.Option(32, "--metric-width", help="Width of ASCII sparklines."),
    plot: bool = typer.Option(False, "--plot/--no-plot", help="Write a live metrics plot during watch."),
    plot_path: Path | None = typer.Option(
        None,
        "--plot-path",
        help="Optional path to write a live metrics plot (PNG) on refresh.",
    ),
    plot_every: int = typer.Option(5, "--plot-every", help="Write live plot every N refreshes."),
) -> None:
    try:
        config_path, run_name = parse_config_and_value(
            args,
            config_option,
            value_label="RUN",
            command_hint="cruncher runs watch <run_name>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if plot or plot_path is not None:
        ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
    try:
        run = get_run(cfg, config_path, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list (or add --config <path>) to see available runs.")
        raise typer.Exit(code=1)
    status = load_run_status(run.run_dir)
    if status is None:
        console.print(f"No run_status.json found for run '{run_name}'.")
        console.print("Hint: watch is only available for active runs writing run_status.json.")
        raise typer.Exit(code=1)
    if metric_points < 1:
        console.print("Error: --metric-points must be >= 1.")
        raise typer.Exit(code=1)
    if metric_width < 1:
        console.print("Error: --metric-width must be >= 1.")
        raise typer.Exit(code=1)
    if plot_every < 1:
        console.print("Error: --plot-every must be >= 1.")
        raise typer.Exit(code=1)

    plot_target = plot_path or (run.run_dir / "live" / "live_metrics.png")

    def _render(payload: dict) -> Table:
        table = Table(title=f"Run status: {run.name}", header_style="bold")
        table.add_column("Field")
        table.add_column("Value")

        def _add_row(label: str, key: str, default: str = "-") -> None:
            value = payload.get(key, default)
            if value is None:
                value = default
            table.add_row(label, str(value))

        _add_row("stage", "stage")
        _add_row("status", "status")
        _add_row("status_message", "status_message")
        _add_row("phase", "phase")
        _add_row("chain", "chain")
        _add_row("step", "step")
        _add_row("total", "total")
        _add_row("progress_pct", "progress_pct")
        if "beta" in payload:
            _add_row("beta", "beta")
        if "beta_min" in payload or "beta_max" in payload:
            _add_row("beta_min", "beta_min")
            _add_row("beta_max", "beta_max")
        if "swap_rate" in payload:
            _add_row("swap_rate", "swap_rate")
        if "current_score" in payload:
            _add_row("current_score", "current_score")
        _add_row("best_score", "best_score")
        _add_row("best_chain", "best_chain")
        _add_row("best_draw", "best_draw")
        _add_row("acceptance_rate", "acceptance_rate")
        _add_row("updated_at", "updated_at", default=payload.get("started_at", "-"))
        if metrics:
            history = _load_live_metrics(run.run_dir, max_points=metric_points)
            if history:
                best_vals = [item.get("best_score") for item in history if item.get("best_score") is not None]
                current_vals = [item.get("current_score") for item in history if item.get("current_score") is not None]
                if best_vals:
                    table.add_row("best_score_trend", _sparkline(best_vals, width=metric_width))
                if current_vals:
                    table.add_row(
                        "current_score_trend",
                        _sparkline(current_vals, width=metric_width),
                    )
                table.add_row("metric_points", str(len(history)))
            else:
                table.add_row("live_metrics", "live_metrics.jsonl not found")
        return table

    tick = 0
    with Live(_render(status), refresh_per_second=4, console=console) as live:
        while True:
            status = load_run_status(run.run_dir)
            if status is None:
                console.print("run_status.json missing.")
                break
            live.update(_render(status))
            if plot or plot_path is not None:
                if tick % plot_every == 0:
                    history = _load_live_metrics(run.run_dir, max_points=metric_points)
                    _plot_live_metrics(history, plot_target)
            if status.get("status") in {"completed", "failed"}:
                break
            time.sleep(interval)
            tick += 1
