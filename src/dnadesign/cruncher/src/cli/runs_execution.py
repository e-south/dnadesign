"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/runs_execution.py

Execution helpers for runs CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from rich.table import Table

from dnadesign.cruncher.analysis.layout import current_analysis_id, list_analysis_entries
from dnadesign.cruncher.app.run_service import RunInfo
from dnadesign.cruncher.artifacts.layout import live_metrics_path, manifest_path


@dataclass(frozen=True)
class StaleRun:
    run_name: str
    run_dir: Path
    payload: dict
    reason: str


@dataclass(frozen=True)
class PruneCandidate:
    run: RunInfo
    age_days: float


def analysis_ids(run_dir: Path) -> list[str]:
    entries = list_analysis_entries(run_dir)
    return sorted({entry["id"] for entry in entries if entry.get("id")})


def latest_analysis_id(run_dir: Path) -> str | None:
    try:
        return current_analysis_id(run_dir)
    except (FileNotFoundError, ValueError):
        return None


def artifact_counts(artifacts: list[dict]) -> list[dict[str, str | int]]:
    counts: dict[tuple[str, str], int] = {}
    for item in artifacts:
        stage = str(item.get("stage") or "unknown")
        kind = str(item.get("type") or "unknown")
        key = (stage, kind)
        counts[key] = counts.get(key, 0) + 1
    payload = [{"stage": stage, "type": kind, "count": count} for (stage, kind), count in sorted(counts.items())]
    return payload


def tail_lines(path: Path, *, max_lines: int, max_bytes: int = 65536) -> list[str]:
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


def load_live_metrics(run_dir: Path, *, max_points: int) -> list[dict]:
    metrics_path = live_metrics_path(run_dir)
    if not metrics_path.exists():
        return []
    lines = tail_lines(metrics_path, max_lines=max_points)
    payloads: list[dict] = []
    for line in lines:
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return payloads


def sparkline(values: list[float], *, width: int = 32) -> str:
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


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def run_timestamp(run: object) -> datetime:
    created_at = parse_timestamp(getattr(run, "created_at", None))
    if created_at is not None:
        return created_at
    run_dir = getattr(run, "run_dir", None)
    if isinstance(run_dir, Path) and run_dir.exists():
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
    raise FileNotFoundError(
        f"Run '{getattr(run, 'name', '-')}' has no created_at and run directory is missing: {run_dir}"
    )


def render_index_issue_table(issues: list[object], *, title: str) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("Run")
    table.add_column("Stage")
    table.add_column("Issue")
    table.add_column("Run directory")
    for issue in issues:
        table.add_row(
            str(getattr(issue, "run_name", "-")),
            str(getattr(issue, "stage", "-")),
            str(getattr(issue, "reason", "-")),
            str(getattr(issue, "run_dir", "-")),
        )
    return table


def plot_live_metrics(history: list[dict], out_path: Path) -> None:
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


def select_best_run(
    runs: Sequence[RunInfo],
    *,
    load_status: Callable[[Path], dict | None],
) -> tuple[RunInfo, float] | None:
    best_run: RunInfo | None = None
    best_score: float | None = None
    for run in runs:
        run_status = run.status
        score = run.best_score
        if score is None or run_status is None:
            status_payload = load_status(run.run_dir)
            if status_payload is not None:
                if score is None:
                    score = status_payload.get("best_score")
                if run_status is None:
                    run_status = status_payload.get("status")
        if str(run_status or "").lower() == "failed":
            continue
        if score is None:
            continue
        score_float = float(score)
        if best_score is None or score_float > best_score:
            best_score = score_float
            best_run = run
    if best_run is None or best_score is None:
        return None
    return best_run, best_score


def collect_stale_runs(
    runs: Sequence[RunInfo],
    *,
    older_than_hours: float,
    now: datetime,
    load_status: Callable[[Path], dict | None],
) -> list[StaleRun]:
    stale_seconds = older_than_hours * 3600
    stale_runs: list[StaleRun] = []
    for run in runs:
        if run.status != "running":
            continue
        status_payload = load_status(run.run_dir)
        last_update = None
        if status_payload is not None:
            last_update = parse_timestamp(
                status_payload.get("updated_at")
                or status_payload.get("finished_at")
                or status_payload.get("started_at")
            )
        if last_update is None:
            reason = "missing status file" if status_payload is None else "missing timestamps"
            is_stale = True
        else:
            age = now - last_update
            is_stale = age.total_seconds() >= stale_seconds
            reason = f"no updates for {age.total_seconds() / 3600:.1f}h"
        if not is_stale:
            continue
        payload = status_payload or {"stage": run.stage}
        payload["run_dir"] = str(run.run_dir.resolve())
        payload.setdefault("started_at", run.created_at or now.isoformat())
        stale_runs.append(
            StaleRun(
                run_name=run.name,
                run_dir=run.run_dir,
                payload=payload,
                reason=reason,
            )
        )
    return stale_runs


def stale_runs_table(stale_runs: Sequence[StaleRun], *, title: str) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("Run")
    table.add_column("Stage")
    table.add_column("Reason")
    for stale_run in stale_runs:
        table.add_row(stale_run.run_name, str(stale_run.payload.get("stage") or "-"), stale_run.reason)
    return table


def select_prune_candidates(
    runs: Sequence[RunInfo],
    *,
    keep_latest: int,
    older_than_days: float,
    now: datetime,
) -> list[PruneCandidate]:
    runs_sorted = sorted(
        runs,
        key=lambda run: (run_timestamp(run), run.name),
        reverse=True,
    )
    keep_names = {run.name for run in runs_sorted[:keep_latest]}

    candidates: list[PruneCandidate] = []
    for run in runs_sorted:
        if run.name in keep_names:
            continue
        if run.status == "running":
            continue
        run_dir = run.run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Indexed run directory is missing: {run_dir}")
        if not manifest_path(run_dir).exists():
            raise FileNotFoundError(f"Indexed run is missing run_manifest.json: {run_dir}")
        age_days = (now - run_timestamp(run)).total_seconds() / 86400.0
        if older_than_days > 0 and age_days < older_than_days:
            continue
        candidates.append(PruneCandidate(run=run, age_days=age_days))
    return candidates


def prune_candidates_table(candidates: Sequence[PruneCandidate], *, stage: str) -> Table:
    table = Table(title=f"Run prune candidates ({stage})", header_style="bold")
    table.add_column("Run")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Age (days)", justify="right")
    for candidate in candidates:
        run = candidate.run
        table.add_row(run.name, run.status or "-", run.created_at or "-", f"{candidate.age_days:.1f}")
    return table


def render_watch_table(
    *,
    run_name: str,
    payload: dict,
    run_dir: Path,
    metrics: bool,
    metric_points: int,
    metric_width: int,
) -> Table:
    table = Table(title=f"Run status: {run_name}", header_style="bold")
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
    if "current_score" in payload:
        _add_row("current_score", "current_score")
    _add_row("best_score", "best_score")
    _add_row("best_chain", "best_chain")
    _add_row("best_draw", "best_draw")
    _add_row("acceptance_rate", "acceptance_rate")
    _add_row("updated_at", "updated_at", default=payload.get("started_at", "-"))

    if metrics:
        history = load_live_metrics(run_dir, max_points=metric_points)
        if history:
            best_vals = [item.get("best_score") for item in history if item.get("best_score") is not None]
            current_vals = [item.get("current_score") for item in history if item.get("current_score") is not None]
            if best_vals:
                table.add_row("best_score_trend", sparkline(best_vals, width=metric_width))
            if current_vals:
                table.add_row("current_score_trend", sparkline(current_vals, width=metric_width))
            table.add_row("metric_points", str(len(history)))
        else:
            table.add_row("live_metrics", "metrics.jsonl not found")

    return table


def validate_watch_options(*, metric_points: int, metric_width: int, plot_every: int) -> None:
    if metric_points < 1:
        raise ValueError("Error: --metric-points must be >= 1.")
    if metric_width < 1:
        raise ValueError("Error: --metric-width must be >= 1.")
    if plot_every < 1:
        raise ValueError("Error: --plot-every must be >= 1.")
