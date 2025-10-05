"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/_console.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Iterable, List

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import install as rich_tb

theme = Theme(
    {
        "ok": "green",
        "warn": "yellow",
        "bad": "red",
        "muted": "dim",
        "accent": "bright_cyan",
        "kv": "bold white",
        "title": "bold bright_cyan",
    }
)
console = Console(theme=theme)


def setup_console_logging(level: str = "INFO", json_logs: bool = False) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):  # idempotent re-init
        root.removeHandler(h)
    root.setLevel(level.upper())

    if json_logs:

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                }
                return json.dumps(payload)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level.upper())
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    else:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            rich_tracebacks=False,
            markup=True,
        )
        handler.setLevel(level.upper())
        root.addHandler(handler)


def rich_tracebacks(enabled: bool = True) -> None:
    if enabled:
        rich_tb(show_locals=False)


def _rounded_table(title: str) -> Table:
    return Table(
        title=Text(title, style="title"),
        show_header=True,
        header_style="bold white",
        border_style="accent",
        row_styles=["", "muted"],
        box=box.ROUNDED,
    )


def render_config_summary(model, jobs: Iterable[Any]) -> None:
    t1 = _rounded_table("Model")
    t1.add_column("id")
    t1.add_column("device")
    t1.add_column("precision")
    t1.add_column("alphabet")
    t1.add_column("batch_size")
    t1.add_row(
        model.id,
        model.device,
        model.precision,
        model.alphabet,
        str(model.batch_size or "—"),
    )
    console.print(t1)

    t2 = _rounded_table("Jobs")
    t2.add_column("id")
    t2.add_column("op")
    t2.add_column("source")
    t2.add_column("details")
    for j in jobs:
        src = j.ingest.source
        if src == "usr":
            details = f"dataset={j.ingest.dataset} field={j.ingest.field}"
        elif src in {"pt_file", "records"}:
            details = f"field={j.ingest.field}"
        else:
            details = "—"
        t2.add_row(j.id, j.operation, src, details)
    console.print(t2)


def _style_for_type(value: object) -> str:
    name = type(value).__name__ if value is not None else "None"
    name_l = name.lower()
    if name_l in {"float", "int"}:
        return "ok"
    if name_l in {"list", "tuple"}:
        return "cyan"
    if name_l in {"dict"}:
        return "yellow"
    if name_l in {"ndarray", "tensor"}:
        return "magenta"
    if name_l in {"nonetype"}:
        return "muted"
    return "white"


def render_outputs_summary(job, columnar: Dict[str, List[object]]) -> None:
    t = _rounded_table(f"Outputs for job '{job.id}'")
    t.add_column("out_id")
    t.add_column("n")
    t.add_column("example type")
    for out_id, col in columnar.items():
        example = col[0] if col else None
        typ = type(example).__name__ if example is not None else "None"
        t.add_row(out_id, str(len(col)), Text(typ, style=_style_for_type(example)))
    console.print(t)


def render_outputs_spec_table(outputs: List[Dict[str, Any]]) -> None:
    t = _rounded_table("Output Specs")
    t.add_column("id")
    t.add_column("fn")
    t.add_column("format")
    t.add_column("params")
    for o in outputs:
        params_str = json.dumps(
            o.get("params", {}), separators=(",", ":"), ensure_ascii=False
        )
        t.add_row(
            o["id"], o["fn"], o.get("format", "—"), Text(params_str, style="muted")
        )
    console.print(t)


def render_adapters_table(models: Dict[str, type]) -> None:
    t = _rounded_table("Registered Models")
    t.add_column("model_id")
    t.add_column("adapter")
    t.add_column("capabilities")
    for model_id, cls in models.items():
        caps = getattr(cls, "supports", {})
        cap_list = [k for k, v in caps.items() if v]
        caps_str = ", ".join(cap_list) if cap_list else "—"
        t.add_row(
            model_id, f"{cls.__module__}.{cls.__name__}", Text(caps_str, style="accent")
        )
    console.print(t)


def render_functions_table(fns: Dict[str, str]) -> None:
    t = _rounded_table("Registered Functions")
    t.add_column("namespaced fn")
    t.add_column("adapter method")
    for k, v in sorted(fns.items()):
        t.add_row(k, Text(v, style="accent"))
    console.print(t)


def render_presets_table(items: List[Dict[str, str]]) -> None:
    t = _rounded_table("Available Presets")
    t.add_column("id")
    t.add_column("kind")
    t.add_column("description")
    for it in items:
        t.add_row(
            it["id"],
            Text(it["kind"], style="accent"),
            Text(it.get("description") or "—", style="muted"),
        )
    console.print(t)


def render_preset_detail(preset: Dict[str, Any]) -> None:
    t = _rounded_table(f"Preset: {preset['id']}")
    t.add_column("Field")
    t.add_column("Value")
    t.add_row("kind", Text(preset["kind"], style="accent"))
    model = preset.get("model") or {}
    t.add_row("model.id", model.get("id", "—"))
    t.add_row("model.precision", model.get("precision", "—"))
    t.add_row("model.alphabet", model.get("alphabet", "—"))
    console.print(t)

    if preset["kind"] == "extract":
        outs = preset.get("outputs", [])
        render_outputs_spec_table(outs)
    else:
        params = preset.get("params", {})
        t2 = _rounded_table("Params")
        t2.add_column("key")
        t2.add_column("value")
        for k, v in params.items():
            t2.add_row(
                k,
                Text(
                    json.dumps(v) if not isinstance(v, str) else str(v), style="muted"
                ),
            )
        console.print(t2)


# ───────────────────────────────────────────────────────────────────────────────
# Progress Management (rounded, modern bar styles)
# ───────────────────────────────────────────────────────────────────────────────


class _RichHandle:
    def __init__(self, prog: Progress, task_id: int):
        self._p = prog
        self._task_id = task_id

    def update(self, n: int) -> None:
        self._p.update(self._task_id, advance=n)

    def close(self) -> None:
        try:
            self._p.remove_task(self._task_id)
        except Exception:
            pass


class RichProgressManager:
    """Context-managed progress manager that exposes a factory for engine hooks."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="grey37",
                    complete_style="accent",
                    finished_style="ok",
                    pulse_style="accent",
                ),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
        else:
            self._progress = None

    def __enter__(self):
        if self.enabled:
            self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._progress.__exit__(exc_type, exc, tb)

    def factory(self, label: str, total: int):
        if not self.enabled:
            return None
        task_id = self._progress.add_task(label, total=total)
        return _RichHandle(self._progress, task_id)
