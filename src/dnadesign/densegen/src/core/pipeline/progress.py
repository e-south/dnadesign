"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/progress.py

Progress and leaderboard formatting for pipeline runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import atexit
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...utils.rich_style import make_panel, make_table

log = logging.getLogger(__name__)


def _summarize_tf_counts(labels: list[str], max_items: int = 6) -> str:
    if not labels:
        return ""
    counts = Counter(labels)
    items = [f"{tf} x {n}" for tf, n in counts.most_common(max_items)]
    extra = len(counts) - min(len(counts), max_items)
    return ", ".join(items) + (f" (+{extra} TFs)" if extra > 0 else "")


def _summarize_leaderboard(counts: dict, *, top: int = 5) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))
    items = items[: max(1, int(top))]
    parts = []
    for key, val in items:
        if isinstance(key, tuple):
            key_label = f"{key[0]}:{key[1]}"
        else:
            key_label = str(key)
        parts.append(f"{key_label}={int(val)}")
    return ", ".join(parts) if parts else "-"


def _leaderboard_items(counts: dict, *, top: int = 5) -> list[dict]:
    if not counts:
        return []
    items = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))
    items = items[: max(1, int(top))]
    out: list[dict] = []
    for key, val in items:
        if isinstance(key, tuple):
            tf = str(key[0])
            tfbs = str(key[1])
            out.append({"tf": tf, "tfbs": tfbs, "count": int(val)})
        else:
            out.append({"tf": str(key), "count": int(val)})
    return out


def _format_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[?]"
    filled = int(width * (current / max(1, total)))
    filled = min(width, max(0, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _short_seq(value: str, *, max_len: int = 16) -> str:
    if not value:
        return "-"
    if len(value) <= max_len:
        return value
    keep = max(1, max_len - 3)
    return value[:keep] + "..."


_DENSE_ARRAY_COMPLEMENT = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "-": "-",
}
_EXTRA_LIBRARY_LABEL = "__densegen__extra__"


def _normalize_extra_label(extra_label: str | None) -> str:
    if extra_label is None:
        return _EXTRA_LIBRARY_LABEL
    label = str(extra_label).strip()
    if not label:
        raise ValueError("extra_label must be a non-empty string")
    return label


def _extend_library_tfs(
    library_tfs: list[str],
    library_len: int,
    *,
    extra_label: str | None = None,
) -> list[str]:
    if len(library_tfs) < library_len:
        extra = library_len - len(library_tfs)
        label = _normalize_extra_label(extra_label)
        return [*list(library_tfs), *([label] * extra)]
    return list(library_tfs)


def _validate_tf_color_mapping(library_tfs: list[str], tf_colors: dict[str, str]) -> dict[str, str]:
    if not tf_colors:
        raise ValueError("logging.visuals.tf_colors must be set when logging.print_visual is true")
    missing = []
    for tf in library_tfs:
        name = str(tf).strip()
        if not name:
            raise ValueError("logging.visuals.tf_colors requires non-empty TF labels in the library")
        if name not in tf_colors:
            missing.append(name)
    if missing:
        preview = ", ".join(sorted(set(missing))[:10])
        raise ValueError(f"logging.visuals.tf_colors missing entries for TFs: {preview}")
    return tf_colors


def _build_color_legend(
    library_tfs: list[str],
    tf_colors: dict[str, str],
    *,
    max_per_row: int = 2,
) -> Table:
    legend = Table.grid(padding=(0, 1))
    legend.add_column(justify="right", no_wrap=True)
    per_row = max(1, int(max_per_row))
    for _ in range(per_row):
        legend.add_column(no_wrap=True)
    seen: set[str] = set()
    labels: list[str] = []
    for tf in library_tfs:
        label = str(tf).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if not labels:
        row = [Text("legend:"), Text("-")]
        while len(row) < 1 + per_row:
            row.append(Text(""))
        legend.add_row(*row)
        return legend
    for row_start in range(0, len(labels), per_row):
        chunk = labels[row_start : row_start + per_row]
        cells: list[Text] = [Text("legend:") if row_start == 0 else Text("")]
        for label in chunk:
            cell = Text()
            color = tf_colors[label]
            style = f"bold {color}"
            cell.append("\u25a0 ", style=style)
            cell.append(label, style=style)
            cells.append(cell)
        while len(cells) < 1 + per_row:
            cells.append(Text(""))
        legend.add_row(*cells)
    return legend


def _dispatch_colored_labels(
    library: list[str],
    offsets: list[int | None],
    *,
    library_tfs: list[str],
    tf_colors: dict[str, str],
    rev: bool = False,
) -> list[Text]:
    lines: list[Text] = []
    if not offsets:
        return lines
    ordered = sorted((offset, idx) for idx, offset in enumerate(offsets) if offset is not None)
    for offset, idx in ordered:
        motif = library[idx][::-1] if rev else library[idx]
        tf = str(library_tfs[idx]).strip()
        color = tf_colors.get(tf)
        if not color:
            raise ValueError(f"logging.visuals.tf_colors missing entry for TF: {tf}")
        for line in lines:
            if len(line.plain) < offset:
                padding = " " * (offset - len(line.plain))
                line.append(padding)
                line.append(motif, style=f"bold {color}")
                break
        else:
            line = Text(" " * offset)
            line.append(motif, style=f"bold {color}")
            lines.append(line)
    return lines


def _render_dense_array_visual(
    sol,
    *,
    library_tfs: list[str],
    tf_colors: dict[str, str],
    extra_label: str | None = None,
) -> Group:
    try:
        library = list(sol.library)
        offsets_fwd = list(sol.offsets_fwd)
        offsets_rev = list(sol.offsets_rev)
        seq_len = int(sol.sequence_length)
        seq = str(sol.sequence)
    except Exception as exc:
        raise RuntimeError("dense-arrays solution missing required attributes for visual rendering") from exc

    library_tfs = _extend_library_tfs(library_tfs, len(library), extra_label=extra_label)
    tf_colors = _validate_tf_color_mapping(library_tfs, tf_colors)

    sequence = seq + "-" * (seq_len - len(seq))
    try:
        seq_rev = "".join(_DENSE_ARRAY_COMPLEMENT[c] for c in sequence)
    except KeyError as exc:
        raise ValueError(f"Dense-array sequence contains invalid base: {exc.args[0]!r}") from exc

    lines_fwd = _dispatch_colored_labels(
        library,
        offsets_fwd,
        library_tfs=library_tfs,
        tf_colors=tf_colors,
        rev=False,
    )
    lines_rev = _dispatch_colored_labels(
        library,
        offsets_rev,
        library_tfs=library_tfs,
        tf_colors=tf_colors,
        rev=True,
    )

    renderables: list[Text] = []
    prefix_fwd = Text("--> ")
    prefix_rev = Text("<-- ")
    for line in reversed(lines_fwd):
        renderables.append(prefix_fwd + line)
    renderables.append(Text("--> " + sequence))
    renderables.append(Text("<-- " + seq_rev))
    for line in lines_rev:
        renderables.append(prefix_rev + line)
    return Group(*renderables)


def _summarize_failure_totals(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
) -> str:
    total = 0
    unique = 0
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in reasons.values())
        if count > 0:
            total += count
            unique += 1
    if total <= 0:
        return "failed_sites=0 total_failures=0"
    return f"failed_sites={unique} total_failures={total}"


def _summarize_failure_leaderboard(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
    top: int = 5,
) -> str:
    if not failure_counts:
        return "-"
    totals: dict[tuple[str, str], int] = {}
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in reasons.values())
        if count <= 0:
            continue
        key = (str(tf), str(tfbs))
        totals[key] = totals.get(key, 0) + int(count)
    if not totals:
        return "-"
    items = sorted(totals.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    items = items[: max(1, int(top))]
    parts = []
    for (tf, tfbs), count in items:
        parts.append(f"{tf}:{_short_seq(tfbs)}={int(count)}")
    return ", ".join(parts) if parts else "-"


def _failure_leaderboard_items(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
    top: int = 5,
) -> list[dict]:
    if not failure_counts:
        return []
    totals: dict[tuple[str, str], int] = {}
    reasons_by_key: dict[tuple[str, str], dict[str, int]] = {}
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in (reasons or {}).values())
        if count <= 0:
            continue
        key = (str(tf), str(tfbs))
        totals[key] = totals.get(key, 0) + int(count)
        reason_counts = reasons_by_key.setdefault(key, {})
        for reason, n in (reasons or {}).items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(n)
    if not totals:
        return []
    items = sorted(totals.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    items = items[: max(1, int(top))]
    out: list[dict] = []
    for (tf, tfbs), count in items:
        reasons = reasons_by_key.get((tf, tfbs), {})
        top_reason = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else ""
        out.append({"tf": tf, "tfbs": tfbs, "failures": int(count), "top_reason": top_reason})
    return out


def _aggregate_failure_counts_for_sampling(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
) -> dict[tuple[str, str], int]:
    if not failure_counts:
        return {}
    totals: dict[tuple[str, str], int] = {}
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in (reasons or {}).values())
        if count <= 0:
            continue
        key = (str(tf), str(tfbs))
        totals[key] = totals.get(key, 0) + int(count)
    return totals


def _normalized_entropy(counts: dict) -> float | None:
    values = np.array(list(counts.values()), dtype=float)
    if values.size == 0:
        return None
    total = float(values.sum())
    if total <= 0:
        return None
    p = values / total
    ent = -np.sum(p * np.log(p))
    max_ent = math.log(len(values)) if len(values) > 1 else 0.0
    if max_ent <= 0:
        return 0.0
    return float(ent / max_ent)


def _summarize_diversity(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    *,
    library_tfs: list[str],
    library_tfbs: list[str],
) -> str:
    lib_tf_count = len(set(library_tfs)) if library_tfs else 0
    if library_tfs:
        lib_tfbs_count = len(set(zip(library_tfs, library_tfbs)))
    else:
        lib_tfbs_count = len(set(library_tfbs))
    used_tf_count = len(tf_usage_counts)
    used_tfbs_count = len(usage_counts)
    tf_cov = used_tf_count / max(1, lib_tf_count) if lib_tf_count else 0.0
    tfbs_cov = used_tfbs_count / max(1, lib_tfbs_count) if lib_tfbs_count else 0.0
    ent = _normalized_entropy(usage_counts)
    ent_label = f"{ent:.3f}" if ent is not None else "n/a"
    return (
        f"tf_coverage={tf_cov:.2f} ({used_tf_count}/{lib_tf_count}) | "
        f"tfbs_coverage={tfbs_cov:.2f} ({used_tfbs_count}/{lib_tfbs_count}) | "
        f"tfbs_entropy={ent_label}"
    )


def _summarize_tfbs_usage_stats(usage_counts: dict[tuple[str, str], int]) -> str:
    if not usage_counts:
        return "unique=0"
    counts = np.array(list(usage_counts.values()), dtype=int)
    if counts.size == 0:
        return "unique=0"
    unique = int(counts.size)
    min_val = int(counts.min())
    med_val = float(np.median(counts))
    max_val = int(counts.max())
    top_vals = sorted(counts.tolist(), reverse=True)[:3]
    top_label = ",".join(str(int(v)) for v in top_vals)
    med_label = f"{med_val:.1f}" if med_val % 1 else str(int(med_val))
    return f"unique={unique} min/med/max={min_val}/{med_label}/{max_val} top={top_label}"


class _ScreenDashboard:
    def __init__(self, *, console: Console, refresh_seconds: float, append: bool = False) -> None:
        refresh_rate = max(1.0, 1.0 / max(refresh_seconds, 0.1))
        self._console = console
        self._append = bool(append)
        self._muted_handlers: list[tuple[logging.Handler, int]] = []
        self._live = (
            Live(
                console=console,
                refresh_per_second=refresh_rate,
                transient=False,
                screen=True,
                vertical_overflow="crop",
            )
            if console.is_terminal and not self._append
            else None
        )
        self._started = False
        self._last_renderable = None
        self._printed = False
        self._disabled = False
        atexit.register(self.close)

    def _console_closed(self) -> bool:
        stream = getattr(self._console, "file", None)
        return bool(getattr(stream, "closed", False))

    def terminal_height(self) -> int | None:
        try:
            height = int(self._console.size.height)
        except Exception:
            return None
        return height if height > 0 else None

    def terminal_width(self) -> int | None:
        try:
            width = int(self._console.size.width)
        except Exception:
            return None
        return width if width > 0 else None

    def _disable(self, message: str, *, exc_info: bool = False) -> None:
        if self._disabled:
            return
        self._disabled = True
        if self._live is not None and self._started:
            try:
                self._live.stop()
            except ValueError:
                pass
            self._started = False
        self._restore_console_handlers()
        log.warning(message, exc_info=exc_info)

    def _mute_console_handlers(self) -> None:
        if self._muted_handlers:
            return
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "console", None) is self._console:
                self._muted_handlers.append((handler, int(handler.level)))
                handler.setLevel(logging.CRITICAL + 100)

    def _restore_console_handlers(self) -> None:
        if not self._muted_handlers:
            return
        for handler, level in self._muted_handlers:
            handler.setLevel(int(level))
        self._muted_handlers.clear()

    def update(self, renderable) -> None:
        if self._disabled:
            return
        if self._console_closed():
            self._disable("Dashboard console stream closed; disabling screen dashboard updates.")
            return
        if self._live is None:
            if self._append:
                try:
                    self._console.print(renderable)
                except ValueError:
                    self._disable(
                        "Dashboard console stream unavailable during append update; disabling screen dashboard.",
                        exc_info=True,
                    )
                return
            self._last_renderable = renderable
            return
        if not self._started:
            try:
                self._mute_console_handlers()
                self._live.start()
            except ValueError:
                self._disable(
                    "Dashboard console stream unavailable while starting live dashboard; disabling screen dashboard.",
                    exc_info=True,
                )
                return
            self._started = True
        try:
            self._live.update(renderable, refresh=True)
        except ValueError:
            self._disable(
                "Dashboard console stream unavailable during live update; disabling screen dashboard.",
                exc_info=True,
            )

    def close(self) -> None:
        if self._disabled:
            return
        if self._console_closed():
            self._disabled = True
            return
        if self._live is not None and self._started:
            try:
                self._live.stop()
            except ValueError:
                self._disable(
                    "Dashboard console stream unavailable during live shutdown; disabling screen dashboard.",
                    exc_info=True,
                )
                return
            self._started = False
        self._restore_console_handlers()
        if self._append:
            return
        if self._live is None and self._last_renderable is not None and not self._printed:
            try:
                self._console.print(self._last_renderable)
            except ValueError:
                self._disable(
                    "Dashboard console stream unavailable during final render; disabling screen dashboard.",
                    exc_info=True,
                )
                return
            self._printed = True


def _slice_visual_preview(
    sequence_preview,
    *,
    max_lines: int | None,
    window_start: int,
) -> tuple[object, str | None, int]:
    if max_lines is None:
        return sequence_preview, None, window_start
    budget = max(1, int(max_lines))
    if isinstance(sequence_preview, Group):
        lines = list(sequence_preview.renderables)
    else:
        return sequence_preview, None, window_start
    total = len(lines)
    if total <= budget:
        return sequence_preview, None, window_start
    start = int(window_start) % total
    subset = [lines[(start + idx) % total] for idx in range(budget)]
    end = ((start + budget - 1) % total) + 1
    label = f"window {start + 1}-{end}/{total}"
    next_start = (start + 1) % total
    return Group(*subset), label, next_start


def _rendered_line_count(renderable, *, width: int) -> int:
    probe_width = max(20, int(width))
    probe = Console(
        width=probe_width,
        force_terminal=False,
        color_system=None,
        no_color=True,
        record=True,
    )
    probe.print(renderable)
    text = probe.export_text(styles=False).rstrip("\n")
    if not text:
        return 0
    return len(text.splitlines())


def _build_screen_dashboard_adaptive(
    *,
    source_label: str,
    plan_name: str,
    bar: str,
    generated: int,
    quota: int,
    pct: float,
    global_bar: str | None,
    global_generated: int | None,
    global_quota: int | None,
    global_pct: float | None,
    local_generated: int,
    local_target: int,
    library_index: int,
    cr_now: float | None,
    cr_avg: float | None,
    resamples: int,
    dup_out: int,
    dup_sol: int,
    fails: int,
    stalls: int,
    failure_totals: str | None,
    tf_usage: dict[str, int],
    tfbs_usage: dict[tuple[str, str], int],
    diversity_label: str,
    legend: object | None,
    show_tfbs: bool,
    show_solutions: bool,
    sequence_preview: object | None,
    solution_label: str = "sequence",
    solver_settings: str | None = None,
    max_lines: int | None = None,
    max_width: int | None = None,
    visual_window_start: int = 0,
) -> tuple[Panel, int]:
    optional = {
        "failure": bool(failure_totals),
        "usage": True,
        "diversity": True,
        "legend": legend is not None,
        "solver": bool(solver_settings),
    }
    visual_enabled = bool(show_solutions and sequence_preview is not None)
    if max_lines is not None:
        # Reserve room for table/panel borders and keep a conservative line budget.
        budget = max(8, int(max_lines) - 8)
        base_rows = 5
        if (
            global_bar is not None
            and global_generated is not None
            and global_quota is not None
            and global_pct is not None
        ):
            base_rows += 1
        optional_rows = sum(1 for enabled in optional.values() if enabled)
        min_visual_rows = 1 if visual_enabled else 0
        drop_order = ["legend", "diversity", "usage", "failure", "solver"]
        while base_rows + optional_rows + min_visual_rows > budget and drop_order:
            key = drop_order.pop(0)
            if optional.get(key, False):
                optional[key] = False
                optional_rows -= 1
    visual_lines_budget = None
    if max_lines is not None and visual_enabled:
        budget = max(8, int(max_lines) - 8)
        non_visual_rows = 5
        if (
            global_bar is not None
            and global_generated is not None
            and global_quota is not None
            and global_pct is not None
        ):
            non_visual_rows += 1
        non_visual_rows += sum(1 for enabled in optional.values() if enabled)
        visual_lines_budget = max(1, budget - non_visual_rows)

    def _build_panel() -> tuple[Panel, int]:
        local_solution_label = solution_label
        local_sequence_preview = sequence_preview
        next_visual_window_start = visual_window_start
        if visual_enabled:
            local_sequence_preview, window_label, next_visual_window_start = _slice_visual_preview(
                local_sequence_preview,
                max_lines=visual_lines_budget,
                window_start=visual_window_start,
            )
            if window_label:
                local_solution_label = f"{solution_label} ({window_label})"

        table = make_table(show_header=False, expand=True)
        header = f"{source_label}/{plan_name}"
        table.add_row("run (pool/plan)", header)
        table.add_row("progress (plan)", f"{bar} {generated}/{quota} ({pct:.2f}%)")
        if (
            global_bar is not None
            and global_generated is not None
            and global_quota is not None
            and global_pct is not None
        ):
            table.add_row(
                "global progress (all plans)",
                f"{global_bar} {global_generated}/{global_quota} ({global_pct:.2f}%)",
            )
        table.add_row("library (index/local)", f"index={library_index} local={local_generated}/{local_target}")
        if cr_now is not None:
            if cr_avg is not None:
                table.add_row("compression (ratio)", f"now={cr_now:.3f} avg={cr_avg:.3f}")
            else:
                table.add_row("compression (ratio)", f"now={cr_now:.3f}")
        table.add_row(
            "counts (resample/dup/fail/stall)",
            f"resamples={resamples} dup_out={dup_out} dup_sol={dup_sol} fails={fails} stalls={stalls}",
        )
        if optional["solver"] and solver_settings:
            table.add_row("solver", solver_settings)
        if optional["failure"] and failure_totals:
            table.add_row("failures", failure_totals)
        if optional["usage"]:
            tfbs_label = (
                _summarize_leaderboard(tfbs_usage, top=5) if show_tfbs else _summarize_tfbs_usage_stats(tfbs_usage)
            )
            table.add_row("TFBS usage (unique tf/tfbs used)", tfbs_label)
        if optional["diversity"]:
            table.add_row("diversity (tf_coverage/tfbs_coverage/tfbs_entropy)", diversity_label)
        if optional["legend"] and legend is not None:
            table.add_row("legend (TF colors)", legend)
        if visual_enabled and local_sequence_preview:
            table.add_row(f"{local_solution_label} (dense-arrays)", local_sequence_preview)
        return make_panel(table, title="DenseGen progress"), int(next_visual_window_start)

    panel, next_visual_window_start = _build_panel()
    if max_lines is None or max_width is None:
        return panel, next_visual_window_start

    safety_target = max(1, int(max_lines) - 1)
    drop_order = ["legend", "diversity", "usage", "failure", "solver"]
    while _rendered_line_count(panel, width=int(max_width)) > safety_target:
        if visual_enabled and visual_lines_budget is not None and visual_lines_budget > 1:
            visual_lines_budget -= 1
            panel, next_visual_window_start = _build_panel()
            continue
        dropped = False
        while drop_order:
            key = drop_order.pop(0)
            if optional.get(key, False):
                optional[key] = False
                dropped = True
                break
        if not dropped:
            break
        panel, next_visual_window_start = _build_panel()
    return panel, next_visual_window_start


def _build_screen_dashboard(
    *,
    source_label: str,
    plan_name: str,
    bar: str,
    generated: int,
    quota: int,
    pct: float,
    global_bar: str | None,
    global_generated: int | None,
    global_quota: int | None,
    global_pct: float | None,
    local_generated: int,
    local_target: int,
    library_index: int,
    cr_now: float | None,
    cr_avg: float | None,
    resamples: int,
    dup_out: int,
    dup_sol: int,
    fails: int,
    stalls: int,
    failure_totals: str | None,
    tf_usage: dict[str, int],
    tfbs_usage: dict[tuple[str, str], int],
    diversity_label: str,
    legend: object | None,
    show_tfbs: bool,
    show_solutions: bool,
    sequence_preview: object | None,
    solution_label: str = "sequence",
) -> Panel:
    panel, _ = _build_screen_dashboard_adaptive(
        source_label=source_label,
        plan_name=plan_name,
        bar=bar,
        generated=generated,
        quota=quota,
        pct=pct,
        global_bar=global_bar,
        global_generated=global_generated,
        global_quota=global_quota,
        global_pct=global_pct,
        local_generated=local_generated,
        local_target=local_target,
        library_index=library_index,
        cr_now=cr_now,
        cr_avg=cr_avg,
        resamples=resamples,
        dup_out=dup_out,
        dup_sol=dup_sol,
        fails=fails,
        stalls=stalls,
        failure_totals=failure_totals,
        tf_usage=tf_usage,
        tfbs_usage=tfbs_usage,
        diversity_label=diversity_label,
        legend=legend,
        show_tfbs=show_tfbs,
        show_solutions=show_solutions,
        sequence_preview=sequence_preview,
        solution_label=solution_label,
    )
    return panel


@dataclass
class PlanProgressState:
    last_screen_refresh: float = 0.0
    cr_sum: float = 0.0
    cr_count: int = 0
    latest_failure_totals: str | None = None
    legend_cache: object | None = None
    legend_key: tuple[str, ...] | None = None
    visual_window_start: int = 0


@dataclass
class PlanProgressReporter:
    source_label: str
    plan_name: str
    quota: int
    max_per_subsample: int
    progress_style: str
    progress_every: int
    progress_refresh_seconds: float
    show_tfbs: bool
    show_solutions: bool
    print_visual: bool
    dashboard: _ScreenDashboard | None
    tf_colors: dict[str, str] | None = None
    extra_library_label: str | None = None
    solver_settings: str | None = None
    display_tf_label: Callable[[str], str] | None = None
    logger: logging.Logger = field(default_factory=lambda: log)
    state: PlanProgressState = field(default_factory=PlanProgressState)

    def record_solution(
        self,
        *,
        global_generated: int,
        local_generated: int,
        library_index: int,
        sol,
        library_tfs: list[str],
        library_tfbs: list[str],
        used_tfbs_detail: list[dict],
        used_tf_list: list[str],
        final_seq: str,
        counters,
        duplicate_records: int,
        duplicate_solutions: int,
        failed_solutions: int,
        stall_events: int,
        usage_counts: dict[tuple[str, str], int],
        tf_usage_counts: dict[str, int],
        tf_usage_display: dict[str, int],
        tfbs_usage_display: dict[tuple[str, str], int],
        global_total_generated: int | None = None,
        global_total_quota: int | None = None,
    ) -> None:
        solution_preview = None
        solution_label = "sequence"
        legend = None
        display_library_tfs = list(library_tfs)
        if self.display_tf_label is not None:
            display_library_tfs = [self.display_tf_label(tf) for tf in display_library_tfs]
        if self.print_visual:
            raw_visual = str(sol)
            if self.progress_style == "screen":
                if self.tf_colors is None:
                    raise ValueError("logging.visuals.tf_colors must be set when logging.print_visual is true")
                display_library_tfs = _extend_library_tfs(
                    display_library_tfs,
                    len(sol.library),
                    extra_label=self.extra_library_label,
                )
                solution_preview = _render_dense_array_visual(
                    sol,
                    library_tfs=display_library_tfs,
                    tf_colors=self.tf_colors,
                    extra_label=self.extra_library_label,
                )
                legend_key = tuple(display_library_tfs)
                if self.state.legend_cache is None or self.state.legend_key != legend_key:
                    self.state.legend_cache = _build_color_legend(display_library_tfs, self.tf_colors)
                    self.state.legend_key = legend_key
                legend = self.state.legend_cache
            else:
                solution_preview = raw_visual
            solution_label = "visual"
        elif self.show_solutions:
            solution_preview = final_seq

        if self.progress_style == "screen" and self.dashboard is not None:
            now = time.monotonic()
            should_refresh = bool(self.print_visual or self.show_solutions)
            if not should_refresh:
                should_refresh = (now - self.state.last_screen_refresh) >= self.progress_refresh_seconds
            if should_refresh:
                self.state.last_screen_refresh = now
                bar = _format_progress_bar(global_generated, self.quota)
                global_bar = None
                global_pct = None
                if global_total_quota is not None and global_total_generated is not None:
                    global_bar = _format_progress_bar(global_total_generated, global_total_quota)
                    global_pct = float(global_total_generated) / float(global_total_quota) * 100.0
                cr_now = float(sol.compression_ratio)
                self.state.cr_sum += cr_now
                self.state.cr_count += 1
                cr_avg = self.state.cr_sum / max(self.state.cr_count, 1)
                diversity_label = _summarize_diversity(
                    usage_counts,
                    tf_usage_counts,
                    library_tfs=display_library_tfs,
                    library_tfbs=library_tfbs,
                )
                max_dashboard_lines = None
                max_dashboard_width = None
                if hasattr(self.dashboard, "terminal_height"):
                    max_dashboard_lines = self.dashboard.terminal_height()
                if hasattr(self.dashboard, "terminal_width"):
                    max_dashboard_width = self.dashboard.terminal_width()
                renderable, next_visual_window_start = _build_screen_dashboard_adaptive(
                    source_label=self.source_label,
                    plan_name=self.plan_name,
                    bar=bar,
                    generated=int(global_generated),
                    quota=int(self.quota),
                    pct=float(global_generated) / float(self.quota) * 100.0,
                    global_bar=global_bar,
                    global_generated=(int(global_total_generated) if global_total_generated is not None else None),
                    global_quota=int(global_total_quota) if global_total_quota is not None else None,
                    global_pct=global_pct,
                    local_generated=int(local_generated),
                    local_target=int(self.max_per_subsample),
                    library_index=int(library_index),
                    cr_now=cr_now,
                    cr_avg=cr_avg,
                    resamples=int(counters.total_resamples),
                    dup_out=int(duplicate_records),
                    dup_sol=int(duplicate_solutions),
                    fails=int(failed_solutions),
                    stalls=int(stall_events),
                    failure_totals=self.state.latest_failure_totals,
                    tf_usage=tf_usage_display,
                    tfbs_usage=tfbs_usage_display,
                    diversity_label=diversity_label,
                    legend=legend,
                    show_tfbs=self.show_tfbs,
                    show_solutions=bool(solution_preview),
                    sequence_preview=solution_preview,
                    solution_label=solution_label,
                    solver_settings=self.solver_settings,
                    max_lines=max_dashboard_lines,
                    max_width=max_dashboard_width,
                    visual_window_start=self.state.visual_window_start,
                )
                self.state.visual_window_start = int(next_visual_window_start)
                self.dashboard.update(renderable)

        if self.progress_style == "stream" and global_generated % max(1, self.progress_every) == 0:
            bar = _format_progress_bar(global_generated, self.quota)
            pct = float(global_generated) / float(self.quota) * 100.0
            if self.show_tfbs:
                tf_label = _short_seq(str(used_tfbs_detail), max_len=80)
            else:
                tf_label = _summarize_tf_counts(used_tf_list)
            cr_now = float(sol.compression_ratio)
            self.state.cr_sum += cr_now
            self.state.cr_count += 1
            if self.print_visual and solution_preview is not None:
                self.logger.info(
                    "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f | TFBS %s\nvisual:\n%s",
                    self.source_label,
                    self.plan_name,
                    bar,
                    global_generated,
                    self.quota,
                    pct,
                    local_generated,
                    self.max_per_subsample,
                    cr_now if cr_now is not None else float("nan"),
                    tf_label,
                    solution_preview,
                )
            elif self.show_solutions and solution_preview is not None:
                self.logger.info(
                    "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f | TFBS %s",
                    self.source_label,
                    self.plan_name,
                    bar,
                    global_generated,
                    self.quota,
                    pct,
                    local_generated,
                    self.max_per_subsample,
                    cr_now if cr_now is not None else float("nan"),
                    tf_label,
                )
            else:
                self.logger.info(
                    "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f",
                    self.source_label,
                    self.plan_name,
                    bar,
                    global_generated,
                    self.quota,
                    pct,
                    local_generated,
                    self.max_per_subsample,
                    cr_now if cr_now is not None else float("nan"),
                )

    def record_leaderboard(
        self,
        *,
        global_generated: int,
        counters,
        duplicate_records: int,
        duplicate_solutions: int,
        failed_solutions: int,
        stall_events: int,
        failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
        leaderboard_every: int,
        log_snapshot,
    ) -> None:
        if leaderboard_every <= 0:
            return
        if global_generated % max(1, leaderboard_every) != 0:
            return
        failure_totals = _summarize_failure_totals(
            failure_counts,
            input_name=self.source_label,
            plan_name=self.plan_name,
        )
        self.state.latest_failure_totals = failure_totals
        if self.progress_style != "screen":
            bar = _format_progress_bar(global_generated, self.quota)
            pct = float(global_generated) / float(self.quota) * 100.0
            self.logger.info(
                "[%s/%s] Progress %s %d/%d (%.2f%%) | resamples=%d dup_out=%d dup_sol=%d fails=%d stalls=%d | %s",
                self.source_label,
                self.plan_name,
                bar,
                global_generated,
                self.quota,
                pct,
                counters.total_resamples,
                duplicate_records,
                duplicate_solutions,
                failed_solutions,
                stall_events,
                failure_totals,
            )
            log_snapshot()


def _diversity_snapshot(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    *,
    library_tfs: list[str],
    library_tfbs: list[str],
) -> dict[str, object]:
    lib_tf_count = len(set(library_tfs)) if library_tfs else 0
    if library_tfs:
        lib_tfbs_count = len(set(zip(library_tfs, library_tfbs)))
    else:
        lib_tfbs_count = len(set(library_tfbs))
    used_tf_count = len(tf_usage_counts)
    used_tfbs_count = len(usage_counts)
    tf_cov = used_tf_count / max(1, lib_tf_count) if lib_tf_count else 0.0
    tfbs_cov = used_tfbs_count / max(1, lib_tfbs_count) if lib_tfbs_count else 0.0
    ent = _normalized_entropy(usage_counts)
    return {
        "tf_coverage": float(tf_cov),
        "tfbs_coverage": float(tfbs_cov),
        "tfbs_entropy": float(ent) if ent is not None else None,
        "used_tf_count": int(used_tf_count),
        "library_tf_count": int(lib_tf_count),
        "used_tfbs_count": int(used_tfbs_count),
        "library_tfbs_count": int(lib_tfbs_count),
    }


def _leaderboard_snapshot(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
    library_tfs: list[str],
    library_tfbs: list[str],
    top: int = 5,
) -> dict[str, object]:
    return {
        "tf": _leaderboard_items(tf_usage_counts, top=top),
        "tfbs": _leaderboard_items(usage_counts, top=top),
        "failed_tfbs": _failure_leaderboard_items(failure_counts, input_name=input_name, plan_name=plan_name, top=top),
        "diversity": _diversity_snapshot(
            usage_counts,
            tf_usage_counts,
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
        ),
    }
