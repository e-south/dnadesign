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
import math
from collections import Counter

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from ...utils.rich_style import make_panel, make_table


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
    def __init__(self, *, console: Console, refresh_seconds: float) -> None:
        refresh_rate = max(1.0, 1.0 / max(refresh_seconds, 0.1))
        self._live = Live(console=console, refresh_per_second=refresh_rate, transient=False)
        self._started = False
        atexit.register(self.close)

    def update(self, renderable) -> None:
        if not self._started:
            self._live.start()
            self._started = True
        self._live.update(renderable, refresh=True)

    def close(self) -> None:
        if self._started:
            self._live.stop()
            self._started = False


def _build_screen_dashboard(
    *,
    source_label: str,
    plan_name: str,
    bar: str,
    generated: int,
    quota: int,
    pct: float,
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
    show_tfbs: bool,
    show_solutions: bool,
    sequence_preview: str | None,
) -> Panel:
    table = make_table(show_header=False, expand=True)
    header = f"{source_label}/{plan_name}"
    table.add_row("run", header)
    table.add_row("progress", f"{bar} {generated}/{quota} ({pct:.2f}%)")
    table.add_row("library", f"index={library_index} local={local_generated}/{local_target}")
    if cr_now is not None:
        if cr_avg is not None:
            table.add_row("compression", f"now={cr_now:.3f} avg={cr_avg:.3f}")
        else:
            table.add_row("compression", f"now={cr_now:.3f}")
    table.add_row("counts", f"resamples={resamples} dup_out={dup_out} dup_sol={dup_sol} fails={fails} stalls={stalls}")
    if failure_totals:
        table.add_row("failures", failure_totals)
    if tf_usage:
        table.add_row("TF usage", _summarize_leaderboard(tf_usage, top=5))
    tfbs_label = _summarize_leaderboard(tfbs_usage, top=5) if show_tfbs else _summarize_tfbs_usage_stats(tfbs_usage)
    table.add_row("TFBS usage", tfbs_label)
    table.add_row("diversity", diversity_label)
    if show_solutions and sequence_preview:
        table.add_row("sequence", sequence_preview)
    return make_panel(table, title="DenseGen progress")


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
