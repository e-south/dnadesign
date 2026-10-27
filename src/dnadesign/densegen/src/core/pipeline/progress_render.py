"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/progress_render.py

Formatting helpers for dense pipeline progress dashboards and summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text


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
