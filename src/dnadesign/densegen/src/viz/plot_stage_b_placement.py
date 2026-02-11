"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_b_placement.py

Stage-B placement map plotting (occupancy + TFBS usage leaderboard).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, to_rgba

from .plot_common import _apply_style, _palette, _stage_b_plan_output_dir, _style

log = logging.getLogger(__name__)


def _plot_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    if isinstance(cfg.get("generation"), dict):
        return cfg
    nested = cfg.get("config")
    if isinstance(nested, dict):
        return nested
    return cfg


def _sequence_length_from_cfg(cfg: dict) -> int:
    cfg = _plot_config(cfg)
    gen = cfg.get("generation") if cfg else None
    if not isinstance(gen, dict):
        raise ValueError("Plot config missing generation block.")
    length = gen.get("sequence_length")
    if length is None:
        raise ValueError("Plot config missing generation.sequence_length.")
    return int(length)


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _promoter_constraints(cfg: dict, plan_name: str) -> list[dict]:
    cfg = _plot_config(cfg)
    gen = cfg.get("generation", {}) if cfg else {}
    plan_items = gen.get("plan", []) or []
    out = []
    for item in plan_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or "") != plan_name:
            continue
        fixed = item.get("fixed_elements") or {}
        pcs = fixed.get("promoter_constraints")
        pcs = [pcs] if isinstance(pcs, dict) else (pcs or [])
        for pc in (p for p in pcs if isinstance(p, dict)):
            upstream = str(pc.get("upstream") or "").strip().upper()
            downstream = str(pc.get("downstream") or "").strip().upper()
            if not upstream or not downstream:
                continue
            name = str(pc.get("name") or "").strip() or None
            spacer = pc.get("spacer_length")
            if isinstance(spacer, (list, tuple)) and spacer:
                spacer_min = min(int(v) for v in spacer)
                spacer_max = max(int(v) for v in spacer)
            elif isinstance(spacer, (int, float)):
                spacer_min = int(spacer)
                spacer_max = int(spacer)
            else:
                spacer_min = None
                spacer_max = None
            upstream_pos = pc.get("upstream_pos")
            downstream_pos = pc.get("downstream_pos")
            out.append(
                {
                    "name": name,
                    "upstream": upstream,
                    "downstream": downstream,
                    "spacer_min": spacer_min,
                    "spacer_max": spacer_max,
                    "upstream_pos": upstream_pos,
                    "downstream_pos": downstream_pos,
                }
            )
    return out


def _fixed_labels(constraints: list[dict]) -> list[str]:
    labels = []
    for idx, pc in enumerate(constraints):
        name = pc.get("name") or f"promoter_{idx + 1}"
        labels.append(f"fixed:{name}:-35")
        labels.append(f"fixed:{name}:-10")
    return labels


def _fixed_label_sequences(constraints: list[dict]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for idx, pc in enumerate(constraints):
        name = str(pc.get("name") or f"promoter_{idx + 1}")
        upstream = str(pc.get("upstream") or "").strip().upper()
        downstream = str(pc.get("downstream") or "").strip().upper()
        if upstream:
            sequences[f"fixed:{name}:-35"] = upstream
        if downstream:
            sequences[f"fixed:{name}:-10"] = downstream
    return sequences


def _sanitize_tf_label(label: str) -> str:
    label = str(label)
    if "_" in label:
        head, tail = label.split("_", 1)
        tail_upper = tail.upper()
        iupac = set("ACGTURYSWKMBDHVN")
        if len(tail_upper) >= 6 and set(tail_upper).issubset(iupac):
            return head
    return label


def _sanitize_fixed_label(label: str) -> str:
    label = str(label)
    if label.startswith("fixed:"):
        label = label[len("fixed:") :]
    if ":" in label:
        name, suffix = label.rsplit(":", 1)
        return f"{name} {suffix}"
    return label


def _normalize_tf_label(label: str, fixed_set: set[str]) -> str:
    label = str(label).strip()
    if not label:
        return ""
    if label in fixed_set or label.startswith("fixed:"):
        return label
    return _sanitize_tf_label(label)


def _find_positions(seq: str, motif: str, pos_range) -> list[int]:
    seq = str(seq).upper()
    motif = str(motif).upper()
    if not motif:
        return []
    lo = None
    hi = None
    if isinstance(pos_range, (list, tuple)) and len(pos_range) == 2:
        lo = int(pos_range[0])
        hi = int(pos_range[1])
    positions = []
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            break
        if lo is not None and idx < lo:
            start = idx + 1
            continue
        if hi is not None and idx > hi:
            start = idx + 1
            continue
        if idx + len(motif) <= len(seq):
            positions.append(idx)
        start = idx + 1
    return positions


def _select_promoter_pair(seq: str, pc: dict) -> tuple[int, int] | None:
    upstream = pc["upstream"]
    downstream = pc["downstream"]
    up_positions = _find_positions(seq, upstream, pc.get("upstream_pos"))
    down_positions = _find_positions(seq, downstream, pc.get("downstream_pos"))
    if not up_positions or not down_positions:
        return None
    spacer_min = pc.get("spacer_min")
    spacer_max = pc.get("spacer_max")
    for up_start in sorted(up_positions):
        up_end = up_start + len(upstream)
        for down_start in sorted(down_positions):
            if down_start < up_end:
                continue
            spacer = down_start - up_end
            if spacer_min is not None and spacer < int(spacer_min):
                continue
            if spacer_max is not None and spacer > int(spacer_max):
                continue
            return up_start, down_start
    return None


def _truncate_tfbs(seq: str, *, head: int = 8, tail: int = 6) -> str:
    seq = str(seq)
    if len(seq) <= head + tail + 3:
        return seq
    return f"{seq[:head]}...{seq[-tail:]}"


def _placement_bounds(row: pd.Series, seq_len: int) -> tuple[int, int] | None:
    start_val = row.get("offset")
    if start_val is None or pd.isna(start_val):
        raise ValueError("composition.parquet has placement row with missing offset.")
    start = int(start_val)
    length_val = row.get("length") if "length" in row else None
    if length_val is not None and not pd.isna(length_val):
        end = start + int(length_val)
    else:
        end_val = row.get("end")
        if end_val is None or pd.isna(end_val):
            raise ValueError("composition.parquet has placement row with missing length/end.")
        end = int(end_val)
    if end <= start:
        return None
    lo = max(0, min(start, seq_len))
    hi = max(lo, min(end, seq_len))
    if hi <= lo:
        return None
    return lo, hi


def _category_display_label(label: str, *, fixed_label_sequences: dict[str, str] | None = None) -> str:
    if label.startswith("fixed:"):
        sequence = (fixed_label_sequences or {}).get(label)
        if ":" in label:
            suffix = label.rsplit(":", 1)[-1]
        else:
            suffix = _sanitize_fixed_label(label)
        if sequence:
            return f"{suffix} ({sequence})"
        return str(suffix)
    if str(label).strip() == "neutral_bg":
        return "background"
    return label


def _colorblind_palette(style: dict, n: int) -> list:
    palette_style = dict(style)
    palette_style["palette"] = "okabe_ito"
    return _palette(palette_style, n)


def _darken_color(color: object, *, factor: float) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    scale = min(1.0, max(0.0, float(factor)))
    return (r * scale, g * scale, b * scale)


def _build_occupancy(
    sub: pd.DataFrame,
    *,
    solutions: pd.DataFrame,
    seq_len: int,
    constraints: list[dict],
    max_categories: int,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
    fixed_labels = _fixed_labels(constraints)
    fixed_set = set(fixed_labels)
    if max_categories < len(fixed_labels):
        msg = (
            "placement_map occupancy_max_categories="
            f"{max_categories} is smaller than fixed categories ({len(fixed_labels)})."
        )
        raise ValueError(msg)

    sub = sub.copy()
    sub["tf_label"] = sub["tf"].map(lambda tf: _normalize_tf_label(tf, fixed_set))
    regulator_labels = sorted(
        {str(tf) for tf in sub["tf_label"].astype(str).tolist() if str(tf).strip() and str(tf) not in fixed_set}
    )
    tf_totals: dict[str, int] = {}
    for _, row in sub.iterrows():
        tf_label = str(row.get("tf_label") or "").strip()
        if not tf_label or tf_label in fixed_set:
            continue
        bounds = _placement_bounds(row, seq_len)
        if bounds is None:
            continue
        lo, hi = bounds
        tf_totals[tf_label] = tf_totals.get(tf_label, 0) + max(0, hi - lo)

    max_regulators = max(0, max_categories - len(fixed_labels))
    if len(regulator_labels) > max_regulators and max_regulators > 0:
        keep = max(0, max_regulators - 1)
        ranked = sorted(regulator_labels, key=lambda t: (-tf_totals.get(t, 0), t))
        selected_tfs = sorted(ranked[:keep])
        other_tfs = ranked[keep:]
    elif max_regulators <= 0 and regulator_labels:
        selected_tfs = []
        other_tfs = regulator_labels
    else:
        selected_tfs = sorted(regulator_labels)
        other_tfs = []

    categories = list(fixed_labels) + list(selected_tfs)
    if other_tfs:
        categories.append("other")
    occupancy = {label: np.zeros(seq_len, dtype=float) for label in categories}

    for _, row in sub.iterrows():
        tf_label = str(row.get("tf_label") or "").strip()
        if not tf_label:
            continue
        if tf_label in fixed_set:
            label = tf_label
        else:
            label = tf_label if tf_label in selected_tfs else ("other" if other_tfs else tf_label)
        if label not in occupancy:
            continue
        bounds = _placement_bounds(row, seq_len)
        if bounds is None:
            continue
        lo, hi = bounds
        occupancy[label][lo:hi] += 1.0

    fixed_from_composition = set(sub["tf_label"]).intersection(fixed_set)
    missing_counts: dict[str, int] = {}
    for pc_idx, pc in enumerate(constraints):
        name = pc.get("name") or f"promoter_{pc_idx + 1}"
        label_up = f"fixed:{name}:-35"
        label_down = f"fixed:{name}:-10"
        for _, row in solutions.iterrows():
            seq = str(row.get("sequence") or "")
            pair = _select_promoter_pair(seq, pc)
            if pair is None:
                missing_counts[name] = missing_counts.get(name, 0) + 1
                continue
            up_start, down_start = pair
            up_end = up_start + len(pc["upstream"])
            down_end = down_start + len(pc["downstream"])
            up_lo = max(0, min(up_start, seq_len))
            up_hi = max(up_lo, min(up_end, seq_len))
            down_lo = max(0, min(down_start, seq_len))
            down_hi = max(down_lo, min(down_end, seq_len))
            if up_hi > up_lo and label_up in occupancy and label_up not in fixed_from_composition:
                occupancy[label_up][up_lo:up_hi] += 1.0
            if down_hi > down_lo and label_down in occupancy and label_down not in fixed_from_composition:
                occupancy[label_down][down_lo:down_hi] += 1.0

    return occupancy, categories, missing_counts


def _build_tfbs_count_records(
    sub: pd.DataFrame,
    *,
    solutions: pd.DataFrame,
    constraints: list[dict],
) -> pd.DataFrame:
    fixed_labels = _fixed_labels(constraints)
    fixed_set = set(fixed_labels)
    records: list[dict[str, str]] = []
    for _, row in sub.iterrows():
        tf_label = _normalize_tf_label(row.get("tf"), fixed_set)
        tfbs = str(row.get("tfbs") or "").strip().upper()
        if not tf_label or not tfbs:
            continue
        records.append({"category_label": tf_label, "tfbs": tfbs})

    composition_has_fixed = bool(set(sub["tf"].astype(str).tolist()).intersection(fixed_set))
    if not composition_has_fixed:
        for pc_idx, pc in enumerate(constraints):
            name = pc.get("name") or f"promoter_{pc_idx + 1}"
            upstream = str(pc.get("upstream") or "").strip().upper()
            downstream = str(pc.get("downstream") or "").strip().upper()
            if not upstream or not downstream:
                continue
            label_up = f"fixed:{name}:-35"
            label_down = f"fixed:{name}:-10"
            for _, row in solutions.iterrows():
                seq = str(row.get("sequence") or "")
                pair = _select_promoter_pair(seq, pc)
                if pair is None:
                    continue
                records.append({"category_label": label_up, "tfbs": upstream})
                records.append({"category_label": label_down, "tfbs": downstream})

    if not records:
        raise ValueError("placement_map found no TFBS usage for the selected solutions.")
    counts = pd.DataFrame(records).groupby(["category_label", "tfbs"]).size().reset_index(name="count")
    counts["category_label"] = counts["category_label"].astype(str)
    counts["tfbs"] = counts["tfbs"].astype(str)
    counts["rank_key"] = counts["category_label"] + ":" + counts["tfbs"]
    counts = counts.sort_values(by=["count", "rank_key"], ascending=[False, True]).reset_index(drop=True)
    return counts


def _selected_library_members(
    library_members_df: pd.DataFrame,
    *,
    input_name: str,
    plan_name: str,
    sub: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(library_members_df, {"input_name", "plan_name", "tf", "tfbs"}, "library_members.parquet")
    members = library_members_df[
        (library_members_df["input_name"].astype(str) == str(input_name))
        & (library_members_df["plan_name"].astype(str) == str(plan_name))
    ].copy()
    if members.empty:
        raise ValueError(f"library_members.parquet has no rows for placement_map scope {input_name}/{plan_name}.")

    filters = []
    if "library_hash" in sub.columns and "library_hash" in members.columns:
        hashes = {str(h) for h in sub["library_hash"].dropna().astype(str).tolist() if str(h).strip()}
        if hashes:
            filters.append(members["library_hash"].astype(str).isin(hashes))
    if "library_index" in sub.columns and "library_index" in members.columns:
        indices = {int(i) for i in pd.to_numeric(sub["library_index"], errors="coerce").dropna().astype(int).tolist()}
        if indices:
            filters.append(pd.to_numeric(members["library_index"], errors="coerce").isin(sorted(indices)))

    if filters:
        mask = filters[0]
        for extra in filters[1:]:
            mask = mask | extra
        scoped = members[mask].copy()
        if scoped.empty:
            raise ValueError(
                f"library_members.parquet rows did not match selected libraries for {input_name}/{plan_name}."
            )
        return scoped
    return members


def _build_available_tfbs_records(
    members: pd.DataFrame,
    *,
    n_solutions: int,
    constraints: list[dict],
) -> pd.DataFrame:
    fixed_labels = _fixed_labels(constraints)
    fixed_set = set(fixed_labels)
    rows: list[dict[str, str | int]] = []
    for _, row in members.iterrows():
        tf_label = _normalize_tf_label(row.get("tf"), fixed_set)
        tfbs = str(row.get("tfbs") or "").strip().upper()
        if not tf_label or not tfbs:
            continue
        rows.append({"category_label": tf_label, "tfbs": tfbs, "weight": int(n_solutions)})

    for pc_idx, pc in enumerate(constraints):
        name = pc.get("name") or f"promoter_{pc_idx + 1}"
        upstream = str(pc.get("upstream") or "").strip().upper()
        downstream = str(pc.get("downstream") or "").strip().upper()
        if not upstream or not downstream:
            continue
        rows.append({"category_label": f"fixed:{name}:-35", "tfbs": upstream, "weight": int(n_solutions)})
        rows.append({"category_label": f"fixed:{name}:-10", "tfbs": downstream, "weight": int(n_solutions)})

    if not rows:
        raise ValueError("placement_map could not derive available TFBS records from library members.")
    available = pd.DataFrame(rows)
    available["category_label"] = available["category_label"].astype(str)
    available["tfbs"] = available["tfbs"].astype(str)
    available["weight"] = pd.to_numeric(available["weight"], errors="coerce").fillna(0).astype(int)
    available = available[available["weight"] > 0]
    if available.empty:
        raise ValueError("placement_map derived no non-empty available TFBS records.")
    return available


def _allocation_summary_lines(
    *,
    placements_used: int,
    placements_possible: int,
    unique_used: int,
    unique_available: int,
    top10_share: float,
    top50_share: float,
) -> list[str]:
    placements_possible = max(1, int(placements_possible))
    unique_available = max(1, int(unique_available))
    placements_ratio = float(placements_used) / float(placements_possible)
    unique_ratio = float(unique_used) / float(unique_available)
    return [
        f"TFBS placements used / possible: {placements_used}/{placements_possible} ({placements_ratio:.1%})",
        f"unique TFBS-pairs used / available: {unique_used}/{unique_available} ({unique_ratio:.1%})",
        f"top10 share (all TFBS-pairs by usage): {top10_share:.2f}",
        f"top50 share (all TFBS-pairs by usage): {top50_share:.2f}",
    ]


def _place_figure_legend_below_xlabel(
    fig: plt.Figure,
    *,
    ax_xlabel: plt.Axes,
    legend: plt.Legend,
    gap: float = 0.012,
    min_bottom: float = 0.01,
    max_bottom: float = 0.60,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel_bbox = (
        ax_xlabel.xaxis.get_label().get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    )
    target_top = float(xlabel_bbox.y0) - float(gap)
    legend.set_bbox_to_anchor((0.5, target_top), transform=fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    if legend_bbox.y0 < float(min_bottom):
        needed = float(min_bottom) - float(legend_bbox.y0)
        new_bottom = min(float(max_bottom), float(fig.subplotpars.bottom) + needed + 0.005)
        fig.subplots_adjust(bottom=new_bottom)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xlabel_bbox = (
            ax_xlabel.xaxis.get_label().get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        )
        target_top = float(xlabel_bbox.y0) - float(gap)
        legend.set_bbox_to_anchor((0.5, target_top), transform=fig.transFigure)
        fig.canvas.draw()


def _render_occupancy(
    occupancy: dict[str, np.ndarray],
    categories: list[str],
    *,
    seq_len: int,
    input_name: str,
    plan_name: str,
    n_solutions: int,
    alpha: float,
    fixed_label_sequences: dict[str, str] | None,
    style: dict,
) -> tuple[plt.Figure, plt.Axes]:
    width = max(8.4, float(seq_len) * 0.2, float(len(categories)) * 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(width, 3.25))
    x_positions = np.arange(seq_len, dtype=float)
    colors = dict(zip(categories, _colorblind_palette(style, len(categories))))
    fixed_categories = [label for label in categories if str(label).startswith("fixed:")]
    regular_categories = [label for label in categories if label not in fixed_categories]
    draw_order = regular_categories + fixed_categories
    for label in draw_order:
        color = colors[label]
        y = occupancy[label]
        if y.shape[0] != seq_len:
            raise ValueError(
                "Occupancy series length mismatch; expected one value per nucleotide "
                f"(seq_len={seq_len}, got={y.shape[0]} for '{label}')."
            )
        is_fixed = str(label).startswith("fixed:")
        fill_alpha = max(0.12, alpha * (1.1 if is_fixed else 1.0))
        fill_rgb = _darken_color(color, factor=0.84 if is_fixed else 0.88)
        edge_rgb = _darken_color(color, factor=0.56 if is_fixed else 0.62)
        line_width = 0.75 if is_fixed else 0.6
        z_base = 8 if is_fixed else 2
        fill_color = to_rgba(fill_rgb, alpha=fill_alpha)
        edge_color = to_rgba(edge_rgb, alpha=1.0)
        ax.bar(
            x_positions,
            y.astype(float),
            width=0.96,
            align="edge",
            color=fill_color,
            edgecolor=edge_color,
            linewidth=line_width,
            label=_category_display_label(label, fixed_label_sequences=fixed_label_sequences),
            zorder=z_base + (0.5 if is_fixed else 0.0),
        )

    x_pad = max(0.8, float(seq_len) * 0.015)
    ax.set_xlim(0.0 - x_pad, float(seq_len) + x_pad)
    ax.set_xticks(np.arange(0, seq_len + 1, 5, dtype=int))
    ax.set_xlabel("Position (nt)", labelpad=8)
    ax.set_ylabel("Occurrences")
    input_label = str(input_name).replace("plan_pool__", "").replace("_", " ")
    plan_label = str(plan_name).replace("_", " ")
    if input_label == plan_label:
        scope = plan_label
    else:
        scope = f"{plan_label} / {input_label}"
    ax.set_title(f"Occupancy across sequence positions for {scope} (n={n_solutions})")
    _apply_style(ax, style)
    handles, labels = ax.get_legend_handles_labels()
    legend_rows = 1
    if handles:
        deduped: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            deduped[str(label)] = handle
        entry_count = max(1, len(deduped))
        ncol = max(1, min(3, int(np.ceil(np.sqrt(entry_count)))))
        legend_rows = int(np.ceil(float(entry_count) / float(ncol)))
        fig.legend(
            deduped.values(),
            deduped.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=ncol,
            frameon=False,
            fontsize=float(style.get("tick_size", style.get("font_size", 13.0) * 0.74)),
        )
    bottom = max(0.31, 0.22 + 0.07 * float(legend_rows))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=bottom, top=0.88)
    return fig, ax


def _render_tfbs_allocation(
    counts: pd.DataFrame,
    *,
    available: pd.DataFrame,
    input_name: str,
    plan_name: str,
    n_solutions: int,
    top_k_annotation: int,
    fixed_label_sequences: dict[str, str] | None,
    style: dict,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    ranks = np.arange(1, len(counts) + 1)
    values = counts["count"].astype(float).to_numpy()
    total = float(values.sum()) if len(values) else 0.0
    cum = np.cumsum(values) / total if total > 0 else np.zeros_like(values)
    available_unique = int(len(available.drop_duplicates(subset=["category_label", "tfbs"])))
    placements_possible = int(pd.to_numeric(available["weight"], errors="coerce").fillna(0).sum())

    figure_width = max(9.2, float(counts["category_label"].nunique()) * 2.0)
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 6.0), sharex=False)
    ax_rank, ax_cum = axes
    palette = _colorblind_palette(style, max(1, counts["category_label"].nunique() + 1))
    category_order = (
        counts.groupby("category_label")["count"].sum().sort_values(ascending=False).index.astype(str).tolist()
    )
    color_map = {label: palette[idx + 1] for idx, label in enumerate(category_order)}
    ax_rank.plot(
        ranks,
        values,
        color=palette[0],
        linewidth=1.5,
        marker="o",
        markersize=2.9,
        linestyle="-",
        label="all TFBS-pairs",
        zorder=4,
    )
    ax_rank.set_yscale("log")
    ax_rank.set_ylabel("Usage count")
    input_label = str(input_name).replace("plan_pool__", "").replace("_", " ")
    plan_label = str(plan_name).replace("_", " ")
    if input_label == plan_label:
        scope = plan_label
    else:
        scope = f"{plan_label} / {input_label}"
    ax_rank.set_title(f"TFBS usage rank and cumulative share for {scope} (n={n_solutions}).")
    ax_cum.plot(
        ranks,
        cum,
        color=palette[0],
        linewidth=1.5,
        marker="o",
        markersize=2.9,
        linestyle="-",
        label="all TFBS-pairs",
        zorder=4,
    )
    ax_cum.set_ylabel("Cumulative share")
    ax_cum.set_xlabel("TFBS rank within category")
    ax_cum.set_ylim(0.0, 1.03)

    available_category_unique = (
        available.groupby("category_label")[["tfbs"]].nunique().rename(columns={"tfbs": "unique_available"})
    )
    for label in category_order:
        category = counts[counts["category_label"] == label].sort_values(by=["count", "tfbs"], ascending=[False, True])
        if category.empty:
            continue
        cat_values = category["count"].astype(float).to_numpy()
        cat_ranks = np.arange(1, len(category) + 1)
        cat_total = float(cat_values.sum())
        cat_cum = np.cumsum(cat_values) / cat_total if cat_total > 0 else np.zeros_like(cat_values)
        available_unique_cat = int(
            available_category_unique.loc[label, "unique_available"] if label in available_category_unique.index else 0
        )
        label_text = _category_display_label(str(label), fixed_label_sequences=fixed_label_sequences)
        placement_share = (cat_total / total) if total > 0 else 0.0
        legend_label = (
            f"{label_text}: placements {int(cat_total)}/{int(total)} ({placement_share:.1%}), "
            f"unique {len(category)}/{max(1, available_unique_cat)}"
        )
        color = color_map[label]
        ax_rank.plot(cat_ranks, cat_values, color=color, linewidth=1.2, marker="o", markersize=2.5, label=legend_label)
        ax_cum.plot(cat_ranks, cat_cum, color=color, linewidth=1.2, marker="o", markersize=2.5, label=legend_label)

    if values.size > 0:
        y_min = max(0.8, float(np.nanmin(values)) * 0.9)
        y_max = float(np.nanmax(values)) * 1.08
        if y_max <= y_min:
            y_max = y_min * 1.1
        ax_rank.set_ylim(y_min, y_max)

    top10 = values[: min(10, len(values))].sum() / total if total > 0 else 0.0
    top50 = values[: min(50, len(values))].sum() / total if total > 0 else 0.0
    summary = "\n".join(
        _allocation_summary_lines(
            placements_used=int(total),
            placements_possible=placements_possible,
            unique_used=int(len(values)),
            unique_available=available_unique,
            top10_share=top10,
            top50_share=top50,
        )
    )
    ax_rank.text(
        0.98,
        0.95,
        summary,
        transform=ax_rank.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
    )

    if top_k_annotation and top_k_annotation > 0:
        k = min(top_k_annotation, len(values))
        for idx in range(k):
            row = counts.iloc[idx]
            label = (
                f"{_category_display_label(row['category_label'], fixed_label_sequences=fixed_label_sequences)}:"
                f"{_truncate_tfbs(row['tfbs'])}"
            )
            ax_rank.annotate(
                label,
                (ranks[idx], values[idx]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                ha="left",
                va="bottom",
            )

    _apply_style(ax_rank, style)
    _apply_style(ax_cum, style)
    fig.tight_layout()
    handles, labels = ax_rank.get_legend_handles_labels()
    if handles:
        deduped: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            deduped[str(label)] = handle
        entry_count = max(1, len(deduped))
        ncol = max(1, min(4, int(np.ceil(np.sqrt(entry_count)))))
        legend = fig.legend(
            deduped.values(),
            deduped.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=False,
            fontsize=float(style.get("tick_size", style.get("font_size", 13.0) * 0.68)),
        )
        _place_figure_legend_below_xlabel(fig, ax_xlabel=ax_cum, legend=legend)
    return fig, {"rank": ax_rank, "cum": ax_cum}


def plot_placement_map(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    dense_arrays_df: pd.DataFrame,
    library_members_df: pd.DataFrame | None,
    cfg: dict,
    style: Optional[dict] = None,
    occupancy_alpha: float | None = None,
    occupancy_max_categories: int | None = None,
) -> list[Path]:
    if composition_df is None or composition_df.empty:
        raise ValueError("placement_map requires composition.parquet with placements.")
    if dense_arrays_df is None or dense_arrays_df.empty:
        raise ValueError("placement_map requires dense_arrays.parquet with final sequences.")
    _require_columns(
        dense_arrays_df,
        {"id", "sequence", "densegen__input_name", "densegen__plan"},
        "dense_arrays.parquet",
    )
    _require_columns(
        composition_df,
        {"solution_id", "input_name", "plan_name", "tf", "tfbs", "offset"},
        "composition.parquet",
    )
    if "length" not in composition_df.columns and "end" not in composition_df.columns:
        raise ValueError("composition.parquet requires length or end columns.")

    style = _style(style)
    alpha_val = float(occupancy_alpha) if occupancy_alpha is not None else 0.22
    if not (0.0 < alpha_val <= 1.0):
        raise ValueError("placement_map occupancy_alpha must be between 0 and 1.")
    max_cats = int(occupancy_max_categories) if occupancy_max_categories is not None else 12
    if max_cats <= 0:
        raise ValueError("placement_map occupancy_max_categories must be positive.")

    seq_len = _sequence_length_from_cfg(cfg)
    dense_arrays_df = dense_arrays_df.copy()
    dense_arrays_df["densegen__input_name"] = dense_arrays_df["densegen__input_name"].astype(str)
    dense_arrays_df["densegen__plan"] = dense_arrays_df["densegen__plan"].astype(str)
    composition_df = composition_df.copy()
    composition_df["input_name"] = composition_df["input_name"].astype(str)
    composition_df["plan_name"] = composition_df["plan_name"].astype(str)

    paths: list[Path] = []
    for (input_name, plan_name), solutions in dense_arrays_df.groupby(["densegen__input_name", "densegen__plan"]):
        solution_ids = solutions["id"].astype(str).dropna().unique().tolist()
        if not solution_ids:
            raise ValueError(f"placement_map has no accepted solutions for {input_name}/{plan_name}.")
        sub = composition_df[
            (composition_df["input_name"] == input_name)
            & (composition_df["plan_name"] == plan_name)
            & (composition_df["solution_id"].astype(str).isin(solution_ids))
        ].copy()
        if sub.empty:
            raise ValueError(f"placement_map has no placements for {input_name}/{plan_name}.")

        constraints = _promoter_constraints(cfg, str(plan_name))
        fixed_label_sequences = _fixed_label_sequences(constraints)
        n_solutions = len(solution_ids)
        occupancy, categories, missing_counts = _build_occupancy(
            sub,
            solutions=solutions,
            seq_len=seq_len,
            constraints=constraints,
            max_categories=max_cats,
        )

        if missing_counts:
            misses = ", ".join(f"{name}({count})" for name, count in sorted(missing_counts.items()))
            log.warning("placement_map promoter motifs not found for %s/%s: %s", input_name, plan_name, misses)

        fig_occ, _ = _render_occupancy(
            occupancy,
            categories,
            seq_len=seq_len,
            input_name=str(input_name),
            plan_name=str(plan_name),
            n_solutions=n_solutions,
            alpha=alpha_val,
            fixed_label_sequences=fixed_label_sequences,
            style=style,
        )
        base_dir = _stage_b_plan_output_dir(out_path, input_name=str(input_name), plan_name=str(plan_name))
        base_dir.mkdir(parents=True, exist_ok=True)
        occ_path = base_dir / f"occupancy{out_path.suffix}"
        fig_occ.savefig(occ_path)
        plt.close(fig_occ)
        paths.append(occ_path)

    return paths
