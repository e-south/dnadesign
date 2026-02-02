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

from .plot_common import _apply_style, _palette, _safe_filename, _style

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


def _sanitize_tf_label(label: str) -> str:
    label = str(label)
    if "_" in label:
        return label.split("_", 1)[0]
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


def _category_display_label(label: str) -> str:
    if label.startswith("fixed:"):
        return _sanitize_fixed_label(label)
    return label


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


def _build_tfbs_counts(sub: pd.DataFrame) -> pd.DataFrame:
    counts = sub.groupby(["tf", "tfbs"]).size().reset_index(name="count")
    if counts.empty:
        raise ValueError("placement_map found no TFBS usage for the selected solutions.")
    counts["tf"] = counts["tf"].astype(str)
    counts["tfbs"] = counts["tfbs"].astype(str)
    counts["rank_key"] = counts["tf"] + ":" + counts["tfbs"]
    counts = counts.sort_values(by=["count", "rank_key"], ascending=[False, True]).reset_index(drop=True)
    return counts


def _render_occupancy(
    occupancy: dict[str, np.ndarray],
    categories: list[str],
    *,
    seq_len: int,
    input_name: str,
    plan_name: str,
    n_solutions: int,
    alpha: float,
    style: dict,
) -> tuple[plt.Figure, plt.Axes]:
    width = max(8.0, float(seq_len) * 0.2)
    fig, ax = plt.subplots(1, 1, figsize=(width, 2.5))
    x = np.arange(seq_len)
    colors = _palette(style, len(categories))
    for label, color in zip(categories, colors):
        y = occupancy[label]
        ax.step(x, y, where="mid", color=color, linewidth=1.0)
        ax.fill_between(x, y, step="mid", alpha=alpha, color=color, label=_category_display_label(label))

    ax.set_xlim(0, seq_len)
    ax.set_xticks(np.arange(0, seq_len + 1, 5, dtype=int))
    ax.set_xlabel("Position (nt)")
    ax.set_ylabel("Occurrences")
    ax.set_title(f"Placement map - {input_name}/{plan_name} (n={n_solutions})")
    ncol = min(4, max(1, len(categories)))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        ncol=ncol,
        frameon=bool(style.get("legend_frame", False)),
    )
    _apply_style(ax, style)
    fig.tight_layout()
    return fig, ax


def _render_tfbs_allocation(
    counts: pd.DataFrame,
    *,
    input_name: str,
    plan_name: str,
    n_solutions: int,
    top_k_annotation: int,
    style: dict,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    ranks = np.arange(1, len(counts) + 1)
    values = counts["count"].astype(float).to_numpy()
    total = float(values.sum()) if len(values) else 0.0
    cum = np.cumsum(values) / total if total > 0 else np.zeros_like(values)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax_rank, ax_cum = axes
    color = _palette(style, 1)[0]
    ax_rank.plot(ranks, values, color=color, linewidth=1.5)
    ax_rank.set_yscale("log")
    ax_rank.set_ylabel("Usage count")
    ax_rank.set_title(f"TFBS allocation - {input_name}/{plan_name} (n={n_solutions})")

    ax_cum.plot(ranks, cum, color=color, linewidth=1.5)
    ax_cum.set_ylabel("Cumulative share")
    ax_cum.set_xlabel("Rank")
    ax_cum.set_ylim(0.0, 1.0)

    top10 = values[: min(10, len(values))].sum() / total if total > 0 else 0.0
    top50 = values[: min(50, len(values))].sum() / total if total > 0 else 0.0
    summary = "\n".join(
        [
            f"placements: {int(total)}",
            f"unique TFBS: {len(values)}",
            f"top10 share: {top10:.2f}",
            f"top50 share: {top50:.2f}",
        ]
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
            label = f"{_sanitize_tf_label(row['tf'])}:{_truncate_tfbs(row['tfbs'])}"
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
    return fig, {"rank": ax_rank, "cum": ax_cum}


def plot_placement_map(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    dense_arrays_df: pd.DataFrame,
    cfg: dict,
    style: Optional[dict] = None,
    occupancy_alpha: float | None = None,
    occupancy_max_categories: int | None = None,
    tfbs_top_k_annotation: int | None = None,
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
    alpha_val = float(occupancy_alpha) if occupancy_alpha is not None else 0.3
    if not (0.0 < alpha_val <= 1.0):
        raise ValueError("placement_map occupancy_alpha must be between 0 and 1.")
    max_cats = int(occupancy_max_categories) if occupancy_max_categories is not None else 12
    if max_cats <= 0:
        raise ValueError("placement_map occupancy_max_categories must be positive.")
    top_k_annotation = int(tfbs_top_k_annotation) if tfbs_top_k_annotation is not None else 0
    if top_k_annotation < 0:
        raise ValueError("placement_map tfbs_top_k_annotation must be >= 0.")

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
            style=style,
        )
        occ_name = (
            f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}__occupancy{out_path.suffix}"
        )
        occ_path = out_path.parent / occ_name
        fig_occ.savefig(occ_path)
        plt.close(fig_occ)
        paths.append(occ_path)

        counts = _build_tfbs_counts(sub)
        fig_tfbs, _ = _render_tfbs_allocation(
            counts,
            input_name=str(input_name),
            plan_name=str(plan_name),
            n_solutions=n_solutions,
            top_k_annotation=top_k_annotation,
            style=style,
        )
        tfbs_name = (
            f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}"
            f"__tfbs_allocation{out_path.suffix}"
        )
        tfbs_path = out_path.parent / tfbs_name
        fig_tfbs.savefig(tfbs_path)
        plt.close(fig_tfbs)
        paths.append(tfbs_path)
    return paths
