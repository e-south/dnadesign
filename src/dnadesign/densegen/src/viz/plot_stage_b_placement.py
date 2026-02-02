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


def plot_placement_map(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    dense_arrays_df: pd.DataFrame,
    cfg: dict,
    style: Optional[dict] = None,
    alpha: float | None = None,
    top_k_tfbs: int | None = None,
    max_categories: int | None = None,
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
    alpha_val = float(alpha) if alpha is not None else 0.3
    if not (0.0 < alpha_val <= 1.0):
        raise ValueError("placement_map alpha must be between 0 and 1.")
    top_k = int(top_k_tfbs) if top_k_tfbs is not None else 20
    if top_k <= 0:
        raise ValueError("placement_map top_k_tfbs must be positive.")
    max_cats = int(max_categories) if max_categories is not None else 12
    if max_cats <= 0:
        raise ValueError("placement_map max_categories must be positive.")

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
        fixed_labels = _fixed_labels(constraints)
        fixed_set = set(fixed_labels)
        if max_cats < len(fixed_labels):
            raise ValueError(
                f"placement_map max_categories={max_cats} is smaller than fixed categories ({len(fixed_labels)})."
            )

        tf_labels = sorted({str(tf) for tf in sub["tf"].astype(str).tolist() if str(tf).strip()})
        regulator_labels = [tf for tf in tf_labels if tf not in fixed_set]
        tf_totals: dict[str, int] = {}
        for _, row in sub.iterrows():
            tf = str(row.get("tf") or "").strip()
            if not tf:
                continue
            bounds = _placement_bounds(row, seq_len)
            if bounds is None:
                continue
            lo, hi = bounds
            if tf not in fixed_set:
                tf_totals[tf] = tf_totals.get(tf, 0) + max(0, hi - lo)

        max_regulators = max(0, max_cats - len(fixed_labels))
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
            tf = str(row.get("tf") or "").strip()
            if not tf:
                continue
            if tf in fixed_set:
                label = tf
            else:
                label = tf if tf in selected_tfs else ("other" if other_tfs else tf)
            if label not in occupancy:
                continue
            bounds = _placement_bounds(row, seq_len)
            if bounds is None:
                continue
            lo, hi = bounds
            occupancy[label][lo:hi] += 1.0

        missing_counts: dict[str, int] = {}
        fixed_from_composition = {tf for tf in tf_labels if tf in fixed_set}
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

        if missing_counts:
            misses = ", ".join(f"{name}({count})" for name, count in sorted(missing_counts.items()))
            log.warning("placement_map promoter motifs not found for %s/%s: %s", input_name, plan_name, misses)

        n_solutions = len(solution_ids)
        x = np.arange(seq_len)
        fig, (ax_occ, ax_tfbs) = plt.subplots(1, 2, figsize=style["figsize"])
        colors = _palette(style, len(categories))
        for label, color in zip(categories, colors):
            y = occupancy[label]
            ax_occ.step(x, y, where="pre", color=color, linewidth=1.0)
            ax_occ.fill_between(x, y, step="pre", alpha=alpha_val, color=color, label=label)

        ax_occ.set_xlim(0, seq_len)
        ax_occ.set_xticks(np.arange(0, seq_len + 1, dtype=int))
        ax_occ.set_xlabel("Position (nt)")
        ax_occ.set_ylabel("Occurrences")
        ax_occ.set_title(f"Placement map - {input_name}/{plan_name} (n={n_solutions})")
        ax_occ.legend(loc="upper right", frameon=bool(style.get("legend_frame", False)))

        counts = sub.groupby(["tf", "tfbs"]).size().reset_index(name="count")
        if counts.empty:
            raise ValueError(f"placement_map found no TFBS usage for {input_name}/{plan_name}.")
        counts = counts.sort_values(by=["count", "tfbs", "tf"], ascending=[False, True, True])
        if len(counts) > top_k:
            top = counts.head(top_k)
            other_count = int(counts["count"].sum() - top["count"].sum())
        else:
            top = counts
            other_count = 0
        labels = [f"{row['tf']}:{_truncate_tfbs(row['tfbs'])}" for _, row in top.iterrows()]
        values = top["count"].astype(int).tolist()
        if other_count > 0:
            labels.append("other")
            values.append(other_count)
        y_pos = np.arange(len(labels))
        ax_tfbs.barh(y_pos, values, color="#4c78a8")
        ax_tfbs.set_yticks(y_pos)
        ax_tfbs.set_yticklabels(labels)
        ax_tfbs.invert_yaxis()
        ax_tfbs.set_xlabel("Usage count")
        ax_tfbs.set_title(f"TFBS usage (top {top_k})")

        _apply_style(ax_occ, style)
        _apply_style(ax_tfbs, style)
        fig.tight_layout()
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths
