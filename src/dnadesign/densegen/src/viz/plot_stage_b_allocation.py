"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_b_allocation.py

Stage-B TFBS allocation summaries factored out from the occupancy plot module.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_stage_b_placement import (
    _apply_style,
    _category_display_label,
    _collapse_fixed_component_label,
    _colorblind_palette,
    _fixed_component_label,
    _fixed_labels,
    _normalize_tf_label,
    _place_figure_legend_below_xlabel,
    _require_columns,
    _select_promoter_pair,
    _truncate_tfbs,
)


def _build_tfbs_count_records(
    sub: pd.DataFrame,
    *,
    solutions: pd.DataFrame,
    constraints: list[dict],
    aggregate_fixed_components: bool = False,
    plot_label: str = "placement_occupancy_map",
) -> pd.DataFrame:
    fixed_labels = _fixed_labels(constraints, aggregate_fixed_components=aggregate_fixed_components)
    fixed_set = set(fixed_labels)
    records: list[dict[str, str]] = []
    for _, row in sub.iterrows():
        tf_label = _collapse_fixed_component_label(
            _normalize_tf_label(row.get("tf"), fixed_set),
            aggregate_fixed_components=aggregate_fixed_components,
        )
        tfbs = str(row.get("tfbs") or "").strip().upper()
        if not tf_label or not tfbs:
            continue
        records.append({"category_label": tf_label, "tfbs": tfbs})

    composition_fixed_labels = {
        _collapse_fixed_component_label(
            _normalize_tf_label(label, fixed_set),
            aggregate_fixed_components=aggregate_fixed_components,
        )
        for label in sub["tf"].astype(str).tolist()
    }
    composition_has_fixed = bool(composition_fixed_labels.intersection(fixed_set))
    if not composition_has_fixed:
        for pc_idx, pc in enumerate(constraints):
            upstream = str(pc.get("upstream") or "").strip().upper()
            downstream = str(pc.get("downstream") or "").strip().upper()
            if not upstream or not downstream:
                continue
            label_up = _fixed_component_label(
                pc,
                pc_idx,
                "-35",
                aggregate_fixed_components=aggregate_fixed_components,
            )
            label_down = _fixed_component_label(
                pc,
                pc_idx,
                "-10",
                aggregate_fixed_components=aggregate_fixed_components,
            )
            for _, row in solutions.iterrows():
                seq = str(row.get("sequence") or "")
                pair = _select_promoter_pair(seq, pc)
                if pair is None:
                    continue
                records.append({"category_label": label_up, "tfbs": upstream})
                records.append({"category_label": label_down, "tfbs": downstream})

    if not records:
        raise ValueError(f"{plot_label} found no TFBS usage for the selected solutions.")
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
    plot_label: str = "placement_occupancy_map",
) -> pd.DataFrame:
    _require_columns(library_members_df, {"input_name", "plan_name", "tf", "tfbs"}, "library_members.parquet")
    members = library_members_df[
        (library_members_df["input_name"].astype(str) == str(input_name))
        & (library_members_df["plan_name"].astype(str) == str(plan_name))
    ].copy()
    if members.empty:
        raise ValueError(f"library_members.parquet has no rows for {plot_label} scope {input_name}/{plan_name}.")

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
    aggregate_fixed_components: bool = False,
    plot_label: str = "placement_occupancy_map",
) -> pd.DataFrame:
    fixed_labels = _fixed_labels(constraints, aggregate_fixed_components=aggregate_fixed_components)
    fixed_set = set(fixed_labels)
    rows: list[dict[str, str | int]] = []
    for _, row in members.iterrows():
        tf_label = _collapse_fixed_component_label(
            _normalize_tf_label(row.get("tf"), fixed_set),
            aggregate_fixed_components=aggregate_fixed_components,
        )
        tfbs = str(row.get("tfbs") or "").strip().upper()
        if not tf_label or not tfbs:
            continue
        rows.append({"category_label": tf_label, "tfbs": tfbs, "weight": int(n_solutions)})

    for pc_idx, pc in enumerate(constraints):
        upstream = str(pc.get("upstream") or "").strip().upper()
        downstream = str(pc.get("downstream") or "").strip().upper()
        if not upstream or not downstream:
            continue
        rows.append(
            {
                "category_label": _fixed_component_label(
                    pc,
                    pc_idx,
                    "-35",
                    aggregate_fixed_components=aggregate_fixed_components,
                ),
                "tfbs": upstream,
                "weight": int(n_solutions),
            }
        )
        rows.append(
            {
                "category_label": _fixed_component_label(
                    pc,
                    pc_idx,
                    "-10",
                    aggregate_fixed_components=aggregate_fixed_components,
                ),
                "tfbs": downstream,
                "weight": int(n_solutions),
            }
        )

    if not rows:
        raise ValueError(f"{plot_label} could not derive available TFBS records from library members.")
    available = pd.DataFrame(rows)
    available["category_label"] = available["category_label"].astype(str)
    available["tfbs"] = available["tfbs"].astype(str)
    available["weight"] = pd.to_numeric(available["weight"], errors="coerce").fillna(0).astype(int)
    available = available[available["weight"] > 0]
    if available.empty:
        raise ValueError(f"{plot_label} derived no non-empty available TFBS records.")
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
    scope = plan_label if input_label == plan_label else f"{plan_label} / {input_label}"
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
