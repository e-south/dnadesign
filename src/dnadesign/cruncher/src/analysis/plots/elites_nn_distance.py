"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/elites_nn_distance.py

Render elite diversity panels that relate joint score to sequence diversity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots._style import apply_axes_style, place_figure_caption


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _resolve_joint_score(elites_df: pd.DataFrame) -> pd.Series:
    for column in ("combined_score_final", "objective_scalar"):
        if column in elites_df.columns:
            values = _safe_numeric(elites_df[column])
            if values.notna().any():
                return values
    score_columns = [col for col in elites_df.columns if col.startswith("score_")]
    if score_columns:
        matrix = elites_df[score_columns].apply(_safe_numeric).to_numpy(dtype=float)
        if matrix.size:
            return pd.Series(np.nanmin(matrix, axis=1), index=elites_df.index, dtype=float)
    return pd.Series(np.nan, index=elites_df.index, dtype=float)


def _normalized_hamming(seq_a: str, seq_b: str) -> float:
    if len(seq_a) != len(seq_b):
        raise ValueError("Elite full-sequence distance requires equal sequence lengths.")
    if len(seq_a) == 0:
        raise ValueError("Elite full-sequence distance requires non-empty sequences.")
    mismatches = sum(int(base_a != base_b) for base_a, base_b in zip(seq_a, seq_b, strict=False))
    value = mismatches / float(len(seq_a))
    if value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError("Elite full-sequence normalized Hamming distance must be in [0, 1].")
    return float(np.clip(value, 0.0, 1.0))


def _full_distance_matrix(elites_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    if "sequence" not in elites_df.columns:
        raise ValueError("Elites table missing required sequence column for full-distance panel.")
    if "id" not in elites_df.columns:
        raise ValueError("Elites table missing required id column for full-distance panel.")
    ids = elites_df["id"].astype(str).tolist()
    seqs = elites_df["sequence"].astype(str).tolist()
    n_items = len(ids)
    matrix = np.zeros((n_items, n_items), dtype=float)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            distance = _normalized_hamming(seqs[i], seqs[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return ids, matrix


def _nearest_neighbor_distances(ids: list[str], matrix: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if matrix.size == 0:
        return result
    for idx, elite_id in enumerate(ids):
        row = np.delete(matrix[idx, :], idx)
        if row.size == 0:
            result[str(elite_id)] = None
            continue
        finite = row[np.isfinite(row)]
        result[str(elite_id)] = float(np.min(finite)) if finite.size else None
    return result


def _text_panel(
    ax: plt.Axes,
    *,
    n_elites: int,
    nn_vals: pd.Series,
    baseline_vals: pd.Series | None,
) -> None:
    ax.axis("off")
    lines = [
        "Elite diversity panel requires at least two elites with sequences.",
        f"n_elites={n_elites}",
    ]
    if not nn_vals.empty:
        lines.append(f"core-NN median={float(nn_vals.median()):.3f}")
    if baseline_vals is not None and not baseline_vals.empty:
        lines.append(f"baseline core-NN median={float(baseline_vals.median()):.3f}")
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=10, color="#444444")


def plot_elites_nn_distance(
    nn_df: pd.DataFrame,
    out_path: Path,
    *,
    elites_df: pd.DataFrame | None = None,
    baseline_nn: pd.Series | None = None,
    dpi: int,
    png_compress_level: int,
) -> dict[str, object]:
    if nn_df is None or nn_df.empty or "nn_dist" not in nn_df.columns:
        raise ValueError("Nearest-neighbor distance table is empty or missing nn_dist.")
    nn_vals = _safe_numeric(nn_df["nn_dist"]).dropna()
    has_finite = not nn_vals.empty
    all_zero_core = bool(
        has_finite
        and np.isclose(float(nn_vals.min()), 0.0, rtol=0.0, atol=1.0e-12)
        and np.isclose(float(nn_vals.max()), 0.0, rtol=0.0, atol=1.0e-12)
    )
    baseline_vals = _safe_numeric(baseline_nn).dropna() if baseline_nn is not None else None
    n_elites = int(len(elites_df)) if elites_df is not None else int(len(nn_df))

    if elites_df is None or elites_df.empty or "sequence" not in elites_df.columns or len(elites_df) < 2:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        _text_panel(ax, n_elites=n_elites, nn_vals=nn_vals, baseline_vals=baseline_vals)
        place_figure_caption(fig, "Core distance only: provide elite sequences for the full diversity panel.")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
        plt.close(fig)
        return {
            "panel_kind": "text",
            "n_elites": n_elites,
            "has_finite_distances": has_finite,
            "all_zero_core": all_zero_core,
            "d_core_median": float(nn_vals.median()) if has_finite else None,
            "d_full_median": None,
            "core_zero_but_full_diverse": False,
        }

    elites = elites_df.copy()
    if "id" not in elites.columns:
        raise ValueError("Elites table missing required id column for diversity panel.")
    elites["id"] = elites["id"].astype(str)
    elites["rank"] = _safe_numeric(elites["rank"]) if "rank" in elites.columns else np.nan
    elites["joint_score"] = _resolve_joint_score(elites)
    core_nn_by_id = {str(elite_id): value for elite_id, value in nn_df[["elite_id", "nn_dist"]].itertuples(index=False)}
    ids, full_matrix = _full_distance_matrix(elites)
    full_nn_by_id = _nearest_neighbor_distances(ids, full_matrix)
    elites["d_core_nn"] = elites["id"].map(core_nn_by_id)
    elites["d_full_nn"] = elites["id"].map(full_nn_by_id)
    elites["d_core_nn"] = _safe_numeric(elites["d_core_nn"])
    elites["d_full_nn"] = _safe_numeric(elites["d_full_nn"])
    elites = elites.sort_values("rank", na_position="last")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax_scatter, ax_heat = axes

    valid_scatter = elites["d_full_nn"].notna() & elites["joint_score"].notna()
    scatter_df = elites[valid_scatter].copy()
    color_values = (
        scatter_df["rank"].to_numpy(dtype=float)
        if "rank" in scatter_df.columns and scatter_df["rank"].notna().any()
        else np.arange(len(scatter_df), dtype=float)
    )
    scatter = ax_scatter.scatter(
        scatter_df["d_full_nn"].to_numpy(dtype=float),
        scatter_df["joint_score"].to_numpy(dtype=float),
        c=color_values,
        cmap="viridis",
        s=48,
        edgecolors="#111111",
        linewidths=0.7,
        alpha=0.9,
    )
    if "rank" in scatter_df.columns:
        for x_val, y_val, rank in zip(
            scatter_df["d_full_nn"].to_numpy(dtype=float),
            scatter_df["joint_score"].to_numpy(dtype=float),
            scatter_df["rank"],
            strict=False,
        ):
            if np.isnan(rank):
                continue
            ax_scatter.annotate(
                str(int(rank)),
                xy=(x_val, y_val),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="#1f1f1f",
            )
    ax_scatter.set_xlabel("Nearest-neighbor distance (full-sequence normalized Hamming)")
    ax_scatter.set_ylabel("Joint score (selected elites)")
    ax_scatter.set_title("Score vs primary-sequence diversity")
    apply_axes_style(ax_scatter, ygrid=True)
    if len(scatter_df) > 0:
        cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
        cbar.set_label("Elite rank", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    rank_order = elites["id"].tolist()
    matrix_order = [ids.index(elite_id) for elite_id in rank_order if elite_id in ids]
    matrix_view = full_matrix[np.ix_(matrix_order, matrix_order)] if matrix_order else full_matrix
    heat = ax_heat.imshow(matrix_view, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    tick_labels = []
    for elite_id in rank_order:
        row = elites[elites["id"] == elite_id].iloc[0]
        rank_value = row.get("rank")
        if pd.notna(rank_value):
            tick_labels.append(str(int(rank_value)))
        else:
            tick_labels.append(elite_id)
    ax_heat.set_xticks(np.arange(len(tick_labels)))
    ax_heat.set_yticks(np.arange(len(tick_labels)))
    ax_heat.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
    ax_heat.set_yticklabels(tick_labels, fontsize=7)
    ax_heat.set_title("Pairwise full-sequence distance matrix")
    ax_heat.set_xlabel("Elite")
    ax_heat.set_ylabel("Elite")
    apply_axes_style(ax_heat, ygrid=False)
    cbar = fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Hamming distance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    d_core_median = float(np.nanmedian(elites["d_core_nn"].to_numpy(dtype=float))) if has_finite else None
    d_full_vals = _safe_numeric(elites["d_full_nn"]).dropna()
    d_full_median = float(d_full_vals.median()) if not d_full_vals.empty else None
    core_zero_but_full_diverse = bool(all_zero_core and d_full_median is not None and d_full_median > 0)
    caption = (
        f"D_core median={d_core_median:.3f} | D_full median={d_full_median:.3f}"
        if d_core_median is not None and d_full_median is not None
        else "Diversity metrics unavailable for one or more elites."
    )
    if core_zero_but_full_diverse:
        caption += " Motif-core signatures are identical; diversity exists in non-core positions."
    place_figure_caption(fig, caption)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "panel_kind": "diversity_panel",
        "n_elites": int(len(elites)),
        "has_finite_distances": has_finite,
        "all_zero_core": all_zero_core,
        "d_core_median": d_core_median,
        "d_full_median": d_full_median,
        "core_zero_but_full_diverse": core_zero_but_full_diverse,
    }
