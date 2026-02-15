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
from dnadesign.cruncher.analysis.plots._style import apply_axes_style


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _score_scale_label(objective_config: dict[str, object] | None) -> str:
    cfg = objective_config if isinstance(objective_config, dict) else {}
    scale = str(cfg.get("score_scale") or "normalized-llr").strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return "raw-LLR"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return "norm-LLR"
    if scale == "logp":
        return "logp"
    return scale


def _objective_scalar_semantics(objective_config: dict[str, object] | None) -> str:
    cfg = objective_config if isinstance(objective_config, dict) else {}
    combine = str(cfg.get("combine") or "min").strip().lower()
    softmin_cfg = cfg.get("softmin")
    softmin_enabled = isinstance(softmin_cfg, dict) and bool(softmin_cfg.get("enabled"))
    scale_label = _score_scale_label(cfg)
    if combine == "sum":
        return f"sum TF best-window {scale_label}"
    if combine == "min" and softmin_enabled:
        return f"soft-min TF best-window {scale_label}"
    return f"min TF best-window {scale_label}"


def _resolve_joint_score(
    elites_df: pd.DataFrame,
    *,
    objective_config: dict[str, object] | None = None,
) -> tuple[pd.Series, str]:
    for column in ("combined_score_final", "objective_scalar"):
        if column in elites_df.columns:
            values = _safe_numeric(elites_df[column])
            if values.notna().any():
                return values, column
    score_columns = [col for col in elites_df.columns if col.startswith("score_")]
    if score_columns:
        matrix = elites_df[score_columns].apply(_safe_numeric).to_numpy(dtype=float)
        if matrix.size and np.isfinite(matrix).any():
            cfg = objective_config if isinstance(objective_config, dict) else {}
            combine = str(cfg.get("combine") or "min").strip().lower()
            softmin_cfg = cfg.get("softmin")
            softmin_enabled = isinstance(softmin_cfg, dict) and bool(softmin_cfg.get("enabled"))
            if combine == "sum":
                return pd.Series(np.nansum(matrix, axis=1), index=elites_df.index, dtype=float), "reconstructed_sum"
            if softmin_enabled:
                beta = cfg.get("softmin_final_beta_used")
                try:
                    beta_val = float(beta) if beta is not None else np.nan
                except (TypeError, ValueError, OverflowError):
                    beta_val = np.nan
                if np.isfinite(beta_val) and beta_val > 0:
                    score = []
                    for row in matrix:
                        vals = row[np.isfinite(row)]
                        if vals.size == 0:
                            score.append(np.nan)
                            continue
                        scaled = -beta_val * vals
                        max_scaled = float(np.max(scaled))
                        logsum = max_scaled + float(np.log(np.exp(scaled - max_scaled).sum()))
                        score.append(float(-logsum / beta_val))
                    return pd.Series(score, index=elites_df.index, dtype=float), "reconstructed_softmin"
            return pd.Series(np.nanmin(matrix, axis=1), index=elites_df.index, dtype=float), "reconstructed_min"
    return pd.Series(np.nan, index=elites_df.index, dtype=float), "missing"


def _joint_score_ylabel(
    *,
    objective_config: dict[str, object] | None,
    source_column: str,
) -> str:
    del source_column
    semantics = _objective_scalar_semantics(objective_config)
    return semantics[:1].upper() + semantics[1:]


def _full_nn_xlabel() -> str:
    return "NN full-seq Hamming to closest selected elite (bp)"


def _hamming_distance_bp(seq_a: str, seq_b: str) -> float:
    if len(seq_a) != len(seq_b):
        raise ValueError("Elite full-sequence distance requires equal sequence lengths.")
    if len(seq_a) == 0:
        raise ValueError("Elite full-sequence distance requires non-empty sequences.")
    mismatches = float(sum(int(base_a != base_b) for base_a, base_b in zip(seq_a, seq_b, strict=False)))
    if mismatches < 0:
        raise ValueError("Elite full-sequence Hamming distance must be non-negative.")
    return mismatches


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
            distance = _hamming_distance_bp(seqs[i], seqs[j])
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
    objective_config: dict[str, object] | None = None,
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
    joint_score, score_source = _resolve_joint_score(elites, objective_config=objective_config)
    elites["joint_score"] = joint_score
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
    ax_scatter.scatter(
        scatter_df["d_full_nn"].to_numpy(dtype=float),
        scatter_df["joint_score"].to_numpy(dtype=float),
        c="#ffffff",
        s=74,
        edgecolors="#9a9a9a",
        linewidths=1.0,
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
                fontsize=8,
                color="#1f1f1f",
            )
    ax_scatter.set_xlabel(_full_nn_xlabel(), fontsize=13)
    ax_scatter.set_ylabel(
        _joint_score_ylabel(objective_config=objective_config, source_column=score_source),
        fontsize=13,
    )
    ax_scatter.set_title("Score vs primary-sequence diversity", fontsize=14)
    apply_axes_style(ax_scatter, ygrid=True, xgrid=True, tick_labelsize=12, title_size=14, label_size=14)

    rank_order = elites["id"].tolist()
    matrix_order = [ids.index(elite_id) for elite_id in rank_order if elite_id in ids]
    matrix_view = full_matrix[np.ix_(matrix_order, matrix_order)] if matrix_order else full_matrix
    sequence_length = int(len(str(elites["sequence"].iloc[0])))
    heat = ax_heat.imshow(matrix_view, cmap="BuGn_r", vmin=0.0, vmax=float(sequence_length), aspect="auto")
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
    ax_heat.set_xticklabels(tick_labels, fontsize=11, rotation=0, ha="center")
    ax_heat.set_yticklabels(tick_labels, fontsize=11)
    ax_heat.set_title("Pairwise full-sequence distance matrix", fontsize=14)
    ax_heat.set_xlabel("Elite", fontsize=14)
    ax_heat.set_ylabel("Elite", fontsize=14)
    apply_axes_style(ax_heat, ygrid=False, xgrid=False, tick_labelsize=12, title_size=14, label_size=14)
    cbar = fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Hamming distance (bp)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    d_core_median = float(np.nanmedian(elites["d_core_nn"].to_numpy(dtype=float))) if has_finite else None
    d_full_vals = _safe_numeric(elites["d_full_nn"]).dropna()
    d_full_median_bp = float(d_full_vals.median()) if not d_full_vals.empty else None
    d_full_median_norm = (
        float(d_full_median_bp / float(sequence_length))
        if d_full_median_bp is not None and sequence_length > 0
        else None
    )
    core_zero_but_full_diverse = bool(all_zero_core and d_full_median_bp is not None and d_full_median_bp > 0)
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
        "d_full_median": d_full_median_norm,
        "d_full_median_bp": d_full_median_bp,
        "core_zero_but_full_diverse": core_zero_but_full_diverse,
    }
