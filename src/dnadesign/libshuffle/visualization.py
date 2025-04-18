"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py
  
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

def plot_scatter(subsamples, config, output_dir, run_info):
    """
    Single-panel scatter plot of subsamples:
      - X-axis: mean pairwise cosine dissimilarity (Evo2).
      - Y-axis: user-selected metric ("billboard", "euclidean", or "log1p_euclidean").
      - Threshold logic (joint_selection) determines which points are 'hit-zone'.
      - Non-hit-zone points: base_color with non_hit_zone_alpha.
      - Hit-zone points: colormap based on unique cluster count, with hit_zone_alpha.
      - A color bar for cluster count is shown on the right.
      - Only the top 5 hit-zone points by cluster count are annotated.
    """
    sns.set_style("ticks")

    # ------------------
    # Extract Plot Config
    # ------------------
    plot_cfg = config.get("plot", {})
    base_color = plot_cfg.get("base_color", "gray")
    highlight_cmap = plt.get_cmap("coolwarm")  # or another colormap
    alpha_non_hit = plot_cfg.get("non_hit_zone_alpha", 0.3)
    alpha_hit = plot_cfg.get("hit_zone_alpha", 0.8)
    marker_size = plot_cfg.get("marker_size", 20)
    size_by_cluster = plot_cfg.get("size_by_cluster", False)
    size_multiplier = plot_cfg.get("size_multiplier", 50)
    fig_size = plot_cfg.get("figure_size", [7, 5])
    dpi = plot_cfg.get("dpi", 600)
    y_axis_metric = plot_cfg.get("y_axis_metric", "billboard")
    annotate_x_threshold = plot_cfg.get("annotate_x_threshold", False)

    # ------------------
    # Determine X, Y Axes
    # ------------------
    x_vals = np.array([s["evo2_metric"] for s in subsamples])
    bm_cfg = config.get("billboard_metric", {})
    log_transform = plot_cfg.get("log_transform", False)

    # Decide Y based on y_axis_metric
    if y_axis_metric == "billboard":
        composite_enabled = bm_cfg.get("composite_score", False)
        if not composite_enabled:
            raw_y = np.array([s["billboard_metric"] for s in subsamples])
            y_vals = np.log2(raw_y) if log_transform else raw_y
            y_label = "log₂(Core Metric)" if log_transform else "Core Metric"
        else:
            y_vals = np.array([s["billboard_metric"] for s in subsamples])
            y_label = "Composite Billboard Metric"
    elif y_axis_metric in ["euclidean", "log1p_euclidean"]:
        l2_vals = np.array([s.get("evo2_metric_l2", 0.0) for s in subsamples])
        if y_axis_metric == "log1p_euclidean":
            y_vals = np.log1p(l2_vals)
            y_label = "log1p(Euclidean Distance)"
        else:
            y_vals = l2_vals
            y_label = "Euclidean Distance"
    else:
        raise ValueError(f"Unsupported y_axis_metric: {y_axis_metric}")

    # ------------------
    # Threshold Logic
    # ------------------
    joint_sel = config.get("joint_selection", {})
    sel_method = joint_sel.get("method", "null").lower()  # "null", "x_only", "both"

    def compute_threshold(values, mode, val):
        if mode == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            return q3 + val * iqr
        elif mode == "percentile":
            return np.percentile(values, val)
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")

    # X-threshold
    threshold_x_mode = joint_sel.get("threshold_x_mode", "iqr")
    threshold_x_value = joint_sel.get("threshold_x_value", 1.5)
    x_threshold = None
    if sel_method in ["x_only", "both"]:
        x_threshold = compute_threshold(x_vals, threshold_x_mode, threshold_x_value)

    # Y-threshold (only if method == "both")
    y_threshold = None
    if sel_method == "both":
        threshold_y_mode = joint_sel.get("threshold_y_mode", "iqr")
        threshold_y_value = joint_sel.get("threshold_y_value", 1.5)
        y_threshold = compute_threshold(y_vals, threshold_y_mode, threshold_y_value)

    # Identify hit-zone mask
    if sel_method == "null":
        hit_mask = np.zeros_like(x_vals, dtype=bool)
    elif sel_method == "x_only":
        hit_mask = (x_vals >= x_threshold)
    elif sel_method == "both":
        hit_mask = (x_vals >= x_threshold) & (y_vals >= y_threshold)
    else:
        raise ValueError(f"Unsupported joint_selection.method: {sel_method}")

    # ------------------
    # Create Figure
    # ------------------
    fig, ax = plt.subplots(figsize=fig_size)

    # Unique cluster counts
    cluster_counts = np.array([s.get("unique_cluster_count", 0) for s in subsamples])
    cmin, cmax = cluster_counts.min(), cluster_counts.max()
    spread = cmax - cmin if cmax > cmin else 1.0  # avoid div-by-zero

    # Prepare arrays for final color, alpha, and size
    final_colors = []
    final_alphas = []
    final_sizes = []
    for i, s in enumerate(subsamples):
        if hit_mask[i]:
            # Hit-zone => color by cluster count
            norm_val = (cluster_counts[i] - cmin) / spread
            color = highlight_cmap(norm_val)
            alpha = alpha_hit
            if size_by_cluster:
                final_size = marker_size + (cluster_counts[i] / max(cmax, 1)) * size_multiplier
            else:
                final_size = marker_size
        else:
            # Non-hit => base color
            color = base_color
            alpha = alpha_non_hit
            final_size = marker_size

        final_colors.append(color)
        final_alphas.append(alpha)
        final_sizes.append(final_size)

    # Scatter all points
    for i in range(len(x_vals)):
        ax.scatter(x_vals[i], y_vals[i],
                   c=[final_colors[i]],
                   alpha=final_alphas[i],
                   s=final_sizes[i],
                   edgecolor="none")

    # Draw threshold lines if relevant
    if sel_method in ["x_only", "both"] and x_threshold is not None:
        ax.axvline(x=x_threshold, linestyle="--", color="lightgray")
    if sel_method == "both" and y_threshold is not None:
        ax.axhline(y=y_threshold, linestyle="--", color="lightgray")

    # Optionally annotate points above x_threshold if "annotate_x_threshold" is True
    if (sel_method in ["x_only", "both"]) and annotate_x_threshold:
        for i, s in enumerate(subsamples):
            if hit_mask[i]:
                sub_id = s["subsample_id"].split("_")[-1]
                ax.annotate(sub_id, (x_vals[i], y_vals[i]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=8, color="gray")

    # Now, among hit-zone points only, find top 5 by cluster count for black annotation
    hit_indices = np.where(hit_mask)[0]
    if len(hit_indices) > 0:
        hit_zone_clusters = cluster_counts[hit_indices]
        top5_hit = np.argsort(-hit_zone_clusters)[:5]  # top 5 in the hit zone
        top5_global_indices = hit_indices[top5_hit]
        # Annotate those top 5 in black
        for idx in top5_global_indices:
            sub_id = subsamples[idx]["subsample_id"].split("_")[-1]
            ax.annotate(sub_id, (x_vals[idx], y_vals[idx]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="black")

    # Colorbar for cluster count (only if we have at least one hit-zone point)
    if hit_mask.any():
        norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=highlight_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Unique Cluster Count")

    # Axes labels, title, etc.
    ax.set_xlabel("Mean pairwise cosine dissimilarity in Evo2 latent space")
    ax.set_ylabel(y_label)
    ax.set_title("Subsample Diversity (Single Panel)")

    # Final figure title
    overall_title = (f"Subsample Diversity Analysis\n"
                     f"Draws: {run_info.get('num_draws', 'N/A')} · "
                     f"Subsample size: {run_info.get('subsample_size', 'N/A')}")
    fig.suptitle(overall_title, fontsize=12, y=0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = plot_cfg.get("filename", "scatter_summary.png")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_kde_coremetrics(subsamples, config, output_dir):
    import pandas as pd
    import seaborn as sns
    sns.set_style("ticks")
    
    bm_config = config.get("billboard_metric", {})
    composite_enabled = bm_config.get("composite_score", False)
    core_metrics = bm_config.get("core_metrics", [])
    data = []
    
    if composite_enabled and bm_config.get("method") == "cds":
        for sub in subsamples:
            cds = sub.get("cds_components")
            if not cds:
                continue
            for key in ["cds_score", "dominance", "program_diversity"]:
                data.append({"Metric": key, "Value": cds.get(key)})
    else:
        if not core_metrics:
            raise ValueError("No core metrics defined in config under billboard_metric/core_metrics")
        for sub in subsamples:
            raw_vector = sub.get("raw_billboard_vector")
            if raw_vector is None:
                raw_vector = [sub.get("billboard_metric")]
            if len(raw_vector) != len(core_metrics):
                continue
            for i, metric in enumerate(core_metrics):
                data.append({"Metric": metric, "Value": raw_vector[i]})
    
    if not data:
        raise ValueError("No raw core metrics data available for KDE plot.")
    
    df_raw = pd.DataFrame(data)
    
    fig, ax = plt.subplots()
    sns.kdeplot(data=df_raw, x="Value", hue="Metric", common_norm=False, ax=ax)
    ax.set_title("Core Metric Distributions (Raw)")
    ax.set_xlabel("Metric Value")
    ax.set_ylabel("Density")
    dpi = config.get("plot", {}).get("dpi", 600)
    raw_path = os.path.join(output_dir, "kde_coremetrics_raw.png")
    plt.savefig(raw_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    df_z = df_raw.copy()
    def zscore(x):
        std = x.std(ddof=1)
        if std == 0:
            return 0
        return (x - x.mean()) / std
    df_z["ZScore"] = df_z.groupby("Metric")["Value"].transform(zscore)
    
    fig, ax = plt.subplots()
    sns.kdeplot(data=df_z, x="ZScore", hue="Metric", common_norm=False, ax=ax)
    ax.set_title("Core Metric Distributions (Z-Score Normalized)")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Density")
    zscore_path = os.path.join(output_dir, "kde_coremetrics_zscore.png")
    plt.savefig(zscore_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return raw_path, zscore_path


def plot_pairplot_coremetrics(subsamples, config, output_dir):
    """
    Pairwise scatter‑plot (and upper‑triangle correlations) of:

      • every Billboard core metric requested in the config **plus**  
      • `evo2_metric`         (cosine dissimilarity)  
      • `evo2_metric_l2`      (Euclidean distance)

    If `billboard_metric.method == "cds"`, the CDS components are used instead
    of the raw core metrics, but Evo2 metrics are *still* appended.
    """
    import pandas as pd

    sns.set_style("ticks")
    plot_cfg = config.get("plot", {})
    dpi = plot_cfg.get("dpi", 600)

    bm_cfg = config.get("billboard_metric", {})
    cds_mode = bm_cfg.get("method") == "cds"

    # ── Decide which metric keys we want ─────────────────────────────────────
    if cds_mode:
        wanted = ["cds_score", "dominance", "program_diversity"]
    else:
        wanted = bm_cfg.get("core_metrics", [])
        if not wanted:
            raise ValueError(
                "No core metrics defined in config under billboard_metric/core_metrics"
            )

    # Always append Evo2 metrics when available
    evo2_keys = ["evo2_metric", "evo2_metric_l2"]
    wanted += evo2_keys

    # ── Collect data ─────────────────────────────────────────────────────────
    rows = []
    for sub in subsamples:
        row = {}
        # 1) Billboard‑derived numbers
        if cds_mode:
            cds = sub.get("cds_components", {})
            for k in ("cds_score", "dominance", "program_diversity"):
                row[k] = cds.get(k)
        else:
            vec = sub.get("raw_billboard_vector")
            if vec is None:
                vec = [sub.get("billboard_metric")]
            core_keys = bm_cfg.get("core_metrics", [])
            for k, v in zip(core_keys, vec):
                row[k] = v
        # 2) Evo2 metrics (may be None)
        row["evo2_metric"]     = sub.get("evo2_metric")
        row["evo2_metric_l2"]  = sub.get("evo2_metric_l2")
        rows.append(row)

    # Build DF → ensure every wanted column exists, then prune empty ones
    import pandas as pd
    df = pd.DataFrame(rows).reindex(columns=wanted)
    df = df.dropna(axis=1, how="all")        # drop metrics that never appeared

    # Need at least 2 numeric columns with some data
    if df.shape[1] < 2 or df.dropna(how="all").empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5,
                "Pairplot not available (insufficient data)",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        out = os.path.join(output_dir, "pairplot_coremetrics.png")
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ── Build pair‑grid ──────────────────────────────────────────────────────
    def corrfunc(x, y, **kws):
        r = np.corrcoef(x, y)[0, 1]
        ax = plt.gca()
        ax.annotate(f"r={r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                    ha="center", va="center", fontsize=9)

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.scatterplot, s=20, alpha=0.6)
    g.map_upper(corrfunc)
    g.map_diag(sns.histplot, kde=True, linewidth=0)

    g.fig.suptitle("Pairwise Scatter Plot of Core + Evo2 Metrics", y=1.02)
    out_path = os.path.join(output_dir, "pairplot_coremetrics.png")
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)
    return out_path