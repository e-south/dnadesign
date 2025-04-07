"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a two-panel figure:
  - Left subplot: Scatter plot of all subsamples with X = mean pairwise cosine 
    dissimilarity (Evo2) and Y determined by config ("billboard", "euclidean", or "log1p_euclidean").
    Points are colored (using a continuous colormap) based on unique cluster count,
    and hit-zone points (those passing 1.5×IQR thresholds on both axes) are highlighted.
  - Right subplot: Only the hit-zone points, with x-axis as subsample ID (sorted by billboard metric)
    and y-axis as the billboard metric. Colors again reflect unique cluster count.
  
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


"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a two-panel figure:
  - Left panel: Full scatter plot (X-axis: mean pairwise cosine dissimilarity; 
    Y-axis: user-selected metric) with hit-zone points colored by unique cluster count.
  - Right panel: Ranking plot for hit-zone points (X-axis: subsample ID [numeric],
    Y-axis: billboard metric), with horizontal threshold lines (1.5×IQR based)
    and outlier annotations.
  
Configuration keys of interest:
  - plot.y_axis_metric: one of {"billboard", "euclidean", "log1p_euclidean"}.
  - joint_selection.method: "null", "x_only", or "both".
  - joint_selection.threshold_x_mode: "iqr" or "percentile"
  - joint_selection.threshold_x_value: (float for IQR multiplier or int for percentile)
  - joint_selection.threshold_y_mode: "iqr" or "percentile"
  - joint_selection.threshold_y_value: (float or int)
  - plot.subplot_width_ratios: list, e.g. [2, 1]
  - plot.figure_size: list, e.g. [16, 6]
  
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

# libshuffle/visualization.py

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a two-panel figure:
  - Left panel: Full scatter plot.
      • X-axis: Mean pairwise cosine dissimilarity in Evo2 latent space.
      • Y-axis: User-selected metric (options: "billboard", "euclidean", or "log1p_euclidean").
      • Points in the hit zone (those passing the X and, if applicable, Y thresholds)
        are colored by unique cluster count. A color bar is shown.
      • Subplot title: "Latent Diversity Scatter: Cosine vs. Billboard Metric"
  - Right panel: Ranking plot for hit-zone points.
      • X-axis: Unique cluster count (with ticks restored).
      • Y-axis: Billboard metric.
      • Points are plotted in gray using the hit-zone alpha.
      • Horizontal threshold lines (computed via 1.5×IQR or a chosen percentile) are drawn,
        and the top 5 and bottom 5 hit-zone points (by billboard metric) are annotated with their subsample IDs.
      • Subplot title: "Ranking by Billboard Metric in Hit Zone"
      
Configuration keys of interest:
  - plot.y_axis_metric: "billboard", "euclidean", or "log1p_euclidean".
  - joint_selection.method: "null", "x_only", or "both".
  - joint_selection.threshold_x_mode: "iqr" or "percentile"
  - joint_selection.threshold_x_value: (float for IQR multiplier or int for percentile)
  - joint_selection.threshold_y_mode: "iqr" or "percentile"
  - joint_selection.threshold_y_value: (float or int)
  - plot.subplot_width_ratios: list, e.g. [2, 1]
  - plot.figure_size: list, e.g. [16, 6]

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

"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a two-panel figure:
  - Left panel: Full scatter plot of all subsamples.
      X-axis: Mean pairwise cosine dissimilarity in Evo2 latent space.
      Y-axis: User-selected metric (billboard, euclidean, or log1p_euclidean).
      Non-hit-zone points are drawn in a base color.
      Hit-zone points are drawn in a highlight color and have marker sizes 
      scaled by their unique cluster count.
      A legend (showing sample marker sizes for a few unique cluster count values)
      is placed inside the left panel (bottom-right).
  - Right panel: Ranking plot for hit-zone points.
      X-axis: Unique cluster count (with ticks shown).
      Y-axis: Billboard metric.
      Points are plotted in gray.
      The top 5 and bottom 5 points (by billboard metric) are annotated with their subsample IDs.
      
Configuration keys of interest:
  - plot.y_axis_metric: "billboard", "euclidean", or "log1p_euclidean".
  - joint_selection.method: "null", "x_only", or "both".
  - joint_selection.threshold_x_mode: "iqr" or "percentile"
  - joint_selection.threshold_x_value: (float for IQR multiplier or int for percentile)
  - joint_selection.threshold_y_mode: "iqr" or "percentile"
  - joint_selection.threshold_y_value: (float or int)
  - plot.subplot_width_ratios: list, e.g. [2, 1]
  - plot.figure_size: list, e.g. [16, 6]
  - plot.base_color: e.g. "gray"
  - plot.highlight_color: e.g. "red"
  - plot.marker_size: base marker size (e.g. 50)
  - plot.size_multiplier: additional size factor for hit-zone points (e.g. 20)
  - plot.alpha: default alpha for non-hit-zone points.
  - plot.hit_zone_alpha: alpha for hit-zone points.
  - plot.annotate_x_threshold: boolean.
  
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
    Generates a two-panel figure:
      Left panel: Full scatter plot.
          - X-axis: Mean pairwise cosine dissimilarity (Evo2).
          - Y-axis: User-selected metric.
          - Non-hit-zone points are drawn in a base color.
          - Hit-zone points are drawn in a highlight color with marker sizes
            scaled by unique cluster count.
          - A legend inside the left panel (bottom-right) shows sample marker sizes.
      Right panel: Ranking plot for hit-zone points.
          - X-axis: Unique cluster count.
          - Y-axis: Billboard metric.
          - Points are plotted in gray.
          - Top 5 and bottom 5 (by billboard metric) are annotated with subsample IDs.
    """
    # ------------------
    # Left Panel Data:
    # ------------------
    # X-axis: Mean pairwise cosine dissimilarity.
    x_vals = np.array([s["evo2_metric"] for s in subsamples])

    # Y-axis: Determined by "plot.y_axis_metric".
    plot_config = config.get("plot", {})
    y_axis_metric = plot_config.get("y_axis_metric", "billboard")
    log_transform = plot_config.get("log_transform", False)
    bm_config = config.get("billboard_metric", {})

    if y_axis_metric == "billboard":
        composite = bm_config.get("composite_score", False)
        if not composite:
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
    # Threshold Settings:
    # ------------------
    joint_sel = config.get("joint_selection", {})
    sel_method = joint_sel.get("method", "null").lower()  # "null", "x_only", or "both"

    def compute_threshold(values, mode, val):
        if mode == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            return q3 + val * iqr
        elif mode == "percentile":
            return np.percentile(values, val)
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")

    # X-threshold:
    threshold_x_mode = joint_sel.get("threshold_x_mode", "iqr")
    threshold_x_value = joint_sel.get("threshold_x_value", 1.5)
    x_threshold = compute_threshold(x_vals, threshold_x_mode, threshold_x_value) if sel_method in ["x_only", "both"] else None

    # Y-threshold (only used if sel_method=="both")
    threshold_y_mode = joint_sel.get("threshold_y_mode", "iqr")
    threshold_y_value = joint_sel.get("threshold_y_value", 1.5)
    y_threshold = compute_threshold(y_vals, threshold_y_mode, threshold_y_value) if sel_method == "both" else None

    # ------------------
    # Identify Hit-Zone Points:
    # ------------------
    if sel_method == "null":
        hit_mask = np.zeros_like(x_vals, dtype=bool)
    elif sel_method == "x_only":
        hit_mask = (x_vals >= x_threshold)
    elif sel_method == "both":
        hit_mask = (x_vals >= x_threshold) & (y_vals >= y_threshold)
    else:
        raise ValueError(f"Unsupported joint_selection.method: {sel_method}")

    # ------------------
    # Set up Figure Layout:
    # ------------------
    subplot_width_ratios = plot_config.get("subplot_width_ratios", [2, 1])
    figure_size = plot_config.get("figure_size", [16, 6])
    fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=figure_size,
                                             gridspec_kw={'width_ratios': subplot_width_ratios})
    sns.set_style("ticks")

    # ------------------
    # Left Panel: Full Scatter Plot
    # ------------------
    base_color = plot_config.get("base_color", "gray")
    highlight_color = plot_config.get("highlight_color", "red")
    # Instead of coloring by unique cluster count, use base colors;
    # for hit-zone points, vary the marker size by unique cluster count.
    marker_size = plot_config.get("marker_size", 50)
    size_multiplier = plot_config.get("size_multiplier", 20)
    all_uc = np.array([s.get("unique_cluster_count", 0) for s in subsamples])
    uc_max = all_uc.max() if all_uc.max() > 0 else 1

    sizes_left = []
    colors_left = []
    for i, s in enumerate(subsamples):
        if hit_mask[i]:
            # Scale marker size by unique cluster count (normalized)
            uc = s.get("unique_cluster_count", 0)
            scaled_size = marker_size + (uc / uc_max) * size_multiplier
            sizes_left.append(scaled_size)
            colors_left.append(highlight_color)
        else:
            sizes_left.append(marker_size)
            colors_left.append(base_color)

    # Plot left panel points:
    for i in range(len(x_vals)):
        ax_left.scatter(x_vals[i], y_vals[i], s=sizes_left[i],
                        c=colors_left[i], alpha=plot_config.get("alpha", 0.35),
                        edgecolor="none")
    # Draw threshold lines if applicable:
    if sel_method in ["x_only", "both"]:
        ax_left.axvline(x=x_threshold, linestyle="--", color="lightgray", zorder=1)
    if sel_method == "both":
        ax_left.axhline(y=y_threshold, linestyle="--", color="lightgray", zorder=1)
    # Annotate hit-zone points:
    if sel_method in ["x_only", "both"] and plot_config.get("annotate_x_threshold", False):
        for i, s in enumerate(subsamples):
            if hit_mask[i]:
                numeric_id = s["subsample_id"].split("_")[-1]
                ax_left.annotate(numeric_id, (x_vals[i], y_vals[i]),
                                 textcoords="offset points", xytext=(5, 5),
                                 fontsize=8, color="gray")
    ax_left.set_xlabel("Mean pairwise cosine dissimilarity in Evo2 latent space")
    ax_left.set_ylabel(y_label)
    ax_left.set_title("Isolating Sublibraries with High Latent Space Coverage")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    
    # Create legend for marker sizes (mapping unique cluster count to marker size)
    # Select representative cluster count values (min, median, max)
    rep_uc = np.array([all_uc.min(), np.median(all_uc), all_uc.max()])
    legend_sizes = [marker_size + (uc / uc_max) * size_multiplier for uc in rep_uc]
    # Create dummy scatter plots:
    handles = [plt.scatter([], [], s=size, color=highlight_color, alpha=plot_config.get("hit_zone_alpha", 0.8))
               for size in legend_sizes]
    labels = [f"UC = {int(uc)}" for uc in rep_uc]
    ax_left.legend(handles, labels, title="Unique Cluster Count", loc="lower right", frameon=False)

    # ------------------
    # Right Panel: Ranking Plot for Hit-Zone Points
    # ------------------
    # For right panel, x-axis: unique cluster count, y-axis: billboard metric.
    # All points are plotted in gray.
    hit_subsamples = [s for i, s in enumerate(subsamples) if hit_mask[i]]
    if not hit_subsamples:
        ax_right.text(0.5, 0.5, "No hit-zone subsamples found",
                      ha='center', va='center', fontsize=14, color="gray")
        ax_right.set_xticks([])
        ax_right.set_yticks([])
    else:
        hit_uc = np.array([s.get("unique_cluster_count", 0) for s in hit_subsamples])
        hit_billboard = np.array([s["billboard_metric"] for s in hit_subsamples])
        # For right panel, use gray color for all points.
        ax_right.scatter(hit_uc, hit_billboard, s=marker_size,
                         c="red", alpha=plot_config.get("hit_zone_alpha", 0.8),
                         edgecolor="none")
        # Set x-ticks to unique cluster count values.
        unique_hit_uc = np.unique(hit_uc)
        ax_right.set_xticks(unique_hit_uc)
        ax_right.set_xlabel("Unique Cluster Count")
        ax_right.set_ylabel("Billboard Metric")
        ax_right.set_title("Stratify by Billboard Metric(s)")
        # Remove threshold lines (as per current specification)
        # Annotate top 5 and bottom 5 hit-zone points by billboard metric.
        if len(hit_billboard) >= 5:
            top5_idx = np.argsort(-hit_billboard)[:5]
            bottom5_idx = np.argsort(hit_billboard)[:5]
            annotate_idx = np.concatenate([top5_idx, bottom5_idx])
        else:
            annotate_idx = np.arange(len(hit_billboard))
        for idx in annotate_idx:
            sub_id = hit_subsamples[idx]["subsample_id"].split("_")[-1]
            ax_right.annotate(sub_id, (hit_uc[idx], hit_billboard[idx]),
                              textcoords="offset points", xytext=(0, 5),
                              fontsize=8, color="gray")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # ------------------
    # Overall Figure Settings:
    # ------------------
    overall_title = (f"Subsample Diversity Analysis\n"
                     f"Draws: {run_info.get('num_draws', 'N/A')} · "
                     f"Subsample size: {run_info.get('subsample_size', 'N/A')}")
    fig.suptitle(overall_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    dpi = plot_config.get("dpi", 600)
    filename = plot_config.get("filename", "scatter_summary.png")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
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
    import pandas as pd
    import seaborn as sns
    sns.set_style("ticks")
    
    bm_config = config.get("billboard_metric", {})
    method = bm_config.get("method", "")
    
    if method == "cds":
        keys = ["cds_score", "dominance", "program_diversity"]
        data = {key: [] for key in keys}
        for sub in subsamples:
            cds = sub.get("cds_components")
            if not cds:
                continue
            for key in keys:
                data[key].append(cds.get(key))
    else:
        core_metrics = bm_config.get("core_metrics", [])
        if not core_metrics:
            raise ValueError("No core metrics defined in config under billboard_metric/core_metrics")
        if len(core_metrics) == 1:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Pairplot not available for single core metric",
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis('off')
            pairplot_path = os.path.join(output_dir, "pairplot_coremetrics.png")
            dpi = config.get("plot", {}).get("dpi", 600)
            plt.savefig(pairplot_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            return pairplot_path
        data = {metric: [] for metric in core_metrics}
        for sub in subsamples:
            raw_vector = sub.get("raw_billboard_vector")
            if raw_vector is None:
                if len(core_metrics) == 1:
                    raw_vector = [sub.get("billboard_metric")]
                else:
                    continue
            if len(raw_vector) != len(core_metrics):
                continue
            for i, metric in enumerate(core_metrics):
                data[metric].append(raw_vector[i])
    
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No data available for pairplot.")
    
    import numpy as np
    def corrfunc(x, y, **kws):
        r = np.corrcoef(x, y)[0, 1]
        ax = plt.gca()
        ax.annotate(f"r={r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                    ha='center', va='center', fontsize=10)
    
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.scatterplot)
    g.map_upper(corrfunc)
    g.map_diag(lambda x, **kws: plt.annotate("r=1.00", xy=(0.5, 0.5),
                                              xycoords=plt.gca().transAxes,
                                              ha='center', va='center', fontsize=10))
    
    dpi = config.get("plot", {}).get("dpi", 600)
    pairplot_path = os.path.join(output_dir, "pairplot_coremetrics.png")
    g.fig.suptitle("Pairwise Scatter Plot of Core Metrics", y=1.02)
    g.fig.savefig(pairplot_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)
    return pairplot_path