"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a scatter plot comparing Evo2 and Composite Billboard diversity metrics.
Selected (threshold-passing) subsamples are highlighted.
Also provides KDE and pairplot diagnostic visualizations for core metrics 
(including CDS components if using the CDS method).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(subsamples, config, output_dir, run_info):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns

    # Convert x (Evo2 metric) and y (billboard metric) values to numpy arrays.
    x_vals = np.array([s["evo2_metric"] for s in subsamples])
    bm_config = config.get("billboard_metric", {})
    composite_score = bm_config.get("composite_score", False)
    log_transform = config.get("plot", {}).get("log_transform", False)

    # Determine y values.
    if not composite_score:
        raw_y = np.array([s["billboard_metric"] for s in subsamples])
        if log_transform:
            y_vals = np.array([np.log2(val) if val > 0 else 0 for val in raw_y])
            y_label = "log₂(Core Metric)"
        else:
            y_vals = raw_y
            y_label = "Core Metric"
    else:
        y_vals = np.array([s["billboard_metric"] for s in subsamples])
        y_label = "Composite Billboard Metric"

    # Get joint selection method.
    joint_sel = config.get("joint_selection", {})
    sel_method = joint_sel.get("method", "null").lower()

    # Get marker size from config if provided.
    plot_config = config.get("plot", {})
    constant_marker_size = plot_config.get("marker_size", None)
    # Use constant marker size if defined; otherwise default to 50.
    marker_size = constant_marker_size if constant_marker_size is not None else 50

    # Get unique cluster counts.
    cluster_counts = np.array([s["unique_cluster_count"] for s in subsamples])
    min_c, max_c = cluster_counts.min(), cluster_counts.max()
    cmap = plt.get_cmap("coolwarm")

    # Prepare arrays for final point colors and alphas.
    final_colors = []
    final_alphas = []

    # Define alphas.
    default_alpha = plot_config.get("alpha", 0.35)
    hit_zone_alpha = plot_config.get("hit_zone_alpha", 0.8)
    non_hit_zone_alpha = plot_config.get("non_hit_zone_alpha", 0.3)

    sns.set_style("ticks")
    fig, ax = plt.subplots()

    if sel_method == "null":
        # No thresholding: color points solely by unique cluster count.
        for count in cluster_counts:
            norm_val = 0.5 if max_c == min_c else (count - min_c) / (max_c - min_c)
            final_colors.append(cmap(norm_val))
            final_alphas.append(default_alpha)
        scatter = ax.scatter(x_vals, y_vals, s=marker_size, c=final_colors,
                             alpha=default_alpha, edgecolor="none", zorder=2)

    elif sel_method == "x_only":
        # Compute x threshold using 1.5 * IQR.
        q1, q3 = np.percentile(x_vals, [25, 75])
        iqr = q3 - q1
        x_threshold = q3 + 1.5 * iqr
        # Draw vertical threshold line behind points.
        ax.axvline(x=x_threshold, linestyle="--", color="lightgray", zorder=1)
        for count, x in zip(cluster_counts, x_vals):
            norm_val = 0.5 if max_c == min_c else (count - min_c) / (max_c - min_c)
            if x >= x_threshold:
                final_colors.append(cmap(norm_val))
                final_alphas.append(hit_zone_alpha)
            else:
                final_colors.append("lightgray")
                final_alphas.append(non_hit_zone_alpha)
        scatter = ax.scatter(x_vals, y_vals, s=marker_size, c=final_colors,
                             alpha=1.0, edgecolor="none", zorder=2)
        # Annotate points exceeding the x threshold if enabled.
        if plot_config.get("annotate_x_threshold", False):
            for s, x, y in zip(subsamples, x_vals, y_vals):
                if x >= x_threshold:
                    suffix = s["subsample_id"].split("_")[-1]
                    ax.annotate(suffix, (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=8, color="gray")

    elif sel_method == "both":
        # Compute x threshold (1.5 * IQR).
        q1, q3 = np.percentile(x_vals, [25, 75])
        iqr = q3 - q1
        x_threshold = q3 + 1.5 * iqr
        # Compute y threshold based on a percentile (default 50th).
        y_percentile = joint_sel.get("y_percentile", 50)
        y_threshold = np.percentile(y_vals, y_percentile)
        # Draw both threshold lines behind points.
        ax.axvline(x=x_threshold, linestyle="--", color="lightgray", zorder=1)
        ax.axhline(y=y_threshold, linestyle="--", color="lightgray", zorder=1)
        for count, x, y in zip(cluster_counts, x_vals, y_vals):
            norm_val = 0.5 if max_c == min_c else (count - min_c) / (max_c - min_c)
            if x >= x_threshold and y >= y_threshold:
                final_colors.append(cmap(norm_val))
                final_alphas.append(hit_zone_alpha)
            else:
                final_colors.append("lightgray")
                final_alphas.append(non_hit_zone_alpha)
        # Plot each point individually using constant marker size.
        for xi, yi, col, alp in zip(x_vals, y_vals, final_colors, final_alphas):
            ax.scatter(xi, yi, s=marker_size, c=[col], alpha=alp, edgecolor="none", zorder=2)
        # Annotate points exceeding the x threshold if enabled.
        if plot_config.get("annotate_x_threshold", False):
            for s, x, y in zip(subsamples, x_vals, y_vals):
                if x >= x_threshold:
                    suffix = s["subsample_id"].split("_")[-1]
                    ax.annotate(suffix, (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=8, color="gray")
    else:
        raise ValueError(f"Unsupported joint_selection method: {sel_method}")

    ax.set_xlabel("Mean pairwise cosine dissimilarity in Evo2 latent space")
    ax.set_ylabel(y_label)
    title = "Subsample diversity analysis"
    subtitle = f"Draws: {run_info.get('num_draws', 'N/A')} · Subsample size: {run_info.get('subsample_size', 'N/A')}"
    ax.set_title(f"{title}\n{subtitle}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a colorbar indicating the unique cluster count.
    norm = mpl.colors.Normalize(vmin=min_c, vmax=max_c)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # for compatibility with older versions of Matplotlib
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Unique Cluster Count")

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