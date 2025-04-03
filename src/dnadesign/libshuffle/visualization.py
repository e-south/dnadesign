"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a scatter plot comparing Evo2 and Composite Billboard diversity metrics.
Selected (threshold-passing) subsamples are highlighted.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# libshuffle/visualization.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(subsamples, config, output_dir, run_info):
    """
    Create and save a scatter plot comparing Evo2 and Composite Billboard diversity metrics.
    
    Parameters:
      subsamples: list of dictionaries with computed metrics for each subsample.
      config: libshuffle configuration dictionary.
      output_dir: directory where the plot should be saved.
      run_info: dict containing run metadata (e.g., number of draws, subsample size).
      
    Returns:
      The full path to the saved PNG file.
    """
    # Extract x and y values.
    x_vals = [s["evo2_metric"] for s in subsamples]
    y_vals = [s["billboard_metric"] for s in subsamples]

    # Apply joint selection thresholds if enabled.
    joint_selection = config.get("joint_selection", {})
    selected = []
    if joint_selection.get("enable", False):
        billboard_min = joint_selection.get("billboard_min", 0)
        evo2_min = joint_selection.get("evo2_min", 0)
        for s in subsamples:
            if s["billboard_metric"] >= billboard_min and s["evo2_metric"] >= evo2_min:
                s["passed_threshold"] = True
                selected.append(s)
            else:
                s["passed_threshold"] = False
    else:
        for s in subsamples:
            s["passed_threshold"] = False

    sns.set_style("ticks")
    fig, ax = plt.subplots()
    
    ax.scatter(
        x_vals,
        y_vals,
        color=config.get("plot", {}).get("base_color", "gray"),
        alpha=config.get("plot", {}).get("alpha", 0.5),
        label="Subsamples"
    )

    if selected:
        sel_x = [s["evo2_metric"] for s in selected]
        sel_y = [s["billboard_metric"] for s in selected]
        ax.scatter(
            sel_x,
            sel_y,
            color=config.get("plot", {}).get("highlight_color", "red"),
            alpha=1.0,
            edgecolors="black",
            s=100,
            label="Selected Subsamples",
            marker="*"
        )

    ax.set_xlabel("Mean pairwise distance in Evo2 latent space")
    ax.set_ylabel("Composite Billboard Metric")
    title = "Subsample diversity analysis"
    subtitle = f"Draws: {run_info.get('num_draws', 'N/A')} · Subsample size: {run_info.get('subsample_size', 'N/A')}"
    ax.set_title(f"{title}\n{subtitle}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    dpi = config.get("plot", {}).get("dpi", 600)
    filename = config.get("plot", {}).get("filename", "scatter_summary.png")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return output_path

def plot_kde_coremetrics(subsamples, config, output_dir):
    """
    Generates kernel density plots for each core Billboard metric across all subsamples.
    Two plots are created:
      - Raw core metrics
      - Z-score normalized core metrics (computed per metric)
    
    Saves the plots as PNG files with DPI=600 in the output directory:
      - kde_coremetrics_raw.png
      - kde_coremetrics_zscore.png
    
    Parameters:
      subsamples: list of subsample dicts with core metric values stored in "raw_billboard_vector"
      config: configuration dict (expects core metric names under billboard_metric/core_metrics)
      output_dir: directory where the plots should be saved.
    
    Returns:
      A tuple (raw_plot_path, zscore_plot_path)
    """
    import pandas as pd
    sns.set_style("ticks")
    
    bm_config = config.get("billboard_metric", {})
    core_metrics = bm_config.get("core_metrics", [])
    if not core_metrics:
        raise ValueError("No core metrics defined in config under billboard_metric/core_metrics")
    
    # Build long-format data for raw metrics.
    data = []
    for sub in subsamples:
        # Prefer raw_billboard_vector; if missing and only one core metric is expected, use billboard_metric.
        raw_vector = sub.get("raw_billboard_vector")
        if raw_vector is None:
            raw_vector = [sub.get("billboard_metric")]
        # Ensure the vector has the same length as core_metrics.
        if len(raw_vector) != len(core_metrics):
            continue
        for i, metric in enumerate(core_metrics):
            data.append({"Metric": metric, "Value": raw_vector[i]})
    
    if not data:
        raise ValueError("No raw core metrics data available for KDE plot.")
    
    df_raw = pd.DataFrame(data)
    
    # Plot raw KDE.
    fig, ax = plt.subplots()
    sns.kdeplot(data=df_raw, x="Value", hue="Metric", common_norm=False, ax=ax)
    ax.set_title("Core Billboard Metric Distributions (Raw)")
    ax.set_xlabel("Metric Value")
    ax.set_ylabel("Density")
    dpi = config.get("plot", {}).get("dpi", 600)
    raw_path = os.path.join(output_dir, "kde_coremetrics_raw.png")
    plt.savefig(raw_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    # Compute z-score normalization for each metric.
    df_z = df_raw.copy()
    def zscore(x):
        std = x.std(ddof=1)
        if std == 0:
            return 0
        return (x - x.mean()) / std
    df_z["ZScore"] = df_z.groupby("Metric")["Value"].transform(zscore)
    
    # Plot z-score KDE.
    fig, ax = plt.subplots()
    sns.kdeplot(data=df_z, x="ZScore", hue="Metric", common_norm=False, ax=ax)
    ax.set_title("Core Billboard Metric Distributions (Z-Score Normalized)")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Density")
    zscore_path = os.path.join(output_dir, "kde_coremetrics_zscore.png")
    plt.savefig(zscore_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return raw_path, zscore_path

def plot_pairplot_coremetrics(subsamples, config, output_dir):
    """
    Generates a pairwise scatter plot for the core Billboard metrics.
    Each subsample contributes one dot per metric pair.
    
    The pairplot shows:
      - Scatter plots in the lower triangle.
      - Pearson r (correlation coefficient) annotated in the upper triangle.
      - Diagonal cells annotated with "r=1.00".
    
    The plot is saved as "pairplot_coremetrics.png" with DPI=600 in the output directory.
    
    Parameters:
      subsamples: list of subsample dicts containing core metric values in "raw_billboard_vector"
      config: configuration dict (expects core metric names under billboard_metric/core_metrics)
      output_dir: directory where the plot should be saved.
      
    Returns:
      The full path to the saved pairplot PNG file.
    """
    import pandas as pd
    sns.set_style("ticks")
    
    bm_config = config.get("billboard_metric", {})
    core_metrics = bm_config.get("core_metrics", [])
    if not core_metrics:
        raise ValueError("No core metrics defined in config under billboard_metric/core_metrics")
    
    # Build DataFrame: one row per subsample, one column per core metric.
    data = {metric: [] for metric in core_metrics}
    for sub in subsamples:
        raw_vector = sub.get("raw_billboard_vector")
        if raw_vector is None:
            # In non-composite mode expect a single metric.
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
    g.fig.suptitle("Pairwise Scatter Plot of Core Billboard Metrics", y=1.02)
    g.fig.savefig(pairplot_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)
    return pairplot_path