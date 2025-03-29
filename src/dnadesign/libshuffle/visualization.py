"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Generates a scatter plot comparing Evo2 and Billboard diversity metrics.
Selected (threshold-passing) subsamples are highlighted.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(subsamples, config, output_dir, run_info):
    """
    Create and save a scatter plot comparing Evo2 and Billboard diversity metrics.
    
    Parameters:
      subsamples: list of dictionaries with computed metrics for each subsample.
      config: libshuffle configuration dictionary.
      output_dir: directory where the plot should be saved.
      run_info: dict containing run metadata (e.g., number of draws, subsample size).
      
    Returns:
      The full path to the saved PNG file.
    """
    # Extract x and y values from subsamples.
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

    # Use seaborn to set the plot style to "ticks".
    sns.set_style("ticks")

    # Create the figure and axis.
    fig, ax = plt.subplots()
    
    # Plot all subsamples.
    ax.scatter(
        x_vals,
        y_vals,
        color=config.get("plot", {}).get("base_color", "gray"),
        alpha=config.get("plot", {}).get("alpha", 0.5),
        label="Subsamples"
    )

    # Highlight selected subsamples.
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
    ax.set_ylabel("Sum of entropy metrics from Billboard")
    title = "Subsample diversity analysis"
    subtitle = f"Draws: {run_info.get('num_draws', 'N/A')} · Subsample size: {run_info.get('subsample_size', 'N/A')}"
    ax.set_title(f"{title}\n{subtitle}")

    # Remove top and right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    # Save the plot as a PNG.
    dpi = config.get("plot", {}).get("dpi", 600)
    filename = config.get("plot", {}).get("filename", "scatter_summary.png")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()  # Ensure the plot is closed and not shown.
    
    return output_path
