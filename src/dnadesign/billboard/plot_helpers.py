"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/plot_helpers.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

# Set seaborn style to "ticks".
sns.set_theme(style="ticks", font_scale=0.8)


def save_tf_frequency_barplot(tf_frequency, tf_entropy, title, output_path, dpi, figsize=(14,7)):
    """
    Save a composite TF frequency plot with two subplots:
      - Left: The original TF frequency bar plot with bars colored by positional entropy.
      - Right: A horizontal stacked bar plot showing the proportion of each TF in the library.
    
    Fixed upstream/downstream keys are filtered out.
    Left plot:
      - Bars are sorted in descending order by frequency.
      - X-tick labels are rotated 90°.
      - A colorbar indicates the mapping from positional entropy to color.
    
    Right plot:
      - Displays a single horizontal stacked bar where each segment's width represents the 
        proportion of that TF's frequency relative to the total.
      - Each segment is colored using a colormap that maps the normalized proportion value to a hue.
      - Left and bottom spines are retained; top and right spines are removed.
      - Annotates each segment with the TF name if the segment is wide enough.
    """
    # Filter out fixed elements (upstream/downstream)
    filtered = {k: v for k, v in tf_frequency.items() if not (k.endswith("_upstream") or k.endswith("_downstream"))}
    filtered_entropy = {k: tf_entropy.get(k, 0) for k in filtered.keys()}
    
    # Sort items in descending order by frequency
    sorted_items = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
    tf_names = [item[0] for item in sorted_items]
    frequencies = [item[1] for item in sorted_items]
    entropies = [filtered_entropy.get(tf, 0) for tf in tf_names]
    
    # Compute proportions for stacked bar plot
    total = sum(frequencies)
    proportions = [freq/total for freq in frequencies]
    
    # Create composite figure with two subplots (side-by-side)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    
    # Left subplot: TF frequency bar plot colored by positional entropy.
    cmap_left = cm.get_cmap("YlGnBu")
    norm_left = Normalize(vmin=min(entropies), vmax=max(entropies))
    colors_left = [cmap_left(norm_left(val)) for val in entropies]
    
    ax_left.bar(tf_names, frequencies, color=colors_left)
    ax_left.set_ylabel("Frequency", fontsize=10)
    ax_left.set_title(title, fontsize=12)
    ax_left.set_xlabel("")
    plt.setp(ax_left.get_xticklabels(), rotation=90, fontsize=8)
    sns.despine(ax=ax_left)  # Use seaborn 'ticks' style
    
    sm = plt.cm.ScalarMappable(cmap=cmap_left, norm=norm_left)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_left, pad=0.01)
    cbar.set_label("Positional Entropy", fontsize=10)
    
    # Right subplot: Horizontal stacked bar plot showing TF frequency proportions.
    # Use a separate colormap (e.g., 'viridis') based on normalized proportion values.
    cmap_right = cm.get_cmap("viridis")
    norm_right = Normalize(vmin=min(proportions), vmax=max(proportions))
    colors_right = [cmap_right(norm_right(prop)) for prop in proportions]
    
    cumulative = 0
    for tf, prop, color in zip(tf_names, proportions, colors_right):
        ax_right.barh(0, width=prop, left=cumulative, color=color, edgecolor='white')
        # Annotate segment if its width is sufficiently large (e.g., > 5% of the bar)
        if prop > 0.05:
            ax_right.text(cumulative + prop/2, 0, tf, va='center', ha='center', fontsize=8, color='white')
        cumulative += prop
    ax_right.set_xlim(0, 1)
    ax_right.set_xlabel("Proportion", fontsize=10)
    ax_right.set_title("TF Frequency Proportions", fontsize=12)
    # Retain left and bottom spines; remove top and right spines
    sns.despine(ax=ax_right, top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_occupancy_plot(forward_matrix, reverse_matrix, tf_list, title, output_path, dpi, figsize=(14,7)):
    """
    Save a combined occupancy heatmap.
    
    Two subplots share a y-axis, where TF names are sorted in descending order by frequency.
    Each subplot forces a 1:1 aspect ratio with no whitespace between cells.
    A common colorbar is added to the right.
    Y-axis label is removed and y-tick font is reduced.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    im0 = axes[0].imshow(forward_matrix, aspect="equal", interpolation="none")
    axes[0].set_title("Forward Strand", fontsize=10)
    axes[0].set_xlabel("Nucleotide Position", fontsize=9)
    axes[0].set_yticks(np.arange(len(tf_list)))
    axes[0].set_yticklabels(tf_list, fontsize=4)
    
    im1 = axes[1].imshow(reverse_matrix, aspect="equal", interpolation="none")
    axes[1].set_title("Reverse Strand", fontsize=10)
    axes[1].set_xlabel("Nucleotide Position", fontsize=9)
    axes[1].tick_params(axis='y', labelleft=False)
    
    fig.suptitle("Library-wide sequence diversity through motif coverage and arrangement variability", fontsize=12, y=0.86)
    plt.subplots_adjust(wspace=0.02, left=0.15, right=0.88)
    
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.65])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Value Count", fontsize=10)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_motif_length_histogram(motif_info, output_path, dpi, figsize=(8,6)):
    """
    Save a density plot of motif lengths using Seaborn's KDE plot.
    
    Uses pastel, colorblind-friendly colors.
    Legend labels are updated: "pass" → "Appeared", "fail" → "Failed to appear".
    Title: "Shorter binding sites are more likely to appear in dense arrays"
    X label: "Length of Binding Site"
    Y label: "Appearances in Dense Arrays"
    Legend frame is removed.
    """
    import pandas as pd
    df = pd.DataFrame(motif_info)
    df["status"] = df["status"].map({"pass": "Appeared", "fail": "Failed to appear"})
    
    plt.figure(figsize=figsize)
    ax = sns.kdeplot(data=df, x="length", hue="status", fill=True, common_norm=False, palette="pastel", alpha=0.7)
    ax.set_xlabel("Length of Binding Site", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Shorter binding sites are more likely to appear in dense arrays", fontsize=12)
    sns.despine(ax=ax, top=True, right=True)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_motif_stats_barplot(motif_stats, output_path, dpi, figsize=(14,7)):
    """
    Save a vertical bar plot with two subplots sharing the x-axis.
    
    The x-axis represents unique TFs sorted in descending order by pass ratio.
    Top subplot: Pass ratio by TF.
    Bottom subplot: Average motif length for passed motifs (with y-axis inverted).
    Both subplots have top and right spines removed, x-tick font reduced, and no x label.
    Uses pastel, colorblind-friendly colors.
    """
    import pandas as pd
    df = pd.DataFrame(motif_stats)
    df_sorted = df.sort_values(by="ratio", ascending=False)
    x = df_sorted["tf"]
    pass_ratio = df_sorted["ratio"]
    avg_length = df_sorted["avg_length"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    pastel_color = sns.color_palette("pastel")[0]
    
    sns.barplot(x=x, y=pass_ratio, ax=ax1, color=pastel_color)
    ax1.set_title("Pass/Fail Ratio by TF\nReflects how often the solver successfully incorporated a binding site for each TF when it was presented as an option", fontsize=10)
    ax1.set_ylabel("Pass Ratio", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlabel("")
    
    sns.barplot(x=x, y=avg_length, ax=ax2, color=pastel_color)
    ax2.set_title("Average Motif Length (Passed) by TF", fontsize=10)
    ax2.set_ylabel("Average Length", fontsize=9)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlabel("")
    
    plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
