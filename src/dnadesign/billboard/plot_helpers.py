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
from scipy.stats import entropy as scipy_entropy

from dnadesign.billboard.core import token_edit_distance

sns.set_theme(style="ticks", font_scale=0.8)

def save_tf_frequency_barplot(tf_frequency, tf_metric, title, output_path, dpi, figsize=(14,7)):
    """
    Save a TF frequency bar plot.
    Displays a bar chart of TF frequencies in grey.
    """
    # Filter out fixed elements (if any)
    filtered = {k: v for k, v in tf_frequency.items() if not (k.endswith("_upstream") or k.endswith("_downstream"))}
    sorted_items = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
    tf_names = [item[0] for item in sorted_items]
    frequencies = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(tf_names, frequencies, color="grey")
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    sns.despine(ax=ax, top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_occupancy_plot(forward_matrix, reverse_matrix, tf_list, title, output_path, dpi, figsize=(14,7)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    im0 = axes[0].imshow(forward_matrix, aspect="equal", interpolation="none")
    axes[0].set_title("Forward Strand", fontsize=10)
    axes[0].set_xlabel("Nucleotide Position", fontsize=9)
    axes[0].set_yticks(np.arange(len(tf_list)))
    axes[0].set_yticklabels(tf_list, fontsize=4)
    sns.despine(ax=axes[0], top=True, right=True)
    
    im1 = axes[1].imshow(reverse_matrix, aspect="equal", interpolation="none")
    axes[1].set_title("Reverse Strand", fontsize=10)
    axes[1].set_xlabel("Nucleotide Position", fontsize=9)
    axes[1].tick_params(axis='y', labelleft=False)
    sns.despine(ax=axes[1], top=True, right=True)
    
    fig.suptitle(title, fontsize=12, y=0.92)
    plt.subplots_adjust(wspace=0.02, left=0.15, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.65])
    fig.colorbar(im1, cax=cbar_ax, label="Count")
    sns.despine(fig=fig, top=True, right=True)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_motif_length_histogram(motif_info, output_path, dpi, figsize=(8,6)):
    """
    Plot a KDE of motif lengths with a separate density curve for each TF.
    Expects motif_info to include keys "motif" and "tf".
    """
    import pandas as pd
    df = pd.DataFrame(motif_info)
    df["motif_length"] = df["motif"].apply(len)
    plt.figure(figsize=figsize)
    ax = sns.kdeplot(data=df, x="motif_length", hue="tf", fill=True, common_norm=False, alpha=0.7, warn_singular=False)
    ax.set_xlabel("Motif Length", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of Motif Lengths by Transcription Factors", fontsize=12)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_tf_entropy_kde_plot(occupancy_forward_matrix, occupancy_reverse_matrix, tf_list, sequence_length, output_path, dpi, figsize=(10,6)):
    """
    Overlay KDE plots of positional occupancy for the top 3 and bottom 3 TFs.
    Combines forward and reverse strand occupancy for each TF.
    """
    import pandas as pd
    # If less than 6 TFs, show all; otherwise, choose top 3 and bottom 3.
    if len(tf_list) < 6:
        selected_tf = tf_list
    else:
        top3 = tf_list[:3]
        bottom3 = tf_list[-3:]
        selected_tf = list(dict.fromkeys(top3 + bottom3))  # preserve order and remove duplicates
    
    positions = np.arange(sequence_length)
    plt.figure(figsize=figsize)
    
    for tf in selected_tf:
        try:
            row = tf_list.index(tf)
        except ValueError:
            continue
        occ_forward = occupancy_forward_matrix[row]
        occ_reverse = occupancy_reverse_matrix[row]
        total_occ = occ_forward + occ_reverse
        if total_occ.sum() > 0:
            sns.kdeplot(x=positions, weights=total_occ, label=tf, fill=True, common_norm=False, alpha=0.5)
    
    plt.xlabel("Nucleotide Position")
    plt.ylabel("Density")
    plt.title("Positional Occupancy KDE by TF (Top 3 & Bottom 3)")
    plt.legend(title="TF")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_gini_lorenz_plot(tf_frequency, output_path, dpi, figsize=(8,6)):
    """
    Plot a Lorenz curve for the TF frequency distribution and annotate with the inverted Gini coefficient.
    """
    freqs = np.array(sorted(tf_frequency.values()))
    n = freqs.size
    cumfreq = np.cumsum(freqs)
    total = cumfreq[-1]
    cumfreq_norm = cumfreq / total
    x = np.linspace(1/n, 1, n)
    
    def compute_gini_original(freqs):
        n = freqs.size
        cumfreq = np.cumsum(np.sort(freqs))
        if cumfreq[-1] == 0:
            return 0.0
        gini_inv = (n + 1 - 2 * np.sum(cumfreq) / cumfreq[-1]) / n
        return 1 - gini_inv  # original Gini
    gini_original = compute_gini_original(freqs)
    inverted_gini = 1 - gini_original
    
    plt.figure(figsize=figsize)
    plt.plot(x, cumfreq_norm, marker='o', label="Lorenz Curve")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Equality")
    plt.xlabel("Cumulative proportion of TFs")
    plt.ylabel("Cumulative proportion of TFBS")
    plt.title("TF Representation Based on TFBS Frequency (Lorenz Curve)")
    plt.annotate(f"Inverted Gini: {inverted_gini:.3f}", xy=(0.05, 0.9), xycoords="axes fraction",
                 fontsize=10, bbox=dict(boxstyle="round", fc="w"))
    plt.legend()
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_jaccard_histogram(sequences, output_path, dpi, figsize=(8,6), sample_size=1000):
    """
    Compute pairwise Jaccard dissimilarities for a (sampled) set of sequences and plot a histogram.
    """
    tf_rosters = []
    for seq in sequences:
        roster = set()
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                roster.add(tf_name.lower().strip())
        tf_rosters.append(roster)
    
    n = len(tf_rosters)
    dissimilarities = []
    num_pairs = int(n*(n-1)/2)
    max_pairs = min(sample_size, num_pairs)
    
    for _ in range(max_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        inter = len(tf_rosters[i].intersection(tf_rosters[j]))
        union = len(tf_rosters[i].union(tf_rosters[j]))
        jaccard = inter / union if union > 0 else 0
        dissimilarities.append(1 - jaccard)
    
    plt.figure(figsize=figsize)
    sns.histplot(dissimilarities, kde=True, color="steelblue")
    plt.xlabel("Jaccard Dissimilarity")
    plt.ylabel("Frequency")
    plt.title("Diversity of TF Combinations Across Sequences (Jaccard Similarities)")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
def save_motif_levenshtein_boxplot(motif_strings, config, output_path, dpi, figsize=(8,6)):
    """
    Compute pairwise normalized Levenshtein distances for motif_strings and generate a boxplot.
    The plot includes a white boxplot with styled ticks and top/right spines removed,
    and a scatter overlay (with low alpha) in the background to show distribution.
    """
    # Retrieve penalty parameters.
    msl_config = config.get("motif_string_levenshtein", {})
    tf_penalty = msl_config.get("tf_penalty", 1.0)
    strand_penalty = msl_config.get("strand_penalty", 0.5)
    partial_penalty = msl_config.get("partial_penalty", 0.8)
    
    distances = []
    n = len(motif_strings)
    for i in range(n):
        tokens_i = motif_strings[i].split(",") if motif_strings[i] else []
        for j in range(i+1, n):
            tokens_j = motif_strings[j].split(",") if motif_strings[j] else []
            if not tokens_i or not tokens_j:
                raw_distance = max(len(tokens_i), len(tokens_j))
            else:
                raw_distance = token_edit_distance(tokens_i, tokens_j, tf_penalty, strand_penalty, partial_penalty)
            norm_factor = max(len(tokens_i), len(tokens_j))
            normalized_distance = raw_distance / norm_factor if norm_factor > 0 else 0
            distances.append(normalized_distance)
    
    import pandas as pd
    df = pd.DataFrame({"Normalized Distance": distances})
    
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df, x="Normalized Distance", color="white", fliersize=0)
    sns.stripplot(data=df, x="Normalized Distance", color="gray", alpha=0.4, ax=ax)
    ax.set_title("Pairwise Normalized Motif Levenshtein Distances")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()