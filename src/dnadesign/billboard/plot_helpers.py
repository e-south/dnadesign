"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/plot_helpers.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dnadesign.billboard.core import robust_parse_tfbs

logger = logging.getLogger(__name__)
sns.set_theme(style="ticks", font_scale=0.8)


def save_tf_frequency_barplot(tf_freq, title, path, dpi):
    logger.info(f"Saving TF frequency barplot to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # sort descending
    names, vals = zip(*sorted(tf_freq.items(), key=lambda x: -x[1]))

    plt.figure(figsize=(10, 6))  # narrower than before
    plt.bar(names, vals, color="grey")
    plt.title(title, fontsize=16)
    plt.xlabel("Transcription Factor", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_occupancy_plot(F, R, tf_list, title, path, dpi):
    """
    - Signature matches new code: (F, R, tf_list, title, path, dpi)
    - Internals borrowed from the older version you preferred:
        * two side-by-side heatmaps, sharey=True
        * vertical colorbar on the right, tight margins
    - TF rows sorted by descending (F+R) sums.
    """
    logger.info(f"Saving occupancy plot to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # sort TFs by descending total occupancy
    total = F + R
    sums = total.sum(axis=1)
    order = np.argsort(sums)[::-1]
    Fs = F[order]
    Rs = R[order]
    sorted_tfs = [tf_list[i] for i in order]

    # make figure & axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # forward strand
    im0 = axes[0].imshow(Fs, aspect="equal", interpolation="none")
    axes[0].set_title("Forward Strand", fontsize=10)
    axes[0].set_xlabel("Nucleotide Position", fontsize=9)
    axes[0].set_yticks(np.arange(len(sorted_tfs)))
    axes[0].set_yticklabels(sorted_tfs, fontsize=6)
    sns.despine(ax=axes[0], top=True, right=True)

    # reverse strand
    im1 = axes[1].imshow(Rs, aspect="equal", interpolation="none")
    axes[1].set_title("Reverse Strand", fontsize=10)
    axes[1].set_xlabel("Nucleotide Position", fontsize=9)
    axes[1].tick_params(axis="y", labelleft=False)  # hide labels on the 2nd plot
    sns.despine(ax=axes[1], top=True, right=True)

    # super title
    fig.suptitle(title, fontsize=12, y=0.92)

    # adjust margins so subplots sit nicely
    plt.subplots_adjust(wspace=0.02, left=0.15, right=0.88)

    # add a vertical colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.65])  # [left, bottom, width, height]
    fig.colorbar(im1, cax=cbar_ax, label="Coverage count")

    sns.despine(fig=fig, top=True, right=True)

    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_motif_length_histogram(motif_info, path, dpi):
    """
    - No legend.
    - Kernel density via seaborn for each TF.
    - Annotate each TF’s KDE curve at its median length.
    - Annotations sit just under the title.
    """
    logger.info(f"Saving motif length histogram to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(motif_info)
    if "motif" not in df.columns:
        logger.error("motif_info missing 'motif' column—nothing to plot")
        return

    df["length"] = df["motif"].str.len()
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot one smooth KDE per TF, no legend
    for tf, sub in df.groupby("tf"):
        lengths = sub["length"]
        if lengths.empty:
            continue

        # seaborn will handle edge behavior gracefully
        sns.kdeplot(data=lengths, fill=False, common_norm=False, alpha=0.8, linewidth=1.5, ax=ax)
        # annotate at the median
        med = lengths.median()
        # get the curve's y-value at that x
        y_at_med = ax.lines[-1].get_ydata()[np.argmin(np.abs(ax.lines[-1].get_xdata() - med))]
        ax.text(med, y_at_med * 1.02, tf, ha="center", va="bottom", fontsize=9, clip_on=False)  # improved visibility

    ax.set_title("Distribution of Motif Lengths by TF", fontsize=16)
    ax.set_xlabel("Motif length", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    sns.despine()
    # leave a little room for the title and annotations
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_tf_entropy_kde_plot(F, R, tf_list, L, path, dpi):
    """
    - Histogram for the single max‐coverage TF vs. the single min‐coverage TF.
    - Narrower figure.
    """
    logger.info(f"Saving TF entropy histogram to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    total = F + R
    sums = total.sum(axis=1)
    idx_sorted = np.argsort(sums)
    if len(idx_sorted) < 2:
        logger.warning("Not enough TFs for entropy plot")
        return

    low_i, high_i = idx_sorted[0], idx_sorted[-1]
    choices = [(tf_list[high_i], total[high_i]), (tf_list[low_i], total[low_i])]

    plt.figure(figsize=(6, 6))  # narrower
    for name, weights in choices:
        positions = np.arange(L)
        sns.histplot(x=positions, weights=weights, bins=L, element="step", fill=True, alpha=0.4, label=name)

    plt.xlim(0, L - 1)
    plt.xlabel("Nucleotide Position", fontsize=14)
    plt.ylabel("Coverage count", fontsize=14)
    plt.title("Positional Occupancy Histogram (Max vs Min TF)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="TF", fontsize=12, title_fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_gini_lorenz_plot(tf_freq, path, dpi):
    # unchanged
    logger.info(f"Saving Lorenz curve to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vals = np.sort(list(tf_freq.values()))
    cum = np.cumsum(vals) / vals.sum()
    x = np.linspace(1 / len(vals), 1, len(vals))

    plt.figure(figsize=(8, 6))
    plt.plot(x, cum, marker="o", label="Lorenz Curve")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Equality")
    inv_gini = 1 - (len(vals) + 1 - 2 * np.sum(np.cumsum(vals) / vals.sum())) / len(vals)
    plt.annotate(f"Inverted Gini: {inv_gini:.3f}", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=12)
    plt.title("TF Representation Based on TFBS Frequency (Lorenz Curve)", fontsize=16)
    plt.xlabel("Cumulative proportion of TFs", fontsize=14)
    plt.ylabel("Cumulative proportion of TFBS", fontsize=14)
    plt.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_jaccard_histogram(seqs, path, dpi, sample_size=1000):
    # unchanged
    logger.info(f"Saving Jaccard histogram to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rosters = []

    for s in seqs:
        rf = set()
        for part in s["meta_tfbs_parts"]:
            try:
                tf, _ = robust_parse_tfbs(part, s.get("id"))
                rf.add(tf)
            except ValueError:
                continue
        rosters.append(rf)

    n = len(rosters)
    dis = []
    for _ in range(min(sample_size, n * (n - 1) // 2)):
        i, j = np.random.choice(n, 2, replace=False)
        a, b = rosters[i], rosters[j]
        u = len(a | b)
        dis.append(1 - len(a & b) / u if u else 0)

    plt.figure(figsize=(8, 6))
    sns.histplot(dis, kde=True)
    plt.title("Diversity of TF Combinations (Jaccard Dissimilarity)", fontsize=16)
    plt.xlabel("Dissimilarity", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_motif_levenshtein_boxplot(*args, **kwargs):
    """
    Disabled: too slow to render.
    """
    logger.info("Skipping motif-string Levenshtein boxplot (disabled).")
