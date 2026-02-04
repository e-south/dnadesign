"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/diagnostics.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import csv
import logging
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)
sns.set_style("ticks")
sns.set_context("talk")


def plot_reconstruction_loss(metrics: dict, output_path: str) -> None:
    ks = sorted(metrics.keys())
    losses = [float(metrics[k]["mean_loss"]) for k in ks]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=ks, y=losses, marker="o")
    plt.xlabel("Number of Programs (k)")
    plt.ylabel("Mean Frobenius Reconstruction Loss")
    plt.title("Reconstruction Loss vs. Factorization Rank")
    sns.despine(top=True, right=True)
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reconstruction loss plot to {output_path}")


def plot_heatmaps(
    W: np.ndarray,
    H: np.ndarray,
    output_dir: str,
    encoding_mode: str = "motif_occupancy",
    feature_names: list = None,
) -> None:
    """
    Plot heatmaps for the W and H matrices with clustering reordering.

    For W (Coefficient Matrix):
      - Rows (sequences) are clustered to group similar program usage.

    For H (Basis Matrix):
      - Both rows (programs) and columns (features/motifs) are clustered to reveal co-activation patterns.

    The reordering is strictly for visual interpretability.
    """
    import scipy.cluster.hierarchy as sch

    os.makedirs(output_dir, exist_ok=True)

    # --- Cluster and reorder W matrix ---
    # W has shape (num_sequences, num_programs)
    # Cluster the rows (sequences) using Ward's method.
    row_linkage = sch.linkage(W, method="ward")
    row_order = sch.leaves_list(row_linkage)
    W_ordered = W[row_order, :]

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(W_ordered, cmap="viridis", cbar_kws={"label": "Normalized Program Contribution"})
    ax.set_xlabel("Program Index")
    ax.set_ylabel("Sequence Index (clustered)")
    ax.set_title("Clustered Heatmap of W (Coefficient Matrix)")
    sns.despine()
    plt.savefig(os.path.join(output_dir, "W_heatmap.png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved clustered W heatmap to {output_dir}")

    # --- Cluster and reorder H matrix ---
    # H has shape (num_programs, num_features)
    # First, cluster the rows (programs):
    row_linkage_H = sch.linkage(H, method="ward")
    row_order_H = sch.leaves_list(row_linkage_H)
    H_ordered = H[row_order_H, :]

    # Next, cluster the columns (features/motifs):
    col_linkage_H = sch.linkage(H_ordered.T, method="ward")
    col_order_H = sch.leaves_list(col_linkage_H)
    H_ordered = H_ordered[:, col_order_H]

    # Reorder the labels accordingly.
    row_labels = [f"Program {i}" for i in row_order_H]

    # For motif_occupancy, optionally render feature labels when the axis isn't too dense.
    xticklabels = False
    if encoding_mode == "motif_occupancy":
        if feature_names is None:
            raise ValueError("Feature names must be provided for motif_occupancy encoding.")
        if H_ordered.shape[1] <= 50:
            xticklabels = [feature_names[i] for i in col_order_H]

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        H_ordered,
        cmap="viridis",
        cbar_kws={"label": "Contribution Weight (clipped at 3)"},
        xticklabels=xticklabels,
        yticklabels=row_labels,
    )
    ax.set_xlabel("Features (Motifs)")
    ax.set_ylabel("Program Index (clustered)")
    ax.set_title("Clustered Heatmap of H (Basis Matrix)")
    sns.despine()
    plt.savefig(os.path.join(output_dir, "H_heatmap.png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved clustered H heatmap to {output_dir}")


def plot_motif_occurrence_heatmap(X: np.ndarray, output_path: str, max_occ: int = 5, transpose: bool = False) -> None:
    """
    Plot a heatmap for motif occupancy. If transpose=True, data is flipped so that
    motifs appear on the Y-axis and sequences on the X-axis.
    """
    if transpose:
        X = X.T
    plt.figure(figsize=(12, 8))
    X_vis = np.clip(X, 0, max_occ)
    ax = sns.heatmap(X_vis, cmap="gray", cbar_kws={"label": f"Motif Occupancy (capped at {max_occ})"})
    ax.invert_yaxis()
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Motif")
    ax.set_title("Motif Occurrence Heatmap")
    if X.shape[1] > 50:
        ax.set_xticklabels([])
    if X.shape[0] > 50:
        ax.set_yticklabels([])
    sns.despine(top=True, right=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved motif occurrence heatmap to {output_path}")


def plot_program_association_stacked_barplot(
    H: np.ndarray,
    feature_names: list,
    output_path: str,
    motif2tf_map: dict = None,
    normalize: bool = True,
) -> None:
    """
    Plot a stacked bar plot where each row corresponds to a transcription factor (TF).
    For each TF, the contributions from each NMF program (from matrix H) are aggregated.
    Then each TF row is normalized to sum to 1 (if normalization is enabled) and plotted as a horizontal stacked bar.
    A colorblind-friendly palette ("Set2") is used and the legend is removed.
    """

    # Function to get the TF label from a feature.
    def get_tf(motif):
        return motif2tf_map.get(motif.upper(), motif.upper()) if motif2tf_map else motif.upper()

    # Convert each feature name to its TF label.
    tf_labels_all = [get_tf(feat) for feat in feature_names]

    # H is assumed to be of shape (num_programs, num_features).
    num_programs = H.shape[0]

    # Determine the unique TFs (sorted for consistency).
    tf_set = sorted(set(tf_labels_all))

    # Initialize an array to hold aggregated contributions:
    # Rows: programs (NMF components); Columns: TFs.
    # We start with a matrix of zeros with shape (num_programs, len(tf_set)).
    agg = np.zeros((num_programs, len(tf_set)))

    # For each program (i.e. for each row in H), aggregate contributions for each TF.
    for prog in range(num_programs):
        for feat_idx, tf in enumerate(tf_labels_all):
            # Add the contribution of feature feat_idx (for this program) to the appropriate TF.
            # (Using index lookup for tf_set; if speed is a concern, one could precompute a mapping.)
            col_idx = tf_set.index(tf)
            agg[prog, col_idx] += H[prog, feat_idx]

    # Now, we want to have rows correspond to TFs and columns to programs.
    df_data = pd.DataFrame(agg, index=[f"Program {i}" for i in range(num_programs)], columns=tf_set)
    df_data = df_data.T  # Now rows = TF, columns = Programs.

    # For each TF (each row), normalize so that the sum is 1.
    if normalize:
        row_sums = df_data.sum(axis=1)
        # To avoid division by zero, set any zero sum to 1.
        row_sums[row_sums == 0] = 1
        df_norm = df_data.div(row_sums, axis=0)
    else:
        df_norm = df_data.copy()

    # Plot the horizontal stacked bar plot using the "Set2" palette.
    ax = df_norm.plot(kind="barh", stacked=True, figsize=(10, 8), colormap="tab20", legend=False)
    ax.set_xlabel("Proportion of Contribution")
    ax.set_ylabel("Transcription Factor")
    ax.set_title("Program Association Stacked Bar Plot")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved program association stacked bar plot to {output_path}")


def plot_top_motifs_grid(
    H: np.ndarray,
    feature_names: list,
    output_dir: str,
    motif2tf_map: dict,
    max_top_tfs: int = 5,
    ncols: int = 4,
    encoding_mode: str = "motif_occupancy",
) -> None:
    """
    For each program (row in H), compute the top motifs (converted to TF names) and plot a horizontal
    bar plot with the TF names on the y-axis. Remove top/right spines and use reduced tick fonts.
    The x-axis ticks are not rotated.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_programs = H.shape[0]
    nrows = int(math.ceil(n_programs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if n_programs > 1 else [axes]
    for i in range(n_programs):
        row = H[i, :]
        tf_contrib = {}
        for val, feat in zip(row, feature_names):
            motif = feat.upper()
            tf_name = motif2tf_map.get(motif, motif)
            tf_contrib[tf_name] = tf_contrib.get(tf_name, 0) + val
        sorted_items = sorted(tf_contrib.items(), key=lambda x: x[1], reverse=True)[:max_top_tfs]
        if sorted_items:
            tf_labels, contribs = zip(*sorted_items)
        else:
            tf_labels, contribs = ([], [])
        ax = axes[i]
        sns.barplot(
            x=list(contribs),
            y=list(tf_labels),
            color="slategray",
            ax=ax,
            errorbar=None,
            dodge=False,
        )
        ax.set_title(f"Program {i}", fontsize=10)
        ax.set_xlabel("Contribution", fontsize=9)
        ax.set_ylabel("", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        sns.despine(ax=ax, top=True, right=True)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "top_motifs_grid.png")
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved top motif barplots grid to {grid_path}")


def plot_sequence_program_composition(W: np.ndarray, output_path: str) -> None:
    """
    Generate a heatmap-style plot representing the unit‑normalized composition of each sequence.
    Each column corresponds to a sequence; the column is divided into colored segments for each program.
    Each sequence is normalized to sum to 1, so even if a sequence only has one nonzero program,
    that program fills the column.
    """
    num_sequences, num_programs = W.shape
    # Re-normalize each sequence to sum to 1.
    W_normalized = np.zeros_like(W)
    for i in range(num_sequences):
        row_sum = np.sum(W[i, :])
        if row_sum > 0:
            W_normalized[i, :] = W[i, :] / row_sum
        else:
            W_normalized[i, :] = W[i, :]

    height = 100  # vertical resolution per column
    img = np.zeros((height, num_sequences), dtype=int)
    for seq_idx in range(num_sequences):
        cumulative = 0.0
        for prog_idx in range(num_programs):
            frac = W_normalized[seq_idx, prog_idx]
            if frac > 0:
                start = int(round(cumulative * height))
                cumulative += frac
                end = int(round(cumulative * height))
                if end > start:
                    img[start:end, seq_idx] = prog_idx + 1  # use prog_idx+1 so that 0 is background
        # Ensure the entire column is filled
        if cumulative < 1:
            img[-1, seq_idx] = np.argmax(W_normalized[seq_idx, :]) + 1
    import matplotlib.colors as mcolors

    # Use a colorblind-friendly palette; here we use "Set2" for consistency.
    if num_programs <= 10:
        base_cmap = plt.get_cmap("Set2")
    elif num_programs <= 20:
        base_cmap = plt.get_cmap("tab20")  # fallback if needed
    else:
        base_cmap = plt.get_cmap("viridis")
    colors = [base_cmap(i % base_cmap.N) for i in range(num_programs)]
    # Do not include white as one of the hues for nonzero values.
    cmap_list = colors  # no white in the colormap; background is not drawn by imshow
    cmap_cat = mcolors.ListedColormap(cmap_list)
    fig_width = max(12, num_sequences * 0.02)
    plt.figure(figsize=(fig_width, 6))
    plt.imshow(img, aspect="auto", cmap=cmap_cat, origin="upper")
    plt.xlabel("Sequence Index")
    plt.ylabel("Program")
    plt.title("Sequence Program Composition")
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sequence program composition heatmap to {output_path}")


def compute_composite_diversity_score(W: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """
    Compute the Composite Diversity Score (CDS) for a given coefficient matrix W.
    W is of shape (n_sequences, k) and is assumed to be row-normalized.

    Returns a tuple: (CDS, Dominance, Program Diversity)

    Dominance is defined as:
       1 - (1/|S|) * sum_{s in S} [H(s) / log2(k)]
    where H(s) = -sum_i p_i log2(p_i) and p_i = W[s,i].

    Program Diversity is defined as:
       D_inv / k, where D_inv = 1 / sum_i (q_i^2)
    and q_i is the fraction of sequences whose dominant program is i.

    Composite Diversity Score is then:
       CDS = α * Dominance + (1 - α) * Program Diversity.

    Lower within-sequence entropy (i.e. higher dominance) and higher across-sequence program balance yield a higher CDS.
    """
    # Assert that each row of W sums to 1 (within a tolerance)
    if not np.allclose(np.sum(W, axis=1), 1, atol=1e-3):
        raise ValueError("The coefficient matrix W is not row-normalized.")

    n_sequences, k = W.shape
    # Compute normalized entropy for each sequence.
    normalized_entropies = []
    for s in range(n_sequences):
        p = W[s, :]
        # Avoid log2(0) by adding a small epsilon.
        entropy = -np.sum(p * np.log2(p + 1e-8))
        normalized_entropy = entropy / np.log2(k)
        normalized_entropies.append(normalized_entropy)
    # Dominance is one minus the average normalized entropy.
    dominance = 1 - np.mean(normalized_entropies)

    # Compute Program Diversity (Inverse Simpson)
    dominant = np.argmax(W, axis=1)
    counts = np.bincount(dominant, minlength=k)
    q = counts / float(n_sequences)
    D_inv = 1.0 / (np.sum(q**2) + 1e-8)
    program_diversity = D_inv / k

    composite = alpha * dominance + (1 - alpha) * program_diversity
    return composite, dominance, program_diversity


def plot_composite_diversity_curve(batch_results_dir: str, k_range: list, alpha: float, output_path: str) -> None:
    composite_scores = []
    dominance_scores = []
    diversity_scores = []
    valid_ks = []

    for k in k_range:
        k_dir = os.path.join(batch_results_dir, f"k_{k}")
        W_path = os.path.join(k_dir, "W.csv")
        if not os.path.exists(W_path):
            continue
        try:
            W_df = pd.read_csv(W_path, index_col=0)
            W = W_df.values
            # Normalize rows (if not already normalized)
            row_sums = np.sum(W, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W_normalized = W / row_sums
            # This call will assert that W_normalized is valid.
            composite, dominance, prog_div = compute_composite_diversity_score(W_normalized, alpha)
            composite_scores.append(composite)
            dominance_scores.append(dominance)
            diversity_scores.append(prog_div)
            valid_ks.append(k)
        except Exception as e:
            logger.error(f"Error computing CDS for k={k}: {str(e)}")

    if not valid_ks:
        logger.error("No valid W matrices found for computing Composite Diversity Score.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(valid_ks, composite_scores, marker="o", label="Composite Diversity Score")
    plt.plot(valid_ks, dominance_scores, marker="o", label="Dominance (1 - Entropy)")
    plt.plot(valid_ks, diversity_scores, marker="o", label="Program Diversity")
    plt.xlabel("Factorization Rank (k)")
    plt.ylabel("Diversity Score")
    plt.title("Composite Diversity vs. Factorization Rank")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved composite diversity curve to {output_path}")


def compute_stability_metrics(batch_results_dir: str, k_range: list) -> dict:
    stability = {
        "Frobenius Error": {"k": [], "values": []},
        "Coefficient Variation": {"k": [], "values": []},
        "Silhouette Score": {"k": [], "values": []},
        "Cophenetic Coefficient": {"k": [], "values": []},
        "Amari Distance": {"k": [], "values": []},
    }
    for k in k_range:
        k_dir = os.path.join(batch_results_dir, f"k_{k}")
        W_path = os.path.join(k_dir, "W.csv")
        metrics_yaml = os.path.join(k_dir, "metrics.yaml")
        try:
            W_df = pd.read_csv(W_path, index_col=0)
            W = W_df.values
            with open(metrics_yaml, "r") as f:
                import yaml

                mdata = yaml.safe_load(f)
            frob = float(mdata.get("mean_loss", 0))
            rep_losses = mdata.get("replicate_losses", [])
            rep_losses = [float(x) for x in rep_losses]
            cv = np.std(rep_losses) / np.mean(rep_losses) if rep_losses and np.mean(rep_losses) > 0 else 0
            labels = np.argmax(W, axis=1)
            sil = silhouette_score(W, labels) if len(np.unique(labels)) > 1 else 0
            Z = linkage(W, method="ward")
            coph, _ = cophenet(Z, pdist(W))
            amari = 0  # Placeholder
            stability["Frobenius Error"]["k"].append(k)
            stability["Frobenius Error"]["values"].append(frob)
            stability["Coefficient Variation"]["k"].append(k)
            stability["Coefficient Variation"]["values"].append(cv)
            stability["Silhouette Score"]["k"].append(k)
            stability["Silhouette Score"]["values"].append(sil)
            stability["Cophenetic Coefficient"]["k"].append(k)
            stability["Cophenetic Coefficient"]["values"].append(coph)
            stability["Amari Distance"]["k"].append(k)
            stability["Amari Distance"]["values"].append(amari)
        except Exception as e:
            logger.error(f"Error computing stability metrics for k={k}: {str(e)}")
    return stability


def plot_stability_metrics_grid(stability_data: dict, output_path: str) -> None:
    metrics_list = list(stability_data.keys())
    ncols = 3
    nrows = int(np.ceil(len(metrics_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics_list):
        k_vals = stability_data[metric]["k"]
        y_vals = stability_data[metric]["values"]
        axes[idx].plot(k_vals, y_vals, marker="o")
        axes[idx].set_xlabel("Number of Programs (k)")
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f"{metric} vs. k")
        sns.despine(ax=axes[idx], top=True, right=True)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved stability metrics grid plot to {output_path}")


def compute_program_flow(batch_results_dir: str, k_range: list, similarity_threshold: float) -> dict:
    from sklearn.metrics.pairwise import cosine_similarity

    nodes = []
    edges = []
    prev_H = None
    prev_k = None
    for k in sorted(k_range):
        k_dir = os.path.join(batch_results_dir, f"k_{k}")
        H_path = os.path.join(k_dir, "H.csv")
        try:
            H_df = pd.read_csv(H_path, index_col=0)
            H = H_df.values
        except Exception:
            continue
        num_programs = H.shape[0]
        for i in range(num_programs):
            nodes.append((k, i))
        if prev_H is not None:
            sim_matrix = cosine_similarity(prev_H, H)
            for i, row in enumerate(sim_matrix):
                j = np.argmax(row)
                sim = row[j]
                if sim >= similarity_threshold:
                    edges.append(((prev_k, i), (k, j), sim))
        prev_H = H
        prev_k = k
    return {"nodes": nodes, "edges": edges}


def plot_signature_stability_riverplot(program_flow: dict, output_path: str, k_range: list) -> None:
    """
    Generate a left-to-right Sankey "riverplot".
    Each column represents a factorization rank (k) and each node is a signature index.
    Edges connect signatures (if cosine similarity ≥ threshold) and the edge value encodes similarity.
    """
    nodes = program_flow.get("nodes", [])
    edges = program_flow.get("edges", [])

    if not nodes or not edges:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No program flow data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
        )
        ax.set_title("Signature Stability Riverplot")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved placeholder riverplot to {output_path}")
        return

    group_by_k: dict = defaultdict(list)
    for k_val, prog_idx in nodes:
        group_by_k[k_val].append(prog_idx)
    for k_val in group_by_k:
        group_by_k[k_val] = sorted(set(group_by_k[k_val]))

    k_sorted = sorted(set(k_range)) if k_range else sorted(group_by_k.keys())
    if not k_sorted:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No program flow data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
        )
        ax.set_title("Signature Stability Riverplot")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved placeholder riverplot to {output_path}")
        return

    k_min, k_max = min(k_sorted), max(k_sorted)
    span = max(1, (k_max - k_min))

    # Assign node positions in a normalized (x,y) canvas.
    node_pos: dict[tuple[int, int], tuple[float, float]] = {}
    for k_val in k_sorted:
        progs = group_by_k.get(k_val, [])
        n = len(progs)
        if n == 0:
            continue
        y_positions = [0.5] if n == 1 else [i / (n - 1) for i in range(n)]
        x = (k_val - k_min) / float(span)
        for prog_idx, y in zip(progs, y_positions):
            node_pos[(k_val, prog_idx)] = (x, y)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw edges as cubic Bezier curves with width proportional to similarity.
    for (src_k, src_p), (dst_k, dst_p), sim in edges:
        src = (src_k, src_p)
        dst = (dst_k, dst_p)
        if src not in node_pos or dst not in node_pos:
            continue
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[dst]
        cx = (x0 + x1) / 2.0
        path = Path(
            [(x0, y0), (cx, y0), (cx, y1), (x1, y1)],
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
        )
        lw = 0.5 + 6.0 * float(sim)
        ax.add_patch(PathPatch(path, fill=False, linewidth=lw, alpha=0.6))

    # Draw nodes.
    xs = [xy[0] for xy in node_pos.values()]
    ys = [xy[1] for xy in node_pos.values()]
    ax.scatter(xs, ys, s=30, zorder=3)

    # Column labels for k.
    for k_val in k_sorted:
        x = (k_val - k_min) / float(span)
        ax.text(x, 1.05, f"K={k_val}", ha="center", va="bottom")
        ax.plot([x, x], [0.0, 1.0], linewidth=0.5, alpha=0.2, zorder=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Signature Stability Riverplot")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved signature stability riverplot to {output_path}")


def export_program_tf_contributions(H: np.ndarray, feature_names: list, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n_programs = H.shape[0]
    contributions = []
    for i in range(n_programs):
        row = H[i, :]
        tf_contrib = {}
        for val, feat in zip(row, feature_names):
            tf_name = feat.split(":")[0] if ":" in feat else feat
            tf_contrib[tf_name] = tf_contrib.get(tf_name, 0) + val
        for tf_name, total in tf_contrib.items():
            contributions.append([i, tf_name, total])
    output_csv = os.path.join(output_dir, "program_tf_contributions.csv")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["program_index", "tf_name", "total_contribution"])
        writer.writerows(contributions)
    logger.info(f"Exported program TF contributions to {output_csv}")
