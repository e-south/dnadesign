import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

def plot_hitzone_details(subsamples, sequences, config, output_dir, run_info):
    """
    1) Single‑panel scatter of only the hit‑zone subsamples:
         - X axis: sublibrary rank (by median_tf_entropy)
         - Y axis: median_tf_entropy core metric
         - Hue: unique_cluster_count
       Only subsamples whose Evo2 cosine dissimilarity ≥ the configured threshold are shown.
    2) Pairwise scatter‑matrix of all core metrics, coloring the bottom 1,000
       and top 1,000 subsamples by Evo2 dissimilarity as 'low_latent_diff'
       vs. 'high_latent_diff'.

    Returns: (scatter_path, pairplot_path) or (None, None) if no hit‑zone subsamples.
    """
    # --- 1) determine the hit‑zone mask by the Evo2 threshold (same as plot_scatter) ---
    evo2 = np.array([s["evo2_metric"] for s in subsamples])
    joint = config.get("joint_selection", {})
    method = joint.get("method", "null").lower()

    def _thresh(vals, mode, val):
        if mode == "iqr":
            q1, q3 = np.percentile(vals, [25, 75])
            return q3 + (q3 - q1) * val
        elif mode == "percentile":
            return np.percentile(vals, val)
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")

    x_t = None
    if method in ("x_only", "both"):
        x_t = _thresh(
            evo2,
            joint.get("threshold_x_mode", "iqr"),
            joint.get("threshold_x_value", 1.5),
        )

    hit_mask = (evo2 >= x_t) if x_t is not None else np.zeros_like(evo2, bool)
    hit_idxs = np.where(hit_mask)[0]
    if hit_idxs.size == 0:
        return None, None

    # --- 2) pull out median_tf_entropy and cluster count for those hits ---
    bm_cfg = config.get("billboard_metric", {})
    core_keys = bm_cfg.get("core_metrics", [])
    try:
        me_idx = core_keys.index("min_tf_entropy")
    except ValueError:
        raise ValueError("`min_tf_entropy` must be listed in billboard_metric/core_metrics")

    y_vals = []
    clusters = []
    for i in hit_idxs:
        sub = subsamples[i]
        raw = sub.get("raw_billboard_vector", [])
        if len(raw) > me_idx:
            y_vals.append(raw[me_idx])
        else:
            # fallback if composite_score=False
            y_vals.append(sub.get("billboard_metric", np.nan))
        clusters.append(sub.get("unique_cluster_count", 0))

    # sort by median_tf_entropy
    order = np.argsort(y_vals)
    y_sorted   = np.array(y_vals)[order]
    clusters_s = np.array(clusters)[order]

    # --- 3) make the scatter plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        np.arange(len(y_sorted)),
        y_sorted,
        c=clusters_s,
        cmap="viridis",
        s=50,
        edgecolor="none"
    )
    # show ticks but no labels
    ax.set_xticks(np.arange(len(y_sorted)))
    ax.set_xticklabels([""] * len(y_sorted))
    ax.set_xlabel("Sublibrary (ranked by min_tf_entropy)")
    ax.set_ylabel("min_tf_entropy")
    ax.set_title("Hit‑zone subsamples: min_tf_entropy vs. Unique Cluster Count")
    sns.despine(ax=ax, top=True, right=True)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Unique Cluster Count")
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, "hitzone_scatter_min_tf_entropy.png")
    dpi = config.get("plot", {}).get("dpi", 600)
    fig.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- 4) build DataFrame of all subsamples for pairplot ---
    records = []
    for sub in subsamples:
        raw = sub.get("raw_billboard_vector")
        if raw is None or len(raw) != len(core_keys):
            continue
        row = {metr: raw[i] for i, metr in enumerate(core_keys)}
        row["evo2_metric"] = sub.get("evo2_metric", np.nan)
        records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return scatter_path, None

    # label bottom 1,000 / top 1,000 by evo2 dissimilarity
    df = df.sort_values("evo2_metric")
    df["latent_diff_group"] = ""
    n = len(df)
    if n >= 2000:
        df.iloc[:1000,  df.columns.get_loc("latent_diff_group")] = "low_latent_diff"
        df.iloc[-1000:, df.columns.get_loc("latent_diff_group")] = "high_latent_diff"
    else:
        half = n // 2
        df.iloc[:half, df.columns.get_loc("latent_diff_group")] = "low_latent_diff"
        df.iloc[half:, df.columns.get_loc("latent_diff_group")] = "high_latent_diff"

    # keep only those two groups
    df = df[df["latent_diff_group"].isin(["low_latent_diff", "high_latent_diff"])]
    if df.empty:
        return scatter_path, None

    # --- 5) pairwise scatter‑matrix ---
    pp = sns.pairplot(
        df,
        vars=core_keys,
        hue="latent_diff_group",
        corner=True,
        diag_kind="hist",
        palette={"low_latent_diff": "tab:blue", "high_latent_diff": "tab:red"},
        plot_kws={"alpha": 0.6, "s": 20},
    )
    pp.fig.suptitle("Pairwise Scatter of Core Metrics (bottom 1,000 vs. top 1,000 Evo2 dissim)", y=1.02)

    pairplot_path = os.path.join(output_dir, "hitzone_pairplot_coremetrics.png")
    pp.fig.savefig(pairplot_path, dpi=dpi, bbox_inches="tight")
    plt.close(pp.fig)

    return scatter_path, pairplot_path