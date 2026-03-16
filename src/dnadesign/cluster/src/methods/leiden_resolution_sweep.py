"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/methods/leiden_resolution_sweep.py

Leiden resolution sweep helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import Progress

from .leiden import run as run_leiden


def run_resolution_sweep(
    X: np.ndarray,
    method_params: Mapping[str, object],
    res_min: float,
    res_max: float,
    step: float,
    seeds: Sequence[int],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolutions = np.arange(res_min, res_max + step / 2, step)
    rows = []
    base_params = dict(method_params)
    neighbors = int(base_params.get("neighbors", 15))
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Sweeping Leiden resolution...",
            total=len(resolutions) * max(1, len(seeds)),
        )
        for res in resolutions:
            n_clusters = []
            cvs = []
            for seed in seeds:
                labels = run_leiden(
                    X,
                    **{
                        **base_params,
                        "neighbors": neighbors,
                        "resolution": float(res),
                        "random_state": int(seed),
                    },
                )
                values, counts = np.unique(labels, return_counts=True)
                n_clusters.append(len(values))
                cv = float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else float("nan")
                cvs.append(cv)
                progress.advance(task, 1)
            rows.append(
                {
                    "resolution": float(res),
                    "num_clusters_mean": float(np.mean(n_clusters)),
                    "num_clusters_std": float(np.std(n_clusters)),
                    "cv_cluster_size_mean": float(np.mean(cvs)),
                    "cv_cluster_size_std": float(np.std(cvs)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "ResolutionSweepSummary.parquet", index=False)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(df["resolution"], df["num_clusters_mean"], marker="o")
    axs[0].fill_between(
        df["resolution"],
        df["num_clusters_mean"] - df["num_clusters_std"],
        df["num_clusters_mean"] + df["num_clusters_std"],
        alpha=0.2,
    )
    axs[0].set_ylabel("# clusters")
    axs[0].set_title("Leiden resolution sweep")
    axs[1].plot(df["resolution"], df["cv_cluster_size_mean"], marker="o")
    axs[1].fill_between(
        df["resolution"],
        df["cv_cluster_size_mean"] - df["cv_cluster_size_std"],
        df["cv_cluster_size_mean"] + df["cv_cluster_size_std"],
        alpha=0.2,
    )
    axs[1].set_ylabel("CV cluster sizes")
    axs[1].set_xlabel("resolution")
    fig.tight_layout()
    fig.savefig(out_dir / "ResolutionSweep.png", dpi=200)
    try:
        best_idx = int(df["cv_cluster_size_mean"].idxmin())
        best_res = float(df.loc[best_idx, "resolution"])
        (out_dir / "ResolutionSweep.suggested.txt").write_text(f"{best_res}\n")
    except Exception:
        pass
    return df
