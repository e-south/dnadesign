"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

matplotlib.use("Agg")

sns.set_theme(style="ticks", palette="colorblind")
plt.rcParams.update({"font.size": 12})


class Plotter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger("libshuffle.visualization")

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_threshold(data: np.ndarray, thr_cfg):
        if thr_cfg.type == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            return q3 + (q3 - q1) * thr_cfg.factor
        if thr_cfg.type == "percentile":
            return np.percentile(data, thr_cfg.factor)
        raise ValueError(f"Unknown threshold type: {thr_cfg.type}")

    # ──────────────────────────────────────────────────────────────────────
    # Helper used by all plots
    def _is_literal(self, bb: dict) -> bool:
        """
        Returns True if the subsample fails either the Hamming threshold
        *or* the TF-richness requirement.
        """
        hd_fail = bb.get("min_hamming_distance", np.inf) < self.cfg.literal_min_bp_diff
        tf_fail = bb.get("tf_richness", 0) < self.cfg.min_tf_richness
        return hd_fail or tf_fail

    # ──────────────────────────────────────────────────────────────────────
    def plot_scatter(self, subs, winner, outdir: Path):
        pc = self.cfg.plot.scatter
        x = np.asarray([s["mean_cosine"] for s in subs], dtype=float)
        y = np.log1p(np.asarray([s["mean_euclidean"] for s in subs], dtype=float))
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        passed_mask = (x >= thr) & ~np.isnan(x)

        fig, ax = plt.subplots(figsize=pc.figsize)
        for i, s in enumerate(subs):
            xi, yi = x[i], y[i]
            bb = s["raw_billboard"]
            if not passed_mask[i]:
                ax.scatter(
                    xi, yi, color="lightgray", alpha=pc.low_alpha, edgecolor="none"
                )
                continue

            if self._is_literal(bb):
                col = pc.colors["literal_drop"]["literal"]
            elif s.get("unique_cluster_count", 0) != self.cfg.subsample_size:
                col = pc.colors["cluster_drop"]
            else:
                col = pc.colors["base"]
            ax.scatter(xi, yi, color=col, alpha=pc.high_alpha, edgecolor="none")

        if pc.threshold_line:
            ax.axvline(thr, ls="--", c=pc.colors.get("threshold_line", "lightgray"))

        # annotate winner
        win_idx = next(
            i for i, s in enumerate(subs) if s["subsample_id"] == winner["subsample_id"]
        )
        ax.scatter(
            x[win_idx],
            y[win_idx],
            marker="*",
            s=pc.star_size,
            c=pc.colors["winner"],
            edgecolor="none",
            zorder=3,
        )
        ax.text(
            x[win_idx],
            y[win_idx],
            winner["subsample_id"],
            fontsize=8,
            va="bottom",
            ha="right",
        )

        ax.set_title("Angular vs. Euclidean Diversity: All 16-member Subsamples")
        ax.set_xlabel("Average Pairwise Cosine Dissimilarity (1 − cosθ)")
        ax.set_ylabel("Average Pairwise log1p E-dist")

        legend_elems = [
            Patch(facecolor=pc.colors["base"], label="Passed", alpha=pc.high_alpha),
            Patch(
                facecolor=pc.colors["literal_drop"]["literal"],
                label="Literal flagged",
                alpha=pc.high_alpha,
            ),
            Patch(
                facecolor=pc.colors["cluster_drop"],
                label="Leiden flagged",
                alpha=pc.high_alpha,
            ),
        ]
        ax.legend(handles=legend_elems, loc="lower right", frameon=False)
        sns.despine(ax=ax)

        fig.savefig(outdir / "scatter_summary.pdf", dpi=pc.dpi, bbox_inches="tight")
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    def plot_flag_composition(self, subs, outdir: Path):
        pc = self.cfg.plot.scatter
        x = np.asarray([s["mean_cosine"] for s in subs], dtype=float)
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        filtered = [s for i, s in enumerate(subs) if x[i] >= thr]

        counts = {"Passed": 0, "Literal flagged": 0, "Leiden flagged": 0}
        for s in filtered:
            bb = s["raw_billboard"]
            if self._is_literal(bb):
                counts["Literal flagged"] += 1
            elif s.get("unique_cluster_count", 0) != self.cfg.subsample_size:
                counts["Leiden flagged"] += 1
            else:
                counts["Passed"] += 1

        fig, ax = plt.subplots(figsize=(3, 6))
        bottom = 0
        for label, color in [
            ("Passed", pc.colors["base"]),
            ("Literal flagged", pc.colors["literal_drop"]["literal"]),
            ("Leiden flagged", pc.colors["cluster_drop"]),
        ]:
            ax.bar(
                "Subsamples",
                counts[label],
                bottom=bottom,
                color=color,
                label=label,
                alpha=pc.high_alpha,
            )
            bottom += counts[label]

        ax.set_title("Composition of IQR-Filtered Subsamples")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, bbox_to_anchor=(1, 1))
        sns.despine(ax=ax)

        fig.savefig(outdir / "flag_composition.png", dpi=pc.dpi, bbox_inches="tight")
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    def plot_hitzone(self, subs, winner, outdir: Path):
        pc, hc = self.cfg.plot.scatter, self.cfg.plot.hitzone
        x = np.asarray([s["mean_cosine"] for s in subs], dtype=float)
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        hits = [
            (idx, math.log1p(s["min_euclidean"]), s)
            for idx, s in enumerate(subs)
            if x[idx] >= thr
        ]
        hits.sort(key=lambda t: t[1])

        fig, ax = plt.subplots(figsize=hc.figsize)
        for pos, (idx, yi, s) in enumerate(hits):
            bb = s["raw_billboard"]
            if self._is_literal(bb):
                col = pc.colors["literal_drop"]["literal"]
            elif s.get("unique_cluster_count", 0) != self.cfg.subsample_size:
                col = pc.colors["cluster_drop"]
            else:
                col = pc.colors["base"]
            ax.scatter(pos, yi, color=col, alpha=pc.high_alpha, edgecolor="none")

        win_idx = next(
            i for i, s in enumerate(subs) if s["subsample_id"] == winner["subsample_id"]
        )
        for pos, (idx, yi, _) in enumerate(hits):
            if idx == win_idx:
                ax.scatter(
                    pos, yi, marker="*", s=pc.star_size, c=pc.colors["winner"], zorder=3
                )
                ax.text(
                    pos, yi, winner["subsample_id"], fontsize=8, va="bottom", ha="right"
                )
                break

        ax.set_title("Outlier Ranking by Minimum Euclidean Gap")
        ax.set_xlabel("Subsample ID")
        ax.set_ylabel("Min Pairwise E-dist (log1p)")

        legend_elems = [
            Patch(facecolor=pc.colors["base"], label="Passed", alpha=pc.high_alpha),
            Patch(
                facecolor=pc.colors["literal_drop"]["literal"],
                label="Literal flagged",
                alpha=pc.high_alpha,
            ),
            Patch(
                facecolor=pc.colors["cluster_drop"],
                label="Leiden flagged",
                alpha=pc.high_alpha,
            ),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc="upper left")
        sns.despine(ax=ax)

        fig.savefig(outdir / "hitzone_summary.png", dpi=hc.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_kde(self, subs, outdir):
        kc = self.cfg.plot.kde
        mets = self.cfg.billboard_core_metrics
        rec = []
        for s in subs:
            bb = s["raw_billboard"]
            for m in mets:
                rec.append({"Metric": m, "Value": bb.get(m)})
        df = pd.DataFrame(rec)

        fig, ax = plt.subplots(figsize=kc.figsize)
        sns.kdeplot(
            data=df,
            x="Value",
            hue="Metric",
            common_norm=False,
            warn_singular=False,
            ax=ax,
        )
        ax.set_title("Core Metric Distributions (Raw)")
        sns.despine(ax=ax)
        fig.savefig(outdir / "kde_coremetrics_raw.png", dpi=kc.dpi, bbox_inches="tight")
        plt.close(fig)

        df["Z"] = df.groupby("Metric")["Value"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0
        )
        fig, ax = plt.subplots(figsize=kc.figsize)
        sns.kdeplot(
            data=df, x="Z", hue="Metric", common_norm=False, warn_singular=False, ax=ax
        )
        ax.set_title("Core Metric Distributions (Z-score)")
        sns.despine(ax=ax)
        fig.savefig(
            outdir / "kde_coremetrics_zscore.png", dpi=kc.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_pairplot(self, subs, outdir):
        pp = self.cfg.plot.pairplot
        rows = []
        for s in subs:
            bb = s["raw_billboard"]
            row = {m: bb.get(m, np.nan) for m in self.cfg.billboard_core_metrics}
            row.update(
                {
                    "mean_cosine": s.get("mean_cosine", np.nan),
                    "min_cosine": s.get("min_cosine", np.nan),
                    "mean_euclidean": s.get("mean_euclidean", np.nan),
                    "min_euclidean": s.get("min_euclidean", np.nan),
                    "log1p_mean_euclidean": math.log1p(s.get("mean_euclidean", 0)),
                    "log1p_min_euclidean": math.log1p(s.get("min_euclidean", 0)),
                    "unique_cluster_count": s.get("unique_cluster_count", np.nan),
                }
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        cols = list(self.cfg.billboard_core_metrics)
        for axis in (self.cfg.plot.scatter.x, self.cfg.plot.scatter.y):
            if axis not in cols:
                cols.append(axis)
        if "unique_cluster_count" not in cols:
            cols.append("unique_cluster_count")

        df_sub = df[cols].dropna(axis=1, how="all")
        if df_sub.shape[1] < 2:
            return

        g = sns.pairplot(
            df_sub, corner=True, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 20}
        )
        g.fig.suptitle("Pairwise Scatter of Core + Evo2 Metrics", y=1.02)
        for ax in g.axes.flatten():
            if ax is not None:
                sns.despine(ax=ax)
        g.fig.savefig(
            outdir / "pairplot_coremetrics.png", dpi=pp.dpi, bbox_inches="tight"
        )
        plt.close(g.fig)
