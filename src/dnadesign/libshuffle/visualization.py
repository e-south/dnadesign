"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/visualization.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""
import math
import logging
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# aesthetic theme: ticks + colorblind palette, remove top/right spines
sns.set_theme(style='ticks', palette='colorblind')
plt.rcParams.update({'font.size': 12})

class Plotter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger('libshuffle.visualization')

    @staticmethod
    def compute_threshold(data: np.ndarray, thr_cfg):
        if thr_cfg.type == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            return q3 + (q3 - q1) * thr_cfg.factor
        if thr_cfg.type == 'percentile':
            return np.percentile(data, thr_cfg.factor)
        raise ValueError(f"Unknown threshold type: {thr_cfg.type}")

    def plot_scatter(self, subs, winner, outdir):
        pc = self.cfg.plot.scatter
        low_a, high_a = pc.low_alpha, pc.high_alpha

        x = np.array([s['mean_cosine'] for s in subs], dtype=float)
        y = np.log1p(np.array([s['mean_euclidean'] for s in subs], dtype=float))
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        self.log.info(f"Scatter threshold (mean_cosine): {thr:.3e}")
        passed = {i for i,m in enumerate(x) if not np.isnan(m) and m >= thr}

        fig, ax = plt.subplots(figsize=pc.figsize)
        for i, (xi, yi, s) in enumerate(zip(x, y, subs)):
            bb = s['raw_billboard']
            if i not in passed:
                # de-emphasized
                ax.scatter(xi, yi, color='lightgray', alpha=low_a, edgecolor='none')
            else:
                # consolidate literal flags
                is_literal = (
                    bb.get('min_jaccard_dissimilarity', 1) == 0
                    or bb.get('min_motif_string_levenshtein', 1) == 0
                )
                if is_literal:
                    c = pc.colors['literal_drop']['literal']
                elif s.get('unique_cluster_count', 0) != self.cfg.subsample_size:
                    c = pc.colors['cluster_drop']
                else:
                    c = pc.colors['base']
                ax.scatter(xi, yi, color=c, alpha=high_a, edgecolor='none')

        if pc.threshold_line:
            ax.axvline(thr, ls='--', c=pc.colors.get('threshold_line','lightgray'))

        # annotate winner
        widx = next(i for i,s in enumerate(subs) if s['subsample_id']==winner['subsample_id'])
        wx, wy = x[widx], y[widx]
        ax.scatter(wx, wy,
                   c=pc.colors['winner'],
                   marker='*',
                   s=pc.star_size,
                   edgecolor='none',
                   zorder=3)
        ax.text(wx, wy, winner['subsample_id'], fontsize=8, va='bottom', ha='right')

        legend_elems = [
            Patch(facecolor=pc.colors['base'],                label='Passed',             alpha=high_a),
            Patch(facecolor=pc.colors['literal_drop']['literal'], label='Literal flagged',    alpha=high_a),
            Patch(facecolor=pc.colors['cluster_drop'],        label='Leiden flagged',     alpha=high_a),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc='lower right')

        ax.set_xlabel("Average Pairwise Cosine Dissimilarity (1 - cosθ)")
        ax.set_ylabel("Average Pairwise log1p E-dist")
        ax.set_title("Angular vs. Euclidean Diversity: All 16-member Subsamples")
        sns.despine(ax=ax)

        fig.savefig(outdir/"scatter_summary.png", dpi=pc.dpi, bbox_inches='tight')
        plt.close(fig)


    def plot_flag_composition(self, subs, outdir):
        pc = self.cfg.plot.scatter
        low_a, high_a = pc.low_alpha, pc.high_alpha

        x = np.array([s['mean_cosine'] for s in subs], dtype=float)
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        filtered = [s for i,s in enumerate(subs) if x[i]>=thr]

        counts = {'Passed':0, 'Literal flagged':0, 'Leiden flagged':0}
        for s in filtered:
            bb = s['raw_billboard']
            if bb.get('min_jaccard_dissimilarity',1)==0 or bb.get('min_motif_string_levenshtein',1)==0:
                counts['Literal flagged'] += 1
            elif s.get('unique_cluster_count',0)!= self.cfg.subsample_size:
                counts['Leiden flagged'] += 1
            else:
                counts['Passed'] += 1

        fig, ax = plt.subplots(figsize=(3,6))
        bottom = 0
        for label, color_key in [
            ('Passed','base'),
            ('Literal flagged',('literal_drop','literal')),
            ('Leiden flagged','cluster_drop'),
        ]:
            if isinstance(color_key, tuple):
                col = pc.colors[color_key[0]][color_key[1]]
            else:
                col = pc.colors[color_key]
            ax.bar('Subsamples', counts[label], bottom=bottom, label=label, color=col, alpha=high_a)
            bottom += counts[label]

        ax.set_xlabel("IQR-Filtered Subsamples")
        ax.set_ylabel("Count")
        ax.set_title("Composition of IQR-Filtered Subsamples")
        ax.legend(frameon=False, bbox_to_anchor=(1,1))
        sns.despine(ax=ax)

        fig.savefig(outdir/"flag_composition.png", dpi=pc.dpi, bbox_inches='tight')
        plt.close(fig)


    def plot_hitzone(self, subs, winner, outdir):
        pc = self.cfg.plot.scatter
        low_a, high_a = pc.low_alpha, pc.high_alpha
        hc = self.cfg.plot.hitzone

        x = np.array([s['mean_cosine'] for s in subs], dtype=float)
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        hits = [(i, np.log1p(s['min_euclidean']), s) for i,s in enumerate(subs) if x[i]>=thr]
        if not hits:
            return

        hits_sorted = sorted(hits, key=lambda t: t[1])

        fig, ax = plt.subplots(figsize=hc.figsize)
        for pos,(i,yi,s) in enumerate(hits_sorted):
            bb = s['raw_billboard']
            is_literal = (
                bb.get('min_jaccard_dissimilarity',1) == 0
                or bb.get('min_motif_string_levenshtein',1) == 0
            )
            if is_literal:
                c = pc.colors['literal_drop']['literal']
            elif s.get('unique_cluster_count',0)!= self.cfg.subsample_size:
                c = pc.colors['cluster_drop']
            else:
                c = pc.colors['base']
            ax.scatter(pos, yi, color=c, alpha=high_a, edgecolor='none')

        widx = next(i for i,s in enumerate(subs) if s['subsample_id']==winner['subsample_id'])
        for pos,(i,yi,_) in enumerate(hits_sorted):
            if i==widx:
                ax.scatter(pos, yi, c=pc.colors['winner'], marker='*', s=pc.star_size, zorder=3)
                ax.text(pos, yi, winner['subsample_id'], fontsize=8, va='bottom', ha='right')
                break

        ax.set_xticks([])
        ax.set_xlabel("Subsample ID")
        ax.set_ylabel("Min Pairwise E-dist (log1p)")
        ax.set_title("Outlier Ranking by Minimum Euclidean Gap")
        legend_elems = [
            Patch(facecolor=pc.colors['base'],                label='Passed',             alpha=high_a),
            Patch(facecolor=pc.colors['literal_drop']['literal'], label='Literal flagged',    alpha=high_a),
            Patch(facecolor=pc.colors['cluster_drop'],        label='Leiden flagged',     alpha=high_a),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc='upper left')
        sns.despine(ax=ax)

        fig.savefig(outdir/"hitzone_summary.png", dpi=hc.dpi, bbox_inches='tight')
        plt.close(fig)


    def plot_kde(self, subs, outdir):
        kc   = self.cfg.plot.kde
        mets = self.cfg.billboard_core_metrics
        rec  = []
        for s in subs:
            bb = s['raw_billboard']
            for m in mets:
                rec.append({'Metric': m, 'Value': bb.get(m)})
        df = pd.DataFrame(rec)

        fig, ax = plt.subplots(figsize=kc.figsize)
        sns.kdeplot(data=df, x='Value', hue='Metric', common_norm=False, warn_singular=False, ax=ax)
        ax.set_title("Core Metric Distributions (Raw)")
        sns.despine(ax=ax)
        fig.savefig(outdir/"kde_coremetrics_raw.png", dpi=kc.dpi, bbox_inches='tight')
        plt.close(fig)

        df['Z'] = df.groupby('Metric')['Value'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0
        )
        fig, ax = plt.subplots(figsize=kc.figsize)
        sns.kdeplot(data=df, x='Z', hue='Metric', common_norm=False, warn_singular=False, ax=ax)
        ax.set_title("Core Metric Distributions (Z-score)")
        sns.despine(ax=ax)
        fig.savefig(outdir/"kde_coremetrics_zscore.png", dpi=kc.dpi, bbox_inches='tight')
        plt.close(fig)


    def plot_pairplot(self, subs, outdir):
        pp = self.cfg.plot.pairplot
        rows = []
        for s in subs:
            bb = s['raw_billboard']
            row = {m: bb.get(m, np.nan) for m in self.cfg.billboard_core_metrics}
            row.update({
                'mean_cosine':         s.get('mean_cosine', np.nan),
                'min_cosine':          s.get('min_cosine', np.nan),
                'mean_euclidean':      s.get('mean_euclidean', np.nan),
                'min_euclidean':       s.get('min_euclidean', np.nan),
                'log1p_mean_euclidean': math.log1p(s.get('mean_euclidean', 0)),
                'log1p_min_euclidean':  math.log1p(s.get('min_euclidean', 0)),
                'unique_cluster_count': s.get('unique_cluster_count', np.nan),
            })
            rows.append(row)

        df = pd.DataFrame(rows)
        cols = list(self.cfg.billboard_core_metrics)
        for axis in (self.cfg.plot.scatter.x, self.cfg.plot.scatter.y):
            if axis not in cols:
                cols.append(axis)
        if 'unique_cluster_count' not in cols:
            cols.append('unique_cluster_count')

        df_sub = df[cols].dropna(axis=1, how='all')
        if df_sub.shape[1] < 2:
            return

        g = sns.pairplot(df_sub, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20})
        g.fig.suptitle("Pairwise Scatter of Core + Evo2 Metrics", y=1.02)
        for ax in g.axes.flatten():
            if ax is not None:
                sns.despine(ax=ax)
        g.fig.savefig(outdir/"pairplot_coremetrics.png", dpi=pp.dpi, bbox_inches='tight')
        plt.close(g.fig)