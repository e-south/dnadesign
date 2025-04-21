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
        low_alpha, high_alpha = 0.25, 0.45

        # 1) extract mean_cosine & log1p(mean_euclidean)
        x     = np.array([s['mean_cosine']       for s in subs], dtype=float)
        y_euc = np.log1p(np.array([s['mean_euclidean'] for s in subs], dtype=float))

        # 2) threshold
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        self.log.info(f"Scatter threshold (mean_cosine): {thr:.3e}")

        # 3) survivors
        survivors = {
            i for i, s in enumerate(subs)
            if x[i] >= thr
            and s['raw_billboard'].get('min_jaccard_dissimilarity',1) > 0
            and s['raw_billboard'].get('min_motif_string_levenshtein',1) > 0
        }
        self.log.info(f"Scatter survivors count: {len(survivors)}")

        # 4) plot
        fig, ax = plt.subplots(figsize=pc.figsize)
        for i, (xi, yi, s) in enumerate(zip(x, y_euc, subs)):
            bb = s['raw_billboard']
            if bb['min_jaccard_dissimilarity'] == 0:
                c = pc.colors['literal_drop']['jaccard']
            elif bb['min_motif_string_levenshtein'] == 0:
                c = pc.colors['literal_drop']['levenshtein']
            else:
                c = pc.colors['base']
            alpha = high_alpha if i in survivors else low_alpha
            ax.scatter(xi, yi, color=c, alpha=alpha, edgecolor='none')

        # 5) threshold line
        if pc.threshold_line:
            ax.axvline(thr, ls='--', c=pc.colors.get('threshold_line','lightgray'))

        # 6) star the overall winner
        widx = next(i for i,s in enumerate(subs) if s['subsample_id']==winner['subsample_id'])
        wx, wy = x[widx], y_euc[widx]
        ax.scatter(wx, wy, c=pc.colors['winner'], marker='*', s=200)
        ax.text(wx, wy, winner['subsample_id'], fontsize=8, va='bottom', ha='right')
        self.log.info(f"Scatter annotated winner: {winner['subsample_id']} "
                      f"(cos={wx:.3e}, log1p_mean_euc={wy:.3f})")

        # 7) legend & labels
        legend_elems = [
            Patch(facecolor=pc.colors['base'],                label='Base',           alpha=low_alpha),
            Patch(facecolor=pc.colors['literal_drop']['jaccard'],    label='Jaccard drop',   alpha=low_alpha),
            Patch(facecolor=pc.colors['literal_drop']['levenshtein'],label='Levenshtein drop',alpha=low_alpha),
            Patch(facecolor=pc.colors['base'],                label='Hitzone',        alpha=high_alpha),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc='lower right')
        ax.set_xlabel("Average Pairwise Cosine Dissimilarity (1 - cosθ)")
        ax.set_ylabel("Average Pairwise log1p E-dist")
        ax.set_title("Subsample Diversity (Scatter)")
        sns.despine(ax=ax)

        fig.savefig(outdir/"scatter_summary.png", dpi=pc.dpi, bbox_inches='tight')
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

        # Z‑score
        df['Z'] = df.groupby('Metric')['Value'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1)>0 else 0
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
                'mean_cosine':        s.get('mean_cosine', np.nan),
                'min_cosine':         s.get('min_cosine', np.nan),
                'mean_euclidean':     s.get('mean_euclidean', np.nan),
                'min_euclidean':      s.get('min_euclidean', np.nan),
                'log1p_mean_euclidean': math.log1p(s.get('mean_euclidean',0)),
                'log1p_min_euclidean':  math.log1p(s.get('min_euclidean',0)),
                'unique_cluster_count': s.get('unique_cluster_count',np.nan),
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

        g = sns.pairplot(df_sub, corner=True, diag_kind='kde', plot_kws={'alpha':0.6,'s':20})
        g.fig.suptitle("Pairwise Scatter of Core + Evo2 Metrics", y=1.02)
        for ax in g.axes.flatten():
            if ax is not None:
                sns.despine(ax=ax)
        g.fig.savefig(outdir/"pairplot_coremetrics.png", dpi=pp.dpi, bbox_inches='tight')
        plt.close(g.fig)

    def plot_hitzone(self, subs, winner, outdir):
        pc = self.cfg.plot.scatter
        hc = self.cfg.plot.hitzone

        # 1) reuse mean_cosine & same threshold
        x   = np.array([s['mean_cosine'] for s in subs], dtype=float)
        thr = self.compute_threshold(x[~np.isnan(x)], pc.threshold)
        self.log.info(f"Hitzone threshold (mean_cosine): {thr:.3e}")

        # 2) compute log1p(min_euclidean)
        y_min = np.log1p(np.array([s['min_euclidean'] for s in subs], dtype=float))

        # 3) collect ALL hits with x ≥ thr
        hits = [(i, y_min[i], s) for i, s in enumerate(subs) if x[i] >= thr]
        self.log.info(f"Hitzone total hits (x ≥ thr): {len(hits)}")
        if not hits:
            return

        # 4) sort by ascending y_min → assign x‑axis positions
        hits_sorted = sorted(hits, key=lambda t: t[1])

        # 5) plot each with the same hue logic
        fig, ax = plt.subplots(figsize=hc.figsize)
        for xpos, (i, yi, s) in enumerate(hits_sorted):
            bb = s['raw_billboard']
            if bb.get('min_jaccard_dissimilarity',0) == 0:
                c = pc.colors['literal_drop']['jaccard']
            elif bb.get('min_motif_string_levenshtein',0) == 0:
                c = pc.colors['literal_drop']['levenshtein']
            else:
                c = pc.colors['base']
            ax.scatter(xpos, yi, color=c, alpha=1.0, edgecolor='none')

        # 6) annotate the same overall winner
        #    only if that winner is in this hit‑zone
        winner_idx = next((i for i,s in enumerate(subs) if s['subsample_id']==winner['subsample_id']), None)
        if winner_idx is not None and any(i==winner_idx for i,_,_ in hits_sorted):
            # find its sorted position
            best_pos = next(pos for pos, (i,_,_) in enumerate(hits_sorted) if i==winner_idx)
            best_y   = y_min[winner_idx]
            ax.scatter(best_pos, best_y, c=pc.colors['winner'], marker='*', s=200)
            ax.text(best_pos, best_y, winner['subsample_id'],
                    fontsize=8, va='bottom', ha='right')
            self.log.info(f"Hitzone annotated winner: {winner['subsample_id']} "
                          f"(rank={best_pos}, log1p_min_euc={best_y:.3f})")
        else:
            self.log.warning("Winner not in hit-zone or no non-literal-drop survivor to annotate.")

        # 7) finalize styling
        ax.set_xticks(range(len(hits_sorted)))
        ax.set_xticklabels([''] * len(hits_sorted))
        ax.set_xlabel('Subsample (ranked by log1p min-E-dist)')
        ax.set_ylabel('Log1p Min Pairwise Euclidean')
        ax.set_title('Hit-zone: Log1p Min Euclidean Dissimilarity')
        sns.despine(ax=ax)

        fig.savefig(outdir/'hitzone_summary.png', dpi=hc.dpi, bbox_inches='tight')
        plt.close(fig)