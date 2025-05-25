"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/plots/arviz.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt, seaborn as sns, arviz as az, numpy as np
from pathlib import Path

def mcmc_diagnostics(idata, outdir: Path):
    score = idata.observed_data["score"].values.ravel()

    # 1. Trace
    az.plot_trace(idata, var_names=["score"])
    plt.savefig(outdir/"trace_score.png", dpi=300); plt.close()

    # 2. KDE
    sns.kdeplot(score, fill=True)
    plt.xlabel("total PWM score"); plt.savefig(outdir/"score_kde.png"); plt.close()

    # 3. Running best
    best = np.maximum.accumulate(score)
    plt.plot(best); plt.ylabel("running best"); plt.xlabel("draw")
    plt.savefig(outdir/"running_best.png"); plt.close()

def scatter_pwm(idata, outdir: Path, pwms=("A","B","C")):
    sA, sB = (idata.observed_data[f"score_{p}"].values.ravel() for p in pwms[:2])
    sns.scatterplot(x=sA, y=sB, alpha=0.3, s=10)
    plt.xlabel("PWM-A score"); plt.ylabel("PWM-B score")
    plt.savefig(outdir/"scatter_AB.png"); plt.close()


