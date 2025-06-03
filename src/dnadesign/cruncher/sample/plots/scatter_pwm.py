"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/plots/scatter_pwm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.sample.plots.scatter_utils import (
    compute_consensus_points,
    generate_random_baseline,
    get_tf_pair,
    load_hits,
    load_samples,
    plot_scatter_plot,
    renorm_samples_df,
    subsample_df,
)
from dnadesign.cruncher.utils.config import CruncherConfig


def plot_scatter_pwm(sample_dir: Path, pwms: dict[str, PWM], cfg: CruncherConfig) -> None:
    """
    Entry point: produce <sample_dir>/scatter_pwm.png by:
      1. reading samples.csv and hits.csv
      2. subsampling MCMC trace
      3. renormalizing raw −log10 p values → logp_norm
      4. generating a random baseline of the same length
      5. computing consensus-only points
      6. delegating to plot_scatter_plot(...) to draw & save the figure
    """

    # 1) Load mandatory CSVs
    df_samples_raw = load_samples(sample_dir)  # raw -log10(p) in score_<TF> columns
    hits_df = load_hits(sample_dir)  # contains top-K sequences

    # 2) Determine sequence length (all hits have identical length)
    length = len(hits_df["sequence"].iloc[0])

    # 3) Subsample MCMC data (keep “chain” & “iter” intact)
    df_sub_raw = subsample_df(df_samples_raw, max_n=2000, sort_by="iter")

    # 4) Renormalize raw −log10 p → logp_norm for every sampled point
    df_sub = renorm_samples_df(df_sub_raw, pwms, cfg)

    # 5) Determine which TFs we’re plotting (X vs Y)
    x_tf, y_tf = get_tf_pair(cfg)

    # 6) Generate a random baseline (logp_norm) of same length & same #points as df_sub
    n_random = len(df_sub)
    df_random = generate_random_baseline(pwms, cfg, length=length, n_samples=n_random)

    # 7) Compute the two consensus-only points (logp_norm, because we call compute_consensus_points)
    consensus_pts = compute_consensus_points(pwms, cfg, length=length)

    # 8) Finally, draw & save the figure
    out_path = sample_dir / "scatter_pwm.png"
    plot_scatter_plot(
        df_samples=df_sub,
        df_random=df_random,
        consensus_pts=consensus_pts,
        x_tf=x_tf,
        y_tf=y_tf,
        scale="logp_norm",
        cfg=cfg,
        out_path=out_path,
    )
