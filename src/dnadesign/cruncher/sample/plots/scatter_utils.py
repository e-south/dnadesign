"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/plots/scatter_utils.py

A collection of small helper functions used by scatter_pwm.py:
  - loading CSVs
  - subsampling
  - generating a random-baseline
  - computing consensus points
  - actually drawing/saving the scatter plot

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState
from dnadesign.cruncher.utils.config import CruncherConfig

# Pre-build an ASCII→int8 lookup for very fast string→vec conversion
_TRANS = np.full(256, -1, dtype=np.int8)
for i, b in enumerate(b"ACGT"):
    _TRANS[b] = i

# Fixed array of characters, used to convert numeric‐encoded sequences back to strings.
_ALPH = np.array(["A", "C", "G", "T"], dtype="<U1")


def load_samples(sample_dir: Path) -> pd.DataFrame:
    """
    Read the raw MCMC samples.csv (with columns: chain, iter, beta, score_<TF1>, score_<TF2>, …,
    and optionally a 'sequence' column if you decided to record it).
    """
    return pd.read_csv(sample_dir / "samples.csv")


def load_hits(sample_dir: Path) -> pd.DataFrame:
    """
    Read the hits.csv (contains the top-K sequences and their per-PWM raw LLR scores).
    """
    return pd.read_csv(sample_dir / "hits.csv")


def get_tf_pair(cfg: CruncherConfig) -> Tuple[str, str]:
    """
    By convention, the first two TF names in cfg.regulator_sets[0] are the X- and Y-axes.
    """
    return tuple(cfg.regulator_sets[0][:2])


def subsample_df(df: pd.DataFrame, max_n: int, sort_by: str) -> pd.DataFrame:
    """
    Uniformly sample up to max_n rows from df (seed=0), then sort by `sort_by` column.
    Used to pick a subset of MCMC points for plotting.
    """
    n = min(len(df), max_n)
    return df.sample(n, random_state=0).sort_values(sort_by)


def renorm_samples_df(
    df: pd.DataFrame,
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    length: int,
) -> pd.DataFrame:
    """
    Convert every `score_<TF>` column from raw −log10 p to the 0–1 logp_norm scale.

    There are two possible code paths:

      1) If `df` contains a "sequence" column (string of A/C/G/T), we re-score each row
         exactly (raw_llr → p_seq → p_cons → logp_norm).

      2) If `df` does NOT have a "sequence" column, we assume the existing
         `score_<TF>` is already `−log10(p_seq)`, so we only need to divide by
         `−log10(p_consensus)` (which we compute once per TF & sequence‐length).

    IMPORTANT:
      * `length` must be the total sequence length used in the MCMC (so that
        we know n_win = length − width + 1 for each PWM).
      * If you never recorded "sequence" in your samples.csv, use code path (2).
      * If you DID record "sequence", we choose path (1) and completely re-compute
        scores via a fresh Scorer(scale="logp_norm") on every row.

    Returns a NEW DataFrame (a copy of df) where all `score_<TF>` columns have been overwritten.
    """
    # 1) detect which TFs appear as columns: any col of the form "score_<TF>"
    tf_names = [col[6:] for col in df.columns if col.startswith("score_")]
    if not tf_names:
        return df.copy()  # nothing to do

    # 2) Build a Scorer that knows how to convert raw_llr → logp_norm for exactly those TFs
    scorer_plot = Scorer(
        {tf: pwms[tf] for tf in tf_names},
        bidirectional=cfg.sample.bidirectional,
        scale="logp_norm",
        penalties=cfg.sample.penalties,
    )

    if "sequence" in df.columns:
        # ——————————— Path (1): we have actual sequence strings, so do a full re-score ———————————
        def _str_to_ints(seq_str: str) -> np.ndarray:
            """
            Convert a sequence string "ACGT…" into numpy int8 array [0..3].
            We do a single np.frombuffer pass + lookup via _TRANS.
            """
            ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
            return _TRANS[ascii_arr]

        def _renorm_row(row: pd.Series) -> pd.Series:
            """
            For one row of df, read "sequence", compute raw_llr for each TF,
            then overwrite row["score_<TF>"] with logp_norm. Return the modified row.
            """
            seq_ints = _str_to_ints(row["sequence"])
            L = seq_ints.size
            for tf in tf_names:
                info = scorer_plot._cache[tf]
                raw_llr, _extras = scorer_plot._best_llr_and_extra_hits(seq_ints, info)
                row[f"score_{tf}"] = scorer_plot._scale_llr(raw_llr, info, seq_length=L)
            return row

        # Apply row-wise (vectorization here isn’t worth the complexity for ≤2000 points)
        return df.apply(_renorm_row, axis=1)

    else:
        # ——————————— Path (2): no "sequence" column, assume existing score_<TF> = −log10(p_seq) ——————
        # We only need to divide each raw score by (−log10(p_consensus)) for that TF and length.

        # Precompute denom[tf] = (−log10 p_consensus)  for each TF at this length
        denom: Dict[str, float] = {}
        for tf in tf_names:
            info = scorer_plot._cache[tf]
            # 1) consensus raw_llr = sum of max log-odds per column
            cons_llr = float(info.lom.max(axis=1).sum())
            # 2) compute p_win_cons and p_cons (taking both strands into account if bidirectional)
            p_win_cons = scorer_plot._interp_tail_p(cons_llr, info)
            n_win = max(1, length - info.width + 1)
            p_cons = 1.0 - (1.0 - p_win_cons) ** n_win
            p_cons = max(p_cons, 1e-300)
            denom[tf] = -np.log10(p_cons)

        # Now divide each raw −log10(p_seq) by denom[tf]
        out = df.copy()
        for tf in tf_names:
            col = f"score_{tf}"
            out[col] = out[col] / denom[tf]
            # (Optionally clamp at exactly 1.0 if any floating rounding pushed >1.0)
            out[col] = out[col].clip(upper=1.0)
        return out


def generate_random_baseline(
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    length: int,
    n_samples: int,
) -> pd.DataFrame:
    """
    Build a DataFrame of `n_samples` random sequences (uniform over A/C/G/T) of length `length`,
    scored against *all* PWMs in `pwms` using logp_norm. Returns columns [score_<TF1>, score_<TF2>, …].

    Uses a fixed RNG seed (0) for reproducibility. This baseline is always logp_norm.
    """
    rng = np.random.default_rng(0)
    scorer = Scorer(
        pwms,
        bidirectional=cfg.sample.bidirectional,
        scale="logp_norm",
        background=(0.25, 0.25, 0.25, 0.25),
        penalties=cfg.sample.penalties,
    )

    tf_list = list(pwms.keys())
    rows = []
    for _ in range(n_samples):
        seq_state = SequenceState.random(length, rng)
        scaled_row: Dict[str, float] = {}
        for tfname in tf_list:
            info = scorer._cache[tfname]
            raw_llr, _extras = scorer._best_llr_and_extra_hits(seq_state.seq, info)
            val = scorer._scale_llr(raw_llr, info, seq_length=length)
            scaled_row[f"score_{tfname}"] = float(val)
        rows.append(scaled_row)

    return pd.DataFrame(rows)


def make_consensus_centered(
    pwm: PWM,
    length: int,
    pad_with: str = "background",
) -> SequenceState:
    """
    Deterministically embed the PWM’s consensus (argmax at each column) in the exact center
    of a sequence of total length = `length`.

    1. cons_vec = argmax row-wise of pwm.matrix → (w,)
    2. If w ≥ length: truncate around center
    3. If w < length: pad on both sides (background or fixed base)
    Returns a SequenceState of length `length`.
    """
    if length < 1:
        raise ValueError(f"Cannot embed consensus into length <1; got {length}")

    cons = np.argmax(pwm.matrix, axis=1).astype(np.int8)  # shape (w,)
    w = cons.size

    if w >= length:
        start = (w - length) // 2
        truncated = cons[start : start + length]
        return SequenceState(truncated)

    pad_n = length - w
    left_pad = pad_n // 2
    right_pad = pad_n - left_pad

    if pad_with == "background":
        rng = np.random.default_rng(0)
        full_pad = rng.integers(0, 4, size=pad_n, dtype=np.int8)
    else:
        pad_base = pad_with.upper()
        if pad_base not in ("A", "C", "G", "T"):
            raise ValueError(f"Invalid pad_with '{pad_with}'; must be one of A/C/G/T or 'background'")
        idx = int(np.where(_ALPH == pad_base)[0])
        full_pad = np.full(pad_n, idx, dtype=np.int8)

    prefix_pad = full_pad[:left_pad]
    suffix_pad = full_pad[left_pad : left_pad + right_pad]
    seq_arr = np.concatenate([prefix_pad, cons, suffix_pad])
    return SequenceState(seq_arr)


def compute_consensus_points(
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    length: int,
) -> list[tuple[float, float, str]]:
    """
    Build a *stand-alone* “pure consensus” (no flanks) for each PWM in the XY pair,
    score it against both TFs, and return a list of (x_val, y_val, tfname).

    Because we call Scorer(scale="logp_norm") under the hood, each TF’s own consensus
    will appear as exactly (1.000, ≤1.000) or (≤1.000, 1.000) on (self, cross) axes.
    """
    x_tf, y_tf = get_tf_pair(cfg)

    pair_scorer = Scorer(
        {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
        bidirectional=cfg.sample.bidirectional,
        scale="logp_norm",
        background=(0.25, 0.25, 0.25, 0.25),
        penalties=cfg.sample.penalties,
    )
    consensus_pts: list[tuple[float, float, str]] = []

    for tfname in (x_tf, y_tf):
        pwm = pwms[tfname]
        w = pwm.matrix.shape[0]

        # 1) Build pure consensus vector (length = w)
        info = pair_scorer._cache[tfname]
        cons_vec = np.argmax(info.lom, axis=1).astype(np.int8)
        seq_state = SequenceState(cons_vec)

        # 2) Score that w-length consensus against both TFs
        raw_vs_x, raw_vs_y = pair_scorer.score_per_pwm(seq_state.seq)
        info_x = pair_scorer._cache[x_tf]
        info_y = pair_scorer._cache[y_tf]

        x_val = pair_scorer._scale_llr(raw_vs_x, info_x, seq_length=w)
        y_val = pair_scorer._scale_llr(raw_vs_y, info_y, seq_length=w)

        print(
            f"[DEBUG] {tfname:5s}  raw_self={raw_vs_x if tfname == x_tf else raw_vs_y:.1f} "
            f"→ coords=({x_val:.3f}, {y_val:.3f})"
        )
        consensus_pts.append((float(x_val), float(y_val), tfname))

    return consensus_pts


def plot_scatter_plot(
    df_samples: pd.DataFrame,
    df_random: pd.DataFrame,
    consensus_pts: list[tuple[float, float, str]],
    x_tf: str,
    y_tf: str,
    scale: str,
    cfg: CruncherConfig,
    out_path: Path,
) -> None:
    """
    Draw and save a single “scatter-PWM” figure:

      1) Gray cloud = df_random (background points)
      2) Colored trajectories & scatter = df_samples (logp_norm‐scaled MCMC points)
      3) Red stars = consensus_pts
      4) Title = MCMC settings summary

    Expects df_samples to already have columns:
      - “chain”, “iter”
      - “score_<x_tf>”, “score_<y_tf>” in [0,1] if scale=="logp_norm"
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection

    if "chain" not in df_samples.columns or "iter" not in df_samples.columns:
        raise ValueError("df_samples must have 'chain' and 'iter' columns.")

    unique_chains = sorted(df_samples["chain"].unique())
    n_chains = len(unique_chains)
    base_palette = sns.color_palette("colorblind", n_chains)
    chain_to_color = {chain: base_palette[i] for i, chain in enumerate(unique_chains)}

    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 6))

    # 1) Gray cloud for random baseline
    if not df_random.empty:
        ax.scatter(
            df_random[f"score_{x_tf}"],
            df_random[f"score_{y_tf}"],
            c="lightgray",
            alpha=0.4,
            s=20,
            label="random",
        )

    # 2) MCMC trajectories & scatter (each chain gets its own color ramp)
    for chain_id in unique_chains:
        df_chain = df_samples[df_samples["chain"] == chain_id]
        iters = df_chain["iter"].to_numpy()
        itmin, itmax = iters.min(), iters.max()
        if itmax == itmin:
            norm_iters = np.zeros_like(iters, dtype=float)
        else:
            norm_iters = (iters - itmin) / (itmax - itmin)

        base_hue = np.array(chain_to_color[chain_id])  # shape (3,)

        whites = np.ones((len(norm_iters), 3))
        alphas = norm_iters.reshape(-1, 1)
        per_point_rgb = (1 - alphas) * whites + (alphas * base_hue)

        pts = df_chain.sort_values("iter")[[f"score_{x_tf}", f"score_{y_tf}"]].to_numpy()
        if len(pts) >= 2:
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_alphas = (norm_iters[:-1] + norm_iters[1:]) / 2.0
            seg_colors = (1 - seg_alphas[:, None]) * np.ones((len(seg_alphas), 3)) + (seg_alphas[:, None] * base_hue)
            lc = LineCollection(segments, colors=seg_colors, linewidths=1, alpha=0.5)
            ax.add_collection(lc)

        ax.scatter(
            df_chain[f"score_{x_tf}"],
            df_chain[f"score_{y_tf}"],
            c=per_point_rgb,
            alpha=0.7,
            s=30,
            linewidth=0,
            label=f"chain {chain_id}",
        )

    # 3) Plot the two consensus points as big red stars
    for cx, cy, tfname in consensus_pts:
        ax.scatter(
            cx,
            cy,
            marker="*",
            s=200,
            edgecolors="none",
            facecolors="red",
            label=f"consensus_{tfname}",
        )
        ax.text(cx, cy, f" {tfname}", va="center", ha="left", fontsize=10)

    # 4) Label axes based on `scale`
    if scale == "llr":
        xlabel = f"LLR_{x_tf}"
        ylabel = f"LLR_{y_tf}"
    elif scale == "z":
        xlabel = f"Z_{x_tf}"
        ylabel = f"Z_{y_tf}"
    elif scale == "p":
        xlabel = f"P_{x_tf}"
        ylabel = f"P_{y_tf}"
    else:  # logp_norm
        xlabel = f"consensus-normalised score_{x_tf}"
        ylabel = f"consensus-normalised score_{y_tf}"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # 5) Build multi-line title from cfg.sample
    sample_cfg = cfg.sample
    opt_cfg = sample_cfg.optimiser
    cooling = opt_cfg.cooling
    if cooling.kind == "linear":
        b0, b1 = cooling.beta
        cooling_str = f"linear (β=[{b0:.2f}→{b1:.2f}])"
    else:
        betas = ", ".join(f"{b:.2f}" for b in cooling.beta)
        cooling_str = f"geometric (β=[{betas}])"

    title1 = f"{x_tf} vs {y_tf} over MCMC"
    title2 = (
        f"chains={sample_cfg.chains} · draws={sample_cfg.draws} · tune={sample_cfg.tune} · "
        f"cooling={cooling_str} · swap_prob={opt_cfg.swap_prob:.2f}"
    )
    ax.set_title(f"{title1}\n{title2}", fontsize=10)

    ax.legend(frameon=False, loc="lower right", fontsize=8)
    sns.despine(ax=ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
