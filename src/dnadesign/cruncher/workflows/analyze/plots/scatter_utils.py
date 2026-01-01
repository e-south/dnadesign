"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/workflows/analyze/plots/scatter_utils.py

A collection of small helper functions used by scatter_pwm.py:
  - loading score tables
  - subsampling
  - generating a random‐baseline
  - computing consensus points

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.state import SequenceState
from dnadesign.cruncher.utils.parquet import read_parquet

# Build a lookup table for A/C/G/T → 0..3
_TRANS = np.full(256, -1, dtype=np.int8)
for i, b in enumerate(b"ACGT"):
    _TRANS[b] = i

# Reverse mapping for int → char (if ever needed)
_ALPH = np.array(["A", "C", "G", "T"], dtype="<U1")


def get_tf_pair(tf_names: Sequence[str]) -> Tuple[str, str]:
    """
    Select the first two TF names for pairwise diagnostics.
    """
    if len(tf_names) < 2:
        raise ValueError("get_tf_pair: need at least two TF names")
    return tuple(tf_names[:2])


def load_per_pwm(sample_dir: Path) -> pd.DataFrame:
    """
    Read <sample_dir>/gathered_per_pwm_everyN.csv into a DataFrame.
    Raises FileNotFoundError if missing.
    Expects columns: 'chain', 'draw', 'score_<TF>' for each TF.
    """
    path = sample_dir / "gathered_per_pwm_everyN.csv"
    if not path.exists():
        raise FileNotFoundError(f"load_per_pwm: '{path}' not found")
    df = pd.read_csv(path)
    if "chain" not in df.columns or "draw" not in df.columns:
        raise ValueError("load_per_pwm: expected 'chain' and 'draw' columns in gathered_per_pwm_everyN.csv")
    return df


def load_elites(sample_dir: Path) -> pd.DataFrame:
    """
    Read the newest <sample_dir>/cruncher_elites_*/<name>.parquet into a DataFrame.
    Expects at least a "sequence" column.
    """
    candidates = list(sample_dir.glob("cruncher_elites_*/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"load_elites: no elites parquet found in '{sample_dir}'")
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    data = read_parquet(path)
    if "sequence" not in data.columns:
        raise ValueError("load_elites: missing 'sequence' column in elites parquet")
    return data


def subsample_df(df: pd.DataFrame, max_n: int, sort_by: str) -> pd.DataFrame:
    """
    Uniformly sample up to max_n rows from df (seed=0), then sort by `sort_by`.
    Returns a new DataFrame. Raises if sort_by missing.
    """
    if sort_by not in df.columns:
        raise ValueError(f"subsample_df: column '{sort_by}' not in DataFrame")
    n = min(len(df), max_n)
    sub = df.sample(n, random_state=0).sort_values(sort_by).reset_index(drop=True)
    return sub


def generate_random_baseline(
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    length: int,
    n_samples: int,
    *,
    bidirectional: bool | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of `n_samples` random sequences (uniform over A/C/G/T) of length `length`,
    scored against all PWMs using cfg.analysis.scatter_scale (which may be "llr", "z", "logp",
    or "consensus-neglop-sum").
    Uses RNG seed=0 for reproducibility.
    If bidirectional is provided, overrides cfg.sample.bidirectional.

    Leverages Scorer to build null distributions and compute per-PWM scores.
    Raises ValueError if length < 1 or n_samples < 1.
    """
    if length < 1:
        raise ValueError("generate_random_baseline: length must be ≥ 1")
    if n_samples < 1:
        raise ValueError("generate_random_baseline: n_samples must be ≥ 1")

    runner_scale = cfg.analysis.scatter_scale.lower()
    if runner_scale not in {"llr", "z", "logp", "consensus-neglop-sum"}:
        raise ValueError(f"generate_random_baseline: unsupported scatter_scale '{runner_scale}'")

    # Build a Scorer to handle all per-PWM scoring logic
    scorer = Scorer(
        pwms,
        bidirectional=bidirectional if bidirectional is not None else cfg.sample.bidirectional,
        scale=runner_scale,
    )

    rng = np.random.default_rng(0)
    records: list[dict[str, float]] = []

    for _ in tqdm(range(n_samples), desc="Random baseline"):
        # Generate a random sequence (0..3) of length `length`
        seq_ints = rng.integers(0, 4, size=length, dtype=np.int8)

        per_tf = scorer.compute_all_per_pwm(seq_ints, length)
        records.append({f"score_{tf}": float(val) for tf, val in per_tf.items()})

    return pd.DataFrame(records)


def compute_consensus_points(
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    length: int,
    tf_pair: Tuple[str, str],
    *,
    bidirectional: bool | None = None,
) -> list[tuple[float, float, str]]:
    """
    Build a “full-length consensus” for each TF in the (x_tf, y_tf) pair.
    We embed each PWM's actual consensus motif into a background of total length `length`
    (using SequenceState.from_consensus), then call Scorer.compute_all_per_pwm(…) on that
    full-length array. Returns a list of (x_val, y_val, tfname).
    If bidirectional is provided, overrides cfg.sample.bidirectional.
    """
    if length < 1:
        raise ValueError("compute_consensus_points: length must be ≥ 1")

    x_tf, y_tf = tf_pair
    scatter_scale = cfg.analysis.scatter_scale.lower()
    if scatter_scale not in {"llr", "z", "logp", "consensus-neglop-sum"}:
        raise ValueError(
            f"Unsupported scatter_scale '{scatter_scale}' in config. Use 'llr', 'z', 'logp', or 'consensus-neglop-sum'."
        )

    # Build a Scorer for exactly the two PWMs (x_tf and y_tf)
    pair_scorer = Scorer(
        {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
        bidirectional=bidirectional if bidirectional is not None else cfg.sample.bidirectional,
        scale=scatter_scale,
    )

    out: list[tuple[float, float, str]] = []
    rng = np.random.default_rng(0)

    # 1) Embed x_tf's consensus into a full-length sequence
    seq_cons_x = SequenceState.from_consensus(pwms[x_tf], length, rng, pad_with="background")
    per_tf_x = pair_scorer.compute_all_per_pwm(seq_cons_x.seq, length)
    out.append((float(per_tf_x[x_tf]), float(per_tf_x[y_tf]), x_tf))

    # 2) Embed y_tf's consensus into a full-length sequence
    seq_cons_y = SequenceState.from_consensus(pwms[y_tf], length, rng, pad_with="background")
    per_tf_y = pair_scorer.compute_all_per_pwm(seq_cons_y.seq, length)
    out.append((float(per_tf_y[x_tf]), float(per_tf_y[y_tf]), y_tf))

    return out
