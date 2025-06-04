"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/per_pwm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# dnadesign/cruncher/analyse/per_pwm.py

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dnadesign.cruncher.analyse.plots.scatter_utils import _TRANS
from dnadesign.cruncher.sample.scorer import Scorer


def gather_per_pwm_scores(
    run_dir: Path,
    every_n: int,
    pwms: dict[str, object],
    bidirectional: bool,
    penalties: dict[str, float],  # (unused—consider dropping)
    scale: str,
) -> None:
    seq_path = run_dir / "sequences.csv"
    if not seq_path.exists():
        raise FileNotFoundError(f"[gather] sequences.csv not found in '{run_dir}'")

    if every_n < 1:
        raise ValueError("gather_per_pwm_scores: every_n must be a positive integer")

    df_all = pd.read_csv(seq_path)
    if "draw" not in df_all.columns or "sequence" not in df_all.columns:
        raise ValueError("gather_per_pwm_scores: sequences.csv must have 'draw' and 'sequence' columns")

    # Always keep the first `initial_keep` draws from each chain, then every `every_n`th draw.
    initial_keep = 10
    # “initial_mask” is True if a row’s draw < initial_keep within its own chain.
    initial_mask = df_all.groupby("chain")["draw"].transform(lambda d: d < initial_keep)
    # “sampling_mask” is True if the draw is a multiple of every_n (for all chains).
    sampling_mask = (df_all["draw"] % every_n) == 0
    mask = initial_mask | sampling_mask
    filtered = df_all.loc[mask].reset_index(drop=True)
    if filtered.empty:
        print(f"[gather] No rows where (draw < {initial_keep}) or (draw % {every_n} == 0) (total rows: {len(df_all)})")
        return

    # Build a single Scorer to do all of the “raw→z/p/logp” logic:
    scorer = Scorer(pwms, bidirectional=bidirectional, scale=scale)

    records: list[dict[str, object]] = []
    for row in tqdm(filtered.itertuples(index=False), total=len(filtered), desc="Scoring per-PWM"):
        c = int(getattr(row, "chain"))
        d = int(getattr(row, "draw"))
        seq_str = getattr(row, "sequence")
        ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        seq_ints = _TRANS[ascii_arr].astype(np.int8)

        per_tf: dict[str, float] = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))
        entry = {"chain": c, "draw": d}
        for tf_name, sc_val in per_tf.items():
            entry[f"score_{tf_name}"] = float(sc_val)
        records.append(entry)

    out_df = pd.DataFrame(records)
    out_df.to_csv(run_dir / "gathered_per_pwm_everyN.csv", index=False)
    print(f"[gather] Wrote per-PWM scores every {every_n} (plus first {initial_keep}) → 'gathered_per_pwm_everyN.csv'")
