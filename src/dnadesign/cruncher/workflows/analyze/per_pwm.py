"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/workflows/analyze/per_pwm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.utils.parquet import read_parquet
from dnadesign.cruncher.workflows.analyze.plots.scatter_utils import _TRANS

logger = logging.getLogger(__name__)


def gather_per_pwm_scores(
    run_dir: Path,
    change_threshold: float,
    pwms: dict[str, object],
    bidirectional: bool,
    scale: str,
    out_path: Path,
) -> None:
    """
    Replace “first N + every_n” subsampling with “keep when per-PWM score changes by ≥ ε.”

    Steps:
      1. Read sequences.parquet (must have 'chain', 'draw', 'sequence').
      2. Build a single Scorer(pwms, bidirectional, scale).
      3. For each chain (grouped & sorted by 'draw'):
         a. Always keep the very first draw (index 0).
         b. Compute per-TF scores for draw i using scorer.compute_all_per_pwm(...).
         c. If Euclidean distance between current per-TF vector and last-kept per-TF vector ≥ change_threshold,
            keep this draw.
         d. After finishing, also force-keep the very last draw of that chain.
      4. Write out all kept rows (chain, draw, score_<tf>...) → gathered_per_pwm_everyN.csv
    """

    seq_path = run_dir / "sequences.parquet"
    if not seq_path.exists():
        raise FileNotFoundError(f"[gather] sequences.parquet not found in '{run_dir}'")

    df_all = read_parquet(seq_path)
    if "draw" not in df_all.columns or "sequence" not in df_all.columns:
        raise ValueError("gather_per_pwm_scores: sequences.parquet must have 'draw' and 'sequence' columns")

    # Build a single Scorer to do all of the “raw→z/p/logp” logic:
    scorer = Scorer(pwms, bidirectional=bidirectional, scale=scale)

    # We will collect “kept” entries here
    records: list[dict[str, object]] = []

    # For each chain, walk through its draws in ascending order
    for chain_id, df_chain in df_all.groupby("chain", sort=False):
        # Sort by draw index
        df_chain = df_chain.sort_values("draw").reset_index(drop=True)
        n_rows = len(df_chain)

        if n_rows == 0:
            continue

        # Keep the first draw unconditionally:
        first_row = df_chain.iloc[0]
        seq_str = first_row["sequence"]
        ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        seq_ints = _TRANS[ascii_arr].astype(np.int8)

        # Compute per-TF scores for the first draw:
        per_tf_last: dict[str, float] = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))
        entry0 = {"chain": int(chain_id), "draw": int(first_row["draw"])}
        for tf_name, sc_val in per_tf_last.items():
            entry0[f"score_{tf_name}"] = float(sc_val)
        records.append(entry0)

        # Now iterate over interior rows, deciding which to keep
        for idx in range(1, n_rows - 1):
            row = df_chain.iloc[idx]
            seq_str = row["sequence"]
            ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
            seq_ints = _TRANS[ascii_arr].astype(np.int8)

            per_tf_curr = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))

            # Compute Euclidean distance in per‐TF‐score space:
            squared_diff_sum = 0.0
            for tf_name in per_tf_curr:
                diff = per_tf_curr[tf_name] - per_tf_last[tf_name]
                squared_diff_sum += diff * diff
            dist = squared_diff_sum**0.5

            # If the chain moved “far enough,” keep this draw
            if dist >= change_threshold:
                rec = {"chain": int(chain_id), "draw": int(row["draw"])}
                for tf_name, sc_val in per_tf_curr.items():
                    rec[f"score_{tf_name}"] = float(sc_val)
                records.append(rec)

                # Update “last kept” to current
                per_tf_last = per_tf_curr

        # Finally, always keep the very last draw of this chain
        last_row = df_chain.iloc[-1]
        seq_str = last_row["sequence"]
        ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        seq_ints = _TRANS[ascii_arr].astype(np.int8)

        per_tf_final = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))
        rec_final = {"chain": int(chain_id), "draw": int(last_row["draw"])}
        for tf_name, sc_val in per_tf_final.items():
            rec_final[f"score_{tf_name}"] = float(sc_val)
        records.append(rec_final)

    # Build a DataFrame and write to CSV
    out_df = pd.DataFrame(records)
    # Sort by chain then draw so the output is nicely ordered
    out_df = out_df.sort_values(["chain", "draw"]).reset_index(drop=True)

    out_df.to_csv(out_path, index=False)
    logger.info("Wrote change-threshold per-PWM scores → %s", out_path)
