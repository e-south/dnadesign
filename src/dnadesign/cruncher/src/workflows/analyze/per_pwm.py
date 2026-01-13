"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workflows/analyze/per_pwm.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.utils.parquet import read_parquet
from dnadesign.cruncher.utils.run_layout import sequences_path
from dnadesign.cruncher.workflows.analyze.plots.scatter_utils import encode_sequence

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
      1. Read artifacts/sequences.parquet (must have 'chain', 'draw', 'sequence').
      2. Build a single Scorer(pwms, bidirectional, scale).
      3. For each chain (grouped & sorted by 'draw'):
         a. Always keep the very first draw (index 0).
         b. Compute per-TF scores for draw i using scorer.compute_all_per_pwm(...).
         c. If Euclidean distance between current per-TF vector and last-kept per-TF vector ≥ change_threshold,
            keep this draw.
         d. After finishing, also force-keep the very last draw of that chain.
      4. Write out all kept rows (chain, draw, score_<tf>...) → gathered_per_pwm_everyN.csv
    """

    seq_path = sequences_path(run_dir)
    if not seq_path.exists():
        raise FileNotFoundError(f"[gather] artifacts/sequences.parquet not found in '{run_dir}'")
    if change_threshold <= 0:
        raise ValueError("gather_per_pwm_scores: change_threshold must be > 0")

    df_all = read_parquet(seq_path)
    if "phase" in df_all.columns:
        df_all = df_all[df_all["phase"] == "draw"].copy()
    if df_all.empty:
        raise ValueError("gather_per_pwm_scores: no draw rows found in sequences.parquet")
    missing_cols = [col for col in ("chain", "draw", "sequence") if col not in df_all.columns]
    if missing_cols:
        raise ValueError(f"gather_per_pwm_scores: sequences.parquet missing columns: {missing_cols}")

    # Build a single Scorer to do all of the “raw→z/p/logp” logic:
    scorer = Scorer(pwms, bidirectional=bidirectional, scale=scale)

    # We will collect “kept” entries here
    records: list[dict[str, object]] = []

    # For each chain, walk through its draws in ascending order
    for chain_id, df_chain in df_all.groupby("chain", sort=False):
        try:
            chain_label = int(chain_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gather_per_pwm_scores: invalid chain id {chain_id!r}") from exc
        # Sort by draw index
        df_chain = df_chain.sort_values("draw").reset_index(drop=True)
        n_rows = len(df_chain)

        if n_rows == 0:
            continue

        # Keep the first draw unconditionally:
        first_row = df_chain.iloc[0]
        seq_str = first_row["sequence"]
        seq_ints = encode_sequence(
            seq_str,
            context=f"sequences.parquet chain={chain_label} draw={first_row['draw']}",
        )

        # Compute per-TF scores for the first draw:
        per_tf_last: dict[str, float] = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))
        entry0 = {"chain": chain_label, "draw": int(first_row["draw"])}
        for tf_name, sc_val in per_tf_last.items():
            entry0[f"score_{tf_name}"] = float(sc_val)
        records.append(entry0)

        # Now iterate over interior rows, deciding which to keep
        for idx in range(1, n_rows - 1):
            row = df_chain.iloc[idx]
            seq_str = row["sequence"]
            seq_ints = encode_sequence(
                seq_str,
                context=f"sequences.parquet chain={chain_label} draw={row['draw']}",
            )

            per_tf_curr = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))

            # Compute Euclidean distance in per‐TF‐score space:
            squared_diff_sum = 0.0
            for tf_name in per_tf_curr:
                diff = per_tf_curr[tf_name] - per_tf_last[tf_name]
                squared_diff_sum += diff * diff
            dist = squared_diff_sum**0.5

            # If the chain moved “far enough,” keep this draw
            if dist >= change_threshold:
                rec = {"chain": chain_label, "draw": int(row["draw"])}
                for tf_name, sc_val in per_tf_curr.items():
                    rec[f"score_{tf_name}"] = float(sc_val)
                records.append(rec)

                # Update “last kept” to current
                per_tf_last = per_tf_curr

        # Finally, always keep the very last draw of this chain (if it isn't already kept).
        if n_rows > 1:
            last_row = df_chain.iloc[-1]
            seq_str = last_row["sequence"]
            seq_ints = encode_sequence(
                seq_str,
                context=f"sequences.parquet chain={chain_label} draw={last_row['draw']}",
            )

            per_tf_final = scorer.compute_all_per_pwm(seq_ints, len(seq_ints))
            rec_final = {"chain": chain_label, "draw": int(last_row["draw"])}
            for tf_name, sc_val in per_tf_final.items():
                rec_final[f"score_{tf_name}"] = float(sc_val)
            records.append(rec_final)

    # Build a DataFrame and write to CSV
    out_df = pd.DataFrame(records)
    # Sort by chain then draw so the output is nicely ordered
    out_df = out_df.sort_values(["chain", "draw"]).reset_index(drop=True)

    out_df.to_csv(out_path, index=False)
    logger.info("Wrote change-threshold per-PWM scores → %s", out_path)
