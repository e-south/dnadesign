"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/summary.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import logging
import pandas as pd
from dnadesign.aligner.metrics import mean_pairwise

logger = logging.getLogger(__name__)

NOTES = {
    "tf_richness":                   "Unique TF count",
    "1_minus_gini":                  "Inverted Gini (TF evenness)",
    "min_jaccard_dissimilarity":     "Min pairwise Jaccard dissimilarity",
    "min_tf_entropy":                "Min per‑TF positional entropy",
    "min_motif_string_levenshtein":  "Min normalized motif‑string Levenshtein",
    "nw_dissimilarity":              "Min normalized Needleman–Wunsch dissimilarity",
}

def generate_csv_summaries(results: dict, csv_dir: os.PathLike) -> None:
    """
    Writes:
      - summary_metrics.csv
      - tf_coverage_summary.csv
      - diversity_summary.csv  (in the order of cfg["diversity_metrics"])
    """
    os.makedirs(csv_dir, exist_ok=True)
    logger.info(f"Writing CSV summaries into {csv_dir}")

    # summary_metrics.csv
    df_sum = pd.DataFrame({
        "num_sequences":      [results["num_sequences"]],
        "sequence_length":    [results["sequence_length"]],
        "total_tfbs_instances":[results["total_tfbs_instances"]],
    })
    df_sum.to_csv(os.path.join(csv_dir, "summary_metrics.csv"), index=False)

    # tf_coverage_summary.csv
    df_tf = (
        pd.DataFrame.from_dict(results["tf_frequency"], orient="index", columns=["frequency"])
          .reset_index()
          .rename(columns={"index":"tf"})
    )
    df_tf.to_csv(os.path.join(csv_dir, "tf_coverage_summary.csv"), index=False)

    # diversity_summary.csv
    rows = []
    for m in results.get("cfg_diversity_metrics", results["core_metrics"].keys()):
        v = results["core_metrics"].get(m)
        rows.append({
            "metric_name": m,
            "value":       v,
            "notes":       NOTES.get(m, "")
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(csv_dir, "diversity_summary.csv"),
        index=False
    )
    logger.info("CSV summaries written")


def generate_entropy_summary_csv(results: dict, csv_dir: os.PathLike) -> None:
    # identical for now
    generate_csv_summaries(results, csv_dir)


def print_summary(results: dict) -> None:
    print("\n=== billboard SUMMARY ===")
    print(f"Sequences: {results['num_sequences']}   Length: {results['sequence_length']}")
    print(f"Total TFBS instances: {results['total_tfbs_instances']}\n")
    print("Core Diversity Metrics:")
    for k, v in results["core_metrics"].items():
        print(f"  {k}: {v}")
    print("=========================\n")
    logger.info("Printed summary to console")