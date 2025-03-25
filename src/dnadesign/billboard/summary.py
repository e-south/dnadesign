"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/summary.py

Module Author(s): Eric J. South
Dunlop Lab

Updates:
- Added a helper function generate_entropy_summary_csv() to output only the entropy summary,
  which is used in DRY_RUN mode.
--------------------------------------------------------------------------------
"""

import os
import pandas as pd

def generate_csv_summaries(results, output_dir):
    """
    Generate CSV files with summary metrics, per-TF coverage statistics,
    mapping failures, and TF combination counts.
    Also generate a new entropy_summary.csv containing scalar entropy metrics.
    This function is used in full-run mode.
    """
    csv_dir = os.path.join(output_dir, "csvs")
    
    summary_data = {
        "num_sequences": [results["num_sequences"]],
        "sequence_length": [results["sequence_length"]],
        "num_unique_tfs": [len(results["unique_tfs"])],
        "total_tfbs_instances": [results["total_tfbs_instances"]],
        "average_tfbs_per_sequence": [results["average_tfbs_per_sequence"]],
        "failed_tfbs_mappings": [len(results["failed_mappings"])],
        "positional_entropy_forward": [results["positional_entropy_forward"]],
        "positional_entropy_reverse": [results["positional_entropy_reverse"]],
        "num_unique_combinations": [len(results["combo_counts"])],
        "combo_entropy": [results["combo_entropy"]]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(csv_dir, "summary_metrics.csv"), index=False)
    
    df_tf = pd.DataFrame(results["per_tf_entropy"])
    df_tf["frequency"] = df_tf["tf"].apply(lambda x: results["tf_frequency"].get(x, 0))
    df_tf.to_csv(os.path.join(csv_dir, "tf_coverage_summary.csv"), index=False)
    
    if results["failed_mappings"]:
        df_fail = pd.DataFrame(results["failed_mappings"])
    else:
        df_fail = pd.DataFrame(columns=["sequence_id", "tf", "motif", "offset", "reason"])
    df_fail.to_csv(os.path.join(csv_dir, "tfbs_failures.csv"), index=False)
    
    combo_data = [{"tf_combo": combo, "count": count} for combo, count in results["combo_counts"].items()]
    df_combo = pd.DataFrame(combo_data)
    df_combo.to_csv(os.path.join(csv_dir, "tf_combinations.csv"), index=False)
    
    # Create the entropy_summary.csv with raw and normalized entropy metrics.
    entropy_summary_data = [
        {
            "metric_name": "global_positional_entropy",
            "value": results["global_positional_entropy"],
            "normalized_value": results["global_positional_entropy_norm"],
            "notes": "Global average of forward and reverse positional entropy"
        },
        {
            "metric_name": "per_tf_entropy_mean",
            "value": results["per_tf_entropy_mean_raw"],
            "normalized_value": results["per_tf_entropy_mean_norm"],
            "notes": "Unweighted mean of per-TF positional entropy"
        },
        {
            "metric_name": "per_tf_entropy_weighted",
            "value": results["per_tf_entropy_weighted_raw"],
            "normalized_value": results["per_tf_entropy_weighted_norm"],
            "notes": "TF frequency weighted mean of per-TF positional entropy"
        },
        {
            "metric_name": "per_tf_entropy_topk",
            "value": results["per_tf_entropy_topk_raw"],
            "normalized_value": results["per_tf_entropy_topk_norm"],
            "notes": "Mean of per-TF positional entropy for top-K most frequent TFs"
        },
        {
            "metric_name": "tf_combination_entropy",
            "value": results["combo_entropy"],
            "normalized_value": results["tf_combo_entropy_norm"],
            "notes": "Shannon entropy over unique TF rosters (combinations)"
        }
    ]
    df_entropy = pd.DataFrame(entropy_summary_data)
    df_entropy.to_csv(os.path.join(csv_dir, "entropy_summary.csv"), index=False)

def generate_entropy_summary_csv(results, output_dir):
    """
    Generate only the entropy_summary.csv containing the scalar entropy metrics.
    This function is used when DRY_RUN is enabled.
    """
    csv_dir = os.path.join(output_dir, "csvs")
    entropy_summary_data = [
        {
            "metric_name": "global_positional_entropy",
            "value": results["global_positional_entropy"],
            "normalized_value": results["global_positional_entropy_norm"],
            "notes": "Global average of forward and reverse positional entropy"
        },
        {
            "metric_name": "per_tf_entropy_mean",
            "value": results["per_tf_entropy_mean_raw"],
            "normalized_value": results["per_tf_entropy_mean_norm"],
            "notes": "Unweighted mean of per-TF positional entropy"
        },
        {
            "metric_name": "per_tf_entropy_weighted",
            "value": results["per_tf_entropy_weighted_raw"],
            "normalized_value": results["per_tf_entropy_weighted_norm"],
            "notes": "TF frequency weighted mean of per-TF positional entropy"
        },
        {
            "metric_name": "per_tf_entropy_topk",
            "value": results["per_tf_entropy_topk_raw"],
            "normalized_value": results["per_tf_entropy_topk_norm"],
            "notes": "Mean of per-TF positional entropy for top-K most frequent TFs"
        },
        {
            "metric_name": "tf_combination_entropy",
            "value": results["combo_entropy"],
            "normalized_value": results["tf_combo_entropy_norm"],
            "notes": "Shannon entropy over unique TF rosters (combinations)"
        }
    ]
    df_entropy = pd.DataFrame(entropy_summary_data)
    df_entropy.to_csv(os.path.join(csv_dir, "entropy_summary.csv"), index=False)

def print_summary(results):
    """Print a concise summary of key analysis metrics to the console."""
    print(f"Loaded {results['num_sequences']} sequences (length: {results['sequence_length']} nt)")
    print(f"Detected {len(results['unique_tfs'])} unique TFs across {results['total_tfbs_instances']} motif instances")
    num_failures = len(results['failed_mappings'])
    percent_fail = (num_failures / results['total_tfbs_instances'] * 100) if results['total_tfbs_instances'] > 0 else 0
    print(f"Failed to map {num_failures} TFBS entries ({percent_fail:.2f}%)")
    print(f"Average TFBS per sequence: {results['average_tfbs_per_sequence']:.2f}")
    print(f"Positional entropy (forward strand): {results['positional_entropy_forward']:.2f}")
    print(f"Positional entropy (reverse strand): {results['positional_entropy_reverse']:.2f}")
    print(f"Number of unique TF combinations: {len(results['combo_counts'])}")
    print(f"Combination entropy: {results['combo_entropy']:.2f}")
