"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/summary.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import pandas as pd

def generate_csv_summaries(results, output_dir):
    csv_dir = os.path.join(output_dir, "csvs")
    
    # Overall summary.
    summary_data = {
        "num_sequences": [results["num_sequences"]],
        "sequence_length": [results["sequence_length"]],
        "num_unique_tfs": [len(results.get("tf_frequency", {}))],
        "total_tfbs_instances": [results["total_tfbs_instances"]]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(csv_dir, "summary_metrics.csv"), index=False)
    
    # TF coverage summary.
    df_tf = pd.DataFrame(list(results.get("tf_frequency", {}).items()), columns=["tf", "frequency"])
    df_tf.to_csv(os.path.join(csv_dir, "tf_coverage_summary.csv"), index=False)
    
    # Diversity Summary (including core and composite metrics).
    diversity_data = [
        {
            "metric_name": "tf_richness",
            "value": results["core_metrics"].get("tf_richness"),
            "notes": "Number of unique TFs in the library"
        },
        {
            "metric_name": "1_minus_gini",
            "value": results["core_metrics"].get("1_minus_gini"),
            "notes": "Inverted Gini coefficient for TF frequency distribution"
        },
        {
            "metric_name": "mean_jaccard",
            "value": results["core_metrics"].get("mean_jaccard"),
            "notes": "Mean pairwise Jaccard dissimilarity of TF rosters"
        },
        {
            "metric_name": "median_tf_entropy",
            "value": results["core_metrics"].get("median_tf_entropy"),
            "notes": "Aggregated (median) positional entropy across TFs"
        }
    ]
    if "motif_string_levenshtein_mean" in results["core_metrics"]:
        diversity_data.extend([
            {
                "metric_name": "motif_string_levenshtein_mean",
                "value": results["core_metrics"].get("motif_string_levenshtein_mean"),
                "notes": "Mean normalized Levenshtein distance of motif strings"
            },
            {
                "metric_name": "motif_string_levenshtein_median",
                "value": results["core_metrics"].get("motif_string_levenshtein_median"),
                "notes": "Median normalized Levenshtein distance of motif strings"
            },
            {
                "metric_name": "motif_string_levenshtein_std",
                "value": results["core_metrics"].get("motif_string_levenshtein_std"),
                "notes": "Standard deviation of normalized Levenshtein distances"
            }
        ])
    
    diversity_data.append({
        "metric_name": "billboard_weighted_sum",
        "value": results["composite_metrics"].get("billboard_weighted_sum"),
        "notes": "Weighted sum composite metric"
    })
    
    df_diversity = pd.DataFrame(diversity_data)
    df_diversity.to_csv(os.path.join(csv_dir, "diversity_summary.csv"), index=False)

def generate_entropy_summary_csv(results, output_dir):
    csv_dir = os.path.join(output_dir, "csvs")
    diversity_data = [
        {
            "metric_name": "tf_richness",
            "value": results["core_metrics"].get("tf_richness"),
            "notes": "Number of unique TFs in the library"
        },
        {
            "metric_name": "1_minus_gini",
            "value": results["core_metrics"].get("1_minus_gini"),
            "notes": "Inverted Gini coefficient for TF frequency distribution"
        },
        {
            "metric_name": "mean_jaccard",
            "value": results["core_metrics"].get("mean_jaccard"),
            "notes": "Mean pairwise Jaccard dissimilarity of TF rosters"
        },
        {
            "metric_name": "median_tf_entropy",
            "value": results["core_metrics"].get("median_tf_entropy"),
            "notes": "Aggregated (median) positional entropy across TFs"
        }
    ]
    if "motif_string_levenshtein_mean" in results["core_metrics"]:
        diversity_data.extend([
            {
                "metric_name": "motif_string_levenshtein_mean",
                "value": results["core_metrics"].get("motif_string_levenshtein_mean"),
                "notes": "Mean normalized Levenshtein distance of motif strings"
            },
            {
                "metric_name": "motif_string_levenshtein_median",
                "value": results["core_metrics"].get("motif_string_levenshtein_median"),
                "notes": "Median normalized Levenshtein distance of motif strings"
            },
            {
                "metric_name": "motif_string_levenshtein_std",
                "value": results["core_metrics"].get("motif_string_levenshtein_std"),
                "notes": "Standard deviation of normalized Levenshtein distances"
            }
        ])
    diversity_data.append({
        "metric_name": "billboard_weighted_sum",
        "value": results["composite_metrics"].get("billboard_weighted_sum"),
        "notes": "Weighted sum composite metric"
    })
    df_diversity = pd.DataFrame(diversity_data)
    df_diversity.to_csv(os.path.join(csv_dir, "diversity_summary.csv"), index=False)

def print_summary(results):
    print(f"Loaded {results['num_sequences']} sequences (length: {results['sequence_length']} nt)")
    print(f"Total TFBS instances: {results['total_tfbs_instances']}")
    print("Core Diversity Metrics:")
    for k, v in results["core_metrics"].items():
        print(f"  {k}: {v}")
    print("Composite Metrics:")
    for k, v in results["composite_metrics"].items():
        print(f"  {k}: {v}")