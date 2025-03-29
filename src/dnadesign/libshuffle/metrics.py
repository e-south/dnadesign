"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/metrics.py

Provides functions to compute the two diversity metrics:
- Billboard diversity: computed as the sum (or selected key) of entropy values.
- Evo2 diversity: computed as the mean pairwise Euclidean (L2) distance between latent vectors.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
import tempfile
import os
import csv
import shutil

from dnadesign.billboard.core import process_sequences
from dnadesign.billboard.summary import generate_entropy_summary_csv

def compute_billboard_metric(subsample, config):
    """
    Computes the billboard diversity metric for a subsample by dynamically calling
    the billboard pipeline in DRY_RUN mode. This function writes the subsample to a temporary
    .pt file, prepares a temporary billboard configuration (forcing dry_run mode),
    and calls the billboard processing routines to generate an entropy_summary CSV.
    The desired metric is then extracted from the CSV.

    The config parameter is the full global config dictionary (which should contain both
    a 'libshuffle' and a 'billboard' section). The 'billboard_metric' section of libshuffle's config
    controls which metric to use:
      - If type is "sum_raw", then the final value is computed as the sum of all default entropy keys.
      - Otherwise, the function will return the value for the specified key.
    """
    # Use a temporary directory to hold the temporary .pt file and billboard outputs.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the current subsample to a temporary .pt file.
        temp_pt_path = os.path.join(tmpdir, "subsample.pt")
        torch.save(subsample, temp_pt_path)

        # Prepare a temporary billboard configuration.
        # Start with a copy of the global 'billboard' config from our YAML.
        billboard_config = config.get("billboard", {}).copy()
        # Force dry_run mode so that billboard only computes summaries.
        billboard_config["dry_run"] = True
        # Set the pt_files list to our temporary subsample file.
        billboard_config["pt_files"] = [temp_pt_path]
        # Optionally, use an ephemeral prefix for the output directory.
        billboard_config["output_dir_prefix"] = "temp_billboard_" + next(tempfile._get_candidate_names())
        # Determine an output directory within our temporary folder.
        temp_output_dir = os.path.join(tmpdir, "billboard_output")
        os.makedirs(os.path.join(temp_output_dir, "csvs"), exist_ok=True)
        os.makedirs(os.path.join(temp_output_dir, "plots"), exist_ok=True)

        # Call billboard processing functions.
        # process_sequences() expects a list of pt file paths.
        results = process_sequences([temp_pt_path], billboard_config)
        # In dry_run mode, only the entropy summary CSV is generated.
        generate_entropy_summary_csv(results, temp_output_dir)

        # Read the generated entropy_summary.csv.
        entropy_csv_path = os.path.join(temp_output_dir, "csvs", "entropy_summary.csv")
        metrics = {}
        with open(entropy_csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metric_name = row["metric_name"]
                try:
                    value = float(row["value"])
                except (ValueError, TypeError):
                    value = 0.0
                try:
                    normalized = float(row["normalized_value"])
                except (ValueError, TypeError):
                    normalized = None
                metrics[metric_name] = {"value": value, "normalized": normalized, "notes": row["notes"]}

        # Determine which metric to use based on the libshuffle configuration.
        metric_type = config.get("billboard_metric", {}).get("type", "sum_raw")
        if metric_type == "sum_raw":
            # Sum the values from a set of keys (adjust these keys as necessary).
            keys = [
                "global_positional_entropy",
                "per_tf_entropy_mean",
                "per_tf_entropy_weighted",
                "per_tf_entropy_topk",
                "tf_combination_entropy",
            ]
            final_value = sum(metrics.get(k, {"value": 0.0})["value"] for k in keys)
        else:
            final_value = metrics.get(metric_type, {"value": 0.0})["value"]

        return final_value


def compute_evo2_metric(subsample, config):
    """
    Compute the mean pairwise Euclidean (L2) distance among the evo2 vectors in the subsample.
    Each subsample entry is expected to have a field "evo2_logits_mean_pooled" (a tensor or nested list).
    """
    vectors = []
    for entry in subsample:
        vector = entry.get("evo2_logits_mean_pooled")
        if vector is None:
            raise ValueError("Entry missing 'evo2_logits_mean_pooled' field required for evo2 metric.")
        # Always convert the vector to a float32 tensor.
        if isinstance(vector, torch.Tensor):
            vector = vector.to(torch.float32).flatten()
        elif isinstance(vector, list):
            vector = torch.tensor(vector, dtype=torch.float32).flatten()
        else:
            vector = torch.tensor(vector, dtype=torch.float32).flatten()
        vectors.append(vector)
    if len(vectors) < 2:
        return 0.0
    # Stack vectors into a matrix and compute pairwise L2 distances.
    mat = torch.stack(vectors)
    distances = torch.cdist(mat, mat, p=2)
    n = distances.size(0)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise = distances[mask]
    mean_distance = pairwise.mean().item()
    metric_type = config.get("evo2_metric", {}).get("type", "l2")
    if metric_type == "log1p_l2":
        import math
        return math.log1p(mean_distance)
    return mean_distance

