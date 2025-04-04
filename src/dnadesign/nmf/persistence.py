"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/persistence.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)

def convert_metrics(m):
    """Recursively convert numpy types to native Python types."""
    if isinstance(m, np.generic):
        return m.item()
    elif isinstance(m, dict):
        return {k: convert_metrics(v) for k, v in m.items()}
    elif isinstance(m, list):
        return [convert_metrics(x) for x in m]
    else:
        return m

def save_nmf_results(W, H, metric_data: dict, output_dir: str) -> None:
    """
    Save NMF matrices (W and H) and metrics to CSV/YAML files in output_dir.
    Converts numpy scalars to native Python types for YAML serialization.
    """
    try:
        W_df = pd.DataFrame(W)
        H_df = pd.DataFrame(H)
        W_csv_path = os.path.join(output_dir, "W.csv")
        H_csv_path = os.path.join(output_dir, "H.csv")
        W_df.to_csv(W_csv_path)
        H_df.to_csv(H_csv_path)
        metrics_yaml_path = os.path.join(output_dir, "metrics.yaml")
        with open(metrics_yaml_path, "w") as f:
            yaml.safe_dump(convert_metrics(metric_data), f, default_flow_style=False)
        logger.info(f"Saved NMF results in {output_dir}")
    except Exception as e:
        logger.error(f"Error saving NMF results in {output_dir}: {str(e)}")

def annotate_sequences_with_nmf(sequences: list, W: np.ndarray, best_k: int, pt_path: str, assert_normalized: bool = True) -> None:
    """
    Annotate each sequence with NMF metadata and save the updated sequences back to the .pt file.

    This function assumes that the coefficient matrix W (shape: n_sequences x k_actual)
    corresponds exactly to the factorization rank best_k. It performs the following steps:

      1. (Optional) Assert that each row of W is normalized (sums to 1) and nonzero.
      2. For each sequence, compute:
         - 'program_composition': the normalized row from W (length best_k)
         - 'dominant_program': the index of the maximum value in that row
         - 'program_entropy': the normalized entropy of that row, computed as:
               H(s) = -∑ p_i log2(p_i + ε) and then normalized by log2(best_k)
      3. Store these values under the key "meta_nmf" in the sequence.
      4. Save the updated sequence list back to the .pt file.

    Parameters:
      sequences: List of sequence dictionaries.
      W: Coefficient matrix from NMF (expected shape: n_sequences x best_k).
      best_k: The factorization rank used (must match the number of columns in W).
      pt_path: Path to the .pt file for saving annotated sequences.
      assert_normalized: If True, check that each row of W is normalized.
    """
    if assert_normalized:
        if not np.allclose(np.sum(W, axis=1), 1, atol=1e-3):
            raise ValueError("The coefficient matrix W is not row-normalized.")
        if np.any(np.sum(W, axis=1) < 1e-6):
            raise ValueError("One or more rows in W sum to zero.")

    n_sequences, k_actual = W.shape
    if k_actual != best_k:
        raise ValueError(f"Dimension mismatch: W has {k_actual} columns but best_k is set to {best_k}. "
                         "Ensure that you load the W matrix from the correct k subdirectory.")
    
    for i in range(n_sequences):
        p = W[i, :]
        dominant_program = int(np.argmax(p))
        entropy = -np.sum(p * np.log2(p + 1e-8))
        normalized_entropy = entropy / np.log2(best_k)
        
        sequences[i]["meta_nmf"] = {
            "k_used": best_k,
            "program_composition": p.tolist(),
            "dominant_program": dominant_program,
            "program_entropy": normalized_entropy
        }
    try:
        import torch
        torch.save(sequences, pt_path)
        logger.info(f"Annotated sequences saved to {pt_path}")
    except Exception as e:
        logger.error(f"Error saving annotated sequences to {pt_path}: {str(e)}")

def export_feature_names(feature_names: list, output_path: str) -> None:
    """
    Export feature names (one per line) to a text file.
    """
    try:
        with open(output_path, "w") as f:
            for feat in feature_names:
                f.write(f"{feat}\n")
        logger.info(f"Exported feature names to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting feature names: {str(e)}")

def export_row_ids(row_ids: list, output_path: str) -> None:
    """
    Export row IDs (one per line) to a text file.
    """
    try:
        with open(output_path, "w") as f:
            for rid in row_ids:
                f.write(f"{rid}\n")
        logger.info(f"Exported row IDs to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting row IDs: {str(e)}")