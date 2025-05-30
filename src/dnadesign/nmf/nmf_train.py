"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/nmf_train.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF as SklearnNMF

from .persistence import save_nmf_results

logger = logging.getLogger(__name__)


def train_nmf(X: np.ndarray, config: dict) -> dict:
    """
    Train NMF on feature matrix X for each k in config["nmf"]["k_range"].
    If k_range is provided as exactly two numbers, treat it as an inclusive range.
    For each k, if cached results exist (i.e. if the batch results directory for that k already
    contains valid W.csv, H.csv, and metrics.yaml), then load them instead of retraining.

    Incorporates L2 regularization if enabled in config["nmf"]["regularization"].

    Post-process:
      - Row-normalize W if normalize_coefficients is True.
      - Clip H values using clip_h_max.

    Returns a dictionary with:
      - "metrics": diagnostic metrics per k
      - "best_k": chosen factorization rank based on minimum mean loss
      - "best_W", "best_H": matrices for the best k (loaded from saved files)
    """
    nmf_config = config["nmf"]
    k_range_raw = nmf_config.get("k_range", [8, 12, 16])
    if isinstance(k_range_raw, list) and len(k_range_raw) == 2:
        k_range = list(range(k_range_raw[0], k_range_raw[1] + 1))
    else:
        k_range = k_range_raw
    n_init = nmf_config.get("n_init", 10)
    max_iter = nmf_config.get("max_iter", 1000)
    init_method = nmf_config.get("init", "nndsvdar")
    solver = nmf_config.get("solver", "cd")
    loss_metric = nmf_config.get("loss", "frobenius")
    normalize_coefficients = nmf_config.get("normalize_coefficients", True)
    clip_h_max = nmf_config.get("clip_h_max", 3.0)

    # Regularization settings
    reg_config = nmf_config.get("regularization", {"enable": False})
    reg_enable = reg_config.get("enable", False)
    reg_type = reg_config.get("type", "none")
    alpha = reg_config.get("alpha", 0.01)
    if reg_enable and reg_type == "l2":
        l1_ratio = 0.0
        alpha_W = alpha
        alpha_H = alpha
    else:
        l1_ratio = 0.0
        alpha_W = 0.0
        alpha_H = 0.0

    # Standardize the results directory path using Path
    batch_results_dir = str(Path(__file__).parent / "batch_results" / nmf_config.get("batch_name", "default"))
    results = {}
    metrics = {}

    # Loop over each k in the k_range
    for k in k_range:
        k_dir = os.path.join(batch_results_dir, f"k_{k}")
        # Check if caching exists (W.csv, H.csv, metrics.yaml)
        cache_exists = (
            os.path.exists(os.path.join(k_dir, "W.csv"))
            and os.path.exists(os.path.join(k_dir, "H.csv"))
            and os.path.exists(os.path.join(k_dir, "metrics.yaml"))
        )
        if cache_exists:
            logger.info(f"Cache found for k={k}. Loading cached results.")
            try:
                W_df = pd.read_csv(os.path.join(k_dir, "W.csv"), index_col=0)
                H_df = pd.read_csv(os.path.join(k_dir, "H.csv"), index_col=0)
                with open(os.path.join(k_dir, "metrics.yaml"), "r") as f:
                    import yaml

                    cached_metrics = yaml.safe_load(f)
                metrics[k] = cached_metrics
                continue  # Skip training for this k
            except Exception as e:
                logger.error(f"Error loading cached data for k={k}: {str(e)}. Proceeding with retraining.")

        # Otherwise, perform training for this k.
        best_loss = np.inf
        best_W = None
        best_H = None
        replicate_losses = []
        for i in range(n_init):
            try:
                model = SklearnNMF(
                    n_components=k,
                    init=init_method,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=i,
                    beta_loss=loss_metric,
                    tol=1e-4,
                    alpha_W=alpha_W,
                    alpha_H=alpha_H,
                    l1_ratio=l1_ratio,
                )
                W = model.fit_transform(X)
                H = model.components_
                loss = model.reconstruction_err_
                replicate_losses.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_W = W.copy()
                    best_H = H.copy()
            except Exception as e:
                logger.error(f"Error during NMF training for k={k}, replicate {i}: {str(e)}")
        metrics[k] = {
            "mean_loss": np.mean(replicate_losses) if replicate_losses else None,
            "min_loss": best_loss,
            "replicate_losses": [float(x) for x in replicate_losses],
        }
        if normalize_coefficients and best_W is not None:
            row_sums = best_W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            best_W = best_W / row_sums
        if best_H is not None:
            best_H = np.clip(best_H, None, clip_h_max)

        os.makedirs(k_dir, exist_ok=True)
        save_nmf_results(best_W, best_H, metrics[k], k_dir)
        logger.info(f"Finished training for k={k} with best loss {best_loss:.4f}")
    results["metrics"] = metrics
    best_k = min(metrics, key=lambda k: metrics[k]["mean_loss"])
    results["best_k"] = best_k
    best_k_dir = os.path.join(batch_results_dir, f"k_{best_k}")
    try:
        W_df = pd.read_csv(os.path.join(best_k_dir, "W.csv"), index_col=0)
        H_df = pd.read_csv(os.path.join(best_k_dir, "H.csv"), index_col=0)
        results["best_W"] = W_df.values
        results["best_H"] = H_df.values
    except Exception as e:
        logger.error(f"Failed to load best NMF results from {best_k_dir}: {str(e)}")
    return results
