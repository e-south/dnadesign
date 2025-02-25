"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/aggregations.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
from logger import get_logger

logger = get_logger(__name__)

def apply_pooling(tensor: torch.Tensor, method: str = "mean", dim: int = 1) -> torch.Tensor:
    """
    Apply a pooling operation on the given tensor along dimension `dim`.
    Currently supports "mean" pooling.
    """
    if method == "mean":
        try:
            pooled = torch.mean(tensor, dim=dim)
            return pooled
        except Exception as e:
            logger.error(f"Error during mean pooling: {str(e)}")
            raise e
    else:
        raise ValueError(f"Unsupported pooling method: {method}")

def augment_outputs(results: dict, output_types: list, save_pooled_only: bool = False) -> dict:
    """
    Augment the raw results from run_model by adding additional keys:
      - For each output tensor, add a key with '_shape' appended that holds the shape (as a list).
      - If a pooling configuration is provided in the output type config, apply pooling (e.g., mean)
        along the specified dimension and add keys with the pooling method suffix and their shape.
    
    If save_pooled_only is True, only the pooled outputs (and their shape) will be added.
    """
    augmented = {}

    for config in output_types:
        typ = config.get("type")
        pooling_conf = config.get("pooling")
        
        if typ == "logits":
            key = "evo2_logits"
            if key in results:
                tensor = results[key]
                if pooling_conf is not None:
                    method = pooling_conf.get("method", "mean")
                    dim = pooling_conf.get("dim", 1)
                    pooled = apply_pooling(tensor, method=method, dim=dim)
                    augmented[f"{key}_{method}_pooled"] = pooled
                    augmented[f"{key}_{method}_pooled_shape"] = list(pooled.shape)
                    if not save_pooled_only:
                        augmented[key] = tensor
                        augmented[f"{key}_shape"] = list(tensor.shape)
                else:
                    # If no pooling is configured, keep raw output only if not in save_pooled_only mode.
                    if not save_pooled_only:
                        augmented[key] = tensor
                        augmented[f"{key}_shape"] = list(tensor.shape)
                        
        elif typ == "embeddings" and "layers" in config:
            for layer in config["layers"]:
                key = f"evo2_embeddings_{layer}"
                if key in results:
                    tensor = results[key]
                    if pooling_conf is not None:
                        method = pooling_conf.get("method", "mean")
                        dim = pooling_conf.get("dim", 1)
                        pooled = apply_pooling(tensor, method=method, dim=dim)
                        augmented[f"{key}_{method}_pooled"] = pooled
                        augmented[f"{key}_{method}_pooled_shape"] = list(pooled.shape)
                        if not save_pooled_only:
                            augmented[key] = tensor
                            augmented[f"{key}_shape"] = list(tensor.shape)
                    else:
                        if not save_pooled_only:
                            augmented[key] = tensor
                            augmented[f"{key}_shape"] = list(tensor.shape)
    return augmented
