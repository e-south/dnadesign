"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import torch


def to_device(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device) if t.device.type != torch.device(device).type or t.device != torch.device(device) else t


def to_format(obj, fmt: str):
    """Convert tensor/ndarray to requested format: tensor|numpy|list|float.
    - If obj is scalar-like tensor/ndarray, convert appropriately.
    - If fmt == float, expect a Python float already or 0-d tensor/array.
    """
    if fmt == "tensor":
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        # list → tensor
        return torch.tensor(obj)

    if fmt == "numpy":
        if isinstance(obj, np.ndarray):
            return obj
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy()
        return np.asarray(obj)

    if fmt == "list":
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return list(obj)

    if fmt == "float":
        if isinstance(obj, (float, int)):
            return float(obj)
        if torch.is_tensor(obj):
            return float(obj.detach().cpu().item())
        if isinstance(obj, np.ndarray):
            return float(obj.item())
        # last resort
        return float(obj)

    raise ValueError(f"Unknown format: {fmt}")


def pool_tensor(t: torch.Tensor, method: str = "mean", dim: int = 1) -> torch.Tensor:
    if method == "mean":
        return t.mean(dim=dim)
    if method == "sum":
        return t.sum(dim=dim)
    if method == "max":
        return t.max(dim=dim).values
    raise ValueError(f"Unsupported pool method: {method}")
