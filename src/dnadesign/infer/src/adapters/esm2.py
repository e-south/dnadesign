"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/adapters/esm2.py

Adapter logic for esm2 infer adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..errors import ModelLoadError


class ESM2Adapter:
    """Stub adapter for ESM2. Implements embedding + PLL log_likelihood in future."""

    alphabet_default: str = "protein"

    supports = {
        "logits": False,
        "embedding": False,  # set True when implemented
        "log_likelihood": False,  # set True when implemented
        "generate": False,
    }
    supported_parallelism_strategies = ("single_device",)

    def __init__(self, model_id: str, device: str, precision: str) -> None:
        raise ModelLoadError("ESM2 adapter is stubbed in v1")
