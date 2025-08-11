"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/adapters/esm2.py

Module Author(s): Eric J. South
Dunlop Lab
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

    def __init__(self, model_id: str, device: str, precision: str) -> None:
        raise ModelLoadError("ESM2 adapter is stubbed in v1")
