"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/adapters/__init__.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..registry import register_fn, register_model

# from .esm2 import ESM2Adapter
from .evo2 import Evo2Adapter

# Register models
register_model("evo2_7b", Evo2Adapter)
register_model("evo2_1b_base", Evo2Adapter)  # convenience
# ESM2 is stubbed but keep an example for future
# register_model("esm2_t33_650M_UR50D", ESM2Adapter)

# Register functions (namespaced)
register_fn("evo2.logits", "logits")
register_fn("evo2.embedding", "embedding")
register_fn("evo2.log_likelihood", "log_likelihood")
register_fn("evo2.generate", "generate")

# ESM2 (to be implemented later)
# register_fn("esm2.embedding", "embedding")
# register_fn("esm2.log_likelihood", "log_likelihood")
