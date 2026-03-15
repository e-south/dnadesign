"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/extract_params.py

Resolves extract output parameters to explicit runtime contracts before adapter
invocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from ..bootstrap import initialize_registry
from ..errors import CapabilityError
from ..registry import get_default_embedding_layer


def resolve_extract_params(
    *,
    model_id: str,
    method_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = dict(params)
    if method_name != "embedding":
        return resolved

    if "layer" in resolved:
        layer = resolved.get("layer")
        if not isinstance(layer, str) or not layer.strip():
            raise CapabilityError(
                "embedding output requires params.layer with a non-empty layer name."
            )
        resolved["layer"] = layer.strip()
        return resolved

    initialize_registry()
    default_layer = get_default_embedding_layer(model_id)
    if default_layer:
        resolved["layer"] = default_layer
        return resolved

    raise CapabilityError(
        f"embedding output requires params.layer for model_id='{model_id}'; "
        "no default embedding layer is registered."
    )
