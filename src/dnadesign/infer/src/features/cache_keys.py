"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/cache_keys.py

Deterministic cache keys for sequence-view-aware Infer execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_forward_pass_key(
    *,
    provider: str,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    requested_layers: list[str],
    normalized_input_sequence: str,
    provider_params: dict[str, Any] | None,
    orientation: str,
) -> str:
    return stable_sha256(
        {
            "provider": provider,
            "model_name": model_name,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "requested_layers": list(requested_layers),
            "normalized_input_sequence": normalized_input_sequence,
            "provider_params": dict(provider_params or {}),
            "orientation": orientation,
        }
    )


def compute_feature_vector_key(
    *,
    forward_pass_key: str,
    representation_kind: str,
    layer_name: str | None,
    pooling_operation: str,
    pooling_start_0: int | None,
    pooling_end_0: int | None,
    dtype_or_storage_format: str | None,
) -> str:
    return stable_sha256(
        {
            "forward_pass_key": forward_pass_key,
            "representation_kind": representation_kind,
            "layer_name": layer_name,
            "pooling_operation": pooling_operation,
            "pooling_start_0": pooling_start_0,
            "pooling_end_0": pooling_end_0,
            "dtype_or_storage_format": dtype_or_storage_format,
        }
    )
