"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/encoded.py

Decode Biohub ESMC encoded SAE tensor payloads into sparse rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any


@dataclass(frozen=True)
class SparseSaeTensor:
    """Sparse nonzero SAE activations for one SAE model."""

    sae_model: str
    residue_indices: list[int]
    feature_indices: list[int]
    values: list[float]
    sequence_residue_count: int
    feature_count: int
    token_count: int
    encoded_sae_bytes: int


def decode_sae_outputs(
    payload: Any,
    *,
    sequence_length: int,
    sae_model: str,
) -> SparseSaeTensor:
    """Decode Biohub `sae_outputs` into sparse residue-feature rows."""

    decoded = _decode_payload(payload)
    if not isinstance(decoded, Mapping) or sae_model not in decoded:
        raise ValueError(f"Biohub logits response did not include sae_outputs for {sae_model!r}")
    return _tensor_to_sparse(
        decoded[sae_model],
        sae_model=sae_model,
        sequence_length=sequence_length,
        encoded_sae_bytes=len(payload) if isinstance(payload, str) else 0,
    )


def _decode_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return payload
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("Biohub logits response must include encoded sae_outputs")
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - torch is a project dependency.
        raise RuntimeError("torch is required to decode Biohub encoded SAE outputs") from error
    compressed = base64.b64decode(payload)
    raw = _zstd_decompress(compressed)
    return torch.load(BytesIO(raw), map_location="cpu", weights_only=False)


def _zstd_decompress(payload: bytes) -> bytes:
    try:
        import zstd  # type: ignore[import-not-found]

        return zstd.ZSTD_uncompress(payload)
    except ModuleNotFoundError:
        pass
    try:
        import zstandard as zstd_lib  # type: ignore[import-not-found]

        return zstd_lib.ZstdDecompressor().decompress(payload)
    except ModuleNotFoundError as error:
        raise RuntimeError("zstd or zstandard is required to decode Biohub encoded SAE outputs") from error


def _tensor_to_sparse(
    tensor: Any,
    *,
    sae_model: str,
    sequence_length: int,
    encoded_sae_bytes: int,
) -> SparseSaeTensor:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - torch is a project dependency.
        raise RuntimeError("torch is required to normalize Biohub SAE outputs") from error
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Biohub SAE output must decode to a torch Tensor")
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError("Biohub SAE output tensor must be two-dimensional after removing batch dim")
    token_count = int(tensor.shape[0])
    feature_count = int(tensor.shape[1])
    if tensor.is_sparse:
        offset = _special_token_offset(token_count, sequence_length=sequence_length)
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        residue_indices = []
        feature_indices = []
        activation_values = []
        for token_index, feature_index, value in zip(
            indices[0].tolist(),
            indices[1].tolist(),
            values.tolist(),
            strict=True,
        ):
            residue_index = int(token_index) - offset
            if residue_index < 0 or residue_index >= sequence_length:
                continue
            residue_indices.append(residue_index)
            feature_indices.append(int(feature_index))
            activation_values.append(float(value))
    else:
        sequence_tensor = _trim_special_tokens(tensor, sequence_length=sequence_length)
        nonzero = sequence_tensor.nonzero(as_tuple=False)
        residue_indices = [int(value) for value in nonzero[:, 0].tolist()]
        feature_indices = [int(value) for value in nonzero[:, 1].tolist()]
        activation_values = [float(value) for value in sequence_tensor[nonzero[:, 0], nonzero[:, 1]].tolist()]
    return SparseSaeTensor(
        sae_model=sae_model,
        residue_indices=residue_indices,
        feature_indices=feature_indices,
        values=activation_values,
        sequence_residue_count=sequence_length,
        feature_count=feature_count,
        token_count=token_count,
        encoded_sae_bytes=encoded_sae_bytes,
    )


def _trim_special_tokens(tensor: Any, *, sequence_length: int) -> Any:
    if int(tensor.shape[0]) == sequence_length:
        return tensor
    if int(tensor.shape[0]) == sequence_length + 2:
        return tensor[1:-1]
    raise ValueError(
        f"Biohub SAE output length {int(tensor.shape[0])} does not match sequence length {sequence_length}"
    )


def _special_token_offset(token_count: int, *, sequence_length: int) -> int:
    if token_count == sequence_length:
        return 0
    if token_count == sequence_length + 2:
        return 1
    raise ValueError(f"Biohub SAE output length {token_count} does not match sequence length {sequence_length}")
