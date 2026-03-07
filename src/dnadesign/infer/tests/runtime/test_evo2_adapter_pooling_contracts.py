"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_evo2_adapter_pooling_contracts.py

Contract tests for Evo2 adapter pooling semantics and fail-fast behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import torch
import pytest

from dnadesign.infer.src.adapters.evo2 import Evo2Adapter
from dnadesign.infer.src.errors import CapabilityError


class _Tokenizer:
    def tokenize(self, sequence: str) -> list[int]:
        return [1] * len(sequence)


class _Model:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def __call__(
        self,
        x: torch.Tensor,
        *,
        return_embeddings: bool = False,
        layer_names: list[str] | None = None,
    ) -> tuple[Any, Any]:
        batch, length = x.shape
        logits = (
            torch.arange(batch * length * 4, dtype=torch.float32)
            .reshape(batch, length, 4)
            .to(x.device)
        )
        if not return_embeddings:
            return (logits,), None
        assert layer_names is not None and len(layer_names) == 1
        embeddings = (
            torch.arange(batch * length * 3, dtype=torch.float32)
            .reshape(batch, length, 3)
            .to(x.device)
        )
        return logits, {layer_names[0]: embeddings}

    def score_sequences(self, seqs: list[str], *, reduce_method: str) -> list[float]:
        assert reduce_method in {"sum", "mean"}
        return [float(len(s)) for s in seqs]


def _adapter() -> Evo2Adapter:
    adapter = Evo2Adapter.__new__(Evo2Adapter)
    adapter.model_id = "evo2_7b"
    adapter.device = "cpu"
    adapter.precision = "fp32"
    adapter.model = _Model()
    adapter._torch_module = None
    return adapter


def test_logits_pooling_sequence_dimension_is_consistent_for_variable_lengths() -> None:
    adapter = _adapter()
    out = adapter.logits(
        ["ACGT", "AC"],
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 2
    assert all(torch.is_tensor(item) for item in out)
    assert out[0].shape == torch.Size([4])
    assert out[1].shape == torch.Size([4])


def test_embedding_pooling_sequence_dimension_is_consistent_for_variable_lengths() -> None:
    adapter = _adapter()
    out = adapter.embedding(
        ["ACGT", "AC"],
        layer="blocks.1.mlp.l3",
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 2
    assert all(torch.is_tensor(item) for item in out)
    assert out[0].shape == torch.Size([3])
    assert out[1].shape == torch.Size([3])


def test_logits_rejects_pool_dim_zero_that_consumes_batch_axis() -> None:
    adapter = _adapter()

    with pytest.raises(CapabilityError, match="pool.dim must be >= 1"):
        adapter.logits(
            ["ACGT", "TGCA"],
            pool={"method": "mean", "dim": 0},
            fmt="tensor",
        )
