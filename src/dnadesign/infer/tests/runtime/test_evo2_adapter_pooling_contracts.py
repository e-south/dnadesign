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
        self.reduce_calls: list[str] = []

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
        self.reduce_calls.append(reduce_method)
        mult = 10.0 if reduce_method == "sum" else 1.0
        return [float(len(s)) * mult for s in seqs]


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
    assert torch.allclose(out[0], torch.tensor([6.0, 7.0, 8.0, 9.0]))
    assert torch.allclose(out[1], torch.tensor([2.0, 3.0, 4.0, 5.0]))


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
    assert torch.allclose(out[0], torch.tensor([4.5, 5.5, 6.5]))
    assert torch.allclose(out[1], torch.tensor([1.5, 2.5, 3.5]))


def test_logits_rejects_pool_dim_zero_that_consumes_batch_axis() -> None:
    adapter = _adapter()

    with pytest.raises(CapabilityError, match="pool.dim must be >= 1"):
        adapter.logits(
            ["ACGT", "TGCA"],
            pool={"method": "mean", "dim": 0},
            fmt="tensor",
        )


def test_log_likelihood_reduction_sum_and_mean_map_directly_to_evo2_api() -> None:
    adapter = _adapter()

    out_sum = adapter.log_likelihood(["AC", "ACGT"], method="native", reduction="sum")
    out_mean = adapter.log_likelihood(["AC", "ACGT"], method="native", reduction="mean")

    assert out_sum == [20.0, 40.0]
    assert out_mean == [2.0, 4.0]
    assert adapter.model.reduce_calls == ["sum", "mean"]


def test_log_likelihood_rejects_unknown_reduction() -> None:
    adapter = _adapter()
    with pytest.raises(CapabilityError, match="reduction='sum' or 'mean'"):
        adapter.log_likelihood(["ACGT"], method="native", reduction="median")
