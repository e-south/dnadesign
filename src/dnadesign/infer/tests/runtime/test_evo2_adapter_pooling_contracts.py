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

import pytest
import torch

from dnadesign.infer.src.adapters.evo2 import Evo2Adapter
from dnadesign.infer.src.errors import CapabilityError


class _Tokenizer:
    def tokenize(self, sequence: str) -> list[int]:
        return [1] * len(sequence)


class _Model:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.reduce_calls: list[tuple[str, int]] = []
        self.forward_calls = 0
        self.embedding_layers: list[str] = []

    def __call__(
        self,
        x: torch.Tensor,
        *,
        return_embeddings: bool = False,
        layer_names: list[str] | None = None,
    ) -> tuple[Any, Any]:
        self.forward_calls += 1
        batch, length = x.shape
        logits = torch.arange(batch * length * 4, dtype=torch.float32).reshape(batch, length, 4).to(x.device)
        if not return_embeddings:
            return (logits,), None
        assert layer_names is not None and len(layer_names) == 1
        self.embedding_layers.append(layer_names[0])
        embeddings = torch.arange(batch * length * 3, dtype=torch.float32).reshape(batch, length, 3).to(x.device)
        return logits, {layer_names[0]: embeddings}

    def score_sequences(self, seqs: list[str], *, batch_size: int, reduce_method: str) -> list[float]:
        assert reduce_method in {"sum", "mean"}
        self.reduce_calls.append((reduce_method, batch_size))
        mult = 10.0 if reduce_method == "sum" else 1.0
        return [float(len(s)) * mult for s in seqs]

    def generate(self, **_kwargs):
        return (["ACGTAA"], [0.5])


class _TorchModule:
    def __init__(self, blocks: tuple[int, ...] = (0, 1, 20, 26, 31)) -> None:
        self._blocks = blocks

    def named_modules(self):
        yield ("", object())
        for block in self._blocks:
            yield (f"blocks.{block}.mlp.l3", object())


def _adapter(*, model_id: str = "evo2_7b", blocks: tuple[int, ...] = (0, 1, 20, 26, 31)) -> Evo2Adapter:
    adapter = Evo2Adapter.__new__(Evo2Adapter)
    adapter.model_id = model_id
    adapter.device = "cpu"
    adapter.precision = "fp32"
    adapter.model = _Model()
    adapter._torch_module = _TorchModule(blocks=blocks)
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


def test_logits_groups_variable_lengths_to_reduce_forward_calls() -> None:
    adapter = _adapter()
    out = adapter.logits(
        ["ACGT", "AC", "TGCA"],
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 3
    assert adapter.model.forward_calls == 2


def test_embedding_groups_variable_lengths_to_reduce_forward_calls() -> None:
    adapter = _adapter()
    out = adapter.embedding(
        ["ACGT", "AC", "TGCA"],
        layer="blocks.1.mlp.l3",
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 3
    assert adapter.model.forward_calls == 2


def test_embedding_alias_mid_uses_registered_default_layer() -> None:
    adapter = _adapter()
    out = adapter.embedding(
        ["ACGT"],
        layer="mid",
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 1
    assert adapter.model.embedding_layers == ["blocks.26.mlp.l3"]


def test_embedding_alias_mid_uses_model_specific_default_layer_for_evo2_20b() -> None:
    adapter = _adapter(model_id="evo2_20b", blocks=(0, 1, 20, 23))
    out = adapter.embedding(
        ["ACGT"],
        layer="mid",
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 1
    assert adapter.model.embedding_layers == ["blocks.23.mlp.l3"]


def test_embedding_alias_final_resolves_to_last_block() -> None:
    adapter = _adapter()
    out = adapter.embedding(
        ["ACGT"],
        layer="final",
        pool={"method": "mean", "dim": 1},
        fmt="tensor",
    )

    assert len(out) == 1
    assert adapter.model.embedding_layers == ["blocks.31.mlp.l3"]


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
    assert adapter.model.reduce_calls == [("sum", 2), ("mean", 2)]


def test_logits_and_embedding_reuses_single_forward_call() -> None:
    adapter = _adapter(model_id="evo2_20b", blocks=(0, 1, 20, 23))

    logits, embeddings = adapter.logits_and_embedding(["ACGT", "TGCA"], layer="mid", fmt="tensor")

    assert len(logits) == 2
    assert len(embeddings) == 2
    assert adapter.model.forward_calls == 1
    assert adapter.model.embedding_layers == ["blocks.23.mlp.l3"]
    assert logits[0].shape == torch.Size([4, 4])
    assert embeddings[0].shape == torch.Size([4, 3])


def test_logits_and_embedding_matches_separate_public_outputs_exactly() -> None:
    seqs = ["ACGT", "AC", "TGCA"]

    fused_adapter = _adapter(model_id="evo2_20b", blocks=(0, 1, 20, 23))
    fused_logits, fused_embeddings = fused_adapter.logits_and_embedding(seqs, layer="mid", fmt="tensor")

    separate_adapter = _adapter(model_id="evo2_20b", blocks=(0, 1, 20, 23))
    separate_logits = separate_adapter.logits(seqs, fmt="tensor")
    separate_embeddings = separate_adapter.embedding(seqs, layer="mid", fmt="tensor")

    assert fused_adapter.model.forward_calls == 2
    assert separate_adapter.model.forward_calls == 4
    assert fused_adapter.model.embedding_layers == ["blocks.23.mlp.l3", "blocks.23.mlp.l3"]
    assert separate_adapter.model.embedding_layers == ["blocks.23.mlp.l3", "blocks.23.mlp.l3"]
    for fused, separate in zip(fused_logits, separate_logits, strict=True):
        torch.testing.assert_close(fused, separate, rtol=0.0, atol=0.0)
    for fused, separate in zip(fused_embeddings, separate_embeddings, strict=True):
        torch.testing.assert_close(fused, separate, rtol=0.0, atol=0.0)


def test_log_likelihood_rejects_unknown_reduction() -> None:
    adapter = _adapter()
    with pytest.raises(CapabilityError, match="reduction='sum' or 'mean'"):
        adapter.log_likelihood(["ACGT"], method="native", reduction="median")


def test_generate_accepts_tuple_sequences_from_evo2_api() -> None:
    adapter = _adapter()

    out = adapter.generate(["ACGT"], max_new_tokens=2)

    assert out == {"gen_seqs": ["ACGTAA"]}
