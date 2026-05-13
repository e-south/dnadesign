"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_evo2_adapter_scoring.py

Contract tests for Evo2 adapter scoring behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

from dnadesign.infer.src.adapters.evo2 import Evo2Adapter


class _PaddingSensitiveScoreModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int, str]] = []

    def score_sequences(self, seqs, *, batch_size: int, reduce_method: str):
        assert len({len(seq) for seq in seqs}) <= 1
        self.calls.append((tuple(seqs), batch_size, reduce_method))
        if reduce_method == "sum":
            return [float(len(seq) - 1) for seq in seqs]
        if reduce_method == "mean":
            return [1.0 if len(seq) > 1 else float("nan") for seq in seqs]
        raise AssertionError(f"unexpected reduction: {reduce_method}")


def _adapter_with_score_model(score_model: _PaddingSensitiveScoreModel) -> Evo2Adapter:
    adapter = object.__new__(Evo2Adapter)
    adapter.model = score_model
    return adapter


def test_evo2_log_likelihood_scores_equal_length_buckets_without_padding() -> None:
    score_model = _PaddingSensitiveScoreModel()
    adapter = _adapter_with_score_model(score_model)

    values = adapter.log_likelihood(["AAAA", "TT", "CCCC"], method="native", reduction="sum")

    assert values == [3.0, 1.0, 3.0]
    assert score_model.calls == [
        (("TT",), 1, "sum"),
        (("AAAA", "CCCC"), 2, "sum"),
    ]


def test_evo2_log_likelihood_total_and_mean_uses_one_native_sum_per_length_bucket() -> None:
    score_model = _PaddingSensitiveScoreModel()
    adapter = _adapter_with_score_model(score_model)

    totals, means = adapter.log_likelihood_total_and_mean(["AAAA", "A", "TT"], method="native")

    assert totals == [3.0, 0.0, 1.0]
    assert means[0] == 1.0
    assert math.isnan(means[1])
    assert means[2] == 1.0
    assert score_model.calls == [
        (("A",), 1, "sum"),
        (("TT",), 1, "sum"),
        (("AAAA",), 1, "sum"),
    ]
