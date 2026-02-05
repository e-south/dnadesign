"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_mmr_selection.py

Unit tests for MMR-based elite selection utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.selection.mmr import (
    MmrCandidate,
    compute_core_distance,
    compute_position_weights,
    select_mmr_elites,
)


def _pwm(name: str, rows: list[list[float]]) -> PWM:
    return PWM(name=name, matrix=np.asarray(rows, dtype=float))


def _candidate(seq: str, *, chain: int, draw: int, combined: float, min_norm: float) -> MmrCandidate:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.array([mapping[ch] for ch in seq], dtype=np.int8)
    return MmrCandidate(
        seq_arr=arr,
        chain_id=chain,
        draw_idx=draw,
        combined_score=combined,
        min_norm=min_norm,
        sum_norm=min_norm,
        per_tf_map={},
        norm_map={},
    )


def test_position_weights_emphasize_low_info_positions() -> None:
    pwm = _pwm("tf", [[0.99, 0.003, 0.003, 0.004], [0.25, 0.25, 0.25, 0.25]])
    weights = compute_position_weights(pwm)
    assert weights.shape == (2,)
    assert weights[0] < weights[1]
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


def test_core_distance_is_normalized() -> None:
    weights = {"tf": np.array([1.0, 1.0, 1.0], dtype=float)}
    cores_a = {"tf": np.array([0, 1, 2], dtype=np.int8)}
    cores_b = {"tf": np.array([0, 1, 3], dtype=np.int8)}
    dist = compute_core_distance(cores_a, cores_b, weights=weights, tf_names=["tf"])
    assert dist == pytest.approx(1.0 / 3.0)
    assert 0.0 <= dist <= 1.0


def test_mmr_selection_is_deterministic() -> None:
    candidates = [
        _candidate("AAAA", chain=0, draw=1, combined=0.9, min_norm=0.9),
        _candidate("AAAT", chain=0, draw=2, combined=0.8, min_norm=0.8),
        _candidate("AATT", chain=0, draw=3, combined=0.7, min_norm=0.7),
    ]
    selection_one = select_mmr_elites(
        candidates,
        k=2,
        pool_size=3,
        alpha=0.8,
        relevance="min_per_tf_norm",
        dsdna=False,
        tf_names=["tf"],
        pwms={"tf": _pwm("tf", [[0.25, 0.25, 0.25, 0.25]] * 4)},
        core_maps={f"{cand.chain_id}:{cand.draw_idx}": {"tf": cand.seq_arr} for cand in candidates},
    )
    selection_two = select_mmr_elites(
        candidates,
        k=2,
        pool_size=3,
        alpha=0.8,
        relevance="min_per_tf_norm",
        dsdna=False,
        tf_names=["tf"],
        pwms={"tf": _pwm("tf", [[0.25, 0.25, 0.25, 0.25]] * 4)},
        core_maps={f"{cand.chain_id}:{cand.draw_idx}": {"tf": cand.seq_arr} for cand in candidates},
    )
    ids_one = [row["candidate_id"] for row in selection_one.meta]
    ids_two = [row["candidate_id"] for row in selection_two.meta]
    assert ids_one == ids_two


def test_mmr_dedupes_reverse_complements() -> None:
    candidates = [
        _candidate("ACGA", chain=0, draw=1, combined=0.9, min_norm=0.9),
        _candidate("TCGT", chain=0, draw=2, combined=0.85, min_norm=0.85),
        _candidate("TTTT", chain=0, draw=3, combined=0.8, min_norm=0.8),
    ]
    result = select_mmr_elites(
        candidates,
        k=2,
        pool_size=3,
        alpha=0.7,
        relevance="min_per_tf_norm",
        dsdna=True,
        tf_names=["tf"],
        pwms={"tf": _pwm("tf", [[0.25, 0.25, 0.25, 0.25]] * 4)},
        core_maps={f"{cand.chain_id}:{cand.draw_idx}": {"tf": cand.seq_arr} for cand in candidates},
    )
    selected = [row["sequence"] for row in result.meta]
    assert len(selected) == 2
    assert len({row["canonical_sequence"] for row in result.meta}) == 2
