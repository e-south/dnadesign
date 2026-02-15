"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_adaptive_controllers.py

Validates adaptive controller behavior for move weights, proposal sizes, and
per-pair PT swap ladder shaping.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.optimizers.policies import (
    MOVE_KINDS,
    AdaptiveMoveController,
    AdaptiveProposalController,
    AdaptiveSwapPairController,
    move_probs_array,
)


def test_adaptive_move_controller_rebalances_probabilities() -> None:
    base_probs = move_probs_array({"S": 0.4, "B": 0.3, "M": 0.2, "L": 0.0, "W": 0.0, "I": 0.1})
    controller = AdaptiveMoveController(
        enabled=True,
        window=4,
        k=0.8,
        min_prob=0.01,
        max_prob=0.95,
        targets={"S": 0.95, "B": 0.40, "M": 0.35, "I": 0.35},
        kinds=("S", "B", "M", "I"),
    )

    for _ in range(4):
        controller.record("B", accepted=False)
        controller.record("M", accepted=True)
        controller.record("I", accepted=True)
        controller.record("S", accepted=True)
    adapted_probs = controller.adapt(base_probs)

    idx_s = MOVE_KINDS.index("S")
    idx_b = MOVE_KINDS.index("B")
    idx_m = MOVE_KINDS.index("M")
    assert np.isclose(float(adapted_probs.sum()), 1.0)
    assert adapted_probs[idx_b] < base_probs[idx_b]
    assert adapted_probs[idx_m] > base_probs[idx_m]
    assert adapted_probs[idx_s] > 0.0


def test_adaptive_proposal_controller_adjusts_ranges() -> None:
    controller = AdaptiveProposalController(
        enabled=True,
        window=4,
        step=0.20,
        min_scale=0.5,
        max_scale=2.0,
        target_low=0.25,
        target_high=0.75,
    )
    base_block = (4, 10)
    base_multi = (3, 9)

    for _ in range(4):
        controller.record("B", accepted=True)
        controller.record("M", accepted=True)
    block_grow, multi_grow = controller.current_ranges(base_block, base_multi, sequence_length=64)
    assert block_grow[0] >= base_block[0]
    assert block_grow[1] > base_block[1]
    assert multi_grow[1] > base_multi[1]

    for _ in range(8):
        controller.record("B", accepted=False)
        controller.record("M", accepted=False)
    block_shrink, multi_shrink = controller.current_ranges(base_block, base_multi, sequence_length=64)
    assert block_shrink[0] <= block_grow[0]
    assert block_shrink[1] <= block_grow[1]
    assert multi_shrink[1] <= multi_grow[1]


def test_adaptive_swap_pair_controller_shapes_ladder_by_pair() -> None:
    base_betas = [0.05, 0.10, 0.20, 0.40]
    controller = AdaptiveSwapPairController(
        n_pairs=3,
        enabled=True,
        target=0.25,
        window=2,
        k=1.0,
        min_scale=0.25,
        max_scale=4.0,
        strict=False,
        saturation_windows=5,
    )

    controller.record(pair_idx=0, accepted=False)
    controller.record(pair_idx=0, accepted=False)
    controller.record(pair_idx=1, accepted=True)
    controller.record(pair_idx=1, accepted=True)
    controller.record(pair_idx=2, accepted=True)
    controller.record(pair_idx=2, accepted=True)
    controller.update()

    ladder = controller.ladder_from_base(base_betas)
    assert ladder[0] == base_betas[0]
    assert ladder[-1] > ladder[0]
    gap0 = ladder[1] / ladder[0]
    gap1 = ladder[2] / ladder[1]
    assert gap0 != gap1
