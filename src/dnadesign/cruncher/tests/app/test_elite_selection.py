"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_elite_selection.py

Tests elite-selection helper behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.app.sample.diagnostics import _EliteCandidate, dsdna_equivalence_enabled, resolve_dsdna_mode
from dnadesign.cruncher.app.sample.elites_mmr import select_elites_mmr
from dnadesign.cruncher.config.schema_v3 import SampleConfig


def test_resolve_dsdna_mode_tracks_bidirectional_flag() -> None:
    assert resolve_dsdna_mode(elites_cfg=object(), bidirectional=True) is True
    assert resolve_dsdna_mode(elites_cfg=object(), bidirectional=False) is False


def test_dsdna_equivalence_enabled_uses_sample_objective_flag() -> None:
    cfg = SampleConfig(
        seed=7,
        sequence_length=6,
        budget={"tune": 0, "draws": 1},
        objective={"bidirectional": True, "score_scale": "normalized-llr"},
    )
    assert dsdna_equivalence_enabled(cfg) is True


def test_select_elites_mmr_with_zero_diversity_uses_score_only_selection() -> None:
    raw_elites = [
        _EliteCandidate(
            seq_arr=np.asarray([0, 0, 0, 0], dtype=np.int8),
            chain_id=0,
            draw_idx=1,
            combined_score=0.90,
            min_norm=0.20,
            sum_norm=0.20,
            per_tf_map={"tf": 0.20},
            norm_map={"tf": 0.20},
            per_tf_hits={},
        ),
        _EliteCandidate(
            seq_arr=np.asarray([0, 0, 0, 1], dtype=np.int8),
            chain_id=0,
            draw_idx=2,
            combined_score=0.85,
            min_norm=0.95,
            sum_norm=0.95,
            per_tf_map={"tf": 0.95},
            norm_map={"tf": 0.95},
            per_tf_hits={},
        ),
        _EliteCandidate(
            seq_arr=np.asarray([0, 0, 1, 1], dtype=np.int8),
            chain_id=0,
            draw_idx=3,
            combined_score=0.80,
            min_norm=0.99,
            sum_norm=0.99,
            per_tf_map={"tf": 0.99},
            norm_map={"tf": 0.99},
            per_tf_hits={},
        ),
    ]

    result = select_elites_mmr(
        raw_elites=raw_elites,
        elite_k=2,
        pool_size=3,
        scorer=object(),
        pwms={},
        dsdna_mode=False,
        diversity=0.0,
        sample_sequence_length=4,
        cooling_config={},
    )

    assert [f"{row.chain_id}:{row.draw_idx}" for row in result.kept_elites] == ["0:1", "0:2"]
    assert result.mmr_summary is not None
    assert result.mmr_summary["selection_policy"] == "score_topk"
    assert result.mmr_summary["score_weight"] == pytest.approx(1.0)
    assert result.mmr_summary["diversity_weight"] == pytest.approx(0.0)
