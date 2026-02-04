"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_elite_norm_map.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.app.sample_workflow import _norm_map_for_elites


class DummyScorer:
    def __init__(self, tf_names: list[str]) -> None:
        self.tf_names = tf_names

    def normalized_llr_map(self, seq_arr: np.ndarray) -> dict[str, float]:
        raise RuntimeError("normalized_llr_map should not be called for normalized-llr")


def test_norm_map_uses_per_tf_for_normalized_llr() -> None:
    scorer = DummyScorer(["tf1", "tf2"])
    per_tf_map = {"tf1": 0.2, "tf2": 0.8}
    seq_arr = np.array([0, 1, 2, 3], dtype=np.int8)
    norm_map = _norm_map_for_elites(seq_arr, per_tf_map, scorer=scorer, score_scale="normalized-llr")
    assert norm_map == {"tf1": 0.2, "tf2": 0.8}
