"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_elite_selection.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.app.sample.diagnostics import _elite_filter_passes


def test_elite_filter_min_per_tf_norm() -> None:
    norm_map = {"tf1": 0.2, "tf2": 0.05}
    assert not _elite_filter_passes(
        norm_map=norm_map,
        min_norm=0.05,
        min_per_tf_norm=0.1,
        require_all_tfs_over_min_norm=True,
    )
    assert not _elite_filter_passes(
        norm_map=norm_map,
        min_norm=0.05,
        min_per_tf_norm=0.1,
        require_all_tfs_over_min_norm=False,
    )


def test_elite_filter_reapplies_min_per_tf_norm() -> None:
    assert _elite_filter_passes(
        norm_map={"tf1": 0.5},
        min_norm=0.5,
        min_per_tf_norm=0.4,
        require_all_tfs_over_min_norm=True,
    )
    assert not _elite_filter_passes(
        norm_map={"tf1": 0.1},
        min_norm=0.1,
        min_per_tf_norm=0.4,
        require_all_tfs_over_min_norm=True,
    )
