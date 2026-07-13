"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_sfxi_comparison.py

Tests for canonical SFXI comparisons over Reader-owned summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    sfxi_comparison,
)


def test_sfxi_comparison_reports_rank_reversal_against_snapshot() -> None:
    rows: list[dict[str, object]] = []
    for summary_id, brightness in (
        ("snapshot_12h", (0.0, 1.0, 2.0, 3.0)),
        ("event_linear_auc_6_12h_post", (3.0, 2.0, 1.0, 0.0)),
    ):
        for index, y11 in enumerate(brightness):
            rows.append(
                {
                    "id": f"id-{index}",
                    "design_id": f"D{index}",
                    "assay_summary_id": summary_id,
                    "assay_summary_method": "snapshot" if summary_id == "snapshot_12h" else "integrated_linear_mean",
                    "v00": 0.0,
                    "v10": 0.0,
                    "v01": 0.0,
                    "v11": 1.0,
                    "y00_star": 0.0,
                    "y10_star": 0.0,
                    "y01_star": 0.0,
                    "y11_star": y11,
                }
            )
    target_view = StressTargetView("and", "AND", (0.0, 0.0, 0.0, 1.0))

    metrics = sfxi_comparison.build_sfxi_comparison_rows(
        pd.DataFrame(rows),
        target_views=(target_view,),
        logic_threshold=0.45,
        scaling_min_n=2,
    )
    stability = sfxi_comparison.summarize_sfxi_comparison(metrics, baseline_summary_id="snapshot_12h")

    alternative = stability.loc[stability["assay_summary_id"] == "event_linear_auc_6_12h_post"].iloc[0]
    assert alternative["score_spearman_to_snapshot"] == pytest.approx(-1.0)
    assert alternative["logic_support_count"] == 4
    assert alternative["correlation_defined"]


def test_assay_comparison_requires_identical_candidate_universes() -> None:
    rows = pd.DataFrame(
        [
            _vec8_row("snapshot_12h", "a"),
            _vec8_row("snapshot_12h", "b"),
            _vec8_row("event_logmean_6_12h_post", "a"),
        ]
    )
    target_view = StressTargetView("and", "AND", (0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="identical candidate-id universe"):
        sfxi_comparison.build_sfxi_comparison_rows(
            rows,
            target_views=(target_view,),
            logic_threshold=0.45,
            scaling_min_n=2,
        )


def _vec8_row(summary_id: str, candidate_id: str) -> dict[str, object]:
    return {
        "id": candidate_id,
        "design_id": candidate_id,
        "assay_summary_id": summary_id,
        "assay_summary_method": "comparison",
        "v00": 0.0,
        "v10": 0.0,
        "v01": 0.0,
        "v11": 1.0,
        "y00_star": 0.0,
        "y10_star": 0.0,
        "y01_star": 0.0,
        "y11_star": 1.0,
    }
