"""Publication-type contracts for the MSRB shadow evidence figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    multistate_behavior_plots as plots,
)


def test_msrb_shadow_plots_use_readable_collision_free_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(plots, "save_figure", lambda figure, _path: captured.append(figure))

    plots.render_multistate_behavior_plots(
        normalization_sensitivity=_normalization_rows(),
        grouped_validation=_validation_rows(),
        allocation_comparison=_allocation_rows(),
        prediction_scores=_prediction_rows(),
        output_dir=tmp_path / "plots",
    )

    assert len(captured) == 3
    try:
        for figure in captured:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            figure_box = figure.get_window_extent(renderer)
            assert figure._suptitle is not None
            assert figure._suptitle.get_fontsize() >= 18
            title_boxes = [figure._suptitle.get_window_extent(renderer)]
            for axis in figure.axes:
                if axis.get_title():
                    assert axis.title.get_fontsize() >= 15
                    title_boxes.append(axis.title.get_window_extent(renderer))
                assert min((tick.get_fontsize() for tick in axis.get_xticklabels()), default=11) >= 11
                assert min((tick.get_fontsize() for tick in axis.get_yticklabels()), default=11) >= 11
                if axis.xaxis.label.get_text():
                    assert axis.xaxis.label.get_fontsize() >= 13
                if axis.yaxis.label.get_text():
                    assert axis.yaxis.label.get_fontsize() >= 13
                legend = axis.get_legend()
                if legend is not None:
                    assert min(text.get_fontsize() for text in legend.get_texts()) >= 11
            for legend in figure.legends:
                assert min(text.get_fontsize() for text in legend.get_texts()) >= 11
            for box in title_boxes:
                assert box.x0 >= figure_box.x0
                assert box.x1 <= figure_box.x1
                assert box.y0 >= figure_box.y0
                assert box.y1 <= figure_box.y1
            for left_index, left in enumerate(title_boxes):
                for right in title_boxes[left_index + 1 :]:
                    assert not left.overlaps(right)
    finally:
        for figure in captured:
            plt.close(figure)


def _normalization_rows() -> pd.DataFrame:
    rows = []
    for view in plots.VIEW_ORDER:
        for scenario_id, scenario_kind, value in (
            ("quantile_q90", "scale_quantile", 1.0),
            ("holdout_e1", "leave_one_source_experiment_out", 0.8),
        ):
            rows.append(
                {
                    "selection_view_id": view,
                    "scenario_id": scenario_id,
                    "scenario_kind": scenario_kind,
                    "score_spearman_vs_primary": value,
                    "raw_top_k_overlap": 5,
                    "raw_top_k": 6,
                }
            )
    return pd.DataFrame(rows)


def _validation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "selection_view_id": view,
                "objective_name": objective,
                "median_within_group_spearman": 0.1 + 0.02 * seed,
                "pooled_oof_spearman": 0.05 + 0.01 * seed,
            }
            for view in plots.VIEW_ORDER
            for objective in ("multistate_response_behavior_v1", "response_magnitude_feasibility_v1")
            for seed in range(2)
        ]
    )


def _allocation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "objective_name": "multistate_response_behavior_v1",
                "selection_view_id": view,
                "id": f"{view}-candidate",
                "allocation_slot": 1,
                "rank": 1,
                "display_label": f"{view} candidate",
            }
            for view in plots.VIEW_ORDER
        ]
    )


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selection_view_id": view,
                "id": f"{view}-candidate",
                "limiting_coordinate": "response:10>01",
                "behavior_score": 0.2,
                "hard_bottleneck_clearance": -0.1,
                "response_family_score": 0.3,
                "on_signal_family_score": 0.4,
                "off_signal_suppression_family_score": 0.1,
            }
            for view in plots.VIEW_ORDER
        ]
    )
