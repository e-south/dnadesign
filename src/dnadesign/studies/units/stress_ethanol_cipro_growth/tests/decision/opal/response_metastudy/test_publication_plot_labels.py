"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_publication_plot_labels.py

Render-level checks for publication vocabulary on representative plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    diagnostic_plots,
    metric_behavior_plots,
    model_validation_plot,
    plot_style,
    primary_plots,
    response_assay_plots,
)

CANONICAL = "sfxi_beta1_gamma1"
COMPARISON = "logic_first_beta4_gamma05"


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[plt.Figure]:
    figures: list[plt.Figure] = []

    def capture(figure: plt.Figure, path: Path) -> None:
        plot_style.save_metastudy_figure(figure, path)
        figures.append(figure)

    for module in (
        diagnostic_plots,
        metric_behavior_plots,
        model_validation_plot,
        primary_plots,
        response_assay_plots,
    ):
        monkeypatch.setattr(module, "save_metastudy_figure", capture)
    return figures


def test_model_validation_exposes_conditions_instead_of_target_slugs(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    rows = []
    for strategy in ("shuffled_kfold", "leave_one_experiment_out"):
        for scope, metric in (
            ("target", "v01"),
            ("target", "y11_star"),
            ("selection_view_objective", "ciprofloxacin"),
        ):
            rows.append(
                {
                    "split_strategy": strategy,
                    "scope": scope,
                    "metric_id": metric,
                    "spearman": 0.25,
                }
            )
    model_validation_plot.write_model_validation(pd.DataFrame(rows), tmp_path / "model_validation.png")

    figure = captured_figures[0]
    labels = _visible_text(figure)
    assert "Logic: Ciprofloxacin (v01)" in labels
    assert "Fluorescence: Both stresses (y*11)" in labels
    assert "Ciprofloxacin objective" in labels
    assert "v01" not in labels
    assert "y11_star" not in labels
    assert all(axis.title.get_ha() == "center" for axis in figure.axes)


def test_panel_roles_and_target_views_use_declared_publication_names(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "panel_role": ["canonical_sfxi_high_effect", "comparison_shape_effect"],
            "selection_view_id": ["ciprofloxacin", "and"],
        }
    )
    metric_behavior_plots.write_policy_comparison_panel_roles(
        frame,
        tmp_path / "policy_comparison_panel_roles.png",
    )

    labels = _visible_text(captured_figures[0])
    assert "Canonical SFXI high effect" in labels
    assert "Shape/effect comparison" in labels
    assert "Ciprofloxacin" in labels
    assert "AND" in labels
    assert "canonical_sfxi_high_effect" not in labels
    assert "ciprofloxacin" not in labels


def test_score_component_facets_hide_policy_and_target_view_ids(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    summary = pd.DataFrame({"policy_id": [CANONICAL, COMPARISON]})
    rows = []
    for policy in (CANONICAL, COMPARISON):
        for selection_view_id in ("ethanol", "ciprofloxacin", "and"):
            for component, value in (("logic_fidelity", 0.2), ("effect_scaled", 0.9)):
                rows.append(
                    {
                        "policy_id": policy,
                        "metric": "within_selection_view",
                        "selection_view_a": selection_view_id,
                        "selection_view_b": component,
                        "pearson": value,
                    }
                )
    primary_plots.write_score_component_dominance(
        summary,
        pd.DataFrame(rows),
        tmp_path / "score_component_dominance.png",
        comparison_policy_id=COMPARISON,
    )

    labels = _visible_text(captured_figures[0])
    assert "Canonical SFXI" in labels
    assert "Logic-weighted" in labels
    assert "Ciprofloxacin" in labels
    assert "Score vs scaled effect" in labels
    assert CANONICAL not in labels
    assert COMPARISON not in labels
    assert "ciprofloxacin" not in labels


def test_selected_profile_heatmap_facets_policy_and_preserves_square_tiles(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    summary = pd.DataFrame({"policy_id": [CANONICAL, COMPARISON]})
    rows = []
    for policy in (CANONICAL, COMPARISON):
        for selection_view_id in ("ethanol", "ciprofloxacin", "and"):
            rows.append(
                {
                    "policy_id": policy,
                    "selection_view_id": selection_view_id,
                    "v00": 0.1,
                    "v10": 0.4,
                    "v01": 0.6,
                    "v11": 0.9,
                }
            )
    diagnostic_plots.write_selected_vec8_profiles(
        summary,
        pd.DataFrame(rows),
        tmp_path / "selected_vec8_profiles.png",
        comparison_policy_id=COMPARISON,
    )

    figure = captured_figures[0]
    matrix_axes = [axis for axis in figure.axes if axis.get_xlabel() == "SFXI logic state"]
    assert len(matrix_axes) == 2
    assert all(axis.get_aspect() == 1.0 for axis in matrix_axes)
    assert [axis.get_title() for axis in matrix_axes] == [
        "Canonical SFXI",
        "Logic-weighted",
    ]
    assert [[tick.get_text() for tick in axis.get_yticklabels()] for axis in matrix_axes] == [
        ["Ethanol", "Ciprofloxacin", "AND"],
        ["Ethanol", "Ciprofloxacin", "AND"],
    ]


def test_response_uncertainty_names_event_envelope_by_its_current_semantics(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "selection_view_id": ["ethanol"],
            "response_separation__bootstrap_sd": [0.2],
            "response_separation__event_half_range": [0.05],
            "on_magnitude_floor__bootstrap_sd": [0.3],
            "on_magnitude_floor__event_half_range": [0.04],
            "off_magnitude_ceiling__bootstrap_sd": [0.25],
            "off_magnitude_ceiling__event_half_range": [0.03],
        }
    )

    response_assay_plots.write_response_uncertainty_sources(
        frame,
        tmp_path / "response_uncertainty_sources.png",
    )

    labels = _visible_text(captured_figures[0])
    assert "Maximum event-bound deviation" in labels
    assert "Event-bound half range" not in labels


def _visible_text(figure: plt.Figure) -> set[str]:
    labels: set[str] = set()
    if figure._suptitle is not None:
        labels.add(figure._suptitle.get_text())
    for axis in figure.axes:
        labels.update(text.get_text() for text in axis.get_xticklabels())
        labels.update(text.get_text() for text in axis.get_yticklabels())
        labels.update(text.get_text() for text in axis.texts)
        labels.update(title for title in (axis.get_title(), axis.get_xlabel(), axis.get_ylabel()) if title)
        legend = axis.get_legend()
        if legend is not None:
            labels.update(text.get_text() for text in legend.get_texts())
    for legend in figure.legends:
        labels.update(text.get_text() for text in legend.get_texts())
    return labels
