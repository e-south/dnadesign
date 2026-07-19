"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_multistate_response_behavior_plots.py

Contract and rendering tests for Multistate Response Behavior decision plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest

from dnadesign.opal.api.multistate_response_behavior import score_multistate_response_behavior
from dnadesign.opal.src.analysis.notebook_components.visual_hierarchy import notebook_visual_group
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.plots import multistate_response_behavior_decomposition as decomposition_plot
from dnadesign.opal.src.plots import multistate_response_behavior_frontier as frontier_plot
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.plots.multistate_response_behavior_data import (
    BEHAVIOR_SCORE_REF,
    HARD_BOTTLENECK_REF,
    OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
    ON_SIGNAL_FAMILY_REF,
    RESPONSE_FAMILY_REF,
    SELECTED_COORDINATE_DETAIL_SCOPE,
    MultistateResponseBehaviorPlotData,
    _selected_coordinate_frame,
    load_multistate_response_behavior_plot_data,
    multistate_response_behavior_observed_frame,
    multistate_response_behavior_plot_frame,
)
from dnadesign.opal.src.plots.multistate_response_behavior_support import state_display_labels
from dnadesign.opal.src.plots.runner import _validate_plot_objective_compatibility
from dnadesign.opal.src.registries.plots import describe_plot_kind
from dnadesign.opal.src.storage.artifacts import run_scoped_artifact_path


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.workdir = outputs_dir.parent


STATE_IDS = ("baseline", "stress-a", "stress-b")
TARGET_MASK = (0, 1, 0)
NORMALIZATION = {"response_scale": 1.0, "signal_scale": 1.0}
VECTORS = np.asarray(
    [
        [0.0, 2.0, 0.2, -1.0, 1.2, -0.6],
        [0.3, 1.0, 0.4, -0.2, 0.5, -0.1],
        [0.8, 0.7, 0.6, 0.1, 0.2, 0.3],
    ],
    dtype=float,
)


def _score_channel(value: float, *, view_id: str = "stress-a") -> list[dict[str, object]]:
    return [{"name": f"{view_id}/{BEHAVIOR_SCORE_REF}", "value": value}]


def _prediction_events(*, drift: float = 0.0) -> pd.DataFrame:
    scored = score_multistate_response_behavior(
        VECTORS,
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
    )
    return pd.DataFrame(
        {
            "as_of_round": [0, 0, 0],
            "run_id": ["r0", "r0", "r0"],
            "id": ["selected-a", "selected-b", "pool-c"],
            "pred__y_hat_model": [row.tolist() for row in VECTORS],
            "pred__score_channels": [
                _score_channel(float(value) + (drift if index == 0 else 0.0))
                for index, value in enumerate(scored.behavior_score)
            ],
            "view__rank_competition": [1, 2, 3],
            "view__is_selected": [True, True, False],
        }
    )


def _plot_data() -> MultistateResponseBehaviorPlotData:
    observed = multistate_response_behavior_observed_frame(
        pd.DataFrame(
            {
                "id": ["observed-a"],
                "observed_round": [0],
                "batch_id": ["batch-0"],
                "display_label": ["Observed A"],
                "y_obs": [VECTORS[0].tolist()],
            }
        ),
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
    )
    frame = multistate_response_behavior_plot_frame(
        _prediction_events(),
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
        selection_view_id="stress-a",
    )
    coordinate_labels = (
        "response:stress-a>baseline",
        "response:stress-a>stress-b",
        "on_signal:stress-a",
        "off_signal_suppression:baseline",
        "off_signal_suppression:stress-b",
    )
    return MultistateResponseBehaviorPlotData(
        frame=frame,
        observed_frame=observed,
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
        coordinate_labels=coordinate_labels,
        round_k=0,
        run_id="r0",
        selected_coordinate_frame=_selected_coordinate_frame(
            frame,
            state_ids=STATE_IDS,
            target_mask=TARGET_MASK,
            normalization=NORMALIZATION,
            coordinate_labels=coordinate_labels,
        ),
    )


def _context(tmp_path: Path, *, filename: str, save_data: bool) -> PlotContext:
    return PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds=[0],
        run_id="r0",
        selection_view_id="stress-a",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename=filename,
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.multistate-behavior-plots"),
        save_data=save_data,
    )


def test_behavior_plot_loader_uses_run_pinned_observed_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs_dir = tmp_path / "outputs"
    labels_path = run_scoped_artifact_path(
        outputs_dir / "rounds" / "round_0",
        run_id="r0",
        artifact_key="labels/observed_events.parquet",
    )
    labels_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["r0"],
            "as_of_round": [0],
            "observed_round": [0],
            "batch_id": ["pre-round-0"],
            "display_label": ["Observed A"],
            "id": ["observed-a"],
            "sequence": ["ACGT"],
            "y_space": ["reader_response_window_vector_v1"],
            "y_obs": [VECTORS[0].tolist()],
            "label_source_kind": ["usr_sidecar"],
        }
    ).write_parquet(labels_path)
    runs = pl.DataFrame(
        {
            "run_id": ["r0"],
            "as_of_round": [0],
            "objective__defs_json": [
                json.dumps(
                    [
                        {
                            "selection_view_id": "stress-a",
                            "objective_name": "multistate_response_behavior_v1",
                            "params": {
                                "state_ids": list(STATE_IDS),
                                "target_mask": list(TARGET_MASK),
                                "normalization": NORMALIZATION,
                            },
                        }
                    ]
                )
            ],
            "selection_views__defs_json": [
                json.dumps(
                    [
                        {
                            "selection_view_id": "stress-a",
                            "score_ref": f"stress-a/{BEHAVIOR_SCORE_REF}",
                        }
                    ]
                )
            ],
            "artifacts": [
                {
                    "labels/observed_events.parquet": [
                        file_sha256(labels_path),
                        str(labels_path.resolve()),
                    ]
                }
            ],
        }
    )
    monkeypatch.setattr(
        "dnadesign.opal.src.plots.multistate_response_behavior_data.read_runs",
        lambda _path: runs,
    )
    monkeypatch.setattr(
        "dnadesign.opal.src.plots.multistate_response_behavior_data.load_events",
        lambda *_args, **_kwargs: _prediction_events(),
    )
    context = _context(tmp_path, filename="review.png", save_data=False)
    context.run_id = None

    data = load_multistate_response_behavior_plot_data(context, detail_scope="summary")

    assert data.run_id == "r0"
    assert data.observed_frame["id"].tolist() == ["observed-a"]
    assert data.observed_frame["batch_id"].tolist() == ["pre-round-0"]
    assert data.selected_coordinate_frame is None
    coordinate_object_columns = {
        "coordinate_clearances",
        "coordinate_weights",
        "coordinate_labels",
    }
    assert coordinate_object_columns.isdisjoint(data.frame.columns)
    assert coordinate_object_columns.isdisjoint(data.observed_frame.columns)
    assert "pred__y_hat_model" not in data.frame.columns
    assert context.data_paths["run_observed_events_parquet"] == labels_path.resolve()
    assert not (outputs_dir / "ledger" / "labels.parquet").exists()

    detailed = load_multistate_response_behavior_plot_data(
        context,
        detail_scope=SELECTED_COORDINATE_DETAIL_SCOPE,
    )
    assert detailed.selected_coordinate_frame is not None
    assert detailed.selected_coordinate_frame["id"].tolist() == ["selected-a", "selected-b"]
    assert coordinate_object_columns <= set(detailed.selected_coordinate_frame.columns)
    assert coordinate_object_columns.isdisjoint(detailed.frame.columns)
    assert "pred__y_hat_model" not in detailed.frame.columns
    assert "pred__y_hat_model" not in detailed.selected_coordinate_frame.columns
    expected_selected = score_multistate_response_behavior(
        VECTORS[:2],
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
    )
    np.testing.assert_allclose(
        np.asarray(detailed.selected_coordinate_frame["coordinate_clearances"].tolist()),
        expected_selected.coordinate_clearances,
    )
    np.testing.assert_allclose(
        np.asarray(detailed.selected_coordinate_frame["coordinate_weights"].tolist()),
        expected_selected.coordinate_weights,
    )
    assert all(
        tuple(labels) == expected_selected.coordinate_labels
        for labels in detailed.selected_coordinate_frame["coordinate_labels"]
    )


def test_behavior_plot_loader_rejects_unknown_detail_scope_before_io() -> None:
    with pytest.raises(OpalError, match="detail_scope"):
        load_multistate_response_behavior_plot_data(object(), detail_scope="all_coordinates")  # type: ignore[arg-type]


def test_behavior_plot_frame_replays_public_math_and_rejects_persisted_score_drift() -> None:
    frame = multistate_response_behavior_plot_frame(
        _prediction_events(),
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
        selection_view_id="stress-a",
    )

    assert {
        BEHAVIOR_SCORE_REF,
        HARD_BOTTLENECK_REF,
        RESPONSE_FAMILY_REF,
        ON_SIGNAL_FAMILY_REF,
        OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
        "limiting_coordinate_label",
        "all_reference_directions_met",
    } <= set(frame)
    assert {
        "coordinate_clearances",
        "coordinate_weights",
        "coordinate_labels",
    }.isdisjoint(frame.columns)
    expected = score_multistate_response_behavior(
        VECTORS,
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
    )
    assert frame["limiting_coordinate_label"].astype(str).tolist() == list(expected.limiting_coordinate_label)
    with pytest.raises(OpalError, match="canonical objective math"):
        multistate_response_behavior_plot_frame(
            _prediction_events(drift=0.1),
            state_ids=STATE_IDS,
            target_mask=TARGET_MASK,
            normalization=NORMALIZATION,
            selection_view_id="stress-a",
        )


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("view__is_selected", ["False", "True", "False"], "selected flags must be exact booleans"),
        ("view__rank_competition", [1.5, 2.0, 3.0], "selection ranks must be positive integers"),
    ],
)
def test_behavior_plot_frame_rejects_truthy_strings_and_fractional_ranks(
    column: str,
    values: list[object],
    message: str,
) -> None:
    events = _prediction_events()
    events[column] = values

    with pytest.raises(OpalError, match=message):
        multistate_response_behavior_plot_frame(
            events,
            state_ids=STATE_IDS,
            target_mask=TARGET_MASK,
            normalization=NORMALIZATION,
            selection_view_id="stress-a",
        )


def test_behavior_observed_replay_preserves_repeated_candidate_events() -> None:
    observed = multistate_response_behavior_observed_frame(
        pd.DataFrame(
            {
                "id": ["candidate-a", "candidate-a"],
                "observed_round": [0, 1],
                "batch_id": ["batch-0", "batch-1"],
                "display_label": ["Candidate A", "Candidate A"],
                "y_obs": [VECTORS[0].tolist(), VECTORS[1].tolist()],
            }
        ),
        state_ids=STATE_IDS,
        target_mask=TARGET_MASK,
        normalization=NORMALIZATION,
    )

    assert observed[["id", "observed_round", "batch_key"]].to_dict(orient="records") == [
        {"id": "candidate-a", "observed_round": 0, "batch_key": "batch-0"},
        {"id": "candidate-a", "observed_round": 1, "batch_key": "batch-1"},
    ]


def test_behavior_state_display_labels_must_be_unique() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        state_display_labels(
            STATE_IDS,
            {
                "baseline": "No stress",
                "stress-a": "Stress",
                "stress-b": "Stress",
            },
        )


@pytest.mark.parametrize(
    ("renderer", "renderer_module", "filename", "tidy_columns"),
    [
        (
            frontier_plot.render_family_frontier,
            frontier_plot,
            "frontier.png",
            {
                "response_family_score",
                "on_signal_family_score",
                "off_signal_suppression_family_score",
                "behavior_score",
                "record_kind",
            },
        ),
        (
            decomposition_plot.render_selected_decomposition,
            decomposition_plot,
            "decomposition.png",
            {"component_kind", "component_id", "value", "limiting"},
        ),
    ],
)
def test_behavior_plot_renderers_write_media_and_tidy_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renderer,
    renderer_module,
    filename: str,
    tidy_columns: set[str],
) -> None:
    monkeypatch.setattr(
        renderer_module,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: _plot_data(),
    )
    context = _context(tmp_path, filename=filename, save_data=True)

    renderer(context, {})

    media = context.output_dir / filename
    tidy = context.output_dir / filename.replace(".png", ".csv")
    assert media.stat().st_size > 1_000
    assert tidy_columns <= set(pd.read_csv(tidy).columns)


def test_behavior_frontier_declares_reference_semantics_without_feasibility_guides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(
        frontier_plot,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: _plot_data(),
    )
    monkeypatch.setattr(frontier_plot, "save_figure", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = _context(tmp_path, filename="frontier.png", save_data=False)

    frontier_plot.render_family_frontier(context, {})

    figure = captured.pop()
    axis = figure.axes[0]
    runtime = context.artifact_metadata["notebook_view"]
    assert runtime["reference_lines"] == {"x": [], "y": []}
    assert runtime["color_scale"]["center"] == pytest.approx(0.0)
    assert runtime["color_scale"]["extend"] == "neither"
    assert "not feasibility" in runtime["color_scale"]["context"]
    assert not axis.lines
    assert axis.get_title(loc="center").startswith("Multistate behavior family landscape · Stress A")
    assert axis.title.get_fontsize() >= 14
    assert axis.xaxis.label.get_fontsize() >= 11.5
    assert axis.yaxis.label.get_fontsize() >= 11.5
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert axis.get_legend().get_bbox_to_anchor()._bbox.y0 < 0.0
    assert min(text.get_fontsize() for text in axis.get_legend().get_texts()) >= 9.5
    legend_handles = axis.get_legend().legend_handles
    assert legend_handles[0].get_markerfacecolor() == "#D0D0D0"
    assert all(handle.get_markerfacecolor() == "none" for handle in legend_handles[1:])
    assert figure.axes[-1].get_ylabel() == r"OFF-signal-suppression family score, $S_{\mathrm{OFF}}$"
    assert runtime["color_scale"]["context"].endswith("not feasibility")
    plt.Figure.clear(figure)


def test_behavior_frontier_discloses_saturated_tail_colors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(
        frontier_plot,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: _plot_data(),
    )
    monkeypatch.setattr(frontier_plot, "save_figure", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = _context(tmp_path, filename="frontier.png", save_data=False)

    frontier_plot.render_family_frontier(context, {"color_extent": 0.05})

    figure = captured.pop()
    color_scale = context.artifact_metadata["notebook_view"]["color_scale"]
    assert color_scale["extent"] == pytest.approx(0.05)
    assert color_scale["extend"] in {"both", "min", "max"}
    assert "visible values saturate" in color_scale["context"]
    assert "points and exact ledger values are retained" in color_scale["context"]
    assert figure.axes[-1]._colorbar.extend == color_scale["extend"]
    assert figure.axes[-1]._colorbar.extendrect is True
    plt.Figure.clear(figure)


def test_behavior_frontier_uses_compact_static_observed_batch_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    data = _plot_data()
    data.observed_frame.loc[:, "batch_key"] = "pre_round0_response_corpus_4_8h_v1"
    monkeypatch.setattr(
        frontier_plot,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: data,
    )
    monkeypatch.setattr(frontier_plot, "save_figure", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = _context(tmp_path, filename="frontier.png", save_data=False)

    frontier_plot.render_family_frontier(context, {})

    figure = captured.pop()
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert "Observed · Pre-round 0 (n=1)" in labels
    assert all("response corpus" not in label.lower() for label in labels)
    plt.Figure.clear(figure)


def test_behavior_frontier_wraps_long_target_context_inside_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(
        frontier_plot,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: _plot_data(),
    )
    monkeypatch.setattr(frontier_plot, "save_figure", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = _context(tmp_path, filename="frontier.png", save_data=False)

    frontier_plot.render_family_frontier(
        context,
        {
            "title": "A deliberately long publication-facing multistate behavior comparison across candidates",
            "state_labels": {
                "baseline": "Baseline vehicle condition",
                "stress-a": "Primary stress condition alpha",
                "stress-b": "Secondary stress condition beta",
            },
        },
    )

    figure = captured.pop()
    figure.canvas.draw()
    title = figure.axes[0].title
    title_box = title.get_window_extent(renderer=figure.canvas.get_renderer())
    assert len(title.get_text().splitlines()) >= 3
    assert title_box.x0 >= figure.bbox.x0
    assert title_box.x1 <= figure.bbox.x1
    color_label_box = figure.axes[-1].yaxis.label.get_window_extent(renderer=figure.canvas.get_renderer())
    legend_box = figure.axes[0].get_legend().get_window_extent(renderer=figure.canvas.get_renderer())
    assert not title_box.overlaps(color_label_box)
    assert not legend_box.overlaps(color_label_box)
    plt.Figure.clear(figure)


def test_behavior_decomposition_is_k_state_and_marks_only_coordinate_bottlenecks() -> None:
    data = _plot_data()
    assert data.selected_coordinate_frame is not None
    tidy = decomposition_plot.decomposition_tidy(
        data.selected_coordinate_frame,
        coordinate_labels=data.coordinate_labels,
    )

    coordinates = tidy.loc[tidy["component_kind"].eq("coordinate")]
    assert coordinates.groupby("id").size().to_dict() == {"selected-a": 5, "selected-b": 5}
    assert coordinates.groupby("id")["limiting"].sum().to_dict() == {"selected-a": 1, "selected-b": 1}
    assert not tidy.loc[~tidy["component_kind"].eq("coordinate"), "limiting"].any()


def test_behavior_decomposition_uses_msrb_symbols_and_compact_colorbar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(
        decomposition_plot,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: _plot_data(),
    )
    monkeypatch.setattr(decomposition_plot, "save_figure", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = _context(tmp_path, filename="decomposition.png", save_data=False)

    decomposition_plot.render_selected_decomposition(context, {})

    figure = captured.pop()
    labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    assert r"$x_{\min}$" in labels
    assert r"$S_{\mathrm{MSRB}}$" in labels
    assert figure.axes[-1].get_ylabel() == "Normalized behavior evidence"
    plt.Figure.clear(figure)


def test_behavior_decomposition_honors_pinned_symmetric_color_extent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(
        decomposition_plot,
        "load_multistate_response_behavior_plot_data",
        lambda _context, **_kwargs: _plot_data(),
    )
    monkeypatch.setattr(decomposition_plot, "save_figure", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = _context(tmp_path, filename="decomposition.png", save_data=False)

    decomposition_plot.render_selected_decomposition(context, {"color_extent": 0.05})

    figure = captured.pop()
    image = figure.axes[0].images[0]
    color_scale = context.artifact_metadata["notebook_view"]["color_scale"]
    assert image.norm.vmin == pytest.approx(-0.05)
    assert image.norm.vmax == pytest.approx(0.05)
    assert color_scale["extent"] == pytest.approx(0.05)
    assert color_scale["extend"] in {"both", "min", "max"}
    assert "values saturate" in color_scale["context"]
    assert figure.axes[-1]._colorbar.extend == color_scale["extend"]
    assert figure.axes[-1]._colorbar.extendrect is True
    plt.Figure.clear(figure)


def test_behavior_plot_registry_and_objective_family_routing_are_explicit() -> None:
    frontier = describe_plot_kind("multistate_response_behavior_frontier")
    decomposition = describe_plot_kind("multistate_response_behavior_selected_decomposition")
    for metadata in (frontier, decomposition):
        assert metadata["premise"]
        assert metadata["decision_value"]
        assert metadata["rationale"]
        assert metadata["alt_text"]
        assert metadata["non_claim_boundary"]
        assert metadata["tier"] == "decision"
        assert metadata["capability"]["objective_family"] == "multistate_response_behavior"
    assert frontier["notebook_view"]["adapter"] == "layered_scatter_v1"
    assert frontier["notebook_view"]["interactive"] == {
        "adapter": "three_axis_scatter_v1",
        "score_column": "behavior_score",
        "score_label": r"Behavior score, $S_{\mathrm{MSRB}}$",
        "prediction_sample_limit": 8_000,
        "sampling_method": "sha256_id_v1",
    }
    assert "labels/observed_events.parquet" in frontier["requires"]
    group, _rank = notebook_visual_group(
        {
            "kind": "multistate_response_behavior_frontier",
            "manifest": {"kind": "multistate_response_behavior_frontier", "metadata": frontier},
        }
    )
    assert group.key == "decision"

    _validate_plot_objective_compatibility(
        plot_name="behavior",
        plot_kind="multistate_response_behavior_frontier",
        plot_family="multistate_response_behavior",
        selection_view_id="stress-a",
        objective_name="multistate_response_behavior_v1",
        objective_family="multistate_response_behavior",
    )
    with pytest.raises(OpalError, match="requires objective family 'multistate_response_behavior'"):
        _validate_plot_objective_compatibility(
            plot_name="behavior",
            plot_kind="multistate_response_behavior_frontier",
            plot_family="multistate_response_behavior",
            selection_view_id="stress-a",
            objective_name="response_magnitude_feasibility_v1",
            objective_family="response_magnitude_feasibility",
        )
