"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_response_magnitude_feasibility_plots.py

Contract and rendering tests for response-separation decision plots.

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

from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.plots import response_magnitude_feasibility as plot_mod
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.plots._mpl_utils import NOTEBOOK_ANNOTATION_FONTSIZE
from dnadesign.opal.src.plots.candidate_annotations import (
    annotate_candidate_aliases,
    observed_candidate_display_labels,
    resolve_candidate_display_aliases,
)
from dnadesign.opal.src.plots.response_magnitude_feasibility_data import (
    FEASIBILITY_REF,
    OFF_MAGNITUDE_REF,
    ON_MAGNITUDE_REF,
    RESPONSE_REF,
    ResponseMagnitudeFeasibilityPlotData,
    load_response_magnitude_feasibility_plot_data,
    parse_response_magnitude_feasibility_channels,
    response_magnitude_feasibility_observed_frame,
    response_magnitude_feasibility_plot_frame,
)
from dnadesign.opal.src.registries.plots import describe_plot_kind
from dnadesign.opal.src.storage.artifacts import run_scoped_artifact_path


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.workdir = outputs_dir.parent


def _channels(*, feasibility: float, response: float, on: float, off: float) -> list[dict[str, object]]:
    return [
        {"name": f"ethanol/{FEASIBILITY_REF}", "value": feasibility},
        {"name": f"ethanol/{RESPONSE_REF}", "value": response},
        {"name": f"ethanol/{ON_MAGNITUDE_REF}", "value": on},
        {"name": f"ethanol/{OFF_MAGNITUDE_REF}", "value": off},
    ]


def _prediction_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_round": [0, 0, 0],
            "run_id": ["r0", "r0", "r0"],
            "id": ["selected-a", "selected-b", "pool-c"],
            "pred__score_channels": [
                _channels(feasibility=0.2, response=0.5, on=0.2, off=-0.3),
                _channels(feasibility=-0.2, response=-0.2, on=0.8, off=-0.4),
                _channels(feasibility=-0.3, response=0.6, on=0.4, off=0.3),
            ],
            "view__rank_competition": [1, 2, 3],
            "view__is_selected": [True, True, False],
        }
    )


def _plot_data() -> ResponseMagnitudeFeasibilityPlotData:
    calibration = {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 1.0,
        "on_magnitude_scale": 1.0,
        "off_magnitude_scale": 1.0,
    }
    events = _prediction_events()
    observed = pd.DataFrame(
        {
            "id": ["observed-a"],
            "observed_round": [0],
            "batch_id": ["batch-0"],
            "batch_key": ["batch-0"],
            "display_label": ["Observed A"],
            RESPONSE_REF: [0.1],
            ON_MAGNITUDE_REF: [0.3],
            OFF_MAGNITUDE_REF: [-0.2],
            "response_constraint_margin": [0.1],
            "on_magnitude_constraint_margin": [0.3],
            "off_magnitude_constraint_margin": [0.2],
            FEASIBILITY_REF: [0.1],
            "feasible": [True],
        }
    )
    return ResponseMagnitudeFeasibilityPlotData(
        frame=response_magnitude_feasibility_plot_frame(
            events,
            calibration=calibration,
            selection_view_id="ethanol",
        ),
        observed_frame=observed,
        calibration=calibration,
        state_ids=("00", "10", "01", "11"),
        target_mask=(0, 1, 0, 1),
        round_k=0,
        run_id="r0",
    )


def test_rmf_plot_loads_all_observations_from_run_pinned_event_snapshot(
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
            "y_obs": [[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]],
            "label_source_kind": ["usr_sidecar"],
        }
    ).write_parquet(labels_path)
    calibration = {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 1.0,
        "on_magnitude_scale": 1.0,
        "off_magnitude_scale": 1.0,
    }
    runs = pl.DataFrame(
        {
            "run_id": ["r0"],
            "as_of_round": [0],
            "objective__defs_json": [
                json.dumps(
                    [
                        {
                            "selection_view_id": "ethanol",
                            "objective_name": "response_magnitude_feasibility_v1",
                            "params": {
                                "state_ids": ["00", "10", "01", "11"],
                                "target_mask": [0, 1, 0, 1],
                                "calibration": calibration,
                            },
                        }
                    ]
                )
            ],
            "selection_views__defs_json": [
                json.dumps(
                    [
                        {
                            "selection_view_id": "ethanol",
                            "score_ref": "ethanol/feasibility_margin",
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
    events = pd.DataFrame(
        {
            "as_of_round": [0],
            "run_id": ["r0"],
            "id": ["pool-a"],
            "pred__score_channels": [_channels(feasibility=0.2, response=0.5, on=0.2, off=-0.3)],
            "view__rank_competition": [1],
            "view__is_selected": [True],
        }
    )
    monkeypatch.setattr(
        "dnadesign.opal.src.plots.response_magnitude_feasibility_data.read_runs",
        lambda _path: runs,
    )
    monkeypatch.setattr(
        "dnadesign.opal.src.plots.response_magnitude_feasibility_data.load_events",
        lambda *_args, **_kwargs: events,
    )
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(outputs_dir),
        rounds=[0],
        run_id=None,
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="review.png",
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.rmf-run-labels"),
        save_data=False,
    )

    data = load_response_magnitude_feasibility_plot_data(context)

    assert data.run_id == "r0"
    assert data.observed_frame["id"].tolist() == ["observed-a"]
    assert data.observed_frame["batch_id"].tolist() == ["pre-round-0"]
    assert data.observed_frame["display_label"].tolist() == ["Observed A"]
    assert context.data_paths["run_observed_events_parquet"] == labels_path.resolve()
    assert not (outputs_dir / "ledger" / "labels.parquet").exists()


def test_channel_parser_fails_on_missing_and_duplicate_channels() -> None:
    valid = _channels(feasibility=0.2, response=0.5, on=0.2, off=-0.3)
    assert parse_response_magnitude_feasibility_channels(
        valid,
        selection_view_id="ethanol",
    )[FEASIBILITY_REF] == pytest.approx(0.2)
    with pytest.raises(OpalError, match="missing score channels"):
        parse_response_magnitude_feasibility_channels(valid[:-1], selection_view_id="ethanol")
    with pytest.raises(OpalError, match="Duplicate score channel"):
        parse_response_magnitude_feasibility_channels(
            [*valid, valid[0]],
            selection_view_id="ethanol",
        )


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("view__is_selected", ["False", "True", "False"], "selected flags must be exact booleans"),
        ("view__rank_competition", [1.5, 2.0, 3.0], "selection ranks must be positive integers"),
    ],
)
def test_rmf_plot_frame_rejects_truthy_strings_and_fractional_ranks(
    column: str,
    values: list[object],
    message: str,
) -> None:
    events = _prediction_events()
    events[column] = values

    with pytest.raises(OpalError, match=message):
        response_magnitude_feasibility_plot_frame(
            events,
            calibration=_plot_data().calibration,
            selection_view_id="ethanol",
        )


def test_observed_rmf_frame_preserves_repeated_candidate_events_by_batch() -> None:
    labels = pd.DataFrame(
        {
            "id": ["candidate-a", "candidate-a"],
            "observed_round": [0, 1],
            "batch_id": ["batch-0", "batch-1"],
            "display_label": ["Candidate A", "Candidate A"],
            "y_obs": [
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                [-0.5, 1.5, -0.5, 1.5, -0.5, 1.5, -0.5, 1.5],
            ],
        }
    )

    observed = response_magnitude_feasibility_observed_frame(
        labels,
        target_mask=[0, 1, 0, 1],
        calibration=_plot_data().calibration,
    )

    assert observed[["id", "observed_round", "batch_id"]].to_dict(orient="records") == [
        {"id": "candidate-a", "observed_round": 0, "batch_id": "batch-0"},
        {"id": "candidate-a", "observed_round": 1, "batch_id": "batch-1"},
    ]
    assert observed["batch_key"].tolist() == ["batch-0", "batch-1"]
    assert observed["display_label"].tolist() == ["Candidate A", "Candidate A"]


def test_observed_rmf_frame_derives_a_round_batch_key_without_fabricating_source_batch() -> None:
    labels = pd.DataFrame(
        {
            "id": ["candidate-a"],
            "observed_round": [2],
            "batch_id": [None],
            "display_label": [None],
            "y_obs": [[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]],
        }
    )

    observed = response_magnitude_feasibility_observed_frame(
        labels,
        target_mask=[0, 1, 0, 1],
        calibration=_plot_data().calibration,
    )

    assert pd.isna(observed.loc[0, "batch_id"])
    assert observed.loc[0, "batch_key"] == "round-2"


def test_plot_frame_rejects_persisted_score_math_drift() -> None:
    data = _plot_data()
    events = data.frame.rename(
        columns={
            FEASIBILITY_REF: "persisted_feasibility",
            RESPONSE_REF: "persisted_response",
            ON_MAGNITUDE_REF: "persisted_on",
            OFF_MAGNITUDE_REF: "persisted_off",
        }
    )
    events["pred__score_channels"] = [
        _channels(
            feasibility=float(row.persisted_feasibility) + 0.1,
            response=float(row.persisted_response),
            on=float(row.persisted_on),
            off=float(row.persisted_off),
        )
        for row in events.itertuples(index=False)
    ]
    events = events.drop(
        columns=[
            "persisted_feasibility",
            "persisted_response",
            "persisted_on",
            "persisted_off",
            "response_constraint_margin",
            "on_magnitude_constraint_margin",
            "off_magnitude_constraint_margin",
            "feasible",
        ]
    )
    with pytest.raises(OpalError, match="canonical objective math"):
        response_magnitude_feasibility_plot_frame(
            events,
            calibration=data.calibration,
            selection_view_id="ethanol",
        )


@pytest.mark.parametrize(
    ("renderer", "filename", "tidy_columns"),
    [
        (
            plot_mod.render_frontier,
            "frontier.png",
            {"response_separation", "on_magnitude_floor", "off_constraint_margin", "feasibility_margin"},
        ),
        (
            plot_mod.render_constraint_decomposition,
            "decomposition.png",
            {"constraint", "signed_margin", "limiting", "feasible"},
        ),
    ],
)
def test_response_separation_plot_renderers_write_media_and_tidy_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renderer,
    filename: str,
    tidy_columns: set[str],
) -> None:
    monkeypatch.setattr(plot_mod, "load_response_magnitude_feasibility_plot_data", lambda _context: _plot_data())
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds=[0],
        run_id="r0",
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename=filename,
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.response-separation-plots"),
        save_data=True,
    )

    renderer(context, {})

    media = context.output_dir / filename
    tidy = context.output_dir / filename.replace(".png", ".csv")
    assert media.stat().st_size > 1_000
    assert tidy_columns <= set(pd.read_csv(tidy).columns)


def test_response_separation_plot_metadata_is_manuscript_explicit() -> None:
    for kind in (
        "response_magnitude_feasibility_frontier",
        "response_magnitude_feasibility_constraint_decomposition",
    ):
        meta = describe_plot_kind(kind)
        assert meta["premise"]
        assert meta["decision_value"]
        assert meta["rationale"]
        assert meta["alt_text"]
        assert meta["non_claim_boundary"]
        assert meta["tier"] == "decision"
        assert meta["capability"]["objective_family"] == "response_magnitude_feasibility"

    frontier_alt_text = describe_plot_kind("response_magnitude_feasibility_frontier")["alt_text"]
    assert "red is greater clearance" in frontier_alt_text
    assert "move right, up, and red" in frontier_alt_text


def test_rmf_frontier_metadata_names_run_pinned_observed_events() -> None:
    requirements = describe_plot_kind("response_magnitude_feasibility_frontier")["requires"]

    assert "labels/observed_events.parquet" in requirements
    assert "labels.parquet" not in requirements


def test_rmf_plots_show_target_boundaries_and_observed_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(plot_mod, "load_response_magnitude_feasibility_plot_data", lambda _context: _plot_data())
    monkeypatch.setattr(plot_mod, "_save", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds=[0],
        run_id="r0",
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="review.png",
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.rmf-plot-semantics"),
        save_data=False,
    )
    params = {
        "response_label": "ON-OFF response separation, $d_{response}$\nWindow mean log2(YFP / CFP)",
        "magnitude_label": ("Minimum target-ON fluorescence relative to pDual-10\n$f_{on}$, log2(YFP / OD600)"),
        "off_constraint_label": "Target-OFF clearance, $q_{off}$",
        "state_labels": {
            "00": "No stress",
            "10": "Ethanol",
            "01": "Ciprofloxacin",
            "11": "Both stresses",
        },
    }

    plot_mod.render_frontier(context, params)
    frontier = captured.pop()
    frontier_axis = frontier.axes[0]
    assert frontier._suptitle is None
    assert frontier_axis.get_title(loc="center") == (
        "RMF candidate constraint landscape · Ethanol view\n"
        "Target ON: Ethanol, Both stresses | OFF: No stress,\nCiprofloxacin"
    )
    assert frontier_axis.get_title(loc="left") == ""
    assert frontier_axis.title.get_fontsize() >= 14
    assert frontier_axis.get_xlabel() == "ON-OFF response separation, $d_{response}$\nWindow mean log2(YFP / CFP)"
    assert frontier_axis.get_ylabel() == (
        "Minimum target-ON fluorescence relative to pDual-10\n$f_{on}$, log2(YFP / OD600)"
    )
    assert frontier_axis.xaxis.label.get_fontsize() >= 11.5
    assert frontier_axis.yaxis.label.get_fontsize() >= 11.5
    assert min(tick.get_fontsize() for tick in frontier_axis.get_xticklabels()) >= 10
    assert min(tick.get_fontsize() for tick in frontier_axis.get_yticklabels()) >= 10
    assert not frontier_axis.spines["top"].get_visible()
    assert not frontier_axis.spines["right"].get_visible()
    assert frontier.axes[-1].get_ylabel() == ("Target-OFF clearance, $q_{off}$\nred = greater clearance; 0 = boundary")
    assert frontier.axes[-1].yaxis.label.get_fontsize() >= 10.5
    assert "Observed · Batch 0 (n=1)" in frontier_axis.get_legend_handles_labels()[1]
    assert frontier_axis.get_legend().get_bbox_to_anchor()._bbox.y0 < 0.0
    assert min(text.get_fontsize() for text in frontier_axis.get_legend().get_texts()) >= 9.5
    selected_collection = frontier_axis.collections[-1]
    assert selected_collection.get_array() is not None
    assert len(selected_collection.get_array()) == 2
    assert selected_collection.cmap.name == "RdBu_r"

    plot_mod.render_constraint_decomposition(context, params)
    decomposition = captured.pop()
    decomposition_axis = decomposition.axes[0]
    assert decomposition._suptitle is None
    decomposition_title = decomposition_axis.get_title(loc="center").replace("\n", " ")
    assert decomposition_title.startswith("Predicted RMF margins for selected candidates · Ethanol view ")
    assert "Target ON: Ethanol, Both stresses | OFF: No stress, Ciprofloxacin" in decomposition_title
    assert decomposition_axis.get_title(loc="left") == ""
    assert decomposition_axis.title.get_fontsize() >= 14
    assert "$S_{\\mathrm{RMF}}=\\min" in decomposition_axis.get_xlabel()
    assert "0 marks each configured boundary" in decomposition_axis.get_xlabel()
    assert "higher is better" not in decomposition_axis.get_xlabel().lower()
    assert decomposition_axis.xaxis.label.get_fontsize() >= 11.5
    assert decomposition_axis.yaxis.label.get_fontsize() >= 11.5
    assert min(tick.get_fontsize() for tick in decomposition_axis.get_xticklabels()) >= 10
    assert min(tick.get_fontsize() for tick in decomposition_axis.get_yticklabels()) >= 10
    assert [tick.get_text() for tick in decomposition_axis.get_xticklabels()] == [
        "$q_R$",
        "$q_{\\mathrm{ON}}$",
        "$q_{\\mathrm{OFF}}$",
        "$S_{\\mathrm{RMF}}$",
    ]
    assert len(decomposition_axis.patches) == 2
    assert decomposition_axis.images[0].cmap.name == "RdBu_r"

    plt.Figure.clear(frontier)
    plt.Figure.clear(decomposition)


def test_rmf_frontier_color_extent_includes_observed_values_outside_prediction_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _plot_data()
    data.observed_frame.loc[:, "off_magnitude_constraint_margin"] = -2.5
    captured: list[plt.Figure] = []
    monkeypatch.setattr(plot_mod, "load_response_magnitude_feasibility_plot_data", lambda _context: data)
    monkeypatch.setattr(plot_mod, "_save", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds=[0],
        run_id="r0",
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="review.png",
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.rmf-observed-color-extent"),
        save_data=False,
    )

    plot_mod.render_frontier(context, {})

    frontier = captured.pop()
    assert context.artifact_metadata["notebook_view"]["color_scale"] == {
        "center": 0.0,
        "extent": pytest.approx(2.5),
        "context": "red = greater clearance; 0 = configured boundary",
    }
    for collection in frontier.axes[0].collections[:3]:
        assert collection.norm.vmin == pytest.approx(-2.5)
        assert collection.norm.vmax == pytest.approx(2.5)
    plt.Figure.clear(frontier)


def test_rmf_plot_state_labels_must_match_objective_states() -> None:
    with pytest.raises(ValueError, match="must match state_ids exactly"):
        plot_mod._target_context(
            _plot_data(),
            {"state_labels": {"00": "No stress", "10": "Ethanol"}},
        )


def test_rmf_decomposition_marks_only_requirement_components_as_limiting() -> None:
    tidy = plot_mod._decomposition_tidy(_plot_data().frame.loc[lambda frame: frame["view__is_selected"]])

    assert tidy.groupby("id")["limiting"].sum().to_dict() == {"selected-a": 1, "selected-b": 1}
    assert not tidy.loc[tidy["constraint"].eq("feasibility_minimum"), "limiting"].any()


def _write_alias_records(path: Path) -> None:
    pl.DataFrame(
        {
            "id": [
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "cccccccccccccccccccccccccccccccccccccccc",
                "dddddddddddddddddddddddddddddddddddddddd",
            ],
            "usr_label__primary": ["SpyP", None, None, None],
            "usr_label__aliases": [None, ["Candidate B"], None, None],
        }
    ).write_parquet(path)


def test_candidate_display_aliases_use_only_projected_labels_then_short_ids(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    _write_alias_records(records_path)

    aliases = resolve_candidate_display_aliases(
        records_path,
        [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccc",
            "dddddddddddddddddddddddddddddddddddddddd",
        ],
    )

    assert aliases == {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "SpyP",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "Candidate B",
        "cccccccccccccccccccccccccccccccccccccccc": "cccccc…ccccc",
        "dddddddddddddddddddddddddddddddddddddddd": "dddddd…ddddd",
    }
    assert all("__" not in value for value in aliases.values())


def test_candidate_display_aliases_disambiguate_shared_id_prefixes(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pl.DataFrame(
        {
            "id": ["abcdef111111", "abcdef222222"],
            "usr_label__primary": ["Same", "Same"],
        }
    ).write_parquet(records_path)

    aliases = resolve_candidate_display_aliases(records_path, ["abcdef111111", "abcdef222222"])

    assert aliases == {
        "abcdef111111": "Same · abcdef111111",
        "abcdef222222": "Same · abcdef222222",
    }
    assert len(set(aliases.values())) == 2


def test_observed_display_labels_disambiguate_shared_id_prefixes() -> None:
    observed = pd.DataFrame(
        {
            "id": ["abcdef111111", "abcdef222222", "abcdef111111"],
            "display_label": ["Same", "Same", "Same"],
        }
    )

    labels = observed_candidate_display_labels(observed, fallbacks={})

    assert labels.tolist() == ["Same · abcdef111111", "Same · abcdef222222", "Same · abcdef111111"]


def test_candidate_annotations_do_not_distort_constrained_layout() -> None:
    figure, axis = plt.subplots(figsize=(7.2, 7.2), layout="constrained")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)

    annotations = annotate_candidate_aliases(
        axis,
        pd.DataFrame({"id": ["a", "b"], "x": [-0.4, 0.4], "y": [-0.2, 0.2]}),
        {"a": "Candidate A", "b": "Candidate B"},
        x_column="x",
        y_column="y",
    )

    assert annotations
    assert all(not annotation.get_in_layout() for annotation in annotations)
    plt.close(figure)


def test_candidate_annotations_use_two_nonoverlapping_lanes_for_dense_labels() -> None:
    figure, axis = plt.subplots(figsize=(4.0, 4.0), layout="constrained")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    count = 33
    frame = pd.DataFrame(
        {
            "id": [f"candidate-{index:02d}" for index in range(count)],
            "x": np.linspace(-0.6, 0.6, count),
            "y": np.zeros(count),
        }
    )
    aliases = {candidate_id: f"C{index:02d}" for index, candidate_id in enumerate(frame["id"])}

    annotations = annotate_candidate_aliases(
        axis,
        frame,
        aliases,
        x_column="x",
        y_column="y",
        font_size=NOTEBOOK_ANNOTATION_FONTSIZE,
        max_lanes=2,
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = axis.get_window_extent(renderer=renderer)
    boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
    assert {annotation.get_ha() for annotation in annotations} == {"left", "right"}
    assert all(box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1 for box in boxes)
    assert all(box.y0 >= axes_box.y0 and box.y1 <= axes_box.y1 for box in boxes)
    for left_index, left_box in enumerate(boxes):
        for right_box in boxes[left_index + 1 :]:
            assert not left_box.overlaps(right_box)
    plt.close(figure)


def test_candidate_annotations_flip_at_axes_edges_without_covering_anchor_points() -> None:
    figure, axis = plt.subplots(figsize=(4.0, 4.0), layout="constrained")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    frame = pd.DataFrame(
        {
            "id": ["left", "right"],
            "x": [-0.98, 0.98],
            "y": [-0.4, 0.4],
        }
    )

    annotations = annotate_candidate_aliases(
        axis,
        frame,
        {"left": "Left boundary", "right": "Right boundary"},
        x_column="x",
        y_column="y",
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    anchor_pixels = axis.transData.transform(frame[["x", "y"]].to_numpy(dtype=float))
    boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
    assert [annotation.get_ha() for annotation in annotations] == ["left", "right"]
    assert all(not box.contains(*anchor) for box, anchor in zip(boxes, anchor_pixels, strict=True))
    plt.close(figure)


def test_candidate_display_aliases_treat_alias_columns_as_optional_but_require_ids(tmp_path: Path) -> None:
    missing_columns = tmp_path / "missing-columns.parquet"
    pl.DataFrame({"id": ["candidate-a"]}).write_parquet(missing_columns)
    assert resolve_candidate_display_aliases(missing_columns, ["candidate-a"]) == {"candidate-a": "candidate-a"}

    missing_id = tmp_path / "missing-id.parquet"
    pl.DataFrame({"usr_label__primary": ["Candidate A"]}).write_parquet(missing_id)
    with pytest.raises(OpalError, match="candidate ID column 'id'"):
        resolve_candidate_display_aliases(missing_id, ["candidate-a"])

    records_path = tmp_path / "records.parquet"
    _write_alias_records(records_path)
    with pytest.raises(OpalError, match="missing requested candidate IDs"):
        resolve_candidate_display_aliases(records_path, ["not-in-records"])


@pytest.mark.parametrize("candidate_id", [" candidate-a", "candidate-a "])
def test_candidate_display_aliases_reject_identity_whitespace(tmp_path: Path, candidate_id: str) -> None:
    records_path = tmp_path / "records.parquet"
    pl.DataFrame({"id": ["candidate-a"]}).write_parquet(records_path)

    with pytest.raises(OpalError, match="leading or trailing whitespace"):
        resolve_candidate_display_aliases(records_path, [candidate_id])


@pytest.mark.parametrize("candidate_id", [None, 7])
def test_candidate_display_aliases_reject_non_string_identity(tmp_path: Path, candidate_id: object) -> None:
    records_path = tmp_path / "records.parquet"
    pl.DataFrame({"id": ["candidate-a"]}).write_parquet(records_path)

    with pytest.raises(OpalError, match="exact strings"):
        resolve_candidate_display_aliases(records_path, [candidate_id])


def test_static_frontier_does_not_require_candidate_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(plot_mod, "load_response_magnitude_feasibility_plot_data", lambda _context: _plot_data())
    monkeypatch.setattr(plot_mod, "_save", lambda _context, figure: captured.append(figure))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds=[0],
        run_id="r0",
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="review.png",
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.rmf-plot-no-aliases"),
        save_data=False,
    )

    plot_mod.render_frontier(context, {})

    figure = captured.pop()
    assert not figure.axes[0].texts
    plt.Figure.clear(figure)
