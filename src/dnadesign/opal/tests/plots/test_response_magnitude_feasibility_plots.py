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
import pandas as pd
import polars as pl
import pytest
import yaml

from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.plots import response_magnitude_feasibility as plot_mod
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.plots.response_magnitude_feasibility_aliases import (
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
    response_magnitude_feasibility_plot_frame,
)
from dnadesign.opal.src.registries.plots import describe_plot_kind


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


def _plot_data() -> ResponseMagnitudeFeasibilityPlotData:
    calibration = {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 1.0,
        "on_magnitude_scale": 1.0,
        "off_magnitude_scale": 1.0,
    }
    events = pd.DataFrame(
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
    observed = pd.DataFrame(
        {
            "id": ["observed-a"],
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


def test_rmf_plot_loads_observations_from_run_pinned_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs_dir = tmp_path / "outputs"
    labels_path = outputs_dir / "rounds" / "round_0" / "labels" / "labels_used.parquet"
    labels_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["r0"],
            "as_of_round": [0],
            "observed_round": [0],
            "id": ["observed-a"],
            "sequence": ["ACGT"],
            "y_obs": [[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]],
            "src": ["training_snapshot"],
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
                    "labels/labels_used.parquet": [
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
    assert context.data_paths["run_labels_used_parquet"] == labels_path.resolve()
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
    assert frontier_axis.get_title() == ("Target ON: Ethanol, Both stresses | OFF: No stress, Ciprofloxacin")
    assert frontier_axis.get_xlabel() == "ON-OFF response separation, $d_{response}$\nWindow mean log2(YFP / CFP)"
    assert frontier_axis.get_ylabel() == (
        "Minimum target-ON fluorescence relative to pDual-10\n$f_{on}$, log2(YFP / OD600)"
    )
    assert frontier.axes[-1].get_ylabel() == "Target-OFF clearance, $q_{off}$\n0 = boundary"
    assert "Measured (n=1)" in frontier_axis.get_legend_handles_labels()[1]
    selected_collection = frontier_axis.collections[-1]
    assert selected_collection.get_array() is not None
    assert len(selected_collection.get_array()) == 2

    plot_mod.render_constraint_decomposition(context, params)
    decomposition = captured.pop()
    decomposition_axis = decomposition.axes[0]
    assert "Target ON: Ethanol, Both stresses | OFF: No stress, Ciprofloxacin" in (decomposition_axis.get_title())
    assert "$S_{\\mathrm{RMF}}=\\min" in decomposition_axis.get_xlabel()
    assert "0 marks each configured boundary" in decomposition_axis.get_xlabel()
    assert "higher is better" not in decomposition_axis.get_xlabel().lower()
    assert [tick.get_text() for tick in decomposition_axis.get_xticklabels()] == [
        "$q_{\\mathrm{response}}$",
        "$q_{\\mathrm{on}}$",
        "$q_{\\mathrm{off}}$",
        "$S_{\\mathrm{RMF}}$",
    ]
    assert len(decomposition_axis.patches) == 2

    plt.Figure.clear(frontier)
    plt.Figure.clear(decomposition)


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
            "densegen__plan": [
                None,
                "ethanol__sig35=b",
                "ethanol_ciprofloxacin__sig35=e",
                "unrecognized_plan_slug",
            ],
        }
    ).write_parquet(path)


def test_candidate_display_aliases_prefer_human_labels_and_never_emit_plan_slugs(tmp_path: Path) -> None:
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
        "cccccccccccccccccccccccccccccccccccccccc": "EtOH + Cipro · σ35e · cccccc",
        "dddddddddddddddddddddddddddddddddddddddd": "dddddd…ddddd",
    }
    assert all("__" not in value and "_slug" not in value for value in aliases.values())


def test_candidate_display_aliases_reject_missing_contract_columns_and_ids(tmp_path: Path) -> None:
    missing_columns = tmp_path / "missing-columns.parquet"
    pl.DataFrame({"id": ["candidate-a"]}).write_parquet(missing_columns)
    with pytest.raises(OpalError, match="missing required alias columns"):
        resolve_candidate_display_aliases(missing_columns, ["candidate-a"])

    records_path = tmp_path / "records.parquet"
    _write_alias_records(records_path)
    with pytest.raises(OpalError, match="missing requested candidate IDs"):
        resolve_candidate_display_aliases(records_path, ["not-in-records"])


def test_annotated_frontier_repels_aliases_and_keeps_them_inside_the_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _plot_data()
    selected = data.frame["view__is_selected"]
    data.frame.loc[selected, RESPONSE_REF] = [0.2, 0.201]
    data.frame.loc[selected, ON_MAGNITUDE_REF] = [0.2, 0.201]
    records_path = tmp_path / "records.parquet"
    pl.DataFrame(
        {
            "id": ["selected-a", "selected-b"],
            "usr_label__primary": ["Promoter A", "Promoter B"],
            "usr_label__aliases": [None, None],
            "densegen__plan": [None, None],
        }
    ).write_parquet(records_path)
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
        data_paths={"records": records_path},
        output_dir=tmp_path / "plots",
        filename="review.png",
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.rmf-plot-alias-layout"),
        save_data=False,
    )

    plot_mod.render_frontier(context, {"annotate_selected_aliases": True})

    figure = captured.pop()
    axis = figure.axes[0]
    figure.canvas.draw()
    annotations = [text for text in axis.texts if text.get_text() in {"Promoter A", "Promoter B"}]
    assert len(annotations) == 2
    renderer = figure.canvas.get_renderer()
    axes_box = axis.get_window_extent(renderer=renderer)
    annotation_boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
    assert all(annotation.get_clip_on() for annotation in annotations)
    assert all(
        axes_box.x0 <= box.x0 <= box.x1 <= axes_box.x1 and axes_box.y0 <= box.y0 <= box.y1 <= axes_box.y1
        for box in annotation_boxes
    )
    assert not annotation_boxes[0].overlaps(annotation_boxes[1])
    plt.Figure.clear(figure)


def test_unannotated_frontier_does_not_require_candidate_records(
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

    plot_mod.render_frontier(context, {"annotate_selected_aliases": False})

    figure = captured.pop()
    assert not figure.axes[0].texts
    plt.Figure.clear(figure)


def test_secg_rmf_plot_config_declares_paired_alias_variants_and_symbolic_labels() -> None:
    config_path = Path(__file__).parents[2] / "campaigns" / "secg_rmf_greedy" / "configs" / "plots.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    presets = payload.get("plot_presets", {})
    frontiers = [
        entry
        for entry in payload["plots"]
        if (entry.get("kind") or presets.get(entry.get("preset"), {}).get("kind"))
        == "response_magnitude_feasibility_frontier"
    ]

    assert [entry["name"] for entry in frontiers] == [
        "rmf_candidate_frontier",
        "rmf_candidate_frontier_with_aliases",
    ]
    merged_params = [
        {**presets.get(entry.get("preset"), {}).get("params", {}), **entry.get("params", {})} for entry in frontiers
    ]
    assert {params["surface_label"] for params in merged_params} == {"RMF candidate frontier"}
    assert {
        (
            params["notebook_toggle"]["id"],
            params["notebook_toggle"]["label"],
            params["notebook_toggle"]["value"],
        )
        for params in merged_params
    } == {
        ("candidate_aliases", "Show candidate aliases", False),
        ("candidate_aliases", "Show candidate aliases", True),
    }
    assert [params["annotate_selected_aliases"] for params in merged_params] == [False, True]
    assert all(r"$d_{\mathrm{response}}$" in params["response_label"] for params in merged_params)
    assert all(r"$f_{\mathrm{on}}$" in params["magnitude_label"] for params in merged_params)
    assert all(r"$q_{\mathrm{off}}$" in params["off_constraint_label"] for params in merged_params)

    decomposition = next(
        entry
        for entry in payload["plots"]
        if entry.get("kind") == "response_magnitude_feasibility_constraint_decomposition"
    )
    assert decomposition["params"]["candidate_label_mode"] == "alias"
