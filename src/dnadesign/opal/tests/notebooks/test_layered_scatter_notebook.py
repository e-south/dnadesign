"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_layered_scatter_notebook.py

Test contracts for generic notebook layered-scatter review controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import io
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.analysis.notebook_components import layered_scatter_rendering
from dnadesign.opal.src.analysis.notebook_components.layered_scatter import (
    build_notebook_layered_scatter_contract,
    build_notebook_layered_scatter_controls,
    filter_notebook_layered_scatter_rows,
    render_notebook_layered_scatter_image,
)
from dnadesign.opal.src.analysis.notebook_components.layered_scatter_rendering import (
    render_layered_scatter_figure,
)
from dnadesign.opal.src.analysis.notebook_components.plot_scopes import select_notebook_plot_scope
from dnadesign.opal.src.analysis.notebook_set_template import layered_scatter_cells as layered_scatter_cell_template
from dnadesign.opal.src.analysis.notebook_set_template.layered_scatter_cells import (
    render_layered_scatter_cells,
)
from dnadesign.opal.src.core.utils import file_sha256
from dnadesign.opal.src.plots import candidate_annotations
from dnadesign.opal.src.registries.plots import describe_plot_kind


def _choice(
    tmp_path: Path,
    *,
    filename: str = "frontier.csv",
    ids: tuple[str, str, str, str] = ("pool", "selected", "observed-a", "observed-b"),
    observed_batch_ids: tuple[str, str] = ("pre_round0_response_corpus_4_8h_v1", "batch_1"),
) -> dict[str, object]:
    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    tidy_path = plot_root / filename
    pd.DataFrame(
        {
            "id": ids,
            "record_kind": ["prediction", "prediction", "observed_label", "observed_label"],
            "selected": [False, True, False, False],
            "batch_id": [None, None, *observed_batch_ids],
            "batch_key": [None, None, *observed_batch_ids],
            "display_label": [None, "Candidate S", "SpyP", "sulAp"],
            "response_separation": [0.1, 0.2, 0.3, 0.4],
            "on_magnitude_floor": [1.1, 1.2, 1.3, 1.4],
            "off_constraint_margin": [-0.1, 0.2, 0.3, -0.4],
        }
    ).to_csv(tidy_path, index=False)
    return {
        "label": "RMF candidate frontier",
        "workdir": str(tmp_path),
        "manifest": {
            "kind": "response_magnitude_feasibility_frontier",
            "run_id": "r0",
            "rounds": [0],
            "selection_view_id": "ethanol",
            "tidy_csv": str(tidy_path),
            "outputs": [
                {
                    "role": "tidy_csv",
                    "path": str(tidy_path),
                    "sha256": file_sha256(tidy_path),
                }
            ],
            "metadata": describe_plot_kind("response_magnitude_feasibility_frontier"),
            "artifact_metadata": {
                "notebook_view": {
                    "title": "RMF candidate constraint landscape",
                    "context": "Ethanol target · ON: 10, 11 · OFF: 00, 01",
                    "x_label": r"Response separation, $d_R$",
                    "y_label": r"ON fluorescence, $f_{\mathrm{ON}}$",
                    "color_label": r"OFF clearance, $q_{\mathrm{OFF}}$",
                    "reference_lines": {
                        "x": [{"value": 0.0, "label": "Configured response boundary"}],
                        "y": [{"value": 0.0, "label": "Configured ON-expression boundary"}],
                    },
                    "color_scale": {
                        "center": 0.0,
                        "extent": 1.0,
                        "context": "red = greater clearance; 0 = configured boundary",
                    },
                    "x_limits": [-0.5, 0.8],
                    "y_limits": [-0.5, 1.8],
                }
            },
        },
    }


def _behavior_choice(tmp_path: Path, *, filename: str, selection_view_id: str) -> dict[str, object]:
    choice = _choice(tmp_path, filename=filename)
    manifest = choice["manifest"]
    assert isinstance(manifest, dict)
    tidy_path = Path(str(manifest["tidy_csv"]))
    tidy = pd.read_csv(tidy_path).rename(
        columns={
            "response_separation": "response_family_score",
            "on_magnitude_floor": "on_signal_family_score",
            "off_constraint_margin": "off_signal_suppression_family_score",
        }
    )
    tidy["behavior_score"] = tidy[
        [
            "response_family_score",
            "on_signal_family_score",
            "off_signal_suppression_family_score",
        ]
    ].min(axis=1)
    tidy.to_csv(tidy_path, index=False)
    manifest["kind"] = "multistate_response_behavior_frontier"
    manifest["selection_view_id"] = selection_view_id
    manifest["metadata"] = describe_plot_kind("multistate_response_behavior_frontier")
    outputs = manifest["outputs"]
    assert isinstance(outputs, list) and isinstance(outputs[0], dict)
    outputs[0]["sha256"] = file_sha256(tidy_path)
    runtime = manifest["artifact_metadata"]["notebook_view"]
    runtime.update(
        {
            "title": "Multistate behavior family landscape",
            "x_label": "Response family score",
            "y_label": "ON-signal family score",
            "color_label": "OFF-signal-suppression family score",
            "reference_lines": {"x": [], "y": []},
            "color_scale": {
                "center": 0.0,
                "extent": 1.0,
                "context": "red = stronger suppression; 0 = reference direction; not feasibility",
            },
        }
    )
    return choice


def test_layered_scatter_contract_discovers_exact_observed_batches(tmp_path: Path) -> None:
    contract = build_notebook_layered_scatter_contract(_choice(tmp_path))

    assert contract is not None
    assert contract["adapter"] == "layered_scatter_v1"
    assert [item["id"] for item in contract["observed_batches"]] == [
        "batch_1",
        "pre_round0_response_corpus_4_8h_v1",
    ]
    assert [item["label"] for item in contract["observed_batches"]] == [
        "Batch 1",
        "Pre-round 0",
    ]
    assert isinstance(contract["rows"], pd.DataFrame)
    assert contract["active_selection_round"] == 0
    assert contract["selection_rounds"] == [0]
    assert list(contract["rows"].columns) == [
        "id",
        "record_kind",
        "selected",
        "batch_key",
        "display_label",
        "response_separation",
        "on_magnitude_floor",
        "off_constraint_margin",
    ]


def test_layered_scatter_render_reuses_the_verified_contract_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Mo:
        @staticmethod
        def image(data: bytes, **kwargs):
            return {"data": data, **kwargs}

    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None

    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: pytest.fail("unexpected CSV reread"))
    rendered = render_notebook_layered_scatter_image(
        choice,
        contract=contract,
        state={
            "show_prediction_pool": True,
            "show_selected": True,
            "observed_batches": ["batch_1"],
            "label_scope": "both",
        },
        mo=_Mo(),
    )

    assert rendered["data"].startswith(b"\x89PNG")


def test_layered_scatter_contract_rejects_more_batches_than_the_marker_vocabulary(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    manifest = choice["manifest"]
    assert isinstance(manifest, dict)
    tidy_path = Path(str(manifest["tidy_csv"]))
    prediction_rows = pd.read_csv(tidy_path).iloc[:2].copy()
    observed_rows = pd.DataFrame(
        {
            "id": [f"observed-{index}" for index in range(13)],
            "record_kind": ["observed_label"] * 13,
            "selected": [False] * 13,
            "batch_id": [f"batch_{index}" for index in range(13)],
            "batch_key": [f"batch_{index}" for index in range(13)],
            "display_label": [f"Candidate {index}" for index in range(13)],
            "response_separation": np.linspace(0.1, 0.3, 13),
            "on_magnitude_floor": np.linspace(1.1, 1.3, 13),
            "off_constraint_margin": np.linspace(-0.2, 0.2, 13),
        }
    )
    pd.concat([prediction_rows, observed_rows], ignore_index=True).to_csv(tidy_path, index=False)
    outputs = manifest["outputs"]
    assert isinstance(outputs, list) and isinstance(outputs[0], dict)
    outputs[0]["sha256"] = file_sha256(tidy_path)

    with pytest.raises(ValueError, match="marker universe contains 13 batches; at most 12 are supported"):
        build_notebook_layered_scatter_contract(choice)


def test_layered_scatter_memory_is_campaign_and_plot_scoped(tmp_path: Path) -> None:
    first = build_notebook_layered_scatter_contract(_choice(tmp_path / "campaign-a"))
    second = build_notebook_layered_scatter_contract(_choice(tmp_path / "campaign-b"))

    assert first is not None and second is not None
    assert first["key"] != second["key"]
    assert first["key"].startswith("layered_scatter_v1:")


def test_layered_scatter_memory_persists_across_selection_views(tmp_path: Path) -> None:
    ethanol_choice = _choice(tmp_path)
    ciprofloxacin_choice = _choice(tmp_path, filename="frontier_ciprofloxacin.csv")
    ciprofloxacin_choice["manifest"]["selection_view_id"] = "ciprofloxacin"

    ethanol = build_notebook_layered_scatter_contract(ethanol_choice)
    ciprofloxacin = build_notebook_layered_scatter_contract(ciprofloxacin_choice)

    assert ethanol is not None and ciprofloxacin is not None
    assert ethanol["key"] == ciprofloxacin["key"]


def test_layered_scatter_memory_persists_across_round_scopes(tmp_path: Path) -> None:
    round_zero_choice = _choice(tmp_path, filename="frontier_r0.csv")
    round_one_choice = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero_choice["manifest"].update({"plot_id": "frontier_r0", "run_id": "r0", "rounds": [0]})
    round_one_choice["manifest"].update({"plot_id": "frontier_r1", "run_id": "r1", "rounds": [1]})

    round_zero = build_notebook_layered_scatter_contract(round_zero_choice)
    round_one = build_notebook_layered_scatter_contract(round_one_choice)

    assert round_zero is not None and round_one is not None
    assert round_zero["key"] == round_one["key"]


def test_behavior_layered_scatter_memory_and_reference_semantics_are_view_neutral(tmp_path: Path) -> None:
    first_choice = _behavior_choice(tmp_path, filename="behavior_a.csv", selection_view_id="factor-a")
    second_choice = _behavior_choice(tmp_path, filename="behavior_b.csv", selection_view_id="factor-b")
    first = build_notebook_layered_scatter_contract(first_choice)
    second = build_notebook_layered_scatter_contract(second_choice)

    assert first is not None and second is not None
    assert first["key"] == second["key"]
    filtered = filter_notebook_layered_scatter_rows(
        pd.read_csv(first["tidy_path"]),
        contract=first,
        state={"observed_batches": ["batch_1"]},
    )
    figure = render_layered_scatter_figure(filtered, contract=first)
    assert not figure.axes[0].lines
    assert figure.axes[-1].get_ylabel() == "OFF-signal-suppression family score"
    plt.close(figure)


def test_layered_scatter_wraps_long_title_and_target_context_inside_square_viewport(tmp_path: Path) -> None:
    choice = _behavior_choice(tmp_path, filename="behavior_long.csv", selection_view_id="factor-a")
    runtime = choice["manifest"]["artifact_metadata"]["notebook_view"]
    runtime["title"] = (
        "Multistate behavior family landscape for a deliberately long publication-facing active-learning view"
    )
    runtime["context"] = (
        "Target ON: stress condition alpha, stress condition beta, recovery condition gamma | "
        "OFF: baseline condition, vehicle condition, unrelated stress condition delta"
    )
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    filtered = filter_notebook_layered_scatter_rows(
        pd.read_csv(contract["tidy_path"]),
        contract=contract,
        state={"observed_batches": ["batch_1"]},
    )

    figure = render_layered_scatter_figure(filtered, contract=contract)
    try:
        figure.canvas.draw()
        title = figure.axes[0].title
        subtitle = next(text for text in figure.axes[0].texts if text.get_gid() == "notebook-plot-subtitle")
        title_box = title.get_window_extent(renderer=figure.canvas.get_renderer())
        subtitle_box = subtitle.get_window_extent(renderer=figure.canvas.get_renderer())
        assert len(title.get_text().splitlines()) >= 2
        assert len(subtitle.get_text().splitlines()) >= 2
        assert title_box.x0 >= figure.bbox.x0
        assert title_box.x1 <= figure.bbox.x1
        assert subtitle_box.x0 >= figure.bbox.x0
        assert subtitle_box.x1 <= figure.bbox.x1
        color_label_box = figure.axes[-1].yaxis.label.get_window_extent(renderer=figure.canvas.get_renderer())
        legend_box = figure.axes[0].get_legend().get_window_extent(renderer=figure.canvas.get_renderer())
        assert not title_box.overlaps(subtitle_box)
        assert not title_box.overlaps(color_label_box)
        assert not subtitle_box.overlaps(color_label_box)
        assert not legend_box.overlaps(color_label_box)
    finally:
        plt.close(figure)


def test_layered_scatter_memory_stays_stable_across_evidence_scopes_but_not_plots(tmp_path: Path) -> None:
    baseline_choice = _choice(tmp_path)
    run_choice = _choice(tmp_path, filename="frontier_run.csv")
    run_choice["manifest"]["run_id"] = "r1"
    round_choice = _choice(tmp_path, filename="frontier_round.csv")
    round_choice["manifest"]["rounds"] = [1]
    plot_choice = _choice(tmp_path, filename="frontier_plot.csv")
    plot_choice["manifest"]["kind"] = "another_layered_scatter"

    contracts = [
        build_notebook_layered_scatter_contract(choice)
        for choice in (baseline_choice, run_choice, round_choice, plot_choice)
    ]

    assert all(contract is not None for contract in contracts)
    keys = [contract["key"] for contract in contracts if contract is not None]
    assert keys[0] == keys[1] == keys[2]
    assert keys[3] != keys[0]


def test_layered_scatter_batch_labels_preserve_meaningful_acronyms(tmp_path: Path) -> None:
    contract = build_notebook_layered_scatter_contract(
        _choice(tmp_path, observed_batch_ids=("batch_2_RMF", "batch_2_RMF"))
    )

    assert contract is not None
    assert contract["observed_batches"] == [{"id": "batch_2_RMF", "label": "Batch 2 RMF"}]


def test_layered_scatter_disambiguates_pretty_batch_label_collisions(tmp_path: Path) -> None:
    contract = build_notebook_layered_scatter_contract(_choice(tmp_path, observed_batch_ids=("batch_0", "batch-0")))

    assert contract is not None
    assert contract["observed_batches"] == [
        {"id": "batch-0", "label": "Batch 0 · batch-0"},
        {"id": "batch_0", "label": "Batch 0 · batch_0"},
    ]


def test_layered_scatter_layers_filter_independently(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    rows = pd.read_csv(contract["tidy_path"])

    filtered = filter_notebook_layered_scatter_rows(
        rows,
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": True,
            "observed_batches": ["batch_1"],
            "label_scope": "both",
        },
    )

    assert filtered["id"].tolist() == ["selected", "observed-b"]
    assert filtered.attrs["annotate_row_positions"] == (0, 1)


@pytest.mark.parametrize(
    ("label_scope", "expected_kind", "expected_label"),
    [
        ("selected", "prediction", "Candidate S"),
        ("observed", "observed_label", "SpyP"),
    ],
)
def test_layered_scatter_annotation_scope_is_row_stable_for_overlapping_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label_scope: str,
    expected_kind: str,
    expected_label: str,
) -> None:
    choice = _choice(
        tmp_path,
        ids=("pool", "shared", "shared", "observed-b"),
        observed_batch_ids=("batch_0", "batch_1"),
    )
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    filtered = filter_notebook_layered_scatter_rows(
        pd.read_csv(contract["tidy_path"]),
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": True,
            "observed_batches": ["batch_0"],
            "label_scope": label_scope,
        },
    )
    captured: list[tuple[list[str], list[str], int]] = []

    def _capture_annotations(_ax, frame, aliases, **_kwargs) -> None:
        captured.append(
            (
                frame["record_kind"].astype(str).tolist(),
                list(aliases.values()),
                int(_kwargs["max_lanes"]),
            )
        )

    monkeypatch.setattr(
        candidate_annotations,
        "annotate_candidate_aliases",
        _capture_annotations,
    )
    figure = layered_scatter_rendering.render_layered_scatter_figure(filtered, contract=contract)
    try:
        assert captured == [([expected_kind], [expected_label], 2)]
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_layered_scatter_observed_markers_are_invariant_to_hidden_batches(tmp_path: Path) -> None:
    choice = _choice(tmp_path, observed_batch_ids=("batch_0", "batch_1"))
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    rows = pd.read_csv(contract["tidy_path"])
    full = filter_notebook_layered_scatter_rows(
        rows,
        contract=contract,
        state={"observed_batches": ["batch_0", "batch_1"]},
    )
    batch_one = filter_notebook_layered_scatter_rows(
        rows,
        contract=contract,
        state={"observed_batches": ["batch_1"]},
    )
    full_figure = layered_scatter_rendering.render_layered_scatter_figure(full, contract=contract)
    batch_one_figure = layered_scatter_rendering.render_layered_scatter_figure(batch_one, contract=contract)
    try:
        np.testing.assert_allclose(
            _marker_vertices(full_figure, "Observed · Batch 1"),
            _marker_vertices(batch_one_figure, "Observed · Batch 1"),
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(full_figure)
        plt.close(batch_one_figure)


def test_layered_scatter_rejects_unknown_batches_and_normalizes_hidden_annotation_layers(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    rows = pd.read_csv(contract["tidy_path"])

    with pytest.raises(ValueError, match="Unknown observed batch"):
        filter_notebook_layered_scatter_rows(
            rows,
            contract=contract,
            state={"observed_batches": ["not-a-batch"]},
        )
    filtered = filter_notebook_layered_scatter_rows(
        rows,
        contract=contract,
        state={"show_selected": False, "label_scope": "selected"},
    )

    assert "selected" in filtered["id"].tolist()
    assert filtered.attrs["effective_label_scope"] == "none"
    assert filtered.attrs["annotate_row_positions"] == ()


def test_prediction_pool_remains_complete_when_selection_highlight_is_hidden(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    filtered = filter_notebook_layered_scatter_rows(
        pd.read_csv(contract["tidy_path"]),
        contract=contract,
        state={
            "show_prediction_pool": True,
            "show_selected": False,
            "observed_batches": [],
        },
    )

    assert filtered["id"].tolist() == ["pool", "selected"]
    figure = render_layered_scatter_figure(filtered, contract=contract)
    try:
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert labels == ["Predicted pool (n=2)"]
    finally:
        plt.close(figure)


def test_layered_scatter_allows_a_bounded_empty_state(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    rows = pd.read_csv(contract["tidy_path"])

    filtered = filter_notebook_layered_scatter_rows(
        rows,
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": False,
            "observed_batches": [],
            "label_scope": "both",
        },
    )

    assert filtered.empty
    assert filtered.attrs["effective_label_scope"] == "none"
    assert filtered.attrs["empty_state"] == "all_layers_hidden"
    assert filtered.attrs["annotate_row_positions"] == ()

    class _Mo:
        @staticmethod
        def md(value: str):
            return {"markdown": value}

        @staticmethod
        def callout(value, *, kind: str):
            return {"callout": value, "kind": kind}

    rendered = render_notebook_layered_scatter_image(
        choice,
        state={
            "show_prediction_pool": False,
            "show_selected": False,
            "observed_batches": [],
            "label_scope": "both",
        },
        mo=_Mo(),
    )
    assert rendered["kind"] == "neutral"
    assert "No scatter layers are visible" in rendered["callout"]["markdown"]


def test_rmf_frontier_registers_generic_notebook_adapter() -> None:
    metadata = describe_plot_kind("response_magnitude_feasibility_frontier")

    assert metadata["notebook_view"]["adapter"] == "layered_scatter_v1"
    assert metadata["notebook_view"]["batch_column"] == "batch_key"
    assert metadata["notebook_view"]["selection_column"] == "selected"


def test_layered_scatter_renderer_returns_a_publication_image(tmp_path: Path) -> None:
    class _Mo:
        @staticmethod
        def image(data: bytes, **kwargs):
            return {"data": data, **kwargs}

    rendered = render_notebook_layered_scatter_image(
        _choice(tmp_path),
        state={
            "show_prediction_pool": True,
            "show_selected": True,
            "observed_batches": ["batch_1"],
            "label_scope": "both",
        },
        mo=_Mo(),
    )

    assert rendered["data"].startswith(b"\x89PNG")
    assert "Batch 1" in rendered["caption"]
    assert "Horizontal: Response separation, d_R" in rendered["caption"]
    assert "Vertical: ON fluorescence, f_ON" in rendered["caption"]
    assert "red = greater clearance; 0 = configured boundary" in rendered["caption"]
    assert "Interpret all three encodings together" in rendered["caption"]
    assert "$" not in rendered["caption"]
    assert "\\" not in rendered["caption"]
    assert "$" not in rendered["alt"]
    assert rendered["style"]["max-height"] == "min(76vh, 860px)"


def test_layered_scatter_export_keeps_one_square_canvas_across_display_controls(tmp_path: Path) -> None:
    class _Mo:
        @staticmethod
        def image(data: bytes, **kwargs):
            return {"data": data, **kwargs}

    choice = _behavior_choice(tmp_path, filename="behavior_canvas.csv", selection_view_id="ethanol")
    states = (
        {"show_prediction_pool": True, "show_selected": True, "observed_batches": [], "label_scope": "none"},
        {"show_prediction_pool": True, "show_selected": False, "observed_batches": [], "label_scope": "none"},
        {"show_prediction_pool": True, "show_selected": True, "observed_batches": [], "label_scope": "selected"},
    )
    dimensions = []
    for state in states:
        payload = render_notebook_layered_scatter_image(choice, state=state, mo=_Mo())["data"]
        dimensions.append((int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")))

    assert dimensions == [(1296, 1296)] * len(states)


def test_layered_scatter_caption_plainly_preserves_msrb_symbols(tmp_path: Path) -> None:
    class _Mo:
        @staticmethod
        def image(data: bytes, **kwargs):
            return {"data": data, **kwargs}

    choice = _behavior_choice(tmp_path, filename="behavior_caption.csv", selection_view_id="ethanol")
    runtime = choice["manifest"]["artifact_metadata"]["notebook_view"]
    runtime["x_label"] = r"Response-ordering family score, $S_R$"
    runtime["y_label"] = r"Intended-ON signal family score, $S_{\mathrm{ON}}$"
    rendered = render_notebook_layered_scatter_image(
        choice,
        state={
            "show_prediction_pool": True,
            "show_selected": True,
            "observed_batches": ["batch_1"],
            "label_scope": "none",
        },
        mo=_Mo(),
    )

    assert "Horizontal: Response-ordering family score, S_R" in rendered["caption"]
    assert "Vertical: Intended-ON signal family score, S_ON" in rendered["caption"]
    assert "$" not in rendered["caption"]
    assert "\\" not in rendered["caption"]


def test_layered_scatter_legend_stays_outside_the_annotation_field(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    rows = pd.read_csv(contract["tidy_path"])
    filtered = filter_notebook_layered_scatter_rows(
        rows,
        contract=contract,
        state={
            "show_prediction_pool": True,
            "show_selected": True,
            "observed_batches": ["batch_1"],
            "label_scope": "both",
        },
    )

    figure = render_layered_scatter_figure(filtered, contract=contract)
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert legend.get_bbox_to_anchor()._bbox.y0 < 0.0
    axis = figure.axes[0]
    assert axis.get_title(loc="center") == "RMF candidate constraint landscape"
    subtitle = next(text for text in axis.texts if text.get_gid() == "notebook-plot-subtitle")
    assert subtitle.get_text() == "Ethanol target · ON: 10, 11 · OFF: 00, 01"
    assert subtitle.get_fontsize() < axis.title.get_fontsize()
    assert axis.get_title(loc="left") == ""
    assert axis.title.get_fontsize() >= 18
    assert axis.xaxis.label.get_fontsize() >= 15
    assert axis.yaxis.label.get_fontsize() >= 15
    assert min(tick.get_fontsize() for tick in axis.get_xticklabels()) >= 12
    assert min(text.get_fontsize() for text in legend.get_texts()) >= 12
    assert legend.legend_handles[0].get_alpha() >= 0.7
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert axis.collections[0].cmap.name == "RdBu_r"
    plt.close(figure)


def test_layered_scatter_all_visible_layers_keep_a_usable_square_panel(tmp_path: Path) -> None:
    choice = _behavior_choice(tmp_path, filename="behavior_all_layers.csv", selection_view_id="ethanol")
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    batch_ids = [item["id"] for item in contract["observed_batches"]]
    filtered = filter_notebook_layered_scatter_rows(
        pd.read_csv(contract["tidy_path"]),
        contract=contract,
        state={
            "show_prediction_pool": True,
            "show_selected": True,
            "observed_batches": batch_ids,
            "label_scope": "both",
        },
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        figure = render_layered_scatter_figure(filtered, contract=contract)
        figure.savefig(io.BytesIO(), format="png", facecolor="white")
        first_axis_bounds = figure.axes[0].get_window_extent().bounds
        figure.canvas.draw()
        second_axis_bounds = figure.axes[0].get_window_extent().bounds
    try:
        assert not any("constrained_layout not applied" in str(item.message) for item in captured)
        np.testing.assert_allclose(first_axis_bounds, second_axis_bounds)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis_box = figure.axes[0].get_window_extent(renderer=renderer)
        legend = figure.axes[0].get_legend()
        assert legend is not None
        legend_box = legend.get_window_extent(renderer=renderer)
        colorbar_axis = figure.axes[-1]
        colorbar_label_box = colorbar_axis.yaxis.label.get_window_extent(renderer=renderer)
        assert axis_box.width / figure.bbox.width >= 0.35
        assert axis_box.height / figure.bbox.height >= 0.35
        assert legend_box.x0 >= figure.bbox.x0
        assert legend_box.x1 <= figure.bbox.x1
        assert legend_box.y0 >= figure.bbox.y0
        assert colorbar_label_box.x0 >= axis_box.x1
        assert colorbar_label_box.x1 <= figure.bbox.x1
        assert colorbar_label_box.y0 >= figure.bbox.y0
        assert colorbar_label_box.y1 <= figure.bbox.y1
        assert len(legend.get_texts()) == 4
        assert any(text.get_text().startswith("Observed · Pre-round 0") for text in legend.get_texts())
        assert all("response corpus" not in text.get_text().lower() for text in legend.get_texts())
    finally:
        plt.close(figure)


def test_annotated_selected_and_observed_markers_remain_visible_above_label_boxes(tmp_path: Path) -> None:
    choice = _behavior_choice(tmp_path, filename="behavior_annotation_visibility.csv", selection_view_id="ethanol")
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    visible = filter_notebook_layered_scatter_rows(
        contract["rows"],
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": True,
            "observed_batches": [item["id"] for item in contract["observed_batches"]],
            "label_scope": "both",
        },
    )

    figure = render_layered_scatter_figure(visible, contract=contract)
    try:
        annotations = figure.axes[0].texts
        marker_layers = [
            collection
            for collection in figure.axes[0].collections
            if str(collection.get_label()).startswith(("Selected", "Observed"))
        ]
        assert annotations
        assert marker_layers
        assert min(layer.get_zorder() for layer in marker_layers) > max(
            annotation.get_zorder() for annotation in annotations
        )
    finally:
        plt.close(figure)


def test_selected_annotations_use_both_sides_of_the_plot_for_a_dense_six_candidate_cohort(tmp_path: Path) -> None:
    choice = _behavior_choice(tmp_path, filename="behavior_dense_labels.csv", selection_view_id="ethanol")
    manifest = choice["manifest"]
    assert isinstance(manifest, dict)
    tidy_path = Path(str(manifest["tidy_csv"]))
    tidy = pd.read_csv(tidy_path)
    selected_template = tidy.loc[tidy["selected"].eq(True)].iloc[0]
    dense = pd.DataFrame(
        [
            {
                **selected_template.to_dict(),
                "id": f"selected-{index}",
                "display_label": f"Selected candidate {index}",
                "response_family_score": 0.20 + index * 0.002,
                "on_signal_family_score": 1.20 + index * 0.002,
            }
            for index in range(6)
        ]
    )
    tidy = pd.concat([tidy.loc[~tidy["selected"].eq(True)], dense], ignore_index=True)
    tidy.to_csv(tidy_path, index=False)
    outputs = manifest["outputs"]
    assert isinstance(outputs, list) and isinstance(outputs[0], dict)
    outputs[0]["sha256"] = file_sha256(tidy_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    visible = filter_notebook_layered_scatter_rows(
        contract["rows"],
        contract=contract,
        state={
            "show_prediction_pool": True,
            "show_selected": True,
            "observed_batches": [],
            "label_scope": "selected",
        },
    )

    figure = render_layered_scatter_figure(visible, contract=contract)
    try:
        annotations = [text for text in figure.axes[0].texts if text.get_gid() != "notebook-plot-subtitle"]
        assert len(annotations) == 6
        assert {annotation.get_ha() for annotation in annotations} == {"left", "right"}
        renderer = figure.canvas.get_renderer()
        boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
        assert not any(left.overlaps(right) for index, left in enumerate(boxes) for right in boxes[index + 1 :])
    finally:
        plt.close(figure)


def test_generated_layered_scatter_controls_are_compact_and_reactive() -> None:
    text = render_layered_scatter_cells()
    contract_cell = layered_scatter_cell_template._contract_cell()
    control_cell = layered_scatter_cell_template._control_cell()
    helper_text = inspect.getsource(build_notebook_layered_scatter_controls)

    assert 'label="Prediction pool"' in helper_text
    assert 'label="Selected overlay"' in helper_text
    assert 'label="Observed batches"' in helper_text
    assert '"2D annotations" if interactive else "Labels"' in helper_text
    assert 'label="Figure"' in helper_text
    assert "build_notebook_layered_scatter_controls(" in text
    assert "plot_view_state" in text
    assert 'scatter_prediction_pool_ui = layered_scatter_controls["prediction_pool"]' in text
    assert 'scatter_figure_ui = layered_scatter_controls["figure"]' in text
    assert 'scatter_selected_ui = layered_scatter_controls["selected"]' in text
    assert 'scatter_selection_rounds_ui = layered_scatter_controls["selection_rounds"]' in text
    assert 'scatter_observed_batches_ui = layered_scatter_controls["observed_batches"]' in text
    assert 'scatter_labels_ui = layered_scatter_controls["labels"]' in text
    assert '"prediction_pool": scatter_prediction_pool_ui' in text
    assert '"figure": scatter_figure_ui' in text
    assert '"selected": scatter_selected_ui' in text
    assert '"selection_rounds": scatter_selection_rounds_ui' in text
    assert '"observed_batches": scatter_observed_batches_ui' in text
    assert '"labels": scatter_labels_ui' in text
    assert "read_notebook_layered_scatter_state(layered_scatter_controls)" not in text
    assert "build_notebook_layered_scatter_contract(_s)" in text
    assert "select_notebook_plot_scope(" in text
    assert "plot_scope_ui.value" in text
    assert "layered_scatter_memory" not in contract_cell
    assert "set_layered_scatter_memory" not in contract_cell
    assert "build_notebook_layered_scatter_contract" in contract_cell
    assert "layered_scatter_memory" in control_cell


def test_layered_scatter_controls_follow_the_selected_manifest_scope(tmp_path: Path) -> None:
    round_zero = _choice(
        tmp_path,
        filename="frontier_r0.csv",
        observed_batch_ids=("batch_0", "batch_0"),
    )
    round_one = _choice(
        tmp_path,
        filename="frontier_r1.csv",
        observed_batch_ids=("batch_1", "batch_1"),
    )
    round_zero["scope_label"] = "round 0"
    round_one["scope_label"] = "round 1"
    plot_choice = {**round_zero, "scope_options": [round_zero, round_one]}

    selected = select_notebook_plot_scope(plot_choice, "round 1")
    contract = build_notebook_layered_scatter_contract(selected)

    assert contract is not None
    assert [item["id"] for item in contract["observed_batches"]] == ["batch_1"]


def test_layered_scatter_contract_loads_exact_selected_cohorts_for_every_round(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    for choice, selected_id in ((round_zero, "selected-r0"), (round_one, "selected-r1")):
        manifest = choice["manifest"]
        tidy_path = Path(str(manifest["tidy_csv"]))
        tidy = pd.read_csv(tidy_path)
        tidy.loc[tidy["selected"].eq(True), "id"] = selected_id
        tidy.to_csv(tidy_path, index=False)
        manifest["outputs"][0]["sha256"] = file_sha256(tidy_path)

    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})

    assert contract is not None
    assert contract["active_selection_round"] == 1
    assert contract["selection_rounds"] == [0, 1]
    assert contract["selection_rows"][["id", "__notebook_selection_round"]].to_dict("records") == [
        {"id": "selected-r0", "__notebook_selection_round": 0},
        {"id": "selected-r1", "__notebook_selection_round": 1},
    ]


def test_layered_scatter_round_overlay_allows_round_specific_display_limits(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    round_zero["manifest"]["artifact_metadata"]["notebook_view"]["y_limits"] = [-3.0, 4.0]

    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})

    assert contract is not None
    assert contract["runtime"]["y_limits"] == [-0.5, 1.8]
    assert contract["selection_rounds"] == [0, 1]


def test_layered_scatter_can_overlay_selected_cohorts_categorically_by_round(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    for choice, selected_id in ((round_zero, "selected-r0"), (round_one, "selected-r1")):
        manifest = choice["manifest"]
        tidy_path = Path(str(manifest["tidy_csv"]))
        tidy = pd.read_csv(tidy_path)
        tidy.loc[tidy["selected"].eq(True), "id"] = selected_id
        tidy.to_csv(tidy_path, index=False)
        manifest["outputs"][0]["sha256"] = file_sha256(tidy_path)
    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})
    assert contract is not None

    visible = filter_notebook_layered_scatter_rows(
        contract["rows"],
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": True,
            "selection_rounds": [0, 1],
            "observed_batches": [],
            "label_scope": "none",
        },
    )
    figure = render_layered_scatter_figure(visible, contract=contract)
    try:
        assert visible[["id", "__notebook_selection_round"]].to_dict("records") == [
            {"id": "selected-r0", "__notebook_selection_round": 0},
            {"id": "selected-r1", "__notebook_selection_round": 1},
        ]
        assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
            "Selected for Round 0 (n=1)",
            "Selected for Round 1 (n=1)",
        ]
    finally:
        plt.close(figure)


def test_layered_scatter_controls_offer_exact_manifest_backed_selection_rounds() -> None:
    class _Ui:
        @staticmethod
        def dropdown(_options, *, value, **_kwargs):
            return SimpleNamespace(value=value)

        @staticmethod
        def switch(*, value, **_kwargs):
            return SimpleNamespace(value=value)

        @staticmethod
        def multiselect(options, *, value, **_kwargs):
            return SimpleNamespace(value=[options[item] for item in value])

    memory_state: dict[str, object] = {}
    controls = build_notebook_layered_scatter_controls(
        {
            "key": "plot",
            "interactive": {"adapter": "three_axis_scatter_v1"},
            "active_selection_round": 1,
            "selection_rounds": [0, 1],
            "observed_batches": [],
        },
        memory=lambda: memory_state,
        set_memory=lambda value: memory_state.update(value),
        mo=SimpleNamespace(ui=_Ui()),
    )

    assert controls["selection_rounds"].value == [1]


def _marker_vertices(figure, label: str) -> np.ndarray:
    for collection in figure.axes[0].collections:
        if str(collection.get_label()).startswith(label):
            return np.asarray(collection.get_paths()[0].vertices, dtype=float)
    raise AssertionError(f"No scatter collection found for {label!r}.")
