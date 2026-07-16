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
from pathlib import Path

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
from dnadesign.opal.src.analysis.notebook_set_template.layered_scatter_cells import (
    render_layered_scatter_cells,
)
from dnadesign.opal.src.core.utils import file_sha256
from dnadesign.opal.src.plots import response_magnitude_feasibility_aliases
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
                    "x_boundary": 0.0,
                    "y_boundary": 0.0,
                    "color_extent": 1.0,
                    "x_limits": [-0.5, 0.8],
                    "y_limits": [-0.5, 1.8],
                }
            },
        },
    }


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
        "Pre-round 0 response corpus 4–8 h · v1",
    ]


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


def test_layered_scatter_memory_is_selection_view_scoped(tmp_path: Path) -> None:
    ethanol_choice = _choice(tmp_path)
    ciprofloxacin_choice = _choice(tmp_path, filename="frontier_ciprofloxacin.csv")
    ciprofloxacin_choice["manifest"]["selection_view_id"] = "ciprofloxacin"

    ethanol = build_notebook_layered_scatter_contract(ethanol_choice)
    ciprofloxacin = build_notebook_layered_scatter_contract(ciprofloxacin_choice)

    assert ethanol is not None and ciprofloxacin is not None
    assert ethanol["key"] != ciprofloxacin["key"]


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
    captured: list[tuple[list[str], list[str]]] = []

    def _capture_annotations(_ax, frame, aliases, **_kwargs) -> None:
        captured.append((frame["record_kind"].astype(str).tolist(), list(aliases.values())))

    monkeypatch.setattr(
        response_magnitude_feasibility_aliases,
        "annotate_candidate_aliases",
        _capture_annotations,
    )
    figure = layered_scatter_rendering.render_layered_scatter_figure(filtered, contract=contract)
    try:
        assert captured == [([expected_kind], [expected_label])]
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
    assert rendered["style"]["max-height"] == "min(62vh, 720px)"


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
    plt.close(figure)


def test_generated_layered_scatter_controls_are_compact_and_reactive() -> None:
    text = render_layered_scatter_cells()
    helper_text = inspect.getsource(build_notebook_layered_scatter_controls)

    assert 'label="Prediction pool"' in helper_text
    assert 'label="Selected overlay"' in helper_text
    assert 'label="Observed batches"' in helper_text
    assert 'label="Labels"' in helper_text
    assert "build_notebook_layered_scatter_controls(" in text
    assert "plot_view_state" in text
    assert 'scatter_prediction_pool_ui = layered_scatter_controls["prediction_pool"]' in text
    assert 'scatter_selected_ui = layered_scatter_controls["selected"]' in text
    assert 'scatter_observed_batches_ui = layered_scatter_controls["observed_batches"]' in text
    assert 'scatter_labels_ui = layered_scatter_controls["labels"]' in text
    assert '"prediction_pool": scatter_prediction_pool_ui' in text
    assert '"selected": scatter_selected_ui' in text
    assert '"observed_batches": scatter_observed_batches_ui' in text
    assert '"labels": scatter_labels_ui' in text
    assert "read_notebook_layered_scatter_state(layered_scatter_controls)" not in text
    assert "build_notebook_layered_scatter_contract(_s)" in text
    assert "select_notebook_plot_scope(" in text
    assert "plot_scope_ui.value" in text


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


def _marker_vertices(figure, label: str) -> np.ndarray:
    for collection in figure.axes[0].collections:
        if str(collection.get_label()).startswith(label):
            return np.asarray(collection.get_paths()[0].vertices, dtype=float)
    raise AssertionError(f"No scatter collection found for {label!r}.")
