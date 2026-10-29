"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_explicit_presentation_metadata.py

Renaming artifact IDs must not change scientific labels or plot geometry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from dnadesign.latentdna.src.contracts.plot import ResolvedPlotSpec
from dnadesign.latentdna.src.io.parquet_io import write_table
from dnadesign.latentdna.src.plots.layout import metric_panel_grid_layout, plot_tight_layout_kwargs
from dnadesign.latentdna.src.plots.renderers.metric import metric_panel_prefers_horizontal_bars
from dnadesign.latentdna.src.presentation.visual_style import legend_layout
from dnadesign.latentdna.src.scalars.builders.representation_scorecard import _candidate_metric_pairs_table
from dnadesign.latentdna.src.services.notebook_geometry_controls import _format_view_label


def test_model_label_does_not_invent_a_model_family():
    assert "Evo" not in _format_view_label(model="esm2", family="intermediate_embedding", scope_name=None)


@pytest.mark.parametrize("candidate_id", ["encoder", "unrelated_7b_intermediate_embedding_anchor_60bp"])
def test_metric_pairs_preserve_declared_candidate_metadata(tmp_path, candidate_id):
    path = tmp_path / "scalars/scorecard/table.parquet"
    path.parent.mkdir(parents=True)
    write_table(
        pa.Table.from_pylist(
            [
                {
                    "candidate_id": candidate_id,
                    "candidate_model": "esm2",
                    "candidate_family": "protein",
                    "candidate_scope": "chain",
                    "candidate_label": "Protein encoder",
                    "metric_id": metric,
                    "display_name": metric,
                    "metric_value": 0.5,
                    "direction": "descriptive",
                    "unit": "score",
                }
                for metric in ("x", "y")
            ]
        ),
        path,
    )
    table, _, _ = _candidate_metric_pairs_table(
        SimpleNamespace(output_root=tmp_path), source_scalar="scorecard", x_metric_id="x", y_metric_id="y"
    )
    row = table.to_pylist()[0]
    assert row["candidate_model"] == "esm2"
    assert row["candidate_label"] == "Protein encoder"
    assert row["candidate_family"] == "protein"


@pytest.mark.parametrize(
    "plot_id",
    [
        "balanced_design_family_margin_gallery",
        "representation_health_summary",
        "appendix_umap_gallery",
        "sigma35_ordinal_audit",
    ],
)
def test_layout_does_not_depend_on_plot_id(plot_id):
    assert legend_layout(["A", "B", "C", "D"], plot_id=plot_id, default_anchor_y=0.02) == legend_layout(
        ["A", "B", "C", "D"], plot_id="renamed", default_anchor_y=0.02
    )
    assert metric_panel_grid_layout(plot_id, 6) == metric_panel_grid_layout("renamed", 6)
    assert plot_tight_layout_kwargs(plot_id, legend_bottom=0.1) == plot_tight_layout_kwargs(
        "renamed", legend_bottom=0.1
    )
    original = ResolvedPlotSpec(plot_id=plot_id, kind="metric_panel_grid")
    assert metric_panel_prefers_horizontal_bars(original) == metric_panel_prefers_horizontal_bars(
        original.model_copy(update={"plot_id": "renamed"})
    )


def test_view_descriptor_preserves_model_identity():
    from dnadesign.latentdna.src.scalars.common import _candidate_descriptor_from_view

    context = SimpleNamespace(require_view=lambda _: SimpleNamespace(tags={"model": "esm2"}))
    descriptor = _candidate_descriptor_from_view(context, view_id="protein")
    assert descriptor["candidate_model"] == "esm2"
    assert "Evo" not in descriptor["candidate_label"]
