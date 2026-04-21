"""Projection rendering regression tests for notebook geometry audit surfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dnadesign.latentdna.src.notebooks import browser_runtime_projection as projection_runtime


def _panel_offsets(fig) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for collection in fig.axes[0].collections:
        collection_offsets = np.asarray(collection.get_offsets())
        if collection_offsets.size == 0:
            continue
        offsets.extend((float(x), float(y)) for x, y in collection_offsets.tolist())
    return sorted(offsets)


def test_render_projection_grid_keeps_point_coordinates_fixed_across_hues(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0, 0.0],
            "design_family": ["ethanol", "ethanol", "cipro", "cipro"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig_without_hue = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column=None,
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )
    fig_with_hue = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert _panel_offsets(fig_without_hue) == _panel_offsets(fig_with_hue)
    finally:
        plt.close(fig_without_hue)
        plt.close(fig_with_hue)


def test_render_projection_grid_suppresses_degenerate_continuous_colorbar(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "wildtype_margin_ethanol_vs_control": [0.25, 0.25, 0.25],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="wildtype_margin_ethanol_vs_control",
        hue_kinds={"wildtype_margin_ethanol_vs_control": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_render_projection_grid_compacts_regulator_composition_legend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 1.0, 2.0, 3.0, 4.0],
            "design_regulator_composition": [
                "cpxR_MANWWHTTTAM",
                "lexA_CTGTATAWAWWHACA",
                "baeR_TTTCTSCVHNA+lexA_CTGTATAWAWWHACA",
                "sig35=b",
                "control",
            ],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="design_regulator_composition",
        hue_kinds={"design_regulator_composition": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["BaeR+LexA", "Bg", "CpxR", "Ctrl", "LexA"]
        assert legend._ncols == 3
    finally:
        plt.close(fig)


def test_render_projection_grid_orders_sig35_legend_by_strength(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "sig35_variant": ["b", "f", "d", "control"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="sig35_variant",
        hue_kinds={"sig35_variant": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        labels = [text.get_text() for text in legend.get_texts()]
        colors = [handle.get_color() for handle in legend.legend_handles]
        assert labels == ["TTGACA (f)", "TTTACA (d)", "CTGACA (b)", "Control"]
        assert colors == ["#B2182B", "#F4A582", "#2166AC", "#7F8894"]
    finally:
        plt.close(fig)


def test_render_projection_grid_prefers_single_row_when_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(4)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame, frame, frame, frame],
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
        prefer_single_row=True,
    )

    try:
        y_positions = {round(axis.get_position().y0, 3) for axis in fig.axes[:4]}
        assert len(y_positions) == 1
    finally:
        plt.close(fig)


def test_render_projection_grid_handles_seven_panel_gallery_without_axis_zip_failures(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(7)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame for _ in range(7)],
        plot_id="appendix_umap_gallery",
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        panel_axes = fig.axes[:7]
        assert len({round(axis.get_position().y0, 3) for axis in panel_axes}) == 2
        assert fig.axes[7].axison is False
    finally:
        plt.close(fig)


def test_render_projection_grid_places_continuous_colorbar_below_panels_and_uses_margin_palette(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [2.0, 1.0, 0.0],
            "synthetic_margin_ethanol_vs_background": [-0.35, 0.0, 0.28],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(2)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame, frame],
        hue_column="synthetic_margin_ethanol_vs_background",
        hue_kinds={"synthetic_margin_ethanol_vs_background": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        panel_axes = fig.axes[:2]
        colorbar_ax = fig.axes[-1]
        assert len(fig.axes) == 3
        assert panel_axes[0].collections[0].cmap.name == "coolwarm"
        assert colorbar_ax.get_position().width > colorbar_ax.get_position().height
        assert colorbar_ax.get_position().y1 < min(axis.get_position().y0 for axis in panel_axes)
    finally:
        plt.close(fig)


def test_enrich_projection_frame_joins_requested_columns_by_same_named_key(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "design_centroid_margins_demo"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [0.25, -0.10],
            "synthetic_margin_cipro_vs_background": [0.05, 0.15],
            "unused_metric": [5.0, 6.0],
        }
    ).to_parquet(scalar_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_demo",
                "relative_path": "scalars/design_centroid_margins_demo/table.parquet",
                "columns": [
                    "id",
                    "synthetic_margin_ethanol_vs_background",
                    "synthetic_margin_cipro_vs_background",
                    "unused_metric",
                ],
            }
        ],
        output_root=tmp_path,
        required_columns=["synthetic_margin_ethanol_vs_background"],
    )

    assert enriched["synthetic_margin_ethanol_vs_background"].tolist() == [0.25, -0.10]
    assert "synthetic_margin_cipro_vs_background" not in enriched.columns
    assert "unused_metric" not in enriched.columns


def test_enrich_projection_frame_joins_context_metrics_from_anchor_id(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "context_delta_distribution_demo"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["context_row_1", "context_row_2"],
            "construct__anchor_id": ["anchor_1", "anchor_2"],
            "context_self_cosine": [0.91, 0.84],
            "context_shift_l2": [0.02, 0.09],
        }
    ).to_parquet(scalar_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "context_delta_distribution_demo",
                "relative_path": "scalars/context_delta_distribution_demo/table.parquet",
                "columns": ["id", "construct__anchor_id", "context_self_cosine", "context_shift_l2"],
            }
        ],
        output_root=tmp_path,
        required_columns=["context_self_cosine"],
    )

    assert enriched["context_self_cosine"].tolist() == [0.91, 0.84]
    assert "construct__anchor_id" not in enriched.columns
    assert "context_shift_l2" not in enriched.columns


def test_enrich_projection_frame_respects_explicit_table_view_targets(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "context_delta_distribution_intermediate_embedding_7b"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["context_row_1", "context_row_2"],
            "construct__anchor_id": ["anchor_1", "anchor_2"],
            "context_self_cosine": [0.91, 0.84],
        }
    ).to_parquet(scalar_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "context_delta_distribution_intermediate_embedding_7b",
                "relative_path": "scalars/context_delta_distribution_intermediate_embedding_7b/table.parquet",
                "columns": ["id", "construct__anchor_id", "context_self_cosine"],
                "view_ids": ["intermediate_embedding_7b_anchor_60bp"],
            }
        ],
        output_root=tmp_path,
        view_id="intermediate_embedding_7b_anchor_plus_full_context_concat",
        required_columns=["context_self_cosine"],
        strict_required_columns=False,
    )

    assert "context_self_cosine" not in enriched.columns


def test_enrich_projection_frame_limits_candidate_metric_tables_to_active_view(tmp_path: Path) -> None:
    wrong_dir = tmp_path / "scalars" / "design_centroid_margins_intermediate_embedding_20b_anchor_60bp"
    wrong_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [np.nan, np.nan],
        }
    ).to_parquet(wrong_dir / "table.parquet")

    right_dir = tmp_path / "scalars" / "design_centroid_margins_pooled_logits_20b_anchor_60bp"
    right_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [0.35, -0.22],
        }
    ).to_parquet(right_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            },
            {
                "artifact_id": "design_centroid_margins_pooled_logits_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_pooled_logits_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            },
        ],
        output_root=tmp_path,
        view_id="pooled_logits_20b_anchor_60bp",
        required_columns=["synthetic_margin_ethanol_vs_background"],
    )

    assert enriched["synthetic_margin_ethanol_vs_background"].tolist() == [0.35, -0.22]


def test_enrich_projection_frame_prefers_authoritative_view_row_metadata(tmp_path: Path) -> None:
    view_dir = tmp_path / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "design_regulator_composition": ["baeR", "background"],
            "spacer_length": [18, 17],
        }
    ).to_parquet(view_dir / "rows.parquet")

    frame = pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_regulator_composition": ["baeR_TTTCTSCVHNA", "sig35=c"],
            "spacer_length": [None, None],
        }
    )
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [],
        output_root=tmp_path,
        view_id="intermediate_embedding_7b_anchor_60bp",
        required_columns=["design_regulator_composition", "spacer_length"],
    )

    assert enriched["design_regulator_composition"].tolist() == ["baeR", "background"]
    assert enriched["spacer_length"].tolist() == [18, 17]


def test_enrich_projection_frame_refreshes_required_column_from_joinable_source(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "design_centroid_margins_intermediate_embedding_20b_anchor_60bp"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [0.25, -0.10],
        }
    ).to_parquet(scalar_dir / "table.parquet")

    frame = pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "synthetic_margin_ethanol_vs_background": [999.0, -999.0],
        }
    )
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            }
        ],
        output_root=tmp_path,
        view_id="intermediate_embedding_20b_anchor_60bp",
        required_columns=["synthetic_margin_ethanol_vs_background"],
    )

    assert enriched["synthetic_margin_ethanol_vs_background"].tolist() == [0.25, -0.10]


def test_enrich_projection_frame_rejects_required_column_when_joined_values_are_empty(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "design_centroid_margins_intermediate_embedding_20b_anchor_60bp"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [None, None],
        }
    ).to_parquet(scalar_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})

    with pytest.raises(ValueError, match="required metadata columns are empty"):
        projection_runtime.enrich_projection_frame(
            frame,
            [
                {
                    "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                    "relative_path": (
                        "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet"
                    ),
                    "columns": ["id", "synthetic_margin_ethanol_vs_background"],
                }
            ],
            output_root=tmp_path,
            view_id="intermediate_embedding_20b_anchor_60bp",
            required_columns=["synthetic_margin_ethanol_vs_background"],
        )


def test_enrich_projection_frame_rejects_ambiguous_required_column_sources(tmp_path: Path) -> None:
    first_dir = tmp_path / "scalars" / "design_centroid_margins_pooled_logits_20b_anchor_60bp"
    first_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [0.35, -0.22],
        }
    ).to_parquet(first_dir / "table.parquet")

    second_dir = tmp_path / "scalars" / "context_delta_distribution_pooled_logits_20b_anchor_60bp"
    second_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [0.10, 0.20],
        }
    ).to_parquet(second_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})

    with pytest.raises(ValueError, match="ambiguous metadata source"):
        projection_runtime.enrich_projection_frame(
            frame,
            [
                {
                    "artifact_id": "design_centroid_margins_pooled_logits_20b_anchor_60bp",
                    "relative_path": "scalars/design_centroid_margins_pooled_logits_20b_anchor_60bp/table.parquet",
                    "columns": ["id", "synthetic_margin_ethanol_vs_background"],
                },
                {
                    "artifact_id": "context_delta_distribution_pooled_logits_20b_anchor_60bp",
                    "relative_path": "scalars/context_delta_distribution_pooled_logits_20b_anchor_60bp/table.parquet",
                    "columns": ["id", "synthetic_margin_ethanol_vs_background"],
                },
            ],
            output_root=tmp_path,
            view_id="pooled_logits_20b_anchor_60bp",
            required_columns=["synthetic_margin_ethanol_vs_background"],
        )


def test_enrich_projection_frame_rejects_required_column_when_join_keys_do_not_resolve(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "design_centroid_margins_intermediate_embedding_20b_anchor_60bp"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["anchor_1", "anchor_2"],
            "synthetic_margin_ethanol_vs_background": [0.25, -0.10],
        }
    ).to_parquet(scalar_dir / "table.parquet")

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})

    with pytest.raises(ValueError, match="cannot join"):
        projection_runtime.enrich_projection_frame(
            frame,
            [
                {
                    "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                    "relative_path": (
                        "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet"
                    ),
                    "columns": ["record_id", "synthetic_margin_ethanol_vs_background"],
                }
            ],
            output_root=tmp_path,
            view_id="intermediate_embedding_20b_anchor_60bp",
            required_columns=["synthetic_margin_ethanol_vs_background"],
        )
