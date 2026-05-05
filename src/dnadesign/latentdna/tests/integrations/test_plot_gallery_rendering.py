"""Integration coverage for multi-panel plot gallery rendering."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.plots.render import _add_figure_legends, _pyplot
from dnadesign.latentdna.src.services.plot_service import render_plot


def _write_workspace_config(workspace_dir: Path) -> None:
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row0"], type=pa.string()),
                "subject_id": pa.array(["row0"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "anchor.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "plot_gallery_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": ["design_family", "sig35_variant"]},
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "reference_sets": {
                    "reference_spyp_sulap": {
                        "ids": ["spyp", "sulap"],
                        "match_column": "usr_label__primary",
                        "label_column": "usr_label__primary",
                        "display_labels": {
                            "spyp": "spyP",
                            "sulap": "sulAp",
                        },
                    }
                },
                "plots": {
                    "design_centroid_margin_gallery": {
                        "kind": "xy_scatter_grid",
                        "scalars": ["margin_a", "margin_b"],
                        "panel_titles": ["panel a", "panel b"],
                        "x_column": "synthetic_margin_ethanol_vs_background",
                        "y_column": "synthetic_margin_cipro_vs_background",
                        "default_hue": "design_family",
                        "hue_options": [
                            {"column": "design_family", "label": "Design family", "type": "categorical"},
                            {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
                        ],
                        "semantics_ref": "plot_semantics/design_centroid_margin_gallery.yaml",
                    },
                    "representation_health_summary": {
                        "kind": "metric_panel_grid",
                        "scalar": "representation_health_summary_metrics",
                        "facet_column": "category",
                        "panel_title_column": "display_name",
                        "category_column": "label",
                        "label_column": "candidate_label",
                        "value_column": "metric_value",
                        "color_column": "candidate_family",
                        "direction_column": "direction",
                        "unit_column": "unit",
                        "sort_rule": "panel_direction",
                        "measure_kind": "metric",
                        "value_kind": "score",
                        "value_label": "Metric value",
                        "semantics_ref": "plot_semantics/representation_health_summary.yaml",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    semantics_dir = workspace_dir / "plot_semantics"
    semantics_dir.mkdir(exist_ok=True)
    (semantics_dir / "design_centroid_margin_gallery.yaml").write_text(
        yaml.safe_dump(
            {
                "plot_id": "design_centroid_margin_gallery",
                "question": "Do internal design centroids orient the candidate space away from the background cohort?",
                "decision_role": "primary",
                "encoding": "Two-panel scatter plot of study-internal design-centroid margins.",
                "scope": "Full population.",
                "guardrails": ["Two-dimensional separation is descriptive only."],
                "caption": "Study-internal design-centroid margin plane.",
                "alt_text": (
                    "Two-panel scatter plot of ethanol-versus-background and ciprofloxacin-versus-background margins."
                ),
                "preprocessing_md": "Fixture semantics do not declare additional preprocessing.",
                "math_md": "Fixture semantics do not declare a mathematical definition.",
                "rationale_md": "Fixture semantics exist only to validate plot-gallery rendering.",
                "limitations_md": "Fixture semantics are not a study-facing scientific contract.",
                "failure_modes_md": "Replace fixture semantics before using the plot outside tests.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (semantics_dir / "representation_health_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "plot_id": "representation_health_summary",
                "question": "Which candidate spaces pass the representation-health gate?",
                "decision_role": "primary",
                "encoding": "Two-panel candidate metric summary chart.",
                "scope": "Candidate-level health metrics.",
                "guardrails": ["Health metrics are a gate, not a biological performance claim."],
                "caption": "Representation-health summary.",
                "alt_text": "Two-panel candidate metric chart for effective rank and pairwise distance spread.",
                "preprocessing_md": "Fixture semantics do not declare additional preprocessing.",
                "math_md": "Fixture semantics do not declare a mathematical definition.",
                "rationale_md": "Fixture semantics exist only to validate plot-gallery rendering.",
                "limitations_md": "Fixture semantics are not a study-facing scientific contract.",
                "failure_modes_md": "Replace fixture semantics before using the plot outside tests.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_scalar(output_root: Path, scalar_id: str, rows: list[dict[str, object]]) -> None:
    scalar_dir = output_root / "scalars" / scalar_id
    scalar_dir.mkdir(parents=True, exist_ok=True)
    table_path = scalar_dir / "table.parquet"
    pq.write_table(pa.Table.from_pylist(rows), table_path)
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "scalar_table",
                "artifact_id": scalar_id,
                "workspace_id": "plot_gallery_demo",
                "created_at": "2026-04-17T00:00:00Z",
                "tool_version": "fixture",
                "command": "fixture",
                "status": "ok",
                "outputs": [{"path": table_path.name, "media_type": "application/x-parquet"}],
                "stats": {"rows": len(rows)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_agreement_summary(output_root: Path, agreement_id: str, payload: dict[str, object]) -> None:
    agreement_dir = output_root / "agreements" / agreement_id
    agreement_dir.mkdir(parents=True, exist_ok=True)
    summary_path = agreement_dir / "summary.json"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    (agreement_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "agreement_set",
                "artifact_id": agreement_id,
                "workspace_id": "plot_gallery_demo",
                "created_at": "2026-04-17T00:00:00Z",
                "tool_version": "fixture",
                "command": "fixture",
                "status": "ok",
                "outputs": [{"path": summary_path.name, "media_type": "application/json"}],
                "stats": payload,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_metric_panel_scalar(output_root: Path, scalar_id: str) -> None:
    _write_scalar(
        output_root,
        scalar_id,
        [
            {
                "category": "effective_rank",
                "label": "candidate_a",
                "display_name": "Effective rank",
                "candidate_label": "candidate a",
                "candidate_family": "intermediate_embedding",
                "direction": "higher_is_better",
                "unit": "rank",
                "metric_value": 8.4,
            },
            {
                "category": "effective_rank",
                "label": "candidate_b",
                "display_name": "Effective rank",
                "candidate_label": "candidate b",
                "candidate_family": "output_layer_mean",
                "direction": "higher_is_better",
                "unit": "rank",
                "metric_value": 1.2,
            },
            {
                "category": "pairwise_cosine_distance_iqr",
                "label": "candidate_a",
                "display_name": "Pairwise cosine distance IQR",
                "candidate_label": "candidate a",
                "candidate_family": "intermediate_embedding",
                "direction": "higher_is_better",
                "unit": "distance",
                "metric_value": 0.18,
            },
            {
                "category": "pairwise_cosine_distance_iqr",
                "label": "candidate_b",
                "display_name": "Pairwise cosine distance IQR",
                "candidate_label": "candidate b",
                "candidate_family": "output_layer_mean",
                "direction": "higher_is_better",
                "unit": "distance",
                "metric_value": 0.01,
            },
        ],
    )


def _render_named_plot(workspace_dir: Path, plot_id: str) -> None:
    render_plot(
        workspace_dir,
        plot_id,
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        agreement_id=None,
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )


def test_plot_gallery_rendering_records_plural_scalar_and_agreement_inputs(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)

    output_root = workspace_dir / "outputs"
    _write_scalar(
        output_root,
        "margin_a",
        [
            {
                "synthetic_margin_ethanol_vs_background": 0.25,
                "synthetic_margin_cipro_vs_background": -0.10,
                "design_family": "ethanol",
                "sig35_variant": "b",
            },
            {
                "synthetic_margin_ethanol_vs_background": -0.05,
                "synthetic_margin_cipro_vs_background": 0.20,
                "design_family": "cipro",
                "sig35_variant": "c",
            },
        ],
    )
    _write_scalar(
        output_root,
        "margin_b",
        [
            {
                "synthetic_margin_ethanol_vs_background": 0.35,
                "synthetic_margin_cipro_vs_background": -0.15,
                "design_family": "ethanol",
                "sig35_variant": "b",
            },
            {
                "synthetic_margin_ethanol_vs_background": -0.15,
                "synthetic_margin_cipro_vs_background": 0.30,
                "design_family": "cipro",
                "sig35_variant": "c",
            },
        ],
    )
    _write_metric_panel_scalar(output_root, "representation_health_summary_metrics")

    _render_named_plot(workspace_dir, "design_centroid_margin_gallery")
    _render_named_plot(workspace_dir, "representation_health_summary")

    reference_manifest = json.loads(
        (output_root / "plots" / "design_centroid_margin_gallery" / "manifest.json").read_text(encoding="utf-8")
    )
    assert reference_manifest["params"]["plot_kind"] == "xy_scatter_grid"
    assert reference_manifest["params"]["scalar_ids"] == ["margin_a", "margin_b"]
    assert reference_manifest["params"]["default_hue"] == "design_family"
    assert [option["column"] for option in reference_manifest["params"]["hue_options"]] == [
        "design_family",
        "sig35_variant",
    ]
    assert "shape_column" not in reference_manifest["params"]
    assert [entry["id"] for entry in reference_manifest["inputs"]] == ["margin_a", "margin_b"]
    assert reference_manifest["semantics"]["decision_role"] == "primary"
    assert reference_manifest["semantics"]["question"].startswith("Do internal design centroids orient")
    assert any(
        entry["role"] == "workspace_config" and entry["path"].endswith("/config.yaml")
        for entry in reference_manifest["source_provenance"]
    )
    assert any(
        entry["role"] == "plot_semantics"
        and entry["path"].endswith("/plot_semantics/design_centroid_margin_gallery.yaml")
        for entry in reference_manifest["source_provenance"]
    )
    reference_svg = output_root / "plots" / "design_centroid_margin_gallery" / "plot.svg"
    assert reference_svg.is_file()
    reference_svg_text = reference_svg.read_text(encoding="utf-8")
    assert (
        "<title>Do internal design centroids orient the candidate space away from the background cohort?</title>"
        in reference_svg_text
    )
    assert (
        "<desc>Two-panel scatter plot of ethanol-versus-background and ciprofloxacin-versus-background margins.</desc>"
        in reference_svg_text
    )

    agreement_manifest = json.loads(
        (output_root / "plots" / "representation_health_summary" / "manifest.json").read_text(encoding="utf-8")
    )
    assert agreement_manifest["params"]["plot_kind"] == "metric_panel_grid"
    assert agreement_manifest["params"]["measure_kind"] == "metric"
    assert agreement_manifest["params"]["value_column"] == "metric_value"
    assert agreement_manifest["params"]["row_column"] == "category"
    assert agreement_manifest["params"]["panel_column"] == "display_name"
    assert [entry["id"] for entry in agreement_manifest["inputs"]] == ["representation_health_summary_metrics"]
    assert (output_root / "plots" / "representation_health_summary" / "plot.png").is_file()


def test_design_centroid_margin_gallery_keeps_reasonable_canvas(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)

    output_root = workspace_dir / "outputs"
    design_rows = [
        {
            "synthetic_margin_ethanol_vs_background": 0.25,
            "synthetic_margin_cipro_vs_background": -0.10,
            "design_family": "ethanol",
            "sig35_variant": "b",
        },
        {
            "synthetic_margin_ethanol_vs_background": -0.05,
            "synthetic_margin_cipro_vs_background": 0.20,
            "design_family": "cipro",
            "sig35_variant": "c",
        },
        {
            "synthetic_margin_ethanol_vs_background": 0.02,
            "synthetic_margin_cipro_vs_background": 0.01,
            "design_family": "background_only",
            "sig35_variant": "f",
        },
    ]
    _write_scalar(output_root, "margin_a", design_rows)
    _write_scalar(output_root, "margin_b", design_rows)

    _render_named_plot(workspace_dir, "design_centroid_margin_gallery")

    reference_svg = output_root / "plots" / "design_centroid_margin_gallery" / "plot.svg"
    root = ET.fromstring(reference_svg.read_text(encoding="utf-8"))
    width = float(root.attrib["width"].removesuffix("pt"))
    height = float(root.attrib["height"].removesuffix("pt"))

    assert width / height < 2.6
    assert width < 700.0


def test_figure_legends_are_reserved_below_the_axes() -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([0.0, 1.0], [0.0, 1.0], s=10)

    bottom_margin = _add_figure_legends(
        fig,
        plt,
        plot_id=None,
        color_categories=["background_only", "ethanol", "ciprofloxacin"],
        color_map={
            "background_only": "#56B4E9",
            "ethanol": "#E69F00",
            "ciprofloxacin": "#009E73",
        },
        color_title="design_family",
        shape_categories=[],
        shape_map={},
        shape_title=None,
    )
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 1.0), pad=0.55)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    legend_boxes = [legend.get_window_extent(renderer) for legend in fig.legends]

    assert bottom_margin >= 0.1
    assert len(fig.legends) == 1
    assert legend_boxes
    assert all(box.y1 <= axes_box.y0 for box in legend_boxes)
    assert fig.legends[0].get_title().get_visible() is False

    plt.close(fig)
