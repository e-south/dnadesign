"""Contract tests for workspace notebook control-plane assembly."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from dnadesign.latentdna.src.services.notebook_controls_service import (
    _plot_controls,
    build_workspace_notebook_controls_payload,
)
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_artifact_manifest(
    artifact_dir: Path,
    *,
    artifact_kind: str,
    artifact_id: str,
    status: str = "ok",
    inputs: list[dict[str, object]] | None = None,
    params: dict[str, object] | None = None,
    stats: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "status": status,
    }
    if inputs is not None:
        payload["inputs"] = inputs
    if params is not None:
        payload["params"] = params
    if stats is not None:
        payload["stats"] = stats
    (artifact_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_workspace_config(workspace_dir: Path) -> None:
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row0", "row1"], type=pa.string()),
                "subject_id": pa.array(["row0", "row1"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "anchor.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "projection_sort_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
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
                "metadata": {"include": []},
                "views": {
                    "intermediate_embedding_7b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "plots": {},
                "notebooks": {
                    "latent_geometry_browser": {
                        "kind": "workspace",
                        "title": "Browser",
                        "default_deliverable": "appendix_umap_gallery",
                    }
                },
                "deliverables": {
                    "appendix_umap_gallery": {
                        "title": "Appendix",
                        "section": "Appendix",
                        "question": "Which projections are available?",
                        "summary": "Projection browser contract test.",
                        "recipe": "noop_recipe",
                        "requires": {"views": ["intermediate_embedding_7b_anchor_60bp"]},
                        "outputs": {"notebooks": ["latent_geometry_browser"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
                "recipes": {
                    "noop_recipe": {
                        "steps": [
                            {
                                "id": "materialize_view",
                                "op": "view.materialize",
                                "params": {"view": "intermediate_embedding_7b_anchor_60bp"},
                            },
                            {
                                "id": "generate_notebook",
                                "op": "notebook.generate",
                                "depends_on": ["materialize_view"],
                                "params": {"notebook": "latent_geometry_browser"},
                            },
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_notebook_controls_sort_projection_ids_by_role_then_full_population(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    for projection_id, role, default_rank in [
        ("audit_umap_anchor", "audit", 50),
        ("umap_anchor", "primary", 10),
    ]:
        projection_dir = context.output_root / "projections" / projection_id
        projection_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
            projection_dir / "coords.parquet",
            index=False,
        )
        _write_artifact_manifest(
            projection_dir,
            artifact_kind="projection",
            artifact_id=projection_id,
            inputs=[{"kind": "view_matrix", "id": "intermediate_embedding_7b_anchor_60bp"}],
            params={
                "projection_role": role,
                "default_rank": default_rank,
                "sampling_strategy": "all",
            },
            stats={
                "rows": 2,
                "projected_rows": 2,
                "population_rows": 2,
                "is_full_population": True,
            },
        )

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.schema_version == "latentdna.workspace_notebook_controls.v4"
    assert controls.plot_controls.default_surface == "plots"
    assert controls.plot_controls.ordered_plot_ids == []
    geometry = next(
        row for row in controls.geometry_controls.geometries if row.view_id == "intermediate_embedding_7b_anchor_60bp"
    )
    assert geometry.projection_ids == ["umap_anchor", "audit_umap_anchor"]


def test_notebook_controls_keep_attention_projection_visible_for_geometry_browser(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    projection_dir = context.output_root / "projections" / "umap_anchor_attention"
    projection_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
        projection_dir / "coords.parquet",
        index=False,
    )
    _write_artifact_manifest(
        projection_dir,
        artifact_kind="projection",
        artifact_id="umap_anchor_attention",
        status="attention",
        inputs=[{"kind": "view_matrix", "id": "intermediate_embedding_7b_anchor_60bp"}],
        params={"projection_role": "appendix", "default_rank": 10},
        stats={
            "rows": 2,
            "projected_rows": 2,
            "population_rows": 2,
            "is_full_population": True,
        },
    )

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    geometry = next(
        row for row in controls.geometry_controls.geometries if row.view_id == "intermediate_embedding_7b_anchor_60bp"
    )
    assert geometry.projection_ids == ["umap_anchor_attention"]


def test_notebook_controls_use_workspace_notebook_geometry_order_and_default_compare(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["views"]["pooled_logits_7b_anchor_60bp"] = {
        "source": "anchor_60bp",
        "vector": {"kind": "column", "name": "embedding"},
        "coordinate_space_id": "demo_space",
        "tags": {"model": "7b", "family": "pooled_logits", "scope": "anchor_60bp"},
    }
    config["notebooks"]["latent_geometry_browser"]["geometry_order"] = [
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_anchor_60bp",
    ]
    config["notebooks"]["latent_geometry_browser"]["candidate_grid_views"] = [
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_anchor_60bp",
    ]
    config["notebooks"]["latent_geometry_browser"]["candidate_grid_panel_titles"] = [
        "Pooled logits",
        "Intermediate",
    ]
    config["notebooks"]["latent_geometry_browser"]["default_layout"] = "candidate_grid"
    config["notebooks"]["latent_geometry_browser"]["default_compare_views"] = [
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_anchor_60bp",
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    geometry_ids = [row.view_id for row in controls.geometry_controls.geometries]
    assert geometry_ids == [
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_anchor_60bp",
    ]
    assert controls.geometry_controls.default_layout == "candidate_grid"
    assert controls.geometry_controls.default_compare_left == "pooled_logits_7b_anchor_60bp"
    assert controls.geometry_controls.default_compare_right == "intermediate_embedding_7b_anchor_60bp"


def test_notebook_controls_resolve_candidate_sets_as_layout_presets(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["views"]["pooled_logits_7b_anchor_60bp"] = {
        "source": "anchor_60bp",
        "vector": {"kind": "column", "name": "embedding"},
        "coordinate_space_id": "demo_logits",
        "tags": {"model": "7b", "family": "pooled_logits", "scope": "anchor_60bp"},
    }
    config["candidate_sets"] = {
        "two_view_x": {
            "label": "Two-view X",
            "description": "Fixture candidate set.",
            "views": [
                "pooled_logits_7b_anchor_60bp",
                "intermediate_embedding_7b_anchor_60bp",
            ],
            "panel_titles": {
                "pooled_logits_7b_anchor_60bp": "Pooled logits",
                "intermediate_embedding_7b_anchor_60bp": "Intermediate",
            },
        }
    }
    config["notebooks"]["latent_geometry_browser"]["candidate_sets"] = ["two_view_x"]
    config["notebooks"]["latent_geometry_browser"]["default_candidate_set"] = "two_view_x"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    geometry_ids = [row.view_id for row in controls.geometry_controls.geometries]
    assert geometry_ids == [
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_anchor_60bp",
    ]
    assert controls.geometry_controls.default_layout == "candidate_set__two_view_x"
    candidate_set = controls.geometry_controls.candidate_sets[0]
    assert candidate_set.candidate_set_id == "two_view_x"
    assert candidate_set.view_ids == geometry_ids
    assert candidate_set.available_view_ids == geometry_ids
    assert candidate_set.panel_titles == ["Pooled logits", "Intermediate"]
    assert [row.status for row in candidate_set.views] == ["missing", "missing"]
    assert {preset.id for preset in controls.geometry_controls.layout_presets} >= {"candidate_set__two_view_x"}


def test_notebook_controls_accept_geometry_browser_as_canonical_surface(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["notebooks"]["latent_geometry_browser"]["default_surface"] = "geometry_browser"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.plot_controls.default_surface == "geometry_browser"


def test_notebook_controls_reject_legacy_surface_aliases(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["notebooks"]["latent_geometry_browser"]["default_surface"] = "geometry_audit"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="default_surface"):
        load_workspace_config(workspace_dir)


def test_notebook_controls_candidate_views_expose_representation_metadata(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["views"]["output_layer_mean_7b_native_source_record_seq_mean"] = {
        "source": "anchor_60bp",
        "vector": {"kind": "column", "name": "embedding"},
        "coordinate_space_id": "evo2_7b_native_source_record_seq_mean",
        "role": "planned",
        "tags": {
            "model": "evo2_7b",
            "family": "output_layer_mean",
            "scope": "native_source_record",
            "pooling": "seq_mean",
        },
    }
    config["candidate_sets"] = {
        "native_output_layer": {
            "label": "Native output layer",
            "views": ["output_layer_mean_7b_native_source_record_seq_mean"],
        }
    }
    config["notebooks"]["latent_geometry_browser"]["candidate_sets"] = ["native_output_layer"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    view = controls.geometry_controls.candidate_sets[0].views[0]
    assert view.status == "planned"
    assert view.model == "evo2_7b"
    assert view.family == "output_layer_mean"
    assert view.scope == "native_source_record"
    assert view.coordinate_space_id == "evo2_7b_native_source_record_seq_mean"
    assert view.label == "Evo 2 7B · Native source record · Output-layer mean"


def test_notebook_controls_expose_reference_set_labels_and_default_from_config(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["reference_sets"] = {
        "heldout_controls": {
            "label": "Heldout controls",
            "match_column": "id",
            "label_column": "subject_id",
            "ids": ["row0"],
        },
        "hidden_controls": {
            "label": "Hidden controls",
            "match_column": "id",
            "ids": ["row1"],
            "notebook_exposed": False,
        },
    }
    config["notebooks"]["latent_geometry_browser"]["default_reference_set"] = "heldout_controls"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.geometry_controls.default_reference_set == "heldout_controls"
    assert [row.reference_set_id for row in controls.geometry_controls.reference_sets] == ["heldout_controls"]
    assert controls.geometry_controls.reference_sets[0].label == "Heldout controls"


def test_notebook_controls_degrade_invalid_plot_manifest_to_error_status(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    semantics_dir = workspace_dir / "plot_semantics"
    semantics_dir.mkdir(parents=True, exist_ok=True)
    (semantics_dir / "atlas_demo_plot.yaml").write_text(
        "plot_id: atlas_demo_plot\n"
        "question: Fixture question?\n"
        "decision_role: appendix\n"
        "encoding: Fixture encoding.\n"
        "scope: Fixture scope.\n"
        "guardrails:\n"
        "  - Fixture guardrail.\n"
        "caption: Fixture caption.\n"
        "alt_text: Fixture alt text.\n"
        "preprocessing_md: Fixture preprocessing.\n"
        "math_md: Fixture math.\n"
        "rationale_md: Fixture rationale.\n"
        "limitations_md: Fixture limits.\n"
        "failure_modes_md: Fixture failure modes.\n",
        encoding="utf-8",
    )
    config["plots"] = {
        "atlas_demo_plot": {
            "kind": "categorical_count",
            "scalar": "dataset_overview_counts",
            "category_column": "category",
            "label_column": "category_label",
            "value_column": "count",
            "semantics_ref": "plot_semantics/atlas_demo_plot.yaml",
        }
    }
    config["deliverables"]["appendix_umap_gallery"]["recipe"] = "plot_recipe"
    config["deliverables"]["appendix_umap_gallery"]["outputs"]["plots"] = ["atlas_demo_plot"]
    config["recipes"]["plot_recipe"] = {
        "steps": [
            {
                "id": "render_plot",
                "op": "plot.render",
                "params": {"plot": "atlas_demo_plot"},
            },
            {
                "id": "generate_notebook",
                "op": "notebook.generate",
                "depends_on": ["render_plot"],
                "params": {"notebook": "latent_geometry_browser"},
            },
        ]
    }
    config["notebooks"]["latent_geometry_browser"]["ordered_plots"] = ["atlas_demo_plot"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    plot_dir = context.output_root / "plots" / "atlas_demo_plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "manifest.json").write_text("{invalid json", encoding="utf-8")

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.plot_controls.ordered_plot_ids == ["atlas_demo_plot"]
    assert controls.plot_controls.plots[0].status == "error"
    assert controls.plot_controls.plots[0].stale is False


def test_notebook_controls_prefer_live_catalog_plot_status(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    semantics_dir = workspace_dir / "plot_semantics"
    semantics_dir.mkdir(parents=True, exist_ok=True)
    (semantics_dir / "atlas_demo_plot.yaml").write_text(
        "plot_id: atlas_demo_plot\n"
        "question: Fixture question?\n"
        "decision_role: appendix\n"
        "encoding: Fixture encoding.\n"
        "scope: Fixture scope.\n"
        "guardrails:\n"
        "  - Fixture guardrail.\n"
        "caption: Fixture caption.\n"
        "alt_text: Fixture alt text.\n"
        "preprocessing_md: Fixture preprocessing.\n"
        "math_md: Fixture math.\n"
        "rationale_md: Fixture rationale.\n"
        "limitations_md: Fixture limits.\n"
        "failure_modes_md: Fixture failure modes.\n",
        encoding="utf-8",
    )
    config["plots"] = {
        "atlas_demo_plot": {
            "kind": "categorical_count",
            "scalar": "dataset_overview_counts",
            "category_column": "category",
            "label_column": "category_label",
            "value_column": "count",
            "semantics_ref": "plot_semantics/atlas_demo_plot.yaml",
        }
    }
    config["deliverables"]["appendix_umap_gallery"]["recipe"] = "plot_recipe"
    config["deliverables"]["appendix_umap_gallery"]["outputs"]["plots"] = ["atlas_demo_plot"]
    config["recipes"]["plot_recipe"] = {
        "steps": [
            {
                "id": "render_plot",
                "op": "plot.render",
                "params": {"plot": "atlas_demo_plot"},
            },
            {
                "id": "generate_notebook",
                "op": "notebook.generate",
                "depends_on": ["render_plot"],
                "params": {"notebook": "latent_geometry_browser"},
            },
        ]
    }
    config["notebooks"]["latent_geometry_browser"]["ordered_plots"] = ["atlas_demo_plot"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    plot_dir = context.output_root / "plots" / "atlas_demo_plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps({"artifact_id": "atlas_demo_plot", "status": "ok", "stale": False, "outputs": []}),
        encoding="utf-8",
    )

    controls = build_workspace_notebook_controls_payload(
        context,
        notebook_id="latent_geometry_browser",
        catalog_payload={"plots": [{"plot_id": "atlas_demo_plot", "status": "attention", "stale": True}]},
    )

    assert controls.plot_controls.plots[0].status == "attention"
    assert controls.plot_controls.plots[0].stale is True


def test_notebook_controls_exclude_hidden_model_joinable_tables(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    for artifact_id, view_id in [
        ("design_centroid_margins_intermediate_embedding_7b_anchor_60bp", "intermediate_embedding_7b_anchor_60bp"),
        ("design_centroid_margins_intermediate_embedding_20b_anchor_60bp", "intermediate_embedding_20b_anchor_60bp"),
    ]:
        scalar_dir = context.output_root / "scalars" / artifact_id
        scalar_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "id": ["row0", "row1"],
                "synthetic_margin_ethanol_vs_background": [0.25, -0.1],
            }
        ).to_parquet(scalar_dir / "table.parquet", index=False)
        _write_artifact_manifest(
            scalar_dir,
            artifact_kind="scalar_table",
            artifact_id=artifact_id,
            inputs=[
                {
                    "kind": "view_matrix",
                    "id": view_id,
                }
            ],
        )

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    joinable_artifact_ids = {table.artifact_id for table in controls.geometry_controls.joinable_tables}
    assert "design_centroid_margins_intermediate_embedding_7b_anchor_60bp" in joinable_artifact_ids
    assert "design_centroid_margins_intermediate_embedding_20b_anchor_60bp" not in joinable_artifact_ids


def test_notebook_controls_only_surface_preferred_hues_backed_by_joinable_tables(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    scalar_dir = context.output_root / "scalars" / "context_delta_distribution_intermediate_embedding_7b_anchor_60bp"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["row0", "row1"],
            "context_shift_l2": [0.1, 0.2],
        }
    ).to_parquet(scalar_dir / "table.parquet", index=False)
    _write_artifact_manifest(
        scalar_dir,
        artifact_kind="scalar_table",
        artifact_id="context_delta_distribution_intermediate_embedding_7b_anchor_60bp",
        inputs=[
            {
                "kind": "view_matrix",
                "id": "intermediate_embedding_7b_anchor_60bp",
            }
        ],
    )

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.geometry_controls.preferred_hues == ["context_shift_l2"]
    assert controls.geometry_controls.hue_kinds == {"context_shift_l2": "continuous"}


def test_notebook_controls_surface_configured_hues_backed_by_view_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["notebooks"]["latent_geometry_browser"]["preferred_hues"] = [
        "source_family",
        "promoter_standard__strength_value_numeric",
    ]
    config["notebooks"]["latent_geometry_browser"]["preferred_hue_kinds"] = {
        "source_family": "categorical",
        "promoter_standard__strength_value_numeric": "continuous",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["row0", "row1"],
            "subject_id": ["row0", "row1"],
            "source_family": ["reference", "densegen"],
            "promoter_standard__strength_value_numeric": [0.35, None],
        }
    ).to_parquet(view_dir / "rows.parquet", index=False)
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.geometry_controls.joinable_tables == []
    assert controls.geometry_controls.preferred_hues == [
        "source_family",
        "promoter_standard__strength_value_numeric",
    ]
    assert controls.geometry_controls.row_metadata_hues == [
        "source_family",
        "promoter_standard__strength_value_numeric",
    ]
    assert controls.geometry_controls.hue_kinds == {
        "source_family": "categorical",
        "promoter_standard__strength_value_numeric": "continuous",
    }


def test_notebook_controls_reuse_materialized_view_shape_reads(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_id = "intermediate_embedding_7b_anchor_60bp"
    view_dir = context.output_root / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    matrix_path = view_dir / "matrix.npy"
    calls: list[Path] = []
    real_load = np.load

    def counted_load(path, *args, **kwargs):
        calls.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert [row.view_id for row in controls.candidate_inventory] == [view_id]
    assert [row.view_id for row in controls.geometry_controls.geometries] == [view_id]
    assert calls == [matrix_path]


def test_notebook_controls_ignore_legacy_scalar_tables_without_manifest_bindings(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    scalar_dir = context.output_root / "scalars" / "legacy_debug_distribution"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["row0", "row1"],
            "context_shift_l2": [0.1, 0.2],
        }
    ).to_parquet(scalar_dir / "table.parquet", index=False)

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    joinable_artifact_ids = {table.artifact_id for table in controls.geometry_controls.joinable_tables}
    assert "legacy_debug_distribution" not in joinable_artifact_ids
    assert "context_shift_l2" not in controls.geometry_controls.preferred_hues


def test_notebook_controls_ignore_stale_scalar_tables_with_manifest_bindings(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    context = load_workspace_config(workspace_dir)

    view_dir = context.output_root / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    scalar_dir = context.output_root / "scalars" / "stale_distribution"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["row0", "row1"],
            "context_shift_l2": [0.1, 0.2],
        }
    ).to_parquet(scalar_dir / "table.parquet", index=False)
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "scalar_table",
                "artifact_id": "stale_distribution",
                "status": "attention",
                "inputs": [{"kind": "view_rows", "id": "intermediate_embedding_7b_anchor_60bp"}],
            }
        ),
        encoding="utf-8",
    )

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    joinable_artifact_ids = {table.artifact_id for table in controls.geometry_controls.joinable_tables}
    assert "stale_distribution" not in joinable_artifact_ids
    assert "context_shift_l2" not in controls.geometry_controls.preferred_hues


def test_notebook_controls_prefer_default_deliverable_for_shared_plots(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    plot_dir = output_root / "plots" / "atlas_demo_plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps({"artifact_id": "atlas_demo_plot", "status": "ok", "stale": False, "outputs": []}),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        output_root=output_root,
        config=SimpleNamespace(
            deliverables={
                "appendix_umap_gallery": SimpleNamespace(title="Appendix", outputs={"plots": ["atlas_demo_plot"]}),
                "shared_review": SimpleNamespace(title="Shared review", outputs={"plots": ["atlas_demo_plot"]}),
            }
        ),
        require_notebook=lambda notebook_id: SimpleNamespace(
            default_surface="plots",
            ordered_plots=["atlas_demo_plot"],
            default_deliverable="shared_review",
        ),
        require_plot=lambda plot_id: SimpleNamespace(visibility_tier="primary"),
    )

    controls = _plot_controls(context, notebook_id="latent_geometry_browser")

    assert controls.plots[0].deliverable_id == "shared_review"
    assert controls.plots[0].deliverable_title == "Shared review"
