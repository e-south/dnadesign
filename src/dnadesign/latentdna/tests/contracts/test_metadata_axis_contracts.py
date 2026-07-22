"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/contracts/test_metadata_axis_contracts.py

Metadata-axis style contract tests for generic LatentDNA runtimes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.metadata.axes import (
    axis_display_text,
    axis_style_map_from_config,
    legend_categories,
    normalize_axis_categories,
    normalize_axis_category,
)
from dnadesign.latentdna.src.services.notebook_controls_service import build_workspace_notebook_controls_payload
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def test_stress_workspace_declares_sigma35_axis_semantics_in_config() -> None:
    context = load_workspace_config(Path("src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth"))

    styles = axis_style_map_from_config(context.config)
    style = styles["sig35_variant"]

    assert style.axis_id == "sigma35"
    assert style.column == "sig35_variant"
    assert style.category_order == ["f", "e", "d", "c", "b", "control"]
    assert style.ordinal_subset == ["f", "e", "d", "c", "b"]
    assert axis_display_text(style, "f") == "TTGACA (f)"
    assert axis_display_text(style, style.noncanonical_bucket) == "Reference/other"
    assert normalize_axis_category(style, "TTTACA", row={"source_class": "manual_or_wildtype"}) == (
        style.noncanonical_bucket
    )
    assert normalize_axis_category(style, "b", row={"source_family": "densegen_generated"}) == "b"
    batch_values = ["TTTACA", "b", "control", None]
    batch_rows = [
        {"source_class": "manual_or_wildtype"},
        {"source_family": "densegen_generated"},
        {"source_family": "densegen_generated"},
        {"source_family": "densegen_generated"},
    ]
    assert normalize_axis_categories(style, batch_values, rows=batch_rows) == [
        normalize_axis_category(style, value, row=row) for value, row in zip(batch_values, batch_rows, strict=True)
    ]
    assert legend_categories(style, [style.noncanonical_bucket, "b", "f", "control"]) == ["f", "b", "control"]


def test_notebook_controls_publish_resolved_metadata_axis_styles(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["row0", "row1"],
                "subject_id": ["row0", "row1"],
                "family": ["b", "a"],
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "features.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "axis_style_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "features": {
                        "kind": "parquet",
                        "path": "inputs/features.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["family"],
                    "axes": {
                        "family_axis": {
                            "column": "family",
                            "label": "Family",
                            "category_order": ["a", "b"],
                            "display_labels": {"a": "Family A", "b": "Family B"},
                            "category_colors": {"a": "#111111", "b": "#0072B2"},
                            "ordinal_subset": ["a", "b"],
                        }
                    },
                },
                "views": {
                    "embedding": {
                        "source": "features",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_embedding",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "demo"},
                    }
                },
                "plots": {},
                "notebooks": {
                    "latent_geometry_browser": {
                        "kind": "workspace",
                        "title": "Browser",
                        "default_deliverable": "browser",
                        "preferred_hues": ["family"],
                        "preferred_hue_kinds": {"family": "ordinal"},
                    }
                },
                "deliverables": {
                    "browser": {
                        "title": "Browser",
                        "section": "Browser",
                        "question": "Can axis styles be inspected?",
                        "summary": "Axis-style controls fixture.",
                        "recipe": "noop",
                        "requires": {"views": ["embedding"]},
                        "outputs": {"views": ["embedding"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
                "recipes": {
                    "noop": {
                        "steps": [
                            {
                                "id": "materialize_embedding",
                                "op": "view.materialize",
                                "params": {"view": "embedding"},
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    context = load_workspace_config(workspace_dir)
    view_dir = context.output_root / "views" / "embedding"
    view_dir.mkdir(parents=True)
    pd.DataFrame({"id": ["row0", "row1"], "subject_id": ["row0", "row1"], "family": ["b", "a"]}).to_parquet(
        view_dir / "rows.parquet",
        index=False,
    )
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.geometry_controls.axis_styles["family"].axis_id == "family_axis"
    assert controls.geometry_controls.axis_styles["family"].category_order == ["a", "b"]
    assert controls.geometry_controls.axis_styles["family"].display_labels == {
        "a": "Family A",
        "b": "Family B",
    }
