"""Contract tests for workspace notebook control-plane assembly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.services.notebook_controls_service import build_workspace_notebook_controls_payload
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


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
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
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
                        "requires": {"views": ["intermediate_embedding_20b_anchor_60bp"]},
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
                                "params": {"view": "intermediate_embedding_20b_anchor_60bp"},
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

    view_dir = context.output_root / "views" / "intermediate_embedding_20b_anchor_60bp"
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
        (projection_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "inputs": [{"kind": "view_matrix", "id": "intermediate_embedding_20b_anchor_60bp"}],
                    "params": {
                        "projection_role": role,
                        "default_rank": default_rank,
                        "sampling_strategy": "all",
                    },
                    "stats": {
                        "rows": 2,
                        "projected_rows": 2,
                        "population_rows": 2,
                        "is_full_population": True,
                    },
                }
            ),
            encoding="utf-8",
        )

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert controls.schema_version == "latentdna.workspace_notebook_controls.v4"
    assert controls.plot_controls.default_surface == "plots"
    assert controls.plot_controls.ordered_plot_ids == []
    geometry = next(
        row for row in controls.geometry_controls.geometries if row.view_id == "intermediate_embedding_20b_anchor_60bp"
    )
    assert geometry.projection_ids == ["umap_anchor", "audit_umap_anchor"]
