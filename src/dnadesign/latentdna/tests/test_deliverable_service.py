"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_deliverable_service.py

Deliverable status service contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.contracts.manifest import ArtifactManifest, ArtifactOutput
from dnadesign.latentdna.src.services._artifact_inputs import artifact_input_from_path
from dnadesign.latentdna.src.services.deliverable_service import deliverable_status


def test_deliverable_status_accepts_recipe_produced_scalar_output_without_scalar_config(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    rows_path = inputs_dir / "rows.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
            }
        ),
        rows_path,
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "recipe_scalar_output_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "rows": {
                        "kind": "parquet",
                        "path": "inputs/rows.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": []},
                "recipes": {
                    "review_recipe": {
                        "steps": [
                            {
                                "id": "build_recipe_scalar",
                                "op": "scalar.build",
                                "params": {"scalar": "recipe_scalar", "kind": "fixture_scalar"},
                            }
                        ]
                    }
                },
                "deliverables": {
                    "review_bundle": {
                        "title": "Review bundle",
                        "section": "Review",
                        "question": "Does status treat recipe scalar outputs as generated artifacts?",
                        "summary": "Regression fixture for recipe-produced scalar outputs.",
                        "recipe": "review_recipe",
                        "requires": {"sources": ["rows"]},
                        "outputs": {"scalars": ["recipe_scalar"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scalar_dir = workspace_dir / "outputs" / "scalars" / "recipe_scalar"
    scalar_dir.mkdir(parents=True)
    pq.write_table(pa.table({"metric_value": pa.array([1.0], type=pa.float64())}), scalar_dir / "table.parquet")
    manifest = ArtifactManifest(
        artifact_kind="scalar_table",
        artifact_id="recipe_scalar",
        workspace_id="recipe_scalar_output_demo",
        created_at=datetime.now(UTC).isoformat(),
        tool_version="test",
        command="scalar build",
        inputs=[artifact_input_from_path("source", "rows", rows_path)],
        params={"builder_kind": "fixture_scalar"},
        outputs=[ArtifactOutput(path="table.parquet", media_type="application/x-parquet")],
        stats={"rows": 1, "columns": 1},
    )
    (scalar_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    status = deliverable_status(workspace_dir, "review_bundle")

    assert status.status == "ok"
    outputs = {entry.name: entry for entry in status.outputs}
    assert outputs["scalar_table:recipe_scalar"].status == "ok"
