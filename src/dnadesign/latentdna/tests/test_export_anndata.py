from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli.app import app
from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.recipe import expected_step_artifacts
from dnadesign.latentdna.src.services.export_service import export_anndata

_RUNNER = CliRunner()


def _write_manifest(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(json.dumps(payload or {}), encoding="utf-8")


def _write_demo_workspace(tmp_path: Path) -> Path:
    workspace_dir = tmp_path / "workspace"
    outputs_dir = workspace_dir / "outputs"
    reduced_dir = outputs_dir / "reduced_views" / "demo_reduced"
    projection_dir = outputs_dir / "projections" / "demo_projection"
    neighbors_dir = outputs_dir / "neighbors" / "demo_neighbors"
    reduced_dir.mkdir(parents=True)
    projection_dir.mkdir(parents=True)
    neighbors_dir.mkdir(parents=True)

    np.save(
        reduced_dir / "matrix.npy",
        np.asarray(
            [
                [1.0, 0.5],
                [0.0, 2.0],
                [3.0, 1.5],
            ],
            dtype=np.float32,
        ),
    )
    rows = pa.table(
        {
            "id": pa.array(["row_a", "row_b", "row_c"], type=pa.string()),
            "label": pa.array(["reference", "candidate", "candidate"], type=pa.string()),
        }
    )
    pq.write_table(rows, reduced_dir / "rows.parquet")

    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a", "row_b", "row_c"], type=pa.string()),
                "x": pa.array([0.0, 1.0, 2.0], type=pa.float32()),
                "y": pa.array([2.0, 1.0, 0.0], type=pa.float32()),
            }
        ),
        projection_dir / "coords.parquet",
    )
    _write_manifest(projection_dir / "manifest.json", {"params": {"method": "umap"}})

    pq.write_table(rows, neighbors_dir / "rows.parquet")
    np.save(neighbors_dir / "indices.npy", np.asarray([[1, 2], [0, 2], [1, 0]], dtype=np.int64))
    np.save(
        neighbors_dir / "distances.npy",
        np.asarray([[0.1, 0.2], [0.1, 0.3], [0.3, 0.2]], dtype=np.float32),
    )
    _write_manifest(neighbors_dir / "manifest.json", {"params": {"k": 2, "metric": "cosine"}})

    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "anndata_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "demo_source": {
                        "kind": "parquet",
                        "path": "inputs/demo.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    }
                },
                "views": {
                    "demo_view": {
                        "source": "demo_source",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
                "exports": {
                    "demo_bundle": {
                        "row_basis": "demo_reduced",
                        "metadata_columns": ["label"],
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "demo",
                                "source": "demo_reduced",
                                "feature_prefix": "demo",
                            }
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace_dir


def test_export_anndata_writes_h5ad_with_optional_projection_and_neighbors(tmp_path: Path) -> None:
    workspace_dir = _write_demo_workspace(tmp_path)

    result = export_anndata(
        workspace_dir,
        "demo_bundle",
        projection_ids=["demo_projection"],
        neighbor_ids=["demo_neighbors"],
    )

    bundle_path = workspace_dir / "outputs" / "exports" / "demo_bundle" / "bundle.h5ad"
    manifest = json.loads((bundle_path.parent / "manifest.json").read_text(encoding="utf-8"))
    exported = ad.read_h5ad(bundle_path)

    assert result.command == "export anndata"
    assert result.metrics["rows"] == 3
    assert result.metrics["dims"] == 2
    assert exported.shape == (3, 2)
    assert list(exported.obs_names) == ["row_a", "row_b", "row_c"]
    assert exported.obs["label"].to_list() == ["reference", "candidate", "candidate"]
    assert list(exported.var_names) == ["demo_pc_001", "demo_pc_002"]
    assert "X_umap_demo_projection" in exported.obsm
    assert exported.obsm["X_umap_demo_projection"].shape == (3, 2)
    assert "neighbors_demo_neighbors_distances" in exported.obsp
    assert exported.obsp["neighbors_demo_neighbors_distances"].shape == (3, 3)
    assert exported.uns["latentdna_export"]["row_basis"] == "demo_reduced"
    assert manifest["outputs"][0]["path"] == "bundle.h5ad"
    assert manifest["params"]["projection_ids"] == ["demo_projection"]
    assert manifest["params"]["neighbor_ids"] == ["demo_neighbors"]
    assert any(item["path"].endswith("/indices.npy") for item in manifest["inputs"])


def test_export_anndata_rejects_misaligned_projection_rows(tmp_path: Path) -> None:
    workspace_dir = _write_demo_workspace(tmp_path)
    projection_path = workspace_dir / "outputs" / "projections" / "demo_projection" / "coords.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_c", "row_b", "row_a"], type=pa.string()),
                "x": pa.array([0.0, 1.0, 2.0], type=pa.float32()),
                "y": pa.array([2.0, 1.0, 0.0], type=pa.float32()),
            }
        ),
        projection_path,
    )

    with pytest.raises(ContractViolationError, match="row ordering"):
        export_anndata(workspace_dir, "demo_bundle", projection_ids=["demo_projection"])


def test_export_anndata_cli_dry_run_and_recipe_artifact_contract(tmp_path: Path) -> None:
    workspace_dir = _write_demo_workspace(tmp_path)

    result = _RUNNER.invoke(
        app,
        [
            "export",
            "anndata",
            "demo_bundle",
            "--workspace",
            str(workspace_dir),
            "--dry-run",
            "--json",
        ],
    )
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["command"] == "export anndata"
    assert payload["dry_run"] is True
    assert expected_step_artifacts("export.anndata", {"export": "demo_bundle"}) == [("export_bundle", "demo_bundle")]
