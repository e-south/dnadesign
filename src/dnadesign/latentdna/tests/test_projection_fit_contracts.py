"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_projection_fit_contracts.py

Projection-fit contract tests for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.projections import fit as projection_fit
from dnadesign.latentdna.src.projections.fit import _fit_projection_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config

_RUNNER = CliRunner()


def _write_source(path: Path) -> None:
    rows = [
        {"id": f"r{index}", "subject_id": f"s{index}", "embedding": [float(index), float(index + 1), float(index + 2)]}
        for index in range(4)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_workspace_config(workspace_dir: Path, source_path: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "projection_fit_contracts_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "demo": {
                        "kind": "parquet",
                        "path": source_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": []},
                "views": {
                    "z_demo": {
                        "source": "demo",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_fit_projection_artifact_reuses_view_matrix_for_full_population_all_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    source_path = tmp_path / "inputs" / "demo.parquet"
    _write_source(source_path)
    _write_workspace_config(workspace_dir, source_path)

    materialize = _RUNNER.invoke(
        app,
        ["view", "materialize", "z_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize.exit_code == 0, materialize.stdout

    from dnadesign.latentdna.src.services.sample_service import build_sample

    build_sample(
        workspace_dir,
        "demo_all",
        view_id="z_demo",
        strategy="all",
        n=None,
        group_column=None,
        seed=17,
    )
    context = load_workspace_config(workspace_dir)

    captured: dict[str, object] = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit_transform(self, data):
            captured["data"] = data
            return np.zeros((data.shape[0], 2), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    monkeypatch.setattr(
        projection_fit,
        "_ordered_indices",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_ordered_indices should not run")),
    )

    artifact_dir, rows = _fit_projection_artifact(
        context,
        view_id="z_demo",
        projection_id="umap_z_demo",
        sample_id="demo_all",
        metric="cosine",
        seed=17,
        artifact_dir=workspace_dir / "outputs" / "projections" / "umap_z_demo",
    )

    assert artifact_dir == workspace_dir / "outputs" / "projections" / "umap_z_demo"
    assert rows == 4
    fitted = captured["data"]
    assert isinstance(fitted, np.memmap)
    assert fitted.shape == (4, 3)
