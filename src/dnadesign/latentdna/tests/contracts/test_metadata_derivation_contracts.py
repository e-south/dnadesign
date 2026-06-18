"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/contracts/test_metadata_derivation_contracts.py

Metadata derivation contract tests for source-backed view rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.views.materialize import materialize_view_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_generic_row_and_workspace_contracts_do_not_import_promoter_metadata() -> None:
    root = _repo_root()
    paths = [
        root / "src/dnadesign/latentdna/src/views/row_contracts.py",
        root / "src/dnadesign/latentdna/src/contracts/workspace.py",
    ]

    offenders = {
        path.name: sorted(name for name in _imported_module_names(path) if "promoter_metadata" in name)
        for path in paths
    }

    assert offenders == {"row_contracts.py": [], "workspace.py": []}


def test_annotation_derivation_materializes_from_explicit_workspace_contract(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["row_a", "row_b"],
                "subject_id": ["row_a", "row_b"],
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
                "usr_label__primary": ["candidate_a", "spyp"],
                "densegen__plan": ["ethanol__sig35=f", None],
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace: {id: explicit_annotation_demo, output_root: ./outputs}
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  records:
    kind: parquet
    path: inputs/records.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: [sig35_variant]
  derivations:
    sig35_variant:
      kind: annotation
      source: row
      handler: dnadesign.latentdna.src.views.promoter_metadata:derive_promoter_metadata_value
      derive: sig35_variant
      required_columns: [usr_label__primary]
      any_required_column_groups:
        - [densegen__plan]
        - [densegen__used_tfbs_detail]
        - [seq_annot__features]
        - [sequence, derived__features_retained]
      missing_policy: error
      value_type: string
views:
  demo_view:
    source: records
    vector: {kind: column, name: embedding}
    coordinate_space_id: demo_space
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    context = load_workspace_config(workspace_dir)

    artifact_dir, *_ = materialize_view_artifact(context, view_id="demo_view")

    rows = read_table(artifact_dir / "rows.parquet")
    assert rows["sig35_variant"].to_pylist() == ["f", "control"]
