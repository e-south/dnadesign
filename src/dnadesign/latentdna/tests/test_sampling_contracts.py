"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_sampling_contracts.py

Sampling contract tests for ledger-first latentdna sample building.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa

from dnadesign.latentdna.src.samples import build as sample_build_module
from dnadesign.latentdna.src.samples.build import build_sample_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_workspace(tmp_path) -> tuple[object, str]:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    outputs = workspace_dir / "outputs" / "views" / "z20_60"
    outputs.mkdir(parents=True)
    rows = pa.table(
        {
            "id": ["row_01", "row_02", "row_03", "row_04"],
            "densegen__plan": ["plan_a", "plan_a", "plan_b", "plan_b"],
            "source_class": ["densegen", "native_regulondb", "manual_or_wildtype", "densegen"],
        }
    )
    from dnadesign.latentdna.src.io.parquet_io import write_table

    write_table(rows, outputs / "rows.parquet")
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: ./inputs/anchor60.parquet
    record_key: id
    subject_key: id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo
    tags: {model: demo}
    role: primary
reference_sets:
  promoter_wt_core:
    ids: [row_01, row_04]
    match_column: id
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    return load_workspace_config(workspace_dir), "z20_60"


def test_sample_build_all_strategy_does_not_require_table_to_pylist(tmp_path, monkeypatch) -> None:
    context, view_id = _write_workspace(tmp_path)

    class _TableWithoutToPylist:
        def __init__(self, inner: pa.Table) -> None:
            self._inner = inner

        def to_pylist(self):
            raise AssertionError("Table.to_pylist() must not be used for ledger-first sample building")

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def __getitem__(self, key):
            return self._inner[key]

    from dnadesign.latentdna.src.io.parquet_io import read_table as _read_table

    monkeypatch.setattr(
        sample_build_module,
        "read_table",
        lambda path: _TableWithoutToPylist(_read_table(path)),
    )

    artifact_dir, rows = build_sample_artifact(
        context,
        sample_id="all_rows",
        view_id=view_id,
        strategy="all",
        n=None,
        group_column=None,
        seed=17,
        explicit_ids=None,
    )

    assert artifact_dir.is_dir()
    assert rows == 4


def test_sample_build_supports_explicit_ids_without_row_table_expansion(tmp_path, monkeypatch) -> None:
    context, view_id = _write_workspace(tmp_path)

    class _TableWithoutToPylist:
        def __init__(self, inner: pa.Table) -> None:
            self._inner = inner

        def to_pylist(self):
            raise AssertionError("Table.to_pylist() must not be used for explicit-id sample building")

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def __getitem__(self, key):
            return self._inner[key]

    from dnadesign.latentdna.src.io.parquet_io import read_table as _read_table

    monkeypatch.setattr(
        sample_build_module,
        "read_table",
        lambda path: _TableWithoutToPylist(_read_table(path)),
    )

    artifact_dir, rows = build_sample_artifact(
        context,
        sample_id="explicit_rows",
        view_id=view_id,
        strategy="explicit_ids",
        n=None,
        group_column=None,
        seed=17,
        explicit_ids=["row_04", "row_02"],
    )

    from dnadesign.latentdna.src.io.parquet_io import read_table

    sample_rows = read_table(artifact_dir / "rows.parquet").to_pydict()
    assert rows == 2
    assert sample_rows["id"] == ["row_02", "row_04"]


def test_sample_build_filters_rows_before_sampling(tmp_path) -> None:
    context, view_id = _write_workspace(tmp_path)

    artifact_dir, rows = build_sample_artifact(
        context,
        sample_id="filtered_candidate_rows",
        view_id=view_id,
        strategy="all",
        n=None,
        group_column=None,
        seed=17,
        explicit_ids=None,
        where={"column": "source_class", "in": ["densegen", "manual_or_wildtype"]},
    )

    from dnadesign.latentdna.src.io.parquet_io import read_table

    sample_rows = read_table(artifact_dir / "rows.parquet").to_pydict()
    assert rows == 3
    assert sample_rows["id"] == ["row_01", "row_03", "row_04"]


def test_sample_build_supports_union_and_intersection(tmp_path) -> None:
    context, view_id = _write_workspace(tmp_path)
    outputs = context.output_root / "samples"
    outputs.mkdir(parents=True, exist_ok=True)

    from dnadesign.latentdna.src.io.parquet_io import read_table, write_table

    base_rows = read_table(context.output_root / "views" / view_id / "rows.parquet")
    write_table(base_rows.take(pa.array([0, 1, 3], type=pa.int64())), outputs / "left" / "rows.parquet")
    write_table(base_rows.take(pa.array([1, 2], type=pa.int64())), outputs / "right" / "rows.parquet")

    union_dir, union_rows = build_sample_artifact(
        context,
        sample_id="union_rows",
        view_id=None,
        strategy="union",
        n=None,
        group_column=None,
        seed=17,
        explicit_ids=None,
        input_sample_ids=["left", "right"],
    )
    intersection_dir, intersection_rows = build_sample_artifact(
        context,
        sample_id="intersection_rows",
        view_id=None,
        strategy="intersection",
        n=None,
        group_column=None,
        seed=17,
        explicit_ids=None,
        input_sample_ids=["left", "right"],
    )

    union_sample_rows = read_table(union_dir / "rows.parquet").to_pydict()
    intersection_sample_rows = read_table(intersection_dir / "rows.parquet").to_pydict()

    assert union_rows == 4
    assert union_sample_rows["id"] == ["row_01", "row_02", "row_04", "row_03"]
    assert intersection_rows == 1
    assert intersection_sample_rows["id"] == ["row_02"]


def test_sample_build_stratified_preserves_reference_set_rows(tmp_path) -> None:
    context, view_id = _write_workspace(tmp_path)

    artifact_dir, rows = build_sample_artifact(
        context,
        sample_id="stratified_reference_rows",
        view_id=view_id,
        strategy="stratified",
        n=1,
        group_column="densegen__plan",
        seed=17,
        reference_set_id="promoter_wt_core",
    )

    from dnadesign.latentdna.src.io.parquet_io import read_table

    sample_rows = read_table(artifact_dir / "rows.parquet").to_pydict()
    assert rows >= 2
    assert {"row_01", "row_04"}.issubset(set(sample_rows["id"]))


def test_sample_build_stratified_preserves_selector_reference_set_rows(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    outputs = workspace_dir / "outputs" / "views" / "z20_60"
    outputs.mkdir(parents=True)
    rows = pa.table(
        {
            "id": ["row_01", "row_02", "row_03", "row_04"],
            "densegen__plan": ["plan_a", "plan_a", "plan_b", "plan_b"],
            "source_family": ["reference_source", "densegen_generated", "construct_derived", "densegen_generated"],
        }
    )
    from dnadesign.latentdna.src.io.parquet_io import write_table

    write_table(rows, outputs / "rows.parquet")
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: ./inputs/anchor60.parquet
    record_key: id
    subject_key: id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo
    tags: {model: demo}
    role: primary
reference_sets:
  reference_rows:
    match_column: id
    where:
      - column: source_family
        in_values: [reference_source, construct_derived]
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    context = load_workspace_config(workspace_dir)

    artifact_dir, rows_count = build_sample_artifact(
        context,
        sample_id="stratified_selector_reference_rows",
        view_id="z20_60",
        strategy="stratified",
        n=1,
        group_column="densegen__plan",
        seed=17,
        reference_set_id="reference_rows",
    )

    from dnadesign.latentdna.src.io.parquet_io import read_table

    sample_rows = read_table(artifact_dir / "rows.parquet").to_pydict()
    assert rows_count >= 2
    assert {"row_01", "row_03"}.issubset(set(sample_rows["id"]))
