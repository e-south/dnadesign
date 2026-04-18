"""Scalar builder coverage for notebook-facing cohort inventory tables."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.build import build_scalar_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_workspace_config(
    workspace_dir: Path,
    *,
    anchor_record_key: str = "id",
    anchor_subject_key: str = "subject_id",
    context_record_key: str = "id",
    context_subject_key: str = "subject_id",
) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "dataset_overview_demo", "output_root": "./outputs"},
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
                        "record_key": anchor_record_key,
                        "subject_key": anchor_subject_key,
                    },
                    "full_context_1kb": {
                        "kind": "parquet",
                        "path": "inputs/context.parquet",
                        "record_key": context_record_key,
                        "subject_key": context_subject_key,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _base_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "row_1",
            "subject_id": "row_1",
            "usr_label__primary": "dense_row_1",
            "densegen__plan": "background_only__sig35=f",
            "densegen__required_regulators": [],
        },
        {
            "id": "row_2",
            "subject_id": "row_2",
            "usr_label__primary": "dense_row_2",
            "densegen__plan": "ethanol__sig35=d",
            "densegen__required_regulators": ["baeR"],
        },
        {
            "id": "row_3",
            "subject_id": "row_3",
            "usr_label__primary": "dense_row_3",
            "densegen__plan": "ciprofloxacin__sig35=e",
            "densegen__required_regulators": ["lexA"],
        },
        {
            "id": "row_4",
            "subject_id": "row_4",
            "usr_label__primary": "dense_row_4",
            "densegen__plan": "ethanol_ciprofloxacin__sig35=b",
            "densegen__required_regulators": ["baeR", "lexA"],
        },
        {
            "id": "row_5",
            "subject_id": "row_5",
            "usr_label__primary": "dense_row_5",
            "densegen__plan": "ethanol__sig35=c",
            "densegen__required_regulators": ["cpxR"],
        },
        {
            "id": "control",
            "subject_id": "control",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": [],
        },
    ]


def test_dataset_overview_builds_dimension_panels_with_shared_denominator(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    rows = _base_rows()
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="dataset_overview_counts",
        builder_kind="dataset_overview",
        params={"source_ids": ["anchor_60bp", "full_context_1kb"]},
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    dimensions = {row["dimension"] for row in table}

    assert artifact.stats["denominator"] == 6
    assert dimensions == {"provenance", "generation_plan", "sig35_variant"}
    assert "anchor_60bp" not in dimensions
    assert "full_context_1kb" not in dimensions
    assert {row["dimension_label"] for row in table} == {"Provenance", "Generation plan", "Sigma-35 variant"}

    for dimension in dimensions:
        dimension_rows = [row for row in table if row["dimension"] == dimension]
        assert sum(int(row["count"]) for row in dimension_rows) == 6
        assert pytest.approx(sum(float(row["fraction"]) for row in dimension_rows), rel=0, abs=1e-9) == 1.0


def test_dataset_overview_rejects_mismatched_partitions_across_sources(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    anchor_rows = _base_rows()
    context_rows = _base_rows()
    context_rows[1] = {
        **context_rows[1],
        "densegen__plan": "background_only__sig35=d",
    }
    _write_source(workspace_dir / "inputs" / "anchor.parquet", anchor_rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", context_rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)

    with pytest.raises(ContractViolationError, match="matching cohort partitions across sources"):
        build_scalar_artifact(
            context,
            scalar_id="dataset_overview_counts",
            builder_kind="dataset_overview",
            params={"source_ids": ["anchor_60bp", "full_context_1kb"]},
        )


def test_dataset_overview_accepts_aligned_subject_populations_with_distinct_record_keys(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(
        workspace_dir,
        anchor_record_key="id",
        anchor_subject_key="subject_id",
        context_record_key="context_id",
        context_subject_key="anchor_id",
    )
    anchor_rows = _base_rows()
    context_rows = [
        {
            "context_id": f"context_{index}",
            "anchor_id": row["id"],
            "usr_label__primary": row["usr_label__primary"],
            "densegen__plan": row["densegen__plan"],
            "densegen__required_regulators": row["densegen__required_regulators"],
        }
        for index, row in enumerate(anchor_rows, start=1)
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", anchor_rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", context_rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="dataset_overview_counts",
        builder_kind="dataset_overview",
        params={"source_ids": ["anchor_60bp", "full_context_1kb"]},
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert sum(int(row["count"]) for row in table if row["dimension"] == "provenance") == len(anchor_rows)
