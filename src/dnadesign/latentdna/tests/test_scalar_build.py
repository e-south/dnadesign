"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_scalar_build.py

Scalar builder coverage for notebook-facing cohort inventory tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.build import _cosine_distance_correlation, build_scalar_artifact
from dnadesign.latentdna.src.scalars.builders.representation_scorecard import _reference_neighbor_metrics
from dnadesign.latentdna.src.scalars.common import _normalized_geometry_rows
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config

_PROMOTER_METADATA_HANDLER = "dnadesign.latentdna.src.views.promoter_metadata:derive_promoter_metadata_value"
_STRESS_RETENTION_AXES = [
    {
        "axis_id": "design_family",
        "column": "design_family",
        "metric_id": "design_family_retention_correlation",
        "exclude_values": ["control"],
    },
    {
        "axis_id": "design_regulator_composition",
        "column": "design_regulator_composition",
        "metric_id": "design_regulator_composition_retention_correlation",
        "exclude_values": ["control"],
    },
    {
        "axis_id": "sigma35",
        "column": "sig35_variant",
        "metric_id": "sig35_variant_retention_correlation",
        "exclude_values": ["control"],
    },
]
_STRESS_METRIC_DEFINITIONS = {
    "sig35_variant_separation_ratio": {
        "display_name": "Sigma-35 separation ratio",
        "mathematical_definition": (
            "Mean between-centroid cosine distance divided by mean within-centroid cosine distance "
            "for configured Sigma-35 cohorts."
        ),
        "metric_family": "cohort_structure",
        "evidence_tier": "primary",
        "unit": "ratio",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_ordinal_spearman": {
        "display_name": "Sigma-35 ordinal Spearman",
        "mathematical_definition": (
            "Spearman correlation between configured Sigma-35 rank gaps and observed centroid cosine-distance gaps."
        ),
        "metric_family": "ordinal_structure",
        "evidence_tier": "primary",
        "unit": "correlation",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_ordinal_kendall": {
        "display_name": "Sigma-35 ordinal Kendall",
        "mathematical_definition": (
            "Kendall tau correlation between configured Sigma-35 rank gaps and observed centroid cosine-distance gaps."
        ),
        "metric_family": "ordinal_structure",
        "evidence_tier": "primary",
        "unit": "correlation",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_balanced_ordinal_spearman": {
        "display_name": "Balanced Sigma-35 ordinal Spearman",
        "mathematical_definition": (
            "Spearman correlation between configured Sigma-35 rank gaps and observed centroid cosine-distance gaps "
            "after config-declared cohort balancing."
        ),
        "metric_family": "ordinal_structure",
        "evidence_tier": "primary",
        "unit": "correlation",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_within_family_mean_spearman": {
        "display_name": "Within-family Sigma-35 Spearman",
        "mathematical_definition": (
            "Mean within-design-family Spearman correlation between configured Sigma-35 rank gaps and "
            "observed centroid cosine-distance gaps."
        ),
        "metric_family": "ordinal_structure",
        "evidence_tier": "primary",
        "unit": "correlation",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_within_regulator_mean_spearman": {
        "display_name": "Within-regulator Sigma-35 Spearman",
        "mathematical_definition": (
            "Mean within-regulator-composition Spearman correlation between configured Sigma-35 rank gaps and "
            "observed centroid cosine-distance gaps."
        ),
        "metric_family": "ordinal_structure",
        "evidence_tier": "primary",
        "unit": "correlation",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_label_permutation_pvalue": {
        "display_name": "Sigma-35 permutation p-value",
        "mathematical_definition": (
            "Two-sided permutation p-value for the global Sigma-35 ordinal Spearman statistic under shuffled "
            "configured Sigma-35 ranks."
        ),
        "metric_family": "ordinal_structure",
        "evidence_tier": "primary",
        "unit": "p_value",
        "direction": "lower_is_better",
        "aggregation_level": "candidate_summary",
    },
    "sig35_variant_retention_correlation": {
        "display_name": "Sigma-35 retention",
        "mathematical_definition": (
            "Pearson correlation between aligned anchor and context centroid-distance vectors for configured "
            "Sigma-35 cohorts."
        ),
        "metric_family": "context_stability",
        "evidence_tier": "primary",
        "unit": "correlation",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    },
}


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
                "metadata": {
                    "derivations": {
                        "design_family": {
                            "kind": "annotation",
                            "source": "row",
                            "handler": _PROMOTER_METADATA_HANDLER,
                            "derive": "design_family",
                            "required_columns": ["densegen__plan", "usr_label__primary"],
                            "missing_policy": "error",
                            "value_type": "string",
                        },
                        "sig35_variant": {
                            "kind": "annotation",
                            "source": "row",
                            "handler": _PROMOTER_METADATA_HANDLER,
                            "derive": "sig35_variant",
                            "required_columns": ["usr_label__primary"],
                            "any_required_column_groups": [
                                ["densegen__plan"],
                                ["densegen__used_tfbs_detail"],
                                ["seq_annot__features"],
                                ["sequence", "derived__features_retained"],
                            ],
                            "missing_policy": "error",
                            "value_type": "string",
                        },
                        "source_class": {
                            "kind": "annotation",
                            "source": "row",
                            "handler": _PROMOTER_METADATA_HANDLER,
                            "derive": "source_class",
                            "required_columns": ["usr_label__primary"],
                            "any_required_column_groups": [
                                ["densegen__plan"],
                                ["source_family"],
                                ["promoter_standard__collection_id"],
                            ],
                            "missing_policy": "error",
                            "value_type": "string",
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _dataset_overview_params(
    *,
    source_ids: list[str],
    sigma35_include_unlisted_categories: bool = True,
) -> dict[str, object]:
    return {
        "source_ids": source_ids,
        "dimensions": [
            {
                "dimension": "provenance",
                "label": "Provenance",
                "column": "source_class",
                "category_order": ["densegen", "manual_or_wildtype"],
            },
            {
                "dimension": "generation_plan",
                "label": "Generation plan",
                "column": "design_family",
                "category_order": [
                    "background_only",
                    "ethanol",
                    "ciprofloxacin",
                    "ethanol_ciprofloxacin",
                    "control",
                ],
            },
            {
                "dimension": "sig35_variant",
                "label": "Sigma-35 variant",
                "column": "sig35_variant",
                "category_order": ["f", "e", "d", "c", "b", "control"],
                "include_unlisted_categories": sigma35_include_unlisted_categories,
                "category_labels": {
                    "f": "Variant f",
                    "e": "Variant e",
                    "d": "Variant d",
                    "c": "Variant c",
                    "b": "Variant b",
                },
            },
        ],
    }


def _write_margin_workspace_config(workspace_dir: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "margin_demo", "output_root": "./outputs"},
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
                "views": {
                    "degenerate_view": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "reference_sets": {
                    "reference_w_collection": {
                        "label": "W collection",
                        "match_column": "usr_label__primary",
                        "label_column": "usr_label__primary",
                        "where_all": [
                            {
                                "column": "promoter_standard__collection_id",
                                "equals": "t7_w_collection",
                            }
                        ],
                    },
                    "reference_anderson_igem": {
                        "label": "Anderson iGEM",
                        "match_column": "usr_label__primary",
                        "label_column": "usr_label__primary",
                        "where_all": [
                            {
                                "column": "promoter_standard__collection_id",
                                "equals": "anderson_igem",
                            }
                        ],
                    },
                },
                "metric_definitions": _STRESS_METRIC_DEFINITIONS,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_scalar_manifest(scalar_dir: Path, *, scalar_id: str, outputs: list[str]) -> None:
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "scalar_table",
                "artifact_id": scalar_id,
                "workspace_id": "test",
                "created_at": "2026-05-15T00:00:00+00:00",
                "tool_version": "test",
                "command": "scalar build",
                "status": "ok",
                "outputs": [{"path": output, "media_type": "application/x-parquet"} for output in outputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_view_artifact(
    workspace_dir: Path,
    *,
    view_id: str,
    rows: list[dict[str, object]],
    matrix: np.ndarray,
    record_key: str,
) -> None:
    view_dir = workspace_dir / "outputs" / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray(matrix, dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": record_key}}, indent=2),
        encoding="utf-8",
    )


def _write_alignment_artifact(
    workspace_dir: Path,
    *,
    alignment_id: str,
    left_view_id: str,
    right_view_id: str,
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    key_column: str,
) -> None:
    alignment_dir = workspace_dir / "outputs" / "alignments" / alignment_id
    alignment_dir.mkdir(parents=True, exist_ok=True)

    right_index_by_key = {row[key_column]: index for index, row in enumerate(right_rows)}
    ledger_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for left_index, row in enumerate(left_rows):
        key = row[key_column]
        if key not in right_index_by_key:
            continue
        right_index = right_index_by_key[key]
        ledger_row = {
            key_column: key,
            "left_count": 1,
            "right_count": 1,
        }
        ledger_rows.append(ledger_row)
        mapping_rows.append(
            {
                **ledger_row,
                "left_indices": [left_index],
                "right_indices": [right_index],
            }
        )

    pq.write_table(pa.Table.from_pylist(ledger_rows), alignment_dir / "rows.parquet")
    pq.write_table(pa.Table.from_pylist(mapping_rows), alignment_dir / "mapping.parquet")
    (alignment_dir / "manifest.json").write_text(
        json.dumps(
            {
                "params": {
                    "left": left_view_id,
                    "right": right_view_id,
                    "key_columns": [key_column],
                    "right_key_columns": [key_column],
                    "left_aggregation": "error",
                    "right_aggregation": "error",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_context_robustness_workspace(
    workspace_dir: Path,
    *,
    anchor_rows: list[dict[str, object]],
    context_rows: list[dict[str, object]],
    anchor_matrix: np.ndarray,
    context_matrix: np.ndarray,
) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "context_robustness_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_source": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
                        "record_key": "anchor_id",
                        "subject_key": "anchor_id",
                    },
                    "context_source": {
                        "kind": "parquet",
                        "path": "inputs/context.parquet",
                        "record_key": "context_id",
                        "subject_key": "anchor_id",
                    },
                },
                "views": {
                    "anchor_view": {
                        "source": "anchor_source",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {
                            "model": "7b",
                            "family": "intermediate_embedding",
                            "scope": "anchor_60bp",
                        },
                    },
                    "context_view": {
                        "source": "context_source",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {
                            "model": "7b",
                            "family": "intermediate_embedding",
                            "scope": "full_context_1kb",
                        },
                    },
                },
                "alignments": {
                    "anchor_ctx": {
                        "left": "anchor_view",
                        "right": "context_view",
                        "left_on": ["anchor_id"],
                        "right_on": ["anchor_id"],
                        "left_aggregation": "error",
                        "right_aggregation": "error",
                    }
                },
                "metric_definitions": _STRESS_METRIC_DEFINITIONS,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _write_source(workspace_dir / "inputs" / "anchor.parquet", anchor_rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", context_rows)
    _write_view_artifact(
        workspace_dir,
        view_id="anchor_view",
        rows=anchor_rows,
        matrix=anchor_matrix,
        record_key="anchor_id",
    )
    _write_view_artifact(
        workspace_dir,
        view_id="context_view",
        rows=context_rows,
        matrix=context_matrix,
        record_key="context_id",
    )
    _write_alignment_artifact(
        workspace_dir,
        alignment_id="anchor_ctx",
        left_view_id="anchor_view",
        right_view_id="context_view",
        left_rows=anchor_rows,
        right_rows=context_rows,
        key_column="anchor_id",
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
        params=_dataset_overview_params(source_ids=["anchor_60bp", "full_context_1kb"]),
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    dimensions = {row["dimension"] for row in table}

    assert artifact.stats["denominator"] == 6
    assert dimensions == {"provenance", "generation_plan", "sig35_variant"}
    assert "anchor_60bp" not in dimensions
    assert "full_context_1kb" not in dimensions
    assert {row["dimension_label"] for row in table} == {"Provenance", "Generation plan", "Sigma-35 variant"}
    sig35_rows = sorted((row for row in table if row["dimension"] == "sig35_variant"), key=lambda row: row["order"])
    assert [row["category"] for row in sig35_rows] == ["f", "e", "d", "c", "b", "control"]

    for dimension in dimensions:
        dimension_rows = [row for row in table if row["dimension"] == dimension]
        assert sum(int(row["count"]) for row in dimension_rows) == 6
        assert pytest.approx(sum(float(row["fraction"]) for row in dimension_rows), rel=0, abs=1e-9) == 1.0


def test_dataset_overview_includes_annotated_sigma35_categories_outside_named_ladder(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    archive_row = {
        "id": "sfxi_archive_row",
        "subject_id": "sfxi_archive_row",
        "usr_label__primary": "pDual-10-ES3p",
        "source_family": "densegen",
        "sequence": None,
        "densegen__plan": "ethanol_ciprofloxacin",
        "densegen__required_regulators": ["cpxR", "lexA"],
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "variant_id": "ACCGCG",
                "core_sequence": "ACCGCG",
            },
            {
                "part_kind": "fixed_element",
                "role": "downstream",
                "constraint_name": "sigma70_core",
                "variant_id": "consensus",
                "core_sequence": "TATAAT",
            },
        ],
        "seq_annot__features": None,
        "derived__features_retained": None,
    }
    seq_annot_row = {
        "id": "projected_reference_row",
        "subject_id": "projected_reference_row",
        "usr_label__primary": "J23104",
        "source_family": "reference_control",
        "sequence": "TTGACATATGCTAGCTAGCTAGCTAGCTAGCTAGC",
        "densegen__plan": None,
        "densegen__required_regulators": None,
        "densegen__used_tfbs_detail": None,
        "seq_annot__features": [
            {
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "qualifiers": [{"key": "note", "value": "feature_sequence=TTGACA"}],
            }
        ],
        "derived__features_retained": None,
    }
    rows = [archive_row, seq_annot_row, *_base_rows()]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="dataset_overview_counts",
        builder_kind="dataset_overview",
        params=_dataset_overview_params(source_ids=["anchor_60bp", "full_context_1kb"]),
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    sig35_rows = [row for row in table if row["dimension"] == "sig35_variant"]
    counts = {row["category"]: row["count"] for row in sig35_rows}

    assert artifact.stats["denominator"] == 8
    assert counts["ACCGCG"] == 1
    assert counts["TTGACA"] == 1
    assert sum(int(row["count"]) for row in sig35_rows) == 8


def test_dataset_overview_can_limit_sigma35_panel_to_configured_ladder(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    rows = [
        {
            "id": "projected_reference_row",
            "subject_id": "projected_reference_row",
            "usr_label__primary": "J23104",
            "source_family": "reference_control",
            "sequence": "TTGACATATGCTAGCTAGCTAGCTAGCTAGCTAGC",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "densegen__used_tfbs_detail": None,
            "seq_annot__features": [
                {
                    "label": "-35",
                    "role_hint": "sigma70_minus35",
                    "qualifiers": [{"key": "note", "value": "feature_sequence=TTGACA"}],
                }
            ],
            "derived__features_retained": None,
        },
        *_base_rows(),
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="dataset_overview_counts",
        builder_kind="dataset_overview",
        params=_dataset_overview_params(
            source_ids=["anchor_60bp", "full_context_1kb"],
            sigma35_include_unlisted_categories=False,
        ),
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    sig35_rows = [row for row in table if row["dimension"] == "sig35_variant"]

    assert artifact.stats["denominator"] == 7
    assert [row["category"] for row in sorted(sig35_rows, key=lambda row: row["order"])] == [
        "f",
        "e",
        "d",
        "c",
        "b",
        "control",
    ]
    assert "TTGACA" not in {row["category"] for row in sig35_rows}
    assert sum(int(row["count"]) for row in sig35_rows) == 6


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
            params=_dataset_overview_params(source_ids=["anchor_60bp", "full_context_1kb"]),
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
        params=_dataset_overview_params(source_ids=["anchor_60bp", "full_context_1kb"]),
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert sum(int(row["count"]) for row in table if row["dimension"] == "provenance") == len(anchor_rows)


def test_context_robustness_summary_projects_alignment_metadata_for_retention_metrics(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    anchor_rows = [
        {
            "anchor_id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "f",
            "embedding": [1.0, 0.15, 0.05],
        },
        {
            "anchor_id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "embedding": [0.25, 1.05, 0.2],
        },
        {
            "anchor_id": "cip_1",
            "subject_id": "cip_1",
            "design_family": "ciprofloxacin",
            "design_regulator_composition": "lexA_only",
            "sig35_variant": "b",
            "embedding": [0.35, 0.45, 1.0],
        },
    ]
    context_rows = [
        {
            "context_id": "ctx_bg_1",
            "anchor_id": "bg_1",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "f",
            "embedding": [0.94, 0.2, 0.06],
        },
        {
            "context_id": "ctx_eth_1",
            "anchor_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "embedding": [0.3, 0.96, 0.18],
        },
        {
            "context_id": "ctx_cip_1",
            "anchor_id": "cip_1",
            "design_family": "ciprofloxacin",
            "design_regulator_composition": "lexA_only",
            "sig35_variant": "b",
            "embedding": [0.42, 0.5, 0.88],
        },
    ]
    _write_context_robustness_workspace(
        workspace_dir,
        anchor_rows=anchor_rows,
        context_rows=context_rows,
        anchor_matrix=np.asarray([row["embedding"] for row in anchor_rows], dtype=np.float32),
        context_matrix=np.asarray([row["embedding"] for row in context_rows], dtype=np.float32),
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="context_robustness_summary_metrics",
        builder_kind="context_robustness_summary",
        params={
            "sample_size": 0,
            "retention_axes": _STRESS_RETENTION_AXES,
            "pairs": [
                {
                    "pair_id": "anchor_vs_context",
                    "label": "7B intermediate: anchor vs 1 kb seq mean",
                    "alignment_id": "anchor_ctx",
                    "anchor_view_id": "anchor_view",
                    "context_view_id": "context_view",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    metric_values = {row["metric_id"]: float(row["metric_value"]) for row in table}

    assert artifact.stats["pair_count"] == 1
    assert any(input_ref.kind == "alignment_set" for input_ref in artifact.inputs)
    assert {
        "context_self_cosine_median",
        "design_family_retention_correlation",
        "design_regulator_composition_retention_correlation",
        "sig35_variant_retention_correlation",
    } == set(metric_values)
    assert all(np.isfinite(value) for value in metric_values.values())


def test_alignment_metrics_support_densegen_only_filters_and_sampled_tables(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    anchor_rows = [
        {
            "anchor_id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "f",
            "source_class": "densegen",
            "embedding": [1.0, 0.15, 0.05],
        },
        {
            "anchor_id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "source_class": "densegen",
            "embedding": [0.25, 1.05, 0.2],
        },
        {
            "anchor_id": "ctrl_1",
            "subject_id": "ctrl_1",
            "design_family": "control",
            "design_regulator_composition": "control",
            "sig35_variant": "control",
            "source_class": "manual_or_wildtype",
            "embedding": [0.35, 0.25, 0.95],
        },
    ]
    context_rows = [
        {
            "context_id": "ctx_bg_1",
            "anchor_id": "bg_1",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "f",
            "source_class": "densegen",
            "embedding": [0.94, 0.2, 0.06],
        },
        {
            "context_id": "ctx_eth_1",
            "anchor_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "source_class": "densegen",
            "embedding": [0.3, 0.96, 0.18],
        },
        {
            "context_id": "ctx_ctrl_1",
            "anchor_id": "ctrl_1",
            "design_family": "control",
            "design_regulator_composition": "control",
            "sig35_variant": "control",
            "source_class": "manual_or_wildtype",
            "embedding": [0.4, 0.3, 0.9],
        },
    ]
    _write_context_robustness_workspace(
        workspace_dir,
        anchor_rows=anchor_rows,
        context_rows=context_rows,
        anchor_matrix=np.asarray([row["embedding"] for row in anchor_rows], dtype=np.float32),
        context_matrix=np.asarray([row["embedding"] for row in context_rows], dtype=np.float32),
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="alignment_metrics_demo",
        builder_kind="alignment_metrics",
        params={
            "alignment_id": "anchor_ctx",
            "left_view_id": "context_view",
            "right_view_id": "anchor_view",
            "metadata_view_id": "context_view",
            "margin_deltas": [],
            "sample_size": 1,
            "sample_group_column": None,
            "where": {"column": "source_class", "equals": "densegen"},
            "table_sample_only": True,
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()

    assert len(table) == 1
    assert {str(row["source_class"]) for row in table} == {"densegen"}
    assert str(table[0]["design_family"]) in {"background_only", "ethanol"}
    assert str(table[0]["anchor_id"]) in {"bg_1", "eth_1"}
    assert artifact.stats["rows"] == 1
    assert artifact.stats["where"] == {"column": "source_class", "equals": "densegen"}
    assert artifact.stats["sample_size"] == 1
    assert artifact.stats["table_sample_only"] is True


def test_context_robustness_summary_skips_only_degenerate_axes(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    anchor_rows = [
        {
            "anchor_id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "d",
            "embedding": [1.0, 0.15, 0.05],
        },
        {
            "anchor_id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "embedding": [0.25, 1.05, 0.2],
        },
        {
            "anchor_id": "cip_1",
            "subject_id": "cip_1",
            "design_family": "ciprofloxacin",
            "design_regulator_composition": "lexA_only",
            "sig35_variant": "d",
            "embedding": [0.35, 0.45, 1.0],
        },
    ]
    context_rows = [
        {
            "context_id": "ctx_bg_1",
            "anchor_id": "bg_1",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "d",
            "embedding": [0.94, 0.2, 0.06],
        },
        {
            "context_id": "ctx_eth_1",
            "anchor_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "embedding": [0.3, 0.96, 0.18],
        },
        {
            "context_id": "ctx_cip_1",
            "anchor_id": "cip_1",
            "design_family": "ciprofloxacin",
            "design_regulator_composition": "lexA_only",
            "sig35_variant": "d",
            "embedding": [0.42, 0.5, 0.88],
        },
    ]
    _write_context_robustness_workspace(
        workspace_dir,
        anchor_rows=anchor_rows,
        context_rows=context_rows,
        anchor_matrix=np.asarray([row["embedding"] for row in anchor_rows], dtype=np.float32),
        context_matrix=np.asarray([row["embedding"] for row in context_rows], dtype=np.float32),
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="context_robustness_summary_metrics",
        builder_kind="context_robustness_summary",
        params={
            "sample_size": 0,
            "retention_axes": _STRESS_RETENTION_AXES,
            "pairs": [
                {
                    "pair_id": "anchor_vs_context",
                    "alignment_id": "anchor_ctx",
                    "anchor_view_id": "anchor_view",
                    "context_view_id": "context_view",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    metric_ids = {row["metric_id"] for row in table}

    assert metric_ids == {
        "context_self_cosine_median",
        "design_family_retention_correlation",
        "design_regulator_composition_retention_correlation",
    }
    assert artifact.stats["skipped_metric_ids"] == ["sig35_variant_retention_correlation"]


def test_context_robustness_summary_skips_fully_degenerate_retention_axes(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    anchor_rows = [
        {
            "anchor_id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "anchor_id": "ctrl_1",
            "subject_id": "ctrl_1",
            "design_family": "control",
            "design_regulator_composition": "control",
            "sig35_variant": "control",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]
    context_rows = [
        {
            "context_id": "ctx_eth_1",
            "anchor_id": "eth_1",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "embedding": [0.9, 0.1, 0.0],
        },
        {
            "context_id": "ctx_ctrl_1",
            "anchor_id": "ctrl_1",
            "design_family": "control",
            "design_regulator_composition": "control",
            "sig35_variant": "control",
            "embedding": [0.1, 0.9, 0.0],
        },
    ]
    _write_context_robustness_workspace(
        workspace_dir,
        anchor_rows=anchor_rows,
        context_rows=context_rows,
        anchor_matrix=np.asarray([row["embedding"] for row in anchor_rows], dtype=np.float32),
        context_matrix=np.asarray([row["embedding"] for row in context_rows], dtype=np.float32),
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="context_robustness_summary_metrics",
        builder_kind="context_robustness_summary",
        params={
            "sample_size": 0,
            "retention_axes": _STRESS_RETENTION_AXES,
            "pairs": [
                {
                    "pair_id": "anchor_vs_context",
                    "alignment_id": "anchor_ctx",
                    "anchor_view_id": "anchor_view",
                    "context_view_id": "context_view",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert {row["metric_id"] for row in table} == {"context_self_cosine_median"}
    assert artifact.stats["skipped_metric_ids"] == [
        "design_family_retention_correlation",
        "design_regulator_composition_retention_correlation",
        "sig35_variant_retention_correlation",
    ]


def test_cohort_similarity_margin_returns_nan_for_degenerate_standardized_view(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "design_family": "background_only", "embedding": [1.0, 1.0]},
        {"id": "bg_2", "subject_id": "bg_2", "design_family": "background_only", "embedding": [1.0, 1.0]},
        {"id": "eth_1", "subject_id": "eth_1", "design_family": "ethanol", "embedding": [1.0, 1.0]},
        {"id": "eth_2", "subject_id": "eth_2", "design_family": "ethanol", "embedding": [1.0, 1.0]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.ones((4, 2), dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="degenerate_design_margins",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "margin_pairs": [
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    values = np.asarray(table["synthetic_margin_ethanol_vs_background"].to_pylist(), dtype=np.float32)
    assert np.isnan(values).all()
    assert artifact.stats["synthetic_margin_ethanol_vs_background_degenerate_reference"] is True


def test_tf_axis_orientation_audit_computes_generic_tf_bins_and_margins(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_1",
            "subject_id": "bg_1",
            "source": "densegen",
            "design_family": "background",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "bg_2",
            "subject_id": "bg_2",
            "source": "densegen",
            "design_family": "background",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "eth_1",
            "subject_id": "eth_1",
            "source": "densegen",
            "design_family": "ethanol",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "eth_2",
            "subject_id": "eth_2",
            "source": "densegen",
            "design_family": "ethanol",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "cip_1",
            "subject_id": "cip_1",
            "source": "densegen",
            "design_family": "ciprofloxacin",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "cip_2",
            "subject_id": "cip_2",
            "source": "densegen",
            "design_family": "ciprofloxacin",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "native_baer",
            "subject_id": "native_baer",
            "source": "regulondb",
            "design_family": "native",
            "has_BaeR": True,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "native_lexa",
            "subject_id": "native_lexa",
            "source": "regulondb",
            "design_family": "native",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": True,
        },
        {
            "id": "native_mixed",
            "subject_id": "native_mixed",
            "source": "regulondb",
            "design_family": "native",
            "has_BaeR": True,
            "has_CpxR": False,
            "has_LexA": True,
        },
        {
            "id": "native_none",
            "subject_id": "native_none",
            "source": "regulondb",
            "design_family": "native",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
    ]
    matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [2.0, 0.0],
            [2.0, 0.2],
            [0.0, 2.0],
            [0.2, 2.0],
            [1.8, 0.1],
            [0.1, 1.8],
            [1.4, 1.4],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    _write_view_artifact(workspace_dir, view_id="tf_view", rows=rows, matrix=matrix, record_key="id")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="tf_axis_orientation_audit",
        builder_kind="tf_axis_orientation_audit",
        params={
            "view_id": "tf_view",
            "cohort_column": "design_family",
            "centroid_groups": {
                "background": ["background"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
            },
            "tf_columns": {
                "ethanol": ["has_BaeR", "has_CpxR"],
                "cipro": ["has_LexA"],
            },
            "embedding_view": "pdual10_core60_bidir",
            "output_filter": {"column": "source", "equals": "regulondb"},
        },
    )

    table_artifact = pq.read_table(artifact.artifact_dir / "table.parquet")
    table = table_artifact.to_pylist()
    by_id = {row["id"]: row for row in table}
    assert set(by_id) == {"native_baer", "native_lexa", "native_mixed", "native_none"}
    assert "design_family" not in table_artifact.column_names
    assert by_id["native_baer"]["tf_bin"] == "ethanol_TF"
    assert by_id["native_lexa"]["tf_bin"] == "lexA_TF"
    assert by_id["native_mixed"]["tf_bin"] == "mixed"
    assert by_id["native_none"]["tf_bin"] == "neither"
    assert by_id["native_baer"]["embedding_view"] == "pdual10_core60_bidir"
    assert by_id["native_baer"]["ethanolness"] > by_id["native_none"]["ethanolness"]
    assert by_id["native_lexa"]["ciproness"] > by_id["native_none"]["ciproness"]
    assert artifact.stats["input_rows"] == 10
    assert artifact.stats["tf_bin_counts"] == {"ethanol_TF": 1, "lexA_TF": 1, "mixed": 1, "neither": 1}


def test_tf_axis_orientation_audit_rejects_unexpected_output_row_count(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    base_tf_flags = {"has_BaeR": False, "has_CpxR": False, "has_LexA": False}
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "source": "densegen", "design_family": "background", **base_tf_flags},
        {"id": "bg_2", "subject_id": "bg_2", "source": "densegen", "design_family": "background", **base_tf_flags},
        {"id": "eth_1", "subject_id": "eth_1", "source": "densegen", "design_family": "ethanol", **base_tf_flags},
        {"id": "eth_2", "subject_id": "eth_2", "source": "densegen", "design_family": "ethanol", **base_tf_flags},
        {
            "id": "cip_1",
            "subject_id": "cip_1",
            "source": "densegen",
            "design_family": "ciprofloxacin",
            **base_tf_flags,
        },
        {
            "id": "cip_2",
            "subject_id": "cip_2",
            "source": "densegen",
            "design_family": "ciprofloxacin",
            **base_tf_flags,
        },
        {
            "id": "native_none",
            "subject_id": "native_none",
            "source": "regulondb",
            "design_family": "native",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
    ]
    matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [2.0, 0.0],
            [2.0, 0.2],
            [0.0, 2.0],
            [0.2, 2.0],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    _write_view_artifact(workspace_dir, view_id="tf_view", rows=rows, matrix=matrix, record_key="id")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="expected_output_rows"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_audit",
            builder_kind="tf_axis_orientation_audit",
            params={
                "view_id": "tf_view",
                "cohort_column": "design_family",
                "centroid_groups": {
                    "background": ["background"],
                    "ethanol": ["ethanol"],
                    "cipro": ["ciprofloxacin"],
                },
                "tf_columns": {
                    "ethanol": ["has_BaeR", "has_CpxR"],
                    "cipro": ["has_LexA"],
                },
                "output_filter": {"column": "source", "equals": "regulondb"},
                "expected_output_rows": 2,
            },
        )


def test_tf_axis_orientation_audit_requires_explicit_output_filter(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "source": "densegen", "design_family": "background"},
        {"id": "eth_1", "subject_id": "eth_1", "source": "densegen", "design_family": "ethanol"},
        {"id": "cip_1", "subject_id": "cip_1", "source": "densegen", "design_family": "ciprofloxacin"},
        {
            "id": "native_1",
            "subject_id": "native_1",
            "source": "regulondb",
            "design_family": "native",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": False,
        },
    ]
    for row in rows[:3]:
        row["has_BaeR"] = False
        row["has_CpxR"] = False
        row["has_LexA"] = False
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    _write_view_artifact(workspace_dir, view_id="tf_view", rows=rows, matrix=matrix, record_key="id")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="requires output_filter"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_audit",
            builder_kind="tf_axis_orientation_audit",
            params={
                "view_id": "tf_view",
                "cohort_column": "design_family",
                "centroid_groups": {
                    "background": ["background"],
                    "ethanol": ["ethanol"],
                    "cipro": ["ciprofloxacin"],
                },
                "tf_columns": {
                    "ethanol": ["has_BaeR", "has_CpxR"],
                    "cipro": ["has_LexA"],
                },
            },
        )


def test_tf_axis_orientation_audit_fails_fast_when_tf_columns_are_missing(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "design_family": "background", "has_BaeR": False},
        {"id": "eth_1", "subject_id": "eth_1", "design_family": "ethanol", "has_BaeR": False},
        {"id": "cip_1", "subject_id": "cip_1", "design_family": "ciprofloxacin", "has_BaeR": False},
    ]
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    _write_view_artifact(workspace_dir, view_id="tf_view", rows=rows, matrix=matrix, record_key="id")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="missing required tf columns"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_audit",
            builder_kind="tf_axis_orientation_audit",
            params={
                "view_id": "tf_view",
                "cohort_column": "design_family",
                "centroid_groups": {
                    "background": ["background"],
                    "ethanol": ["ethanol"],
                    "cipro": ["ciprofloxacin"],
                },
                "tf_columns": {
                    "ethanol": ["has_BaeR", "has_CpxR"],
                    "cipro": ["has_LexA"],
                },
                "output_filter": {"column": "design_family", "equals": "native"},
            },
        )


def test_tf_axis_orientation_audit_can_join_generic_tf_association_overlay(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "source": "densegen", "design_family": "background"},
        {"id": "bg_2", "subject_id": "bg_2", "source": "densegen", "design_family": "background"},
        {"id": "eth_1", "subject_id": "eth_1", "source": "densegen", "design_family": "ethanol"},
        {"id": "cip_1", "subject_id": "cip_1", "source": "densegen", "design_family": "ciprofloxacin"},
        {"id": "native_cpx", "subject_id": "native_cpx", "source": "regulondb", "design_family": "native"},
        {"id": "native_lexa", "subject_id": "native_lexa", "source": "regulondb", "design_family": "native"},
    ]
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.1, 1.0],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )
    _write_view_artifact(workspace_dir, view_id="tf_view", rows=rows, matrix=matrix, record_key="id")
    overlay_path = workspace_dir / "inputs" / "regulatory_interactions.parquet"
    overlay_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"usr_id": "native_cpx", "regulator_abbrev": "CpxR"},
                {"usr_id": "native_lexa", "regulator_abbrev": "LexA"},
            ]
        ),
        overlay_path,
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="tf_axis_orientation_audit",
        builder_kind="tf_axis_orientation_audit",
        params={
            "view_id": "tf_view",
            "cohort_column": "design_family",
            "centroid_groups": {
                "background": ["background"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
            },
            "tf_columns": {
                "ethanol": ["has_BaeR", "has_CpxR"],
                "cipro": ["has_LexA"],
            },
            "association_overlay": {
                "path": "inputs/regulatory_interactions.parquet",
                "row_key": "id",
                "relation_key": "usr_id",
                "regulator_column": "regulator_abbrev",
                "tf_aliases": {
                    "has_BaeR": ["BaeR"],
                    "has_CpxR": ["CpxR"],
                    "has_LexA": ["LexA"],
                },
            },
            "output_filter": {"column": "source", "equals": "regulondb"},
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    by_id = {row["id"]: row for row in table}
    assert set(by_id) == {"native_cpx", "native_lexa"}
    assert by_id["native_cpx"]["has_CpxR"] is True
    assert by_id["native_cpx"]["tf_bin"] == "ethanol_TF"
    assert by_id["native_lexa"]["has_LexA"] is True
    assert by_id["native_lexa"]["tf_bin"] == "lexA_TF"
    assert any(input_ref.kind == "association_overlay" for input_ref in artifact.inputs)
    assert artifact.stats["association_overlay_rows"] == 2


def test_tf_axis_orientation_audit_requires_explicit_association_overlay_keys(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "source": "densegen", "design_family": "background"},
        {"id": "eth_1", "subject_id": "eth_1", "source": "densegen", "design_family": "ethanol"},
        {"id": "cip_1", "subject_id": "cip_1", "source": "densegen", "design_family": "ciprofloxacin"},
        {"id": "native_cpx", "subject_id": "native_cpx", "source": "regulondb", "design_family": "native"},
    ]
    _write_view_artifact(
        workspace_dir,
        view_id="tf_view",
        rows=rows,
        matrix=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.1, 1.0]], dtype=np.float32),
        record_key="id",
    )
    overlay_path = workspace_dir / "inputs" / "regulatory_interactions.parquet"
    overlay_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"usr_id": "native_cpx", "regulator_abbrev": "CpxR"}]),
        overlay_path,
    )
    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)

    with pytest.raises(ContractViolationError, match="association_overlay requires row_key"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_audit",
            builder_kind="tf_axis_orientation_audit",
            params={
                "view_id": "tf_view",
                "cohort_column": "design_family",
                "centroid_groups": {
                    "background": ["background"],
                    "ethanol": ["ethanol"],
                    "cipro": ["ciprofloxacin"],
                },
                "tf_columns": {
                    "ethanol": ["has_CpxR"],
                    "cipro": ["has_LexA"],
                },
                "association_overlay": {
                    "path": "inputs/regulatory_interactions.parquet",
                    "relation_key": "usr_id",
                    "regulator_column": "regulator_abbrev",
                    "tf_aliases": {"has_CpxR": ["CpxR"], "has_LexA": ["LexA"]},
                },
                "output_filter": {"column": "source", "equals": "regulondb"},
            },
        )

    with pytest.raises(ContractViolationError, match="legacy keys"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_audit",
            builder_kind="tf_axis_orientation_audit",
            params={
                "view_id": "tf_view",
                "cohort_column": "design_family",
                "centroid_groups": {
                    "background": ["background"],
                    "ethanol": ["ethanol"],
                    "cipro": ["ciprofloxacin"],
                },
                "tf_columns": {
                    "ethanol": ["has_CpxR"],
                    "cipro": ["has_LexA"],
                },
                "association_overlay": {
                    "path": "inputs/regulatory_interactions.parquet",
                    "join_key": "usr_id",
                    "row_key": "id",
                    "relation_key": "usr_id",
                    "regulator_column": "regulator_abbrev",
                    "tf_aliases": {"has_CpxR": ["CpxR"], "has_LexA": ["LexA"]},
                },
                "output_filter": {"column": "source", "equals": "regulondb"},
            },
        )


def test_tf_axis_orientation_audit_can_use_separate_centroid_and_audit_views(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    centroid_rows = [
        {"id": "bg_1", "subject_id": "bg_1", "source": "densegen", "design_family": "background"},
        {"id": "eth_1", "subject_id": "eth_1", "source": "densegen", "design_family": "ethanol"},
        {"id": "cip_1", "subject_id": "cip_1", "source": "densegen", "design_family": "ciprofloxacin"},
    ]
    audit_rows = [
        {
            "id": "native_baer",
            "subject_id": "native_baer",
            "source": "regulondb",
            "has_BaeR": True,
            "has_CpxR": False,
            "has_LexA": False,
        },
        {
            "id": "native_lexa",
            "subject_id": "native_lexa",
            "source": "regulondb",
            "has_BaeR": False,
            "has_CpxR": False,
            "has_LexA": True,
        },
    ]
    _write_view_artifact(
        workspace_dir,
        view_id="synthetic_centroids",
        rows=centroid_rows,
        matrix=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        record_key="id",
    )
    _write_view_artifact(
        workspace_dir,
        view_id="native_pdual_context",
        rows=audit_rows,
        matrix=np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="tf_axis_orientation_audit",
        builder_kind="tf_axis_orientation_audit",
        params={
            "view_id": "synthetic_centroids",
            "audit_view_id": "native_pdual_context",
            "cohort_column": "design_family",
            "centroid_groups": {
                "background": ["background"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
            },
            "tf_columns": {
                "ethanol": ["has_BaeR", "has_CpxR"],
                "cipro": ["has_LexA"],
            },
            "output_filter": {"column": "source", "equals": "regulondb"},
        },
    )

    rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert {row["id"] for row in rows} == {"native_baer", "native_lexa"}
    assert artifact.stats["centroid_view_id"] == "synthetic_centroids"
    assert artifact.stats["audit_view_id"] == "native_pdual_context"
    assert artifact.stats["centroid_rows"] == 3
    assert artifact.stats["input_rows"] == 2
    assert {input_ref.artifact_id for input_ref in artifact.inputs} >= {"synthetic_centroids", "native_pdual_context"}


def test_tf_axis_orientation_tests_reports_effect_sizes_and_excludes_mixed_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    audit_rows = [
        {"id": "baer_1", "source": "regulondb", "tf_bin": "ethanol_TF", "ethanolness": 1.2, "ciproness": 0.0},
        {"id": "cpxr_1", "source": "regulondb", "tf_bin": "ethanol_TF", "ethanolness": 1.0, "ciproness": 0.1},
        {"id": "lexa_1", "source": "regulondb", "tf_bin": "lexA_TF", "ethanolness": 0.1, "ciproness": 1.2},
        {"id": "lexa_2", "source": "regulondb", "tf_bin": "lexA_TF", "ethanolness": 0.0, "ciproness": 1.1},
        {"id": "mixed_1", "source": "regulondb", "tf_bin": "mixed", "ethanolness": 10.0, "ciproness": 10.0},
        {"id": "none_1", "source": "regulondb", "tf_bin": "neither", "ethanolness": 0.2, "ciproness": 0.1},
        {"id": "none_2", "source": "regulondb", "tf_bin": "neither", "ethanolness": 0.1, "ciproness": 0.2},
        {"id": "dense_1", "source": "densegen", "tf_bin": "neither", "ethanolness": 99.0, "ciproness": 99.0},
    ]
    scalar_dir = workspace_dir / "outputs" / "scalars" / "tf_axis_orientation_audit"
    scalar_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(audit_rows), scalar_dir / "table.parquet")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="tf_axis_orientation_tests",
        builder_kind="tf_axis_orientation_tests",
        params={
            "source_scalar": "tf_axis_orientation_audit",
            "where": {"column": "source", "equals": "regulondb"},
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    by_axis = {row["axis"]: row for row in rows_table}
    assert by_axis["ethanolness"]["target_bin"] == "ethanol_TF"
    assert by_axis["ethanolness"]["n_target"] == 2
    assert by_axis["ethanolness"]["n_background"] == 2
    assert by_axis["ethanolness"]["median_difference"] > 0
    assert by_axis["ethanolness"]["rank_biserial"] == pytest.approx(1.0)
    assert by_axis["ethanolness"]["p_value_method"] == "exact_enumeration"
    assert by_axis["ethanolness"]["tie_group_count"] == 0
    assert by_axis["ciproness"]["target_bin"] == "lexA_TF"
    assert by_axis["ciproness"]["rank_biserial"] == pytest.approx(1.0)
    assert artifact.stats["tested_axes"] == ["ethanolness", "ciproness"]


def test_tf_axis_orientation_tests_fails_fast_when_where_column_is_missing(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    scalar_dir = workspace_dir / "outputs" / "scalars" / "tf_axis_orientation_audit"
    scalar_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "baer_1", "tf_bin": "ethanol_TF", "ethanolness": 1.0, "ciproness": 0.0},
                {"id": "none_1", "tf_bin": "neither", "ethanolness": 0.0, "ciproness": 0.0},
            ]
        ),
        scalar_dir / "table.parquet",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="where column"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_tests",
            builder_kind="tf_axis_orientation_tests",
            params={
                "source_scalar": "tf_axis_orientation_audit",
                "where": {"column": "source", "equals": "regulondb"},
            },
        )


def test_tf_axis_orientation_tests_requires_explicit_where_filter(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    scalar_dir = workspace_dir / "outputs" / "scalars" / "tf_axis_orientation_audit"
    scalar_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "baer_1", "source": "regulondb", "tf_bin": "ethanol_TF", "ethanolness": 1.0, "ciproness": 0.0},
                {"id": "none_1", "source": "regulondb", "tf_bin": "neither", "ethanolness": 0.0, "ciproness": 0.0},
            ]
        ),
        scalar_dir / "table.parquet",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="requires where"):
        build_scalar_artifact(
            context,
            scalar_id="tf_axis_orientation_tests",
            builder_kind="tf_axis_orientation_tests",
            params={"source_scalar": "tf_axis_orientation_audit"},
        )


def test_native_regulator_plan_margin_enrichment_scalar_writes_contract_tables(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "alias_id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "bg_1",
        },
        {
            "alias_id": "bg_2",
            "subject_id": "bg_2",
            "design_family": "background_only",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "bg_2",
        },
        {
            "alias_id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "eth_1",
        },
        {
            "alias_id": "eth_2",
            "subject_id": "eth_2",
            "design_family": "ethanol",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "eth_2",
        },
        {
            "alias_id": "cip_1",
            "subject_id": "cip_1",
            "design_family": "ciprofloxacin",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "cip_1",
        },
        {
            "alias_id": "cip_2",
            "subject_id": "cip_2",
            "design_family": "ciprofloxacin",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "cip_2",
        },
        {
            "alias_id": "dual_1",
            "subject_id": "dual_1",
            "design_family": "ethanol_ciprofloxacin",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "dual_1",
        },
        {
            "alias_id": "dual_2",
            "subject_id": "dual_2",
            "design_family": "ethanol_ciprofloxacin",
            "derived__parent_dataset": "densegen",
            "derived__parent_id": "dual_2",
        },
        {
            "alias_id": "native_eth",
            "subject_id": "native_eth",
            "design_family": "native",
            "derived__parent_dataset": "usr_regulondb_native_promoters",
            "derived__parent_id": "rp_eth",
            "regulondb__primary_promoter_id": "p_eth",
            "regulondb__primary_promoter_name": "p_eth",
        },
        {
            "alias_id": "native_cipro",
            "subject_id": "native_cipro",
            "design_family": "native",
            "derived__parent_dataset": "usr_regulondb_native_promoters",
            "derived__parent_id": "rp_cipro",
            "regulondb__primary_promoter_id": "p_cipro",
            "regulondb__primary_promoter_name": "p_cipro",
        },
        {
            "alias_id": "native_bg",
            "subject_id": "native_bg",
            "design_family": "native",
            "derived__parent_dataset": "usr_regulondb_native_promoters",
            "derived__parent_id": "rp_bg",
            "regulondb__primary_promoter_id": "p_bg",
            "regulondb__primary_promoter_name": "p_bg",
        },
    ]
    for row in rows:
        row.setdefault("regulondb__primary_promoter_id", None)
        row.setdefault("regulondb__primary_promoter_name", None)
    matrix = np.asarray(
        [
            [5.0, 0.0, 0.0, 0.0],
            [5.0, 0.2, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 0.2, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0, 0.2],
            [0.0, 0.0, 0.0, 5.0],
            [0.2, 0.0, 0.0, 5.0],
            [0.0, 4.7, 0.0, 0.1],
            [0.0, 0.0, 4.8, 0.2],
            [4.8, 0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    _write_view_artifact(workspace_dir, view_id="bidir_context", rows=rows, matrix=matrix, record_key="alias_id")
    overlay_path = workspace_dir / "inputs" / "regulatory_interactions.parquet"
    overlay_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "usr_id": "rp_eth",
                    "regulator_abbrev": "CpxR",
                    "source_release": "RegulonDB_13",
                    "source_route": "binding_sites/TF-RISet.tsv",
                    "regulatory_interaction_id": "ri_eth",
                    "confidence": "strong",
                    "evidence": "test",
                },
                {
                    "usr_id": "rp_cipro",
                    "regulator_abbrev": "LexA",
                    "source_release": "RegulonDB_13",
                    "source_route": "binding_sites/TF-RISet.tsv",
                    "regulatory_interaction_id": "ri_cipro",
                    "confidence": "strong",
                    "evidence": "test",
                },
                {
                    "usr_id": "rp_bg",
                    "regulator_abbrev": "CRP",
                    "source_release": "RegulonDB_13",
                    "source_route": "binding_sites/TF-RISet.tsv",
                    "regulatory_interaction_id": "ri_bg",
                    "confidence": "strong",
                    "evidence": "test",
                },
            ]
        ),
        overlay_path,
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="native_regulator_plan_margin_enrichment",
        builder_kind="native_regulator_plan_margin_enrichment",
        params={
            "view_id": "bidir_context",
            "cohort_column": "design_family",
            "centroid_groups": {
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            "native_filter": {
                "column": "derived__parent_dataset",
                "equals": "usr_regulondb_native_promoters",
            },
            "expected_output_rows": 3,
            "regulatory_interactions": {
                "path": "inputs/regulatory_interactions.parquet",
                "relation_key": "usr_id",
                "regulator_column": "regulator_abbrev",
                "required_columns": [
                    "source_release",
                    "source_route",
                    "regulatory_interaction_id",
                    "confidence",
                    "evidence",
                ],
            },
            "native_parent_column": "derived__parent_id",
            "native_metadata_columns": [
                "alias_id",
                "regulondb__primary_promoter_id",
                "regulondb__primary_promoter_name",
            ],
            "thresholds": [0.5],
            "tail_modes": ["margin_top_quantile"],
            "min_global_promoters": 1,
            "min_tail_hits": 1,
            "common_regulators": ["CRP"],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    assert {"regulator_abbrev", "plan", "threshold", "enrichment_ratio", "q_value"}.issubset(table.column_names)
    score_table = pq.read_table(artifact.artifact_dir / "native_plan_margin_scores.parquet")
    tail_table = pq.read_table(artifact.artifact_dir / "native_plan_margin_tail_membership.parquet")
    rank_table = pq.read_table(artifact.artifact_dir / "native_regulator_plan_rank_tests.parquet")
    assert "regulondb__primary_promoter_id" in score_table.column_names
    assert score_table.num_rows == 3
    assert tail_table.num_rows > 0
    assert {"regulator_abbrev", "plan", "auc", "rank_biserial", "p_value_method"}.issubset(rank_table.column_names)
    assert rank_table.num_rows == 12
    assert ("native_plan_margin_scores.parquet", "application/x-parquet") in artifact.outputs
    assert ("native_regulator_plan_rank_tests.parquet", "application/x-parquet") in artifact.outputs
    assert artifact.stats["native_rows"] == 3
    assert artifact.stats["rank_test_rows"] == 12
    assert artifact.stats["native_metadata_columns"] == [
        "alias_id",
        "regulondb__primary_promoter_id",
        "regulondb__primary_promoter_name",
    ]

    with pytest.raises(ContractViolationError, match="legacy regulatory_interactions keys"):
        build_scalar_artifact(
            context,
            scalar_id="native_regulator_plan_margin_enrichment_legacy",
            builder_kind="native_regulator_plan_margin_enrichment",
            params={
                "view_id": "bidir_context",
                "cohort_column": "design_family",
                "centroid_groups": {
                    "background": ["background_only"],
                    "ethanol": ["ethanol"],
                    "cipro": ["ciprofloxacin"],
                    "dual": ["ethanol_ciprofloxacin"],
                },
                "native_filter": {
                    "column": "derived__parent_dataset",
                    "equals": "usr_regulondb_native_promoters",
                },
                "regulatory_interactions": {
                    "path": "inputs/regulatory_interactions.parquet",
                    "row_key": "derived__parent_id",
                    "relation_key": "usr_id",
                    "regulator_column": "regulator_abbrev",
                },
                "thresholds": [0.5],
                "tail_modes": ["margin_top_quantile"],
                "min_global_promoters": 1,
                "min_tail_hits": 1,
            },
        )


def test_plan_margin_feature_enrichment_scalar_reuses_source_side_tables(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    source_dir = workspace_dir / "outputs" / "scalars" / "native_regulator_plan_margin_enrichment"
    source_dir.mkdir(parents=True)
    _write_scalar_manifest(
        source_dir,
        scalar_id="native_regulator_plan_margin_enrichment",
        outputs=[
            "table.parquet",
            "native_plan_margin_scores.parquet",
            "native_plan_margin_tail_membership.parquet",
        ],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "native_parent_id": "rp_eth",
                    "nearest_plan": "ethanol",
                    "margin_background": -0.2,
                    "margin_cipro": -0.3,
                    "margin_dual": -0.1,
                    "margin_ethanol": 0.8,
                },
                {
                    "native_parent_id": "rp_cipro",
                    "nearest_plan": "cipro",
                    "margin_background": -0.1,
                    "margin_cipro": 0.7,
                    "margin_dual": -0.2,
                    "margin_ethanol": -0.3,
                },
                {
                    "native_parent_id": "rp_bg",
                    "nearest_plan": "background",
                    "margin_background": 0.6,
                    "margin_cipro": -0.2,
                    "margin_dual": -0.3,
                    "margin_ethanol": -0.4,
                },
                {
                    "native_parent_id": "rp_dual",
                    "nearest_plan": "dual",
                    "margin_background": -0.4,
                    "margin_cipro": -0.1,
                    "margin_dual": 0.5,
                    "margin_ethanol": -0.2,
                },
            ]
        ),
        source_dir / "native_plan_margin_scores.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "native_parent_id": "rp_eth",
                    "plan": "ethanol",
                    "threshold": 0.5,
                    "tail_mode": "margin_top_quantile",
                },
                {
                    "native_parent_id": "rp_cipro",
                    "plan": "cipro",
                    "threshold": 0.5,
                    "tail_mode": "margin_top_quantile",
                },
                {
                    "native_parent_id": "rp_bg",
                    "plan": "background",
                    "threshold": 0.5,
                    "tail_mode": "margin_top_quantile",
                },
                {
                    "native_parent_id": "rp_dual",
                    "plan": "dual",
                    "threshold": 0.5,
                    "tail_mode": "margin_top_quantile",
                },
            ]
        ),
        source_dir / "native_plan_margin_tail_membership.parquet",
    )
    feature_path = workspace_dir / "inputs" / "promoter_regulator_go_terms.parquet"
    feature_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "usr_id": "rp_eth",
                    "go_id": "GO:stress",
                    "go_name": "response to stress",
                    "go_namespace": "biological_process",
                    "biocyc_kb_version": "29.6",
                    "smarttable_id": "st-1",
                    "source_terms_sha256": "abc",
                },
                {
                    "usr_id": "rp_eth",
                    "go_id": "GO:stress",
                    "go_name": "response to stress",
                    "go_namespace": "biological_process",
                    "biocyc_kb_version": "29.6",
                    "smarttable_id": "st-1",
                    "source_terms_sha256": "abc",
                },
                {
                    "usr_id": "rp_cipro",
                    "go_id": "GO:sos",
                    "go_name": "SOS response",
                    "go_namespace": "biological_process",
                    "biocyc_kb_version": "29.6",
                    "smarttable_id": "st-1",
                    "source_terms_sha256": "abc",
                },
                {
                    "usr_id": "rp_bg",
                    "go_id": "GO:dna",
                    "go_name": "DNA binding",
                    "go_namespace": "molecular_function",
                    "biocyc_kb_version": "29.6",
                    "smarttable_id": "st-1",
                    "source_terms_sha256": "abc",
                },
            ]
        ),
        feature_path,
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="native_regulator_go_bp_plan_margin_enrichment",
        builder_kind="plan_margin_feature_enrichment",
        params={
            "source_scalar": "native_regulator_plan_margin_enrichment",
            "scores_table": "native_plan_margin_scores.parquet",
            "tail_membership_table": "native_plan_margin_tail_membership.parquet",
            "feature_membership": {
                "path": "inputs/promoter_regulator_go_terms.parquet",
                "subject_column": "usr_id",
                "feature_id_column": "go_id",
                "feature_label_column": "go_name",
                "feature_namespace_column": "go_namespace",
                "namespace_filter": "biological_process",
                "exclude_label_prefixes": ["obsolete "],
                "source_metadata_columns": ["biocyc_kb_version", "smarttable_id", "source_terms_sha256"],
            },
            "min_global_subjects": 1,
            "min_tail_hits": 1,
            "rank_test_alternative": "greater",
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    rank_table = pq.read_table(artifact.artifact_dir / "plan_margin_feature_rank_tests.parquet")
    assert {"feature_id", "feature_label", "feature_namespace", "n_feature_tail"}.issubset(table.column_names)
    assert {"feature_id", "feature_label", "plan", "auc", "rank_biserial", "p_value_method"}.issubset(
        rank_table.column_names
    )
    rows = table.to_pylist()
    stress_ethanol = next(row for row in rows if row["feature_id"] == "GO:stress" and row["plan"] == "ethanol")
    assert stress_ethanol["feature_label"] == "response to stress"
    assert stress_ethanol["n_feature_tail"] == 1
    rank_rows = rank_table.to_pylist()
    stress_ethanol_rank = next(
        row for row in rank_rows if row["feature_id"] == "GO:stress" and row["plan"] == "ethanol"
    )
    assert stress_ethanol_rank["rank_biserial"] > 0.0
    assert artifact.stats["source_scalar"] == "native_regulator_plan_margin_enrichment"
    assert artifact.stats["matched_features"] == 2
    assert artifact.stats["rank_test_rows"] == 8
    assert artifact.stats["excluded_label_prefixes"] == ["obsolete "]
    assert ("plan_margin_feature_rank_tests.parquet", "application/x-parquet") in artifact.outputs


def test_plan_margin_feature_enrichment_scalar_requires_matching_source_manifest(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    source_dir = workspace_dir / "outputs" / "scalars" / "native_regulator_plan_margin_enrichment"
    source_dir.mkdir(parents=True)
    _write_scalar_manifest(
        source_dir,
        scalar_id="wrong_source",
        outputs=[
            "table.parquet",
            "native_plan_margin_scores.parquet",
            "native_plan_margin_tail_membership.parquet",
        ],
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="manifest id mismatch"):
        build_scalar_artifact(
            context,
            scalar_id="native_regulator_go_bp_plan_margin_enrichment",
            builder_kind="plan_margin_feature_enrichment",
            params={
                "source_scalar": "native_regulator_plan_margin_enrichment",
                "scores_table": "native_plan_margin_scores.parquet",
                "tail_membership_table": "native_plan_margin_tail_membership.parquet",
                "feature_membership": {
                    "path": "inputs/promoter_regulator_go_terms.parquet",
                    "subject_column": "usr_id",
                    "feature_id_column": "go_id",
                    "feature_label_column": "go_name",
                    "feature_namespace_column": "go_namespace",
                    "namespace_filter": "biological_process",
                },
                "min_global_subjects": 1,
                "min_tail_hits": 1,
            },
        )


def test_cohort_similarity_margin_honors_sample_scope(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "design_family": "background_only", "embedding": [0.0, 1.0]},
        {"id": "bg_2", "subject_id": "bg_2", "design_family": "background_only", "embedding": [0.1, 0.9]},
        {"id": "eth_1", "subject_id": "eth_1", "design_family": "ethanol", "embedding": [1.0, 0.0]},
        {"id": "eth_2", "subject_id": "eth_2", "design_family": "ethanol", "embedding": [0.9, 0.1]},
        {"id": "cip_1", "subject_id": "cip_1", "design_family": "ciprofloxacin", "embedding": [0.6, 0.4]},
        {"id": "cip_2", "subject_id": "cip_2", "design_family": "ciprofloxacin", "embedding": [0.4, 0.6]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    sample_rows = rows[:4]
    sample_dir = workspace_dir / "outputs" / "samples" / "ethanol_background_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sample_rows), sample_dir / "rows.parquet")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="sampled_design_margins",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "sample_id": "ethanol_background_sample",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "margin_pairs": [
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    assert table.num_rows == 4
    assert set(table["id"].to_pylist()) == {"bg_1", "bg_2", "eth_1", "eth_2"}
    assert artifact.stats["sample_id"] == "ethanol_background_sample"
    assert any(input_ref.kind == "sample_set" for input_ref in artifact.inputs)


def test_cohort_similarity_margin_supports_per_pair_cohort_columns_and_best_stress_margin(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_b",
            "subject_id": "bg_b",
            "design_family": "background_only",
            "sig35_variant": "b",
            "embedding": [0.0, 1.0],
        },
        {
            "id": "bg_f",
            "subject_id": "bg_f",
            "design_family": "background_only",
            "sig35_variant": "f",
            "embedding": [0.2, 0.8],
        },
        {
            "id": "eth_f",
            "subject_id": "eth_f",
            "design_family": "ethanol",
            "sig35_variant": "f",
            "embedding": [0.9, 0.2],
        },
        {
            "id": "eth_b",
            "subject_id": "eth_b",
            "design_family": "ethanol",
            "sig35_variant": "b",
            "embedding": [0.7, 0.3],
        },
        {
            "id": "cip_f",
            "subject_id": "cip_f",
            "design_family": "ciprofloxacin",
            "sig35_variant": "f",
            "embedding": [0.6, 0.4],
        },
        {
            "id": "cip_b",
            "subject_id": "cip_b",
            "design_family": "ciprofloxacin",
            "sig35_variant": "b",
            "embedding": [0.4, 0.6],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="sigma35_stress_margins",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "margin_pairs": [
                {
                    "cohort_column": "sig35_variant",
                    "target_values": ["f"],
                    "control_values": ["b"],
                    "output_column": "sig35_margin_f_vs_b",
                },
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                },
                {
                    "target_values": ["ciprofloxacin"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_cipro_vs_background",
                },
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    assert "sig35_margin_f_vs_b" in table.column_names
    assert "synthetic_best_stress_margin" in table.column_names
    ethanol = np.asarray(table["synthetic_margin_ethanol_vs_background"].to_pylist(), dtype=np.float32)
    cipro = np.asarray(table["synthetic_margin_cipro_vs_background"].to_pylist(), dtype=np.float32)
    best = np.asarray(table["synthetic_best_stress_margin"].to_pylist(), dtype=np.float32)
    assert np.allclose(best, np.maximum(ethanol, cipro), equal_nan=True)
    assert artifact.stats["sig35_margin_f_vs_b_cohort_column"] == "sig35_variant"


def test_cohort_similarity_margin_can_restrict_rows_to_balanced_design_family_subset(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_f_16",
            "subject_id": "bg_f_16",
            "design_family": "background_only",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [0.0, 1.0],
        },
        {
            "id": "bg_f_17",
            "subject_id": "bg_f_17",
            "design_family": "background_only",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.1, 0.9],
        },
        {
            "id": "bg_b_16",
            "subject_id": "bg_b_16",
            "design_family": "background_only",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.2, 0.8],
        },
        {
            "id": "eth_f_16",
            "subject_id": "eth_f_16",
            "design_family": "ethanol",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "eth_f_17",
            "subject_id": "eth_f_17",
            "design_family": "ethanol",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.9, 0.1],
        },
        {
            "id": "eth_b_16",
            "subject_id": "eth_b_16",
            "design_family": "ethanol",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.8, 0.2],
        },
        {
            "id": "cip_f_16",
            "subject_id": "cip_f_16",
            "design_family": "ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [0.7, 0.3],
        },
        {
            "id": "cip_f_17",
            "subject_id": "cip_f_17",
            "design_family": "ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.6, 0.4],
        },
        {
            "id": "cip_b_16",
            "subject_id": "cip_b_16",
            "design_family": "ciprofloxacin",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.5, 0.5],
        },
        {
            "id": "dual_f_16",
            "subject_id": "dual_f_16",
            "design_family": "ethanol_ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [0.4, 0.6],
        },
        {
            "id": "dual_f_17",
            "subject_id": "dual_f_17",
            "design_family": "ethanol_ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.3, 0.7],
        },
        {
            "id": "dual_b_16",
            "subject_id": "dual_b_16",
            "design_family": "ethanol_ciprofloxacin",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.2, 0.8],
        },
        {
            "id": "ctrl",
            "subject_id": "ctrl",
            "design_family": "control",
            "sig35_variant": "control",
            "spacer_length": 16,
            "embedding": [0.1, 0.1],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="balanced_design_margins",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "balance_group_column": "design_family",
            "balance_columns": ["sig35_variant", "spacer_length"],
            "required_group_values": [
                "background_only",
                "ethanol",
                "ciprofloxacin",
                "ethanol_ciprofloxacin",
            ],
            "exclude_group_values": ["control"],
            "margin_pairs": [
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                },
                {
                    "target_values": ["ciprofloxacin"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_cipro_vs_background",
                },
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    balanced_ids = set(table["id"].to_pylist())
    assert balanced_ids == {
        "bg_f_16",
        "bg_f_17",
        "bg_b_16",
        "eth_f_16",
        "eth_f_17",
        "eth_b_16",
        "cip_f_16",
        "cip_f_17",
        "cip_b_16",
        "dual_f_16",
        "dual_f_17",
        "dual_b_16",
    }
    assert table.num_rows == 12
    assert artifact.stats["balanced_group_column"] == "design_family"
    assert artifact.stats["balanced_row_count"] == 12


def test_cohort_similarity_margin_can_balance_references_without_dropping_full_population(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_f_16",
            "subject_id": "bg_f_16",
            "design_family": "background_only",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [0.0, 1.0],
        },
        {
            "id": "bg_f_17",
            "subject_id": "bg_f_17",
            "design_family": "background_only",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.1, 0.9],
        },
        {
            "id": "bg_b_16",
            "subject_id": "bg_b_16",
            "design_family": "background_only",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.2, 0.8],
        },
        {
            "id": "eth_f_16",
            "subject_id": "eth_f_16",
            "design_family": "ethanol",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "eth_f_17",
            "subject_id": "eth_f_17",
            "design_family": "ethanol",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.9, 0.1],
        },
        {
            "id": "eth_b_16",
            "subject_id": "eth_b_16",
            "design_family": "ethanol",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.8, 0.2],
        },
        {
            "id": "cip_f_16",
            "subject_id": "cip_f_16",
            "design_family": "ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [0.7, 0.3],
        },
        {
            "id": "cip_f_17",
            "subject_id": "cip_f_17",
            "design_family": "ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.6, 0.4],
        },
        {
            "id": "cip_b_16",
            "subject_id": "cip_b_16",
            "design_family": "ciprofloxacin",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.5, 0.5],
        },
        {
            "id": "dual_f_16",
            "subject_id": "dual_f_16",
            "design_family": "ethanol_ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [0.4, 0.6],
        },
        {
            "id": "dual_f_17",
            "subject_id": "dual_f_17",
            "design_family": "ethanol_ciprofloxacin",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.3, 0.7],
        },
        {
            "id": "dual_b_16",
            "subject_id": "dual_b_16",
            "design_family": "ethanol_ciprofloxacin",
            "sig35_variant": "b",
            "spacer_length": 16,
            "embedding": [0.2, 0.8],
        },
        {
            "id": "ctrl",
            "subject_id": "ctrl",
            "design_family": "control",
            "sig35_variant": "control",
            "spacer_length": 16,
            "embedding": [0.1, 0.1],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="balanced_design_margins_full_population",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "balance_group_column": "design_family",
            "balance_columns": ["sig35_variant", "spacer_length"],
            "balance_reference_only": True,
            "required_group_values": [
                "background_only",
                "ethanol",
                "ciprofloxacin",
                "ethanol_ciprofloxacin",
            ],
            "exclude_group_values": ["control"],
            "margin_pairs": [
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                },
                {
                    "target_values": ["ciprofloxacin"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_cipro_vs_background",
                },
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    assert table.num_rows == len(rows)
    assert set(table["id"].to_pylist()) == {row["id"] for row in rows}
    assert artifact.stats["balanced_group_column"] == "design_family"
    assert artifact.stats["balanced_reference_only"] is True
    assert artifact.stats["balanced_row_count"] == 12
    assert artifact.stats["synthetic_margin_ethanol_vs_background_target_members"] == 3
    assert artifact.stats["synthetic_margin_ethanol_vs_background_target_reference_members"] == 3


def test_reference_alignment_summary_fails_when_required_references_are_missing(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "usr_label__primary": "dense_bg",
            "embedding": [0.0, 1.0],
        },
        {
            "id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "usr_label__primary": "dense_eth",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "cip_1",
            "subject_id": "cip_1",
            "design_family": "ciprofloxacin",
            "usr_label__primary": "dense_cip",
            "embedding": [0.5, 0.5],
        },
        {
            "id": "spyp",
            "subject_id": "spyp",
            "design_family": "control",
            "usr_label__primary": "spyp",
            "embedding": [0.8, 0.2],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)

    with pytest.raises(ContractViolationError, match="requires carried SpyP and SulA rows"):
        build_scalar_artifact(
            context,
            scalar_id="reference_alignment_summary_metrics",
            builder_kind="reference_alignment_summary",
            params={"candidates": [{"view_id": "degenerate_view"}]},
        )


def test_reference_alignment_summary_emits_collection_collapse_without_spyp_sulap(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "w1",
            "subject_id": "w1",
            "design_family": "control",
            "usr_label__primary": "W1_core60",
            "promoter_standard__collection_id": "t7_w_collection",
            "embedding": [0.8, 0.2, 0.0],
        },
        {
            "id": "w2",
            "subject_id": "w2",
            "design_family": "control",
            "usr_label__primary": "W2_core60",
            "promoter_standard__collection_id": "t7_w_collection",
            "embedding": [0.75, 0.25, 0.0],
        },
        {
            "id": "j1",
            "subject_id": "j1",
            "design_family": "control",
            "usr_label__primary": "J23101_core60",
            "promoter_standard__collection_id": "anderson_igem",
            "embedding": [0.1, 0.9, 0.0],
        },
        {
            "id": "j2",
            "subject_id": "j2",
            "design_family": "control",
            "usr_label__primary": "J23102_core60",
            "promoter_standard__collection_id": "anderson_igem",
            "embedding": [0.2, 0.8, 0.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="reference_alignment_summary_metrics",
        builder_kind="reference_alignment_summary",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "reference_group_columns": ["promoter_standard__collection_id"],
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    metric_rows = {
        (row["reference_group_column"], row["reference_group"], row["metric_id"]): row
        for row in table_rows
        if row.get("reference_group_column")
    }
    assert (
        "promoter_standard__collection_id",
        "t7_w_collection",
        "reference_group_pairwise_cosine_distance_median",
    ) in metric_rows
    assert (
        "promoter_standard__collection_id",
        "anderson_igem",
        "reference_group_size",
    ) in metric_rows
    anderson_size = metric_rows[
        (
            "promoter_standard__collection_id",
            "anderson_igem",
            "reference_group_size",
        )
    ]
    assert anderson_size["display_name"] == "Reference group size\nStd: Anderson iGEM"
    assert all(row["metric_value"] >= 0.0 for row in table_rows)


def test_reference_alignment_summary_keeps_group_columns_with_legacy_alignment_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "cip_1",
            "subject_id": "cip_1",
            "design_family": "ciprofloxacin",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [0.3, 0.4, 1.0],
        },
        {
            "id": "spyp",
            "subject_id": "spyp",
            "design_family": "control",
            "usr_label__primary": "spyp",
            "promoter_standard__collection_id": "diagnostic_controls",
            "embedding": [0.8, 0.2, 0.0],
        },
        {
            "id": "sulap",
            "subject_id": "sulap",
            "design_family": "control",
            "usr_label__primary": "sulAp",
            "promoter_standard__collection_id": "diagnostic_controls",
            "embedding": [0.2, 0.8, 1.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="reference_alignment_summary_metrics",
        builder_kind="reference_alignment_summary",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "reference_group_columns": ["promoter_standard__collection_id"],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    assert "reference_group_column" in table.column_names
    table_rows = table.to_pylist()
    assert any(row["metric_id"] == "reference_alignment_ethanol_background_relative" for row in table_rows)
    assert any(
        row["reference_group_column"] == "promoter_standard__collection_id"
        and row["reference_group"] == "diagnostic_controls"
        for row in table_rows
    )


def test_reference_alignment_summary_emits_configured_reference_set_status_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "w1",
            "subject_id": "w1",
            "design_family": "control",
            "usr_label__primary": "W1_core60",
            "promoter_standard__collection_id": "t7_w_collection",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "w2",
            "subject_id": "w2",
            "design_family": "control",
            "usr_label__primary": "W2_core60",
            "promoter_standard__collection_id": "t7_w_collection",
            "embedding": [0.8, 0.2, 0.0],
        },
        {
            "id": "design_row",
            "subject_id": "design_row",
            "design_family": "background_only",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [0.0, 1.0, 0.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="reference_alignment_summary_metrics",
        builder_kind="reference_alignment_summary",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "reference_sets": ["reference_w_collection", "reference_anderson_igem"],
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    metrics = {(row["reference_set_id"], row["metric_id"]): row for row in table_rows if row.get("reference_set_id")}
    w_size = metrics[("reference_w_collection", "reference_group_size")]
    assert w_size["metric_value"] == 2.0
    assert w_size["reference_set_status"] == "ok"
    assert w_size["reference_set_complete"] is True
    assert w_size["display_name"] == "Reference group size\nReference set: W collection"
    normalized = _normalized_geometry_rows(np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    expected_w_distance = 1.0 - float(np.dot(normalized[0], normalized[1]))
    w_distance = metrics[
        (
            "reference_w_collection",
            "reference_group_pairwise_cosine_distance_median",
        )
    ]
    assert w_distance["metric_value"] == pytest.approx(expected_w_distance)
    anderson_size = metrics[("reference_anderson_igem", "reference_group_size")]
    assert anderson_size["metric_value"] == 0.0
    assert anderson_size["reference_set_status"] == "absent"
    assert anderson_size["reference_set_complete"] is False
    anderson_distance = metrics[
        (
            "reference_anderson_igem",
            "reference_group_pairwise_cosine_distance_median",
        )
    ]
    assert np.isnan(anderson_distance["metric_value"])


def test_reference_alignment_summary_reports_missing_reference_set_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "row_1",
            "subject_id": "row_1",
            "design_family": "control",
            "usr_label__primary": "W1_core60",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "row_2",
            "subject_id": "row_2",
            "design_family": "control",
            "usr_label__primary": "W2_core60",
            "embedding": [0.0, 1.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="reference_alignment_summary_metrics",
        builder_kind="reference_alignment_summary",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "reference_sets": ["reference_w_collection"],
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    size_row = next(row for row in table_rows if row["metric_id"] == "reference_group_size")
    assert size_row["metric_value"] == 0.0
    assert size_row["reference_set_status"] == "missing_columns"
    assert size_row["reference_set_missing_columns"] == "promoter_standard__collection_id"


def test_sigma35_ordinal_audit_emits_confidence_intervals_for_spearman_summary_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "a_f",
            "subject_id": "a_f",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "a_d",
            "subject_id": "a_d",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "d",
            "spacer_length": 16,
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": "a_e",
            "subject_id": "a_e",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "e",
            "spacer_length": 16,
            "embedding": [-1.0, 0.0, 0.0],
        },
        {
            "id": "b_f",
            "subject_id": "b_f",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.9, 0.1, 0.0],
        },
        {
            "id": "b_d",
            "subject_id": "b_d",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "spacer_length": 17,
            "embedding": [0.1, 0.9, 0.0],
        },
        {
            "id": "b_e",
            "subject_id": "b_e",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "e",
            "spacer_length": 17,
            "embedding": [-0.9, 0.1, 0.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    study_inputs_dir = workspace_dir / "study_inputs"
    study_inputs_dir.mkdir(parents=True, exist_ok=True)
    (study_inputs_dir / "sig35_order.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "test fixture",
                "exploratory": False,
                "order": [
                    {"variant_id": "f", "sequence": "TTGACA", "rank": 1},
                    {"variant_id": "d", "sequence": "TTTACA", "rank": 2},
                    {"variant_id": "e", "sequence": "TAGACA", "rank": 3},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="sigma35_ordinal_audit_metrics",
        builder_kind="ordinal_axis_audit",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "axis": {
                "axis_id": "sigma35",
                "label": "Sigma-35",
                "column": "sig35_variant",
                "order_path": "study_inputs/sig35_order.yaml",
                "exclude_values": ["control"],
                "metric_ids": {
                    "spearman": "sig35_ordinal_spearman",
                    "kendall": "sig35_ordinal_kendall",
                    "balanced_spearman": "sig35_balanced_ordinal_spearman",
                    "permutation_pvalue": "sig35_label_permutation_pvalue",
                },
                "within_groups": [
                    {"column": "design_family", "metric_id": "sig35_within_family_mean_spearman"},
                    {
                        "column": "design_regulator_composition",
                        "metric_id": "sig35_within_regulator_mean_spearman",
                    },
                ],
            },
            "bootstrap_iterations": 20,
            "permutations": 20,
            "balance_columns": ["design_family", "spacer_length"],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    ci_metric_ids = {
        "sig35_ordinal_spearman",
        "sig35_balanced_ordinal_spearman",
        "sig35_within_family_mean_spearman",
        "sig35_within_regulator_mean_spearman",
    }
    assert {row["ordinal_metric_role"] for row in rows_table} == {
        "spearman",
        "kendall",
        "balanced_spearman",
        "permutation_pvalue",
        "within_group_mean_spearman",
    }
    for row in rows_table:
        if row["metric_id"] not in ci_metric_ids:
            continue
        assert row["ci_lower"] is not None
        assert row["ci_upper"] is not None
        assert row["ordinal_axis_id"] == "sigma35"
        assert row["ordinal_axis_column"] == "sig35_variant"


def test_ordinal_axis_audit_supports_numeric_rank_metadata_without_sig35_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "low_1",
            "subject_id": "low_1",
            "standard_id": "std_low",
            "strength": 1.0,
            "batch": "a",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "low_2",
            "subject_id": "low_2",
            "standard_id": "std_low",
            "strength": 1.0,
            "batch": "b",
            "embedding": [0.9, 0.1],
        },
        {
            "id": "mid_1",
            "subject_id": "mid_1",
            "standard_id": "std_mid",
            "strength": 5.0,
            "batch": "a",
            "embedding": [0.1, 0.9],
        },
        {
            "id": "mid_2",
            "subject_id": "mid_2",
            "standard_id": "std_mid",
            "strength": 5.0,
            "batch": "b",
            "embedding": [0.0, 1.0],
        },
        {
            "id": "high_1",
            "subject_id": "high_1",
            "standard_id": "std_high",
            "strength": 10.0,
            "batch": "a",
            "embedding": [-1.0, 0.0],
        },
        {
            "id": "high_2",
            "subject_id": "high_2",
            "standard_id": "std_high",
            "strength": 10.0,
            "batch": "b",
            "embedding": [-0.9, 0.1],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="standard_strength_ordinal_metrics",
        builder_kind="ordinal_axis_audit",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "axis": {
                "axis_id": "standard_strength",
                "label": "Standard strength",
                "column": "standard_id",
                "rank_column": "strength",
                "within_groups": [{"column": "batch"}],
            },
            "bootstrap_iterations": 5,
            "permutations": 5,
            "balance_columns": ["batch"],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    metric_ids = {row["metric_id"] for row in rows_table}

    assert artifact.stats["axis_id"] == "standard_strength"
    assert metric_ids == {
        "ordinal_axis_spearman",
        "ordinal_axis_kendall",
        "ordinal_axis_balanced_spearman",
        "ordinal_axis_within_group_mean_spearman",
        "ordinal_axis_label_permutation_pvalue",
    }
    assert all(row["ordinal_axis_column"] == "standard_id" for row in rows_table)
    assert all(row["ordinal_order_source"] == "strength" for row in rows_table)


def test_ordinal_axes_audit_combines_multiple_ordered_metadata_axes(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "a1", "subject_id": "a1", "khan_bin": "low", "crawford_bin": "very_low", "embedding": [1.0, 0.0]},
        {"id": "a2", "subject_id": "a2", "khan_bin": "low", "crawford_bin": "very_low", "embedding": [0.9, 0.1]},
        {"id": "b1", "subject_id": "b1", "khan_bin": "medium", "crawford_bin": "medium", "embedding": [0.1, 0.9]},
        {"id": "b2", "subject_id": "b2", "khan_bin": "medium", "crawford_bin": "medium", "embedding": [0.0, 1.0]},
        {"id": "c1", "subject_id": "c1", "khan_bin": "high", "crawford_bin": "very_high", "embedding": [-1.0, 0.0]},
        {"id": "c2", "subject_id": "c2", "khan_bin": "high", "crawford_bin": "very_high", "embedding": [-0.9, 0.1]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )
    study_inputs_dir = workspace_dir / "study_inputs"
    study_inputs_dir.mkdir(parents=True, exist_ok=True)
    (study_inputs_dir / "khan_order.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "khan fixture",
                "order": [{"value": "low", "rank": 1}, {"value": "medium", "rank": 2}, {"value": "high", "rank": 3}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_inputs_dir / "crawford_order.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "crawford fixture",
                "order": [
                    {"value": "very_low", "rank": 1},
                    {"value": "medium", "rank": 2},
                    {"value": "very_high", "rank": 3},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="overlay_ordinal_audit_metrics",
        builder_kind="ordinal_axes_audit",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "axes": [
                {"axis_id": "khan", "column": "khan_bin", "order_path": "study_inputs/khan_order.yaml"},
                {
                    "axis_id": "crawford",
                    "column": "crawford_bin",
                    "order_path": "study_inputs/crawford_order.yaml",
                },
            ],
            "bootstrap_iterations": 3,
            "permutations": 3,
            "balance_columns": [],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert artifact.stats["axis_ids"] == ["khan", "crawford"]
    assert {row["ordinal_axis_id"] for row in rows_table} == {"khan", "crawford"}
    assert {row["metric_id"] for row in rows_table} == {
        "ordinal_axis_spearman",
        "ordinal_axis_kendall",
        "ordinal_axis_balanced_spearman",
        "ordinal_axis_label_permutation_pvalue",
    }
    assert {row["ordinal_metric_role"] for row in rows_table} == {
        "spearman",
        "kendall",
        "balanced_spearman",
        "permutation_pvalue",
    }


def test_reference_to_centroid_similarity_maps_references_to_plan_centroids(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "background",
            "subject_id": "background",
            "design_family": "background_only",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "ethanol",
            "subject_id": "ethanol",
            "design_family": "ethanol",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": "cipro",
            "subject_id": "cipro",
            "design_family": "ciprofloxacin",
            "usr_label__primary": None,
            "promoter_standard__collection_id": None,
            "embedding": [0.0, 0.0, 1.0],
        },
        {
            "id": "w_eth",
            "subject_id": "w_eth",
            "design_family": "control",
            "usr_label__primary": "W_eth",
            "promoter_standard__collection_id": "t7_w_collection",
            "embedding": [0.0, 0.9, 0.1],
        },
        {
            "id": "w_cipro",
            "subject_id": "w_cipro",
            "design_family": "control",
            "usr_label__primary": "W_cipro",
            "promoter_standard__collection_id": "t7_w_collection",
            "embedding": [0.1, 0.0, 0.9],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="reference_to_plan_centroid_metrics",
        builder_kind="reference_to_centroid_similarity",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "centroid_axis": {
                "column": "design_family",
                "groups": [
                    {"value": "background_only", "label": "Background"},
                    {"value": "ethanol", "label": "Ethanol"},
                    {"value": "ciprofloxacin", "label": "Ciprofloxacin"},
                ],
            },
            "reference_sets": [{"reference_set_id": "reference_w_collection", "aggregation": "rows"}],
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert {row["metric_id"] for row in table_rows} == {"reference_to_centroid_similarity"}
    assert {row["reference_entity_id"] for row in table_rows} == {"W_eth", "W_cipro"}
    assert {row["centroid_group"] for row in table_rows} == {"background_only", "ethanol", "ciprofloxacin"}
    assert all(row["nearest_centroid_group"] in {"background_only", "ethanol", "ciprofloxacin"} for row in table_rows)
    assert all(np.isfinite(float(row["nearest_centroid_margin"])) for row in table_rows)


def test_collection_strength_ordinal_audit_keeps_reference_strength_scales_separate(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "w_low",
            "subject_id": "w_low",
            "standard_id": "W_low",
            "promoter_standard__collection_id": "t7_w_collection",
            "promoter_standard__strength_value_numeric": 1.0,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "w_mid",
            "subject_id": "w_mid",
            "standard_id": "W_mid",
            "promoter_standard__collection_id": "t7_w_collection",
            "promoter_standard__strength_value_numeric": 5.0,
            "embedding": [0.0, 1.0],
        },
        {
            "id": "w_high",
            "subject_id": "w_high",
            "standard_id": "W_high",
            "promoter_standard__collection_id": "t7_w_collection",
            "promoter_standard__strength_value_numeric": 10.0,
            "embedding": [-1.0, 0.0],
        },
        {
            "id": "a_low",
            "subject_id": "a_low",
            "standard_id": "A_low",
            "promoter_standard__collection_id": "anderson_igem",
            "promoter_standard__strength_value_numeric": 100.0,
            "embedding": [0.0, -1.0],
        },
        {
            "id": "a_mid",
            "subject_id": "a_mid",
            "standard_id": "A_mid",
            "promoter_standard__collection_id": "anderson_igem",
            "promoter_standard__strength_value_numeric": 200.0,
            "embedding": [1.0, 1.0],
        },
        {
            "id": "a_high",
            "subject_id": "a_high",
            "standard_id": "A_high",
            "promoter_standard__collection_id": "anderson_igem",
            "promoter_standard__strength_value_numeric": 300.0,
            "embedding": [-1.0, -1.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="reference_standard_strength_audit_metrics",
        builder_kind="collection_strength_ordinal_audit",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "collection_column": "promoter_standard__collection_id",
            "group_column": "standard_id",
            "rank_column": "promoter_standard__strength_value_numeric",
            "collections": [
                {"collection_id": "t7_w_collection", "label": "W collection"},
                {"collection_id": "anderson_igem", "label": "Anderson iGEM"},
            ],
            "permutations": 5,
            "seed": 17,
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert {row["reference_collection_id"] for row in table_rows} == {"t7_w_collection", "anderson_igem"}
    assert {(row["reference_collection_id"], row["ordinal_ranked_group_count"]) for row in table_rows} == {
        ("t7_w_collection", 3),
        ("anderson_igem", 3),
    }
    assert all(row["ordinal_order_source"] == "promoter_standard__strength_value_numeric" for row in table_rows)


def test_ordinal_ladder_rows_emit_sigma35_and_collection_strength_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "sig_f",
            "subject_id": "sig_f",
            "sig35_variant": "f",
            "source_class": "densegen",
            "promoter_standard__collection_id": None,
            "usr_label__primary": "sig_f",
            "promoter_standard__strength_value_numeric": None,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "sig_b",
            "subject_id": "sig_b",
            "sig35_variant": "b",
            "source_class": "densegen",
            "promoter_standard__collection_id": None,
            "usr_label__primary": "sig_b",
            "promoter_standard__strength_value_numeric": None,
            "embedding": [-1.0, 0.0],
        },
        {
            "id": "w1",
            "subject_id": "w1",
            "sig35_variant": "control",
            "source_class": "construct_derived",
            "promoter_standard__collection_id": "t7_w_collection",
            "usr_label__primary": "W1_core60",
            "promoter_standard__display_name": "W1",
            "promoter_standard__strength_value_numeric": 1.0,
            "selection_basis": "template_window_center",
            "embedding": [0.0, -1.0],
        },
        {
            "id": "w9",
            "subject_id": "w9",
            "sig35_variant": "control",
            "source_class": "construct_derived",
            "promoter_standard__collection_id": "t7_w_collection",
            "usr_label__primary": "W9_core60",
            "promoter_standard__display_name": "W9",
            "promoter_standard__strength_value_numeric": 9.0,
            "selection_basis": "template_window_center",
            "embedding": [0.0, 1.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )
    study_inputs_dir = workspace_dir / "study_inputs"
    study_inputs_dir.mkdir(parents=True, exist_ok=True)
    (study_inputs_dir / "sig35_order.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "test fixture",
                "exploratory": False,
                "order": [
                    {"variant_id": "f", "sequence": "TTGACA", "rank": 1},
                    {"variant_id": "b", "sequence": "CTGACA", "rank": 5},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="ordinal_ladder_rows",
        builder_kind="ordinal_ladder_rows",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "groups": [
                {
                    "group_id": "sigma35",
                    "label": "Sigma-35",
                    "axis": {
                        "axis_id": "sigma35",
                        "label": "Sigma-35",
                        "column": "sig35_variant",
                        "order_path": "study_inputs/sig35_order.yaml",
                        "exclude_values": ["control"],
                        "stronger_rank": "min",
                    },
                    "where": [{"column": "source_class", "equals": "densegen"}],
                },
                {
                    "group_id": "t7_w_collection_core60",
                    "label": "W collection core60",
                    "source_value_column": "promoter_standard__strength_value_numeric",
                    "source_value_label": "W collection measured strength",
                    "axis": {
                        "axis_id": "t7_w_collection_strength",
                        "label": "W collection strength",
                        "column": "usr_label__primary",
                        "rank_column": "promoter_standard__strength_value_numeric",
                        "stronger_rank": "max",
                    },
                    "where": [
                        {"column": "promoter_standard__collection_id", "equals": "t7_w_collection"},
                        {"column": "usr_label__primary", "regex": ".*_core60$"},
                    ],
                    "label_column": "promoter_standard__display_name",
                },
            ],
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert {row["ordinal_group_id"] for row in table_rows} == {"sigma35", "t7_w_collection_core60"}
    assert {row["ordinal_label"] for row in table_rows if row["ordinal_group_id"] == "sigma35"} == {"B", "F"}
    assert {row["ordinal_label"] for row in table_rows if row["ordinal_group_id"] == "t7_w_collection_core60"} == {
        "W1",
        "W9",
    }
    source_values = {
        row["ordinal_label"]: row["ordinal_source_value"]
        for row in table_rows
        if row["ordinal_group_id"] == "t7_w_collection_core60"
    }
    assert source_values == {"W1": 1.0, "W9": 9.0}
    assert {
        row["ordinal_source_value_label"] for row in table_rows if row["ordinal_group_id"] == "t7_w_collection_core60"
    } == {"W collection measured strength"}
    assert all(row["ordinal_margin"] is not None for row in table_rows)
    assert artifact.stats["ordinal_group_count"] == 2


def test_ordinal_ladder_rows_use_filtered_core60_collection_extremes(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "w1_rc",
            "subject_id": "w1_rc",
            "promoter_standard__collection_id": "t7_w_collection",
            "usr_label__primary": "W1_core60_context1kb_rc",
            "promoter_standard__display_name": "W1",
            "promoter_standard__strength_value_numeric": 1.0,
            "embedding": [0.0, -1.0],
        },
        {
            "id": "t7_rc",
            "subject_id": "t7_rc",
            "promoter_standard__collection_id": "t7_w_collection",
            "usr_label__primary": "T7A1_core60_context1kb_rc",
            "promoter_standard__display_name": "T7A1",
            "promoter_standard__strength_value_numeric": 10.0,
            "embedding": [0.0, 1.0],
        },
        {
            "id": "w_native_decoy",
            "subject_id": "w_native_decoy",
            "promoter_standard__collection_id": "t7_w_collection",
            "usr_label__primary": "W9_native",
            "promoter_standard__display_name": "W9 native decoy",
            "promoter_standard__strength_value_numeric": 999.0,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "j23103_rc",
            "subject_id": "j23103_rc",
            "promoter_standard__collection_id": "anderson_igem",
            "usr_label__primary": "J23103_core60_context1kb_rc",
            "promoter_standard__display_name": "J23103",
            "promoter_standard__strength_value_numeric": 0.01,
            "embedding": [-1.0, 0.0],
        },
        {
            "id": "j23113_rc",
            "subject_id": "j23113_rc",
            "promoter_standard__collection_id": "anderson_igem",
            "usr_label__primary": "J23113_core60_context1kb_rc",
            "promoter_standard__display_name": "J23113",
            "promoter_standard__strength_value_numeric": 0.01,
            "embedding": [-1.0, 0.0],
        },
        {
            "id": "j23100_rc",
            "subject_id": "j23100_rc",
            "promoter_standard__collection_id": "anderson_igem",
            "usr_label__primary": "J23100_core60_context1kb_rc",
            "promoter_standard__display_name": "J23100",
            "promoter_standard__strength_value_numeric": 1.0,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "j23119_rc",
            "subject_id": "j23119_rc",
            "promoter_standard__collection_id": "anderson_igem",
            "usr_label__primary": "J23119_core60_context1kb_rc",
            "promoter_standard__display_name": "J23119",
            "promoter_standard__strength_value_numeric": 1.0,
            "embedding": [1.0, 0.0],
        },
        {
            "id": "anderson_native_decoy",
            "subject_id": "anderson_native_decoy",
            "promoter_standard__collection_id": "anderson_igem",
            "usr_label__primary": "J23199_native",
            "promoter_standard__display_name": "J23199 native decoy",
            "promoter_standard__strength_value_numeric": 2.0,
            "embedding": [0.0, 1.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="ordinal_ladder_rows",
        builder_kind="ordinal_ladder_rows",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "groups": [
                {
                    "group_id": "t7_w_collection_core60",
                    "label": "W collection core60",
                    "axis": {
                        "axis_id": "t7_w_collection_strength",
                        "label": "W collection strength",
                        "column": "usr_label__primary",
                        "rank_column": "promoter_standard__strength_value_numeric",
                        "stronger_rank": "max",
                    },
                    "where": [
                        {"column": "promoter_standard__collection_id", "equals": "t7_w_collection"},
                        {"column": "usr_label__primary", "regex": ".*_core60(?:_context1kb_(?:forward|rc))?$"},
                    ],
                    "label_column": "promoter_standard__display_name",
                },
                {
                    "group_id": "anderson_igem_core60",
                    "label": "Anderson iGEM core60",
                    "axis": {
                        "axis_id": "anderson_igem_strength",
                        "label": "Anderson iGEM strength",
                        "column": "usr_label__primary",
                        "rank_column": "promoter_standard__strength_value_numeric",
                        "stronger_rank": "max",
                    },
                    "where": [
                        {"column": "promoter_standard__collection_id", "equals": "anderson_igem"},
                        {"column": "usr_label__primary", "regex": ".*_core60(?:_context1kb_(?:forward|rc))?$"},
                    ],
                    "label_column": "promoter_standard__display_name",
                },
            ],
        },
    )

    table_rows = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    rows_by_group = {
        group_id: [row for row in table_rows if row["ordinal_group_id"] == group_id]
        for group_id in {"t7_w_collection_core60", "anderson_igem_core60"}
    }
    assert {row["usr_label__primary"] for row in rows_by_group["t7_w_collection_core60"]} == {
        "W1_core60_context1kb_rc",
        "T7A1_core60_context1kb_rc",
    }
    assert {row["ordinal_target_values"] for row in rows_by_group["t7_w_collection_core60"]} == {
        "T7A1_core60_context1kb_rc"
    }
    assert {row["ordinal_control_values"] for row in rows_by_group["t7_w_collection_core60"]} == {
        "W1_core60_context1kb_rc"
    }
    assert {row["ordinal_target_members"] for row in rows_by_group["t7_w_collection_core60"]} == {1}
    assert {row["ordinal_control_members"] for row in rows_by_group["t7_w_collection_core60"]} == {1}

    assert {row["usr_label__primary"] for row in rows_by_group["anderson_igem_core60"]} == {
        "J23103_core60_context1kb_rc",
        "J23113_core60_context1kb_rc",
        "J23100_core60_context1kb_rc",
        "J23119_core60_context1kb_rc",
    }
    assert {row["ordinal_target_values"] for row in rows_by_group["anderson_igem_core60"]} == {
        "J23100_core60_context1kb_rc,J23119_core60_context1kb_rc"
    }
    assert {row["ordinal_control_values"] for row in rows_by_group["anderson_igem_core60"]} == {
        "J23103_core60_context1kb_rc,J23113_core60_context1kb_rc"
    }
    assert {row["ordinal_target_members"] for row in rows_by_group["anderson_igem_core60"]} == {2}
    assert {row["ordinal_control_members"] for row in rows_by_group["anderson_igem_core60"]} == {2}


def test_axis_centroid_distance_includes_unranked_axis_values(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "ranked_b_1",
            "subject_id": "ranked_b_1",
            "sig35_variant": "b",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "ranked_b_2",
            "subject_id": "ranked_b_2",
            "sig35_variant": "b",
            "embedding": [0.9, 0.1],
        },
        {
            "id": "ranked_f_1",
            "subject_id": "ranked_f_1",
            "sig35_variant": "f",
            "embedding": [0.0, 1.0],
        },
        {
            "id": "ranked_f_2",
            "subject_id": "ranked_f_2",
            "sig35_variant": "f",
            "embedding": [0.1, 0.9],
        },
        {
            "id": "annotated_sequence_1",
            "subject_id": "annotated_sequence_1",
            "sig35_variant": "ACCGCG",
            "embedding": [-1.0, 0.0],
        },
        {
            "id": "annotated_sequence_2",
            "subject_id": "annotated_sequence_2",
            "sig35_variant": "ACCGCG",
            "embedding": [-0.9, -0.1],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )
    study_inputs_dir = workspace_dir / "study_inputs"
    study_inputs_dir.mkdir(parents=True, exist_ok=True)
    (study_inputs_dir / "sig35_order.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "test fixture",
                "exploratory": True,
                "order": [
                    {"variant_id": "f", "sequence": "TTGACA", "rank": 1},
                    {"variant_id": "b", "sequence": "CTGACA", "rank": 2},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="sigma35_centroid_distance_metrics",
        builder_kind="axis_centroid_distance",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "axis": {
                "axis_id": "sigma35",
                "column": "sig35_variant",
                "order_path": "study_inputs/sig35_order.yaml",
            },
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    labels = {row["row_variant"] for row in rows_table} | {row["column_variant"] for row in rows_table}

    assert "ACCGCG (unranked)" in labels
    assert any(
        row["row_variant"] == "ACCGCG (unranked)"
        and row["column_variant"] == "ACCGCG (unranked)"
        and row["metric_value"] == pytest.approx(0.0, abs=1e-6)
        for row in rows_table
    )


def test_representation_health_summary_records_effective_rank_basis(tmp_path: Path) -> None:
    workspace_dir = tmp_path
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "a", "subject_id": "a", "embedding": [1.0, 0.0, 0.0, 0.0]},
        {"id": "b", "subject_id": "b", "embedding": [0.0, 1.0, 0.0, 0.0]},
        {"id": "c", "subject_id": "c", "embedding": [0.0, 0.0, 1.0, 0.0]},
        {"id": "d", "subject_id": "d", "embedding": [0.0, 0.0, 0.0, 1.0]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )
    reducer_dir = workspace_dir / "outputs" / "reducers" / "pca_degenerate_view"
    reducer_dir.mkdir(parents=True)
    (reducer_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "pca",
                "pca_method": "dense_svd",
                "fit_rows": 2048,
                "input_dims": 4096,
                "output_dims": 16,
                "scope_kind": "sample_set",
                "scope_id": "scorecard_sample_demo",
                "explained_variance_ratio": [0.6, 0.2, 0.1, 0.05],
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="representation_health_summary_metrics",
        builder_kind="representation_health_summary",
        params={
            "candidates": [
                {
                    "view_id": "degenerate_view",
                    "reducer_id": "pca_degenerate_view",
                }
            ],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    effective_rank = next(row for row in rows_table if row["metric_id"] == "effective_rank")
    pc1_fraction = next(row for row in rows_table if row["metric_id"] == "pc1_variance_fraction")

    assert effective_rank["effective_rank_basis"] == "retained_pca_components"
    assert effective_rank["effective_rank_component_count"] == 4
    assert effective_rank["pca_output_dims"] == 16
    assert effective_rank["pca_fit_rows"] == 2048
    assert effective_rank["pca_fit_scope_kind"] == "sample_set"
    assert effective_rank["pca_fit_scope_id"] == "scorecard_sample_demo"
    assert effective_rank["explained_variance_captured"] == pytest.approx(0.95)
    assert pc1_fraction["metric_value"] == pytest.approx(0.6 / 0.95)
    assert pc1_fraction["explained_variance_captured"] == pytest.approx(0.95)


def test_representation_health_summary_reports_planned_candidates_without_ranking(tmp_path: Path) -> None:
    workspace_dir = tmp_path
    _write_margin_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["views"]["planned_output_layer"] = {
        "source": "anchor_60bp",
        "vector": {"kind": "column", "name": "embedding"},
        "coordinate_space_id": "demo_output_layer_space",
        "tags": {"model": "7b", "family": "output_layer_mean", "scope": "anchor_60bp"},
        "role": "planned",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rows = [
        {"id": "a", "subject_id": "a", "embedding": [1.0, 0.0, 0.0, 0.0]},
        {"id": "b", "subject_id": "b", "embedding": [0.0, 1.0, 0.0, 0.0]},
        {"id": "c", "subject_id": "c", "embedding": [0.0, 0.0, 1.0, 0.0]},
        {"id": "d", "subject_id": "d", "embedding": [0.0, 0.0, 0.0, 1.0]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )
    reducer_dir = workspace_dir / "outputs" / "reducers" / "pca_degenerate_view"
    reducer_dir.mkdir(parents=True)
    (reducer_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "pca",
                "fit_rows": 4,
                "input_dims": 4,
                "output_dims": 4,
                "explained_variance_ratio": [0.5, 0.25, 0.15, 0.1],
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="representation_health_summary_metrics",
        builder_kind="representation_health_summary",
        params={
            "candidates": [{"view_id": "degenerate_view", "reducer_id": "pca_degenerate_view"}],
            "omitted_candidates": [
                {"view_id": "planned_output_layer", "reason": "awaiting output-layer materialization"}
            ],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    planned_rows = [row for row in rows_table if row["candidate_id"] == "planned_output_layer"]

    assert {row["metric_id"] for row in planned_rows} == {
        "effective_rank",
        "pc1_variance_fraction",
        "pairwise_cosine_distance_median",
        "pairwise_cosine_distance_iqr",
    }
    assert all(np.isnan(float(row["metric_value"])) for row in planned_rows)
    assert all(row["candidate_status"] == "planned" for row in planned_rows)
    assert all(row["omitted_from_ranking"] for row in planned_rows)
    assert all(row["health_status"] == "planned" for row in planned_rows)
    assert all(row["omission_reason"] == "awaiting output-layer materialization" for row in planned_rows)
    assert planned_rows[0]["candidate_family"] == "output_layer_mean"


def test_cohort_structure_summary_uses_configured_metadata_axis(tmp_path: Path) -> None:
    workspace_dir = tmp_path
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "rpoD_1", "subject_id": "rpoD_1", "regulondb__sigma_factor_set": "rpoD", "embedding": [1.0, 0.0]},
        {"id": "rpoD_2", "subject_id": "rpoD_2", "regulondb__sigma_factor_set": "rpoD", "embedding": [0.9, 0.1]},
        {"id": "rpoS_1", "subject_id": "rpoS_1", "regulondb__sigma_factor_set": "rpoS", "embedding": [0.0, 1.0]},
        {"id": "rpoS_2", "subject_id": "rpoS_2", "regulondb__sigma_factor_set": "rpoS", "embedding": [0.1, 0.9]},
        {"id": "rare", "subject_id": "rare", "regulondb__sigma_factor_set": "rare", "embedding": [0.5, 0.5]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_view_artifact(
        workspace_dir,
        view_id="degenerate_view",
        rows=rows,
        matrix=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
        record_key="id",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="cohort_structure_summary_metrics",
        builder_kind="cohort_structure_summary",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "axes": [
                {
                    "axis_id": "regulondb_sigma_factor_set",
                    "label": "Sigma-factor set",
                    "column": "regulondb__sigma_factor_set",
                    "min_group_size": 2,
                }
            ],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    row = rows_table[0]

    assert artifact.stats["axis_count"] == 1
    assert artifact.stats["skipped_axes"] == []
    assert row["metric_id"] == "cohort_separation_ratio"
    assert row["display_name"] == "Sigma-factor set"
    assert row["cohort_column"] == "regulondb__sigma_factor_set"
    assert row["cohort_group_count"] == 2
    assert row["cohort_usable_row_count"] == 4
    assert np.isfinite(float(row["metric_value"]))


def test_reference_neighbor_metrics_pool_row_level_censored_ranks_across_tasks() -> None:
    rows = [
        {"usr_label__primary": "dense_eth_1", "design_family": "ethanol"},
        {"usr_label__primary": "dense_eth_2", "design_family": "ethanol"},
        {"usr_label__primary": "dense_cipro_1", "design_family": "ciprofloxacin"},
        {"usr_label__primary": "spyp", "design_family": "control"},
        {"usr_label__primary": "sulAp", "design_family": "control"},
    ]
    indices = np.asarray(
        [
            [3, 1, 2, 4],
            [2, 4, 3, 0],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )

    metrics = _reference_neighbor_metrics(
        rows,
        indices,
        label_column="design_family",
        ethanol_values={"ethanol"},
        cipro_values={"ciprofloxacin"},
    )

    assert metrics["reference_in_knn_rate"] == pytest.approx(1.0)
    assert metrics["reference_neighbor_topk_censored_rank_median"] == pytest.approx(3.0)


def test_cosine_distance_correlation_returns_nan_for_single_row_inputs() -> None:
    left = np.asarray([[1.0, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = _cosine_distance_correlation(left, right)

    assert np.isnan(result)
