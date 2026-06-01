"""Contracts for latentdna view metadata materialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.views.materialize import _derived_metadata_array, materialize_view_artifact
from dnadesign.latentdna.src.views.promoter_metadata_sequence import sig35_variant, spacer_length
from dnadesign.latentdna.src.views.promoter_metadata_stress import (
    design_family,
    design_regulator_composition,
    is_control_row,
    source_class,
)
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config

_PROMOTER_METADATA_HANDLER = "dnadesign.latentdna.src.views.promoter_metadata:derive_promoter_metadata_value"


def _annotation_derivation(
    derive: str,
    *,
    required_columns: list[str],
    any_required_column_groups: list[list[str]] | None = None,
    value_type: str = "string",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "annotation",
        "source": "row",
        "handler": _PROMOTER_METADATA_HANDLER,
        "derive": derive,
        "required_columns": required_columns,
        "missing_policy": "error",
        "value_type": value_type,
    }
    if any_required_column_groups:
        payload["any_required_column_groups"] = any_required_column_groups
    return payload


def _sig35_derivation() -> dict[str, object]:
    return _annotation_derivation(
        "sig35_variant",
        required_columns=["usr_label__primary"],
        any_required_column_groups=[
            ["densegen__plan"],
            ["densegen__used_tfbs_detail"],
            ["seq_annot__features"],
            ["sequence", "derived__features_retained"],
        ],
    )


def test_materialize_view_emits_sig35_variant_without_sigma70_alias(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["synthetic_a", "control_a"], type=pa.string()),
                "subject_id": pa.array(["synthetic_a", "control_a"], type=pa.string()),
                "usr_label__primary": pa.array(["synthetic_a", "J23105"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b", None], type=pa.string()),
                "densegen__required_regulators": pa.array(["cpxR", None], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
                "template_id": pa.array(["tpl_a", "wt"], type=pa.string()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "sig35_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": [
                        "usr_label__primary",
                        "construct_template_id",
                        "design_family",
                        "sig35_variant",
                    ],
                    "derivations": {
                        "construct_template_id": {
                            "kind": "coalesce",
                            "sources": ["construct_template_id", "template_id", "construct__template_id"],
                            "value_type": "string",
                        },
                        "design_family": _annotation_derivation(
                            "design_family",
                            required_columns=["densegen__plan", "usr_label__primary"],
                        ),
                        "sig35_variant": _sig35_derivation(),
                    },
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["sig35_variant"] for row in rows] == ["b", "control"]
    assert [row["construct_template_id"] for row in rows] == ["tpl_a", "wt"]
    assert "sigma70_variant" not in rows[0]


def test_materialize_view_rejects_synthetic_rows_without_sig35_token(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["synthetic_a"], type=pa.string()),
                "subject_id": pa.array(["synthetic_a"], type=pa.string()),
                "usr_label__primary": pa.array(["synthetic_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol"], type=pa.string()),
                "densegen__required_regulators": pa.array(["cpxR"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "sig35_required_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["sig35_variant"],
                    "derivations": {"sig35_variant": _sig35_derivation()},
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    with pytest.raises(ContractViolationError, match="sig35_variant"):
        materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")


def test_materialize_view_uses_lookup_derivation_for_parent_metadata(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["parent_a", "parent_b"], type=pa.string()),
                "sigma_factor": pa.array(["sigma70", "sigma38"], type=pa.string()),
            }
        ),
        inputs_dir / "parents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["child_a", "child_b"], type=pa.string()),
                "subject_id": pa.array(["child_a", "child_b"], type=pa.string()),
                "parent_id": pa.array(["parent_a", "parent_b"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "children.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "lookup_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "parents": {
                        "kind": "parquet",
                        "path": "inputs/parents.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "children": {
                        "kind": "parquet",
                        "path": "inputs/children.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": ["sigma_factor"],
                    "derivations": {
                        "sigma_factor": {
                            "kind": "lookup",
                            "source": "parents",
                            "left_key": "parent_id",
                            "right_key": "id",
                            "value_column": "sigma_factor",
                        }
                    },
                },
                "views": {
                    "child_embedding": {
                        "source": "children",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="child_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["sigma_factor"] for row in rows] == ["sigma70", "sigma38"]


def test_materialize_view_lookup_derivation_fails_on_missing_parent_match(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["parent_a"], type=pa.string()),
                "sigma_factor": pa.array(["sigma70"], type=pa.string()),
            }
        ),
        inputs_dir / "parents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["child_a", "child_b"], type=pa.string()),
                "subject_id": pa.array(["child_a", "child_b"], type=pa.string()),
                "parent_id": pa.array(["parent_a", "missing_parent"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "children.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "lookup_missing_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "parents": {
                        "kind": "parquet",
                        "path": "inputs/parents.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "children": {
                        "kind": "parquet",
                        "path": "inputs/children.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": ["sigma_factor"],
                    "derivations": {
                        "sigma_factor": {
                            "kind": "lookup",
                            "source": "parents",
                            "left_key": "parent_id",
                            "right_key": "id",
                            "value_column": "sigma_factor",
                        }
                    },
                },
                "views": {
                    "child_embedding": {
                        "source": "children",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    with pytest.raises(ContractViolationError, match="missing lookup matches"):
        materialize_view_artifact(context, view_id="child_embedding")


def test_materialize_view_lookup_derivation_can_fill_missing_matches_with_default(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["parent_a"], type=pa.string()),
                "label_status": pa.array(["observed"], type=pa.string()),
            }
        ),
        inputs_dir / "parents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["child_a", "child_b"], type=pa.string()),
                "subject_id": pa.array(["child_a", "child_b"], type=pa.string()),
                "parent_id": pa.array(["parent_a", "missing_parent"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "children.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "lookup_default_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "parents": {
                        "kind": "parquet",
                        "path": "inputs/parents.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "children": {
                        "kind": "parquet",
                        "path": "inputs/children.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": ["label_status"],
                    "derivations": {
                        "label_status": {
                            "kind": "lookup",
                            "source": "parents",
                            "left_key": "parent_id",
                            "right_key": "id",
                            "value_column": "label_status",
                            "missing_policy": "null",
                            "default": "missing",
                        }
                    },
                },
                "views": {
                    "child_embedding": {
                        "source": "children",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="child_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["label_status"] for row in rows] == ["observed", "missing"]


def test_sig35_variant_uses_upstream_sigma70_core_annotation_when_plan_lacks_sig35() -> None:
    row = {
        "usr_label__primary": "pDual-10-ES3p",
        "densegen__plan": "ethanol_ciprofloxacin",
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
    }

    assert sig35_variant(row) == "ACCGCG"


def test_sig35_variant_uses_annotation_before_control_fallback_when_plan_is_missing() -> None:
    row = {
        "usr_label__primary": "pDual-10-archive-row",
        "densegen__plan": None,
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "variant_id": "GCAGGT",
                "core_sequence": "GCAGGT",
            }
        ],
    }

    assert sig35_variant(row) == "GCAGGT"


def test_sig35_variant_uses_sequence_annotation_feature_when_densegen_detail_is_missing() -> None:
    row = {
        "usr_label__primary": "J23104",
        "densegen__plan": None,
        "seq_annot__features": [
            {
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "qualifiers": [
                    {"key": "label", "value": "-35"},
                    {"key": "note", "value": "feature_sequence=TTGACA"},
                ],
            },
            {
                "label": "-10",
                "role_hint": "sigma70_minus10",
                "qualifiers": [{"key": "note", "value": "feature_sequence=TATTGT"}],
            },
        ],
    }

    assert sig35_variant(row) == "TTGACA"


def test_sig35_variant_does_not_slice_projected_parent_annotation_bounds_from_context_sequence() -> None:
    row = {
        "usr_label__primary": "J23104_context1kb_forward",
        "densegen__plan": None,
        "source_family": "genbank_projected_reference",
        "sequence": "A" * 1000,
        "seq_annot__sequence_region_start_0": 0,
        "seq_annot__sequence_region_end_0": 60,
        "seq_annot__features": [
            {
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "start_0": 10,
                "end_0": 16,
                "qualifiers": [],
            }
        ],
    }

    assert sig35_variant(row) == "control"


def test_sig35_variant_uses_derived_retained_sigma35_bounds_for_analysis_window() -> None:
    row = {
        "usr_label__primary": "micFp_core60",
        "densegen__plan": None,
        "sequence": "ttcttaagtatttgacagcactgaatgtcaaaacaaaaccttcactcgcaactagaataa",
        "derived__target_length": 60,
        "derived__features_retained": [
            {
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "derived_intervals_0": [{"start_0": 26, "end_0": 32, "strand": 1, "partial": False}],
            }
        ],
    }

    assert sig35_variant(row) == "GTCAAA"


def test_sig35_variant_does_not_slice_projected_analysis_window_retention_from_context_sequence() -> None:
    row = {
        "usr_label__primary": "micFp_core60_context1kb_forward",
        "densegen__plan": None,
        "source_family": "genbank_projected_reference",
        "sequence": "A" * 1000,
        "derived__target_length": 60,
        "derived__features_retained": [
            {
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "derived_intervals_0": [{"start_0": 26, "end_0": 32, "strand": 1, "partial": False}],
            }
        ],
    }

    assert sig35_variant(row) == "control"


def test_source_class_prefers_sequence_view_semantics_and_promoter_standard_metadata() -> None:
    assert source_class({"source_family": "genbank_projected_reference", "densegen__plan": None}) == (
        "reference_control"
    )
    assert source_class({"source_family": "densegen_generated", "densegen__plan": "ethanol__sig35=b"}) == ("densegen")
    assert source_class({"source_family": "construct_derived", "densegen__plan": "ethanol__sig35=b"}) == ("densegen")
    assert source_class({"regulondb__primary_promoter_name": "lexAp", "densegen__plan": None}) == ("native_regulondb")
    assert source_class({"derived__parent_dataset": "usr_regulondb_native_promoters", "densegen__plan": None}) == (
        "native_regulondb"
    )
    assert source_class({"promoter_standard__collection_id": "anderson", "densegen__plan": None}) == (
        "synthetic_reference_standard"
    )


def test_densegen_plan_takes_precedence_over_reference_like_source_family() -> None:
    row = {
        "usr_label__primary": "synthetic_cpxr",
        "source_family": "genbank_projected_reference",
        "densegen__plan": "ethanol__sig35=b",
        "densegen__required_regulators": ["cpxR"],
    }

    assert is_control_row(row) is False
    assert design_family(row) == "ethanol"
    assert design_regulator_composition(row) == "cpxR"
    assert source_class(row) == "densegen"


def test_promoter_metadata_missing_plan_without_control_provenance_fails_fast() -> None:
    row = {"usr_label__primary": "unclassified_row", "densegen__plan": None}

    with pytest.raises(ContractViolationError, match="design_family could not be derived"):
        design_family(row)

    with pytest.raises(ContractViolationError, match="source_class could not be derived"):
        source_class(row)


def test_materialize_view_canonicalizes_design_regulator_composition(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["synthetic_a", "synthetic_b", "background_a", "control_a"], type=pa.string()),
                "subject_id": pa.array(["synthetic_a", "synthetic_b", "background_a", "control_a"], type=pa.string()),
                "usr_label__primary": pa.array(["synthetic_a", "synthetic_b", "background_a", "J23105"]),
                "densegen__plan": pa.array(
                    [
                        "ethanol__sig35=b",
                        "ethanol_ciprofloxacin__sig35=b",
                        "background_only__sig35=f",
                        None,
                    ],
                    type=pa.string(),
                ),
                "densegen__required_regulators": pa.array(
                    [
                        ["cpxR_MANWWHTTTAM"],
                        ["lexA_CTGTATAWAWWHACA", "baeR_TTTCTSCVHNA"],
                        [],
                        None,
                    ],
                    type=pa.list_(pa.string()),
                ),
                "embedding": pa.array(
                    [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]],
                    type=pa.list_(pa.float32()),
                ),
                "template_id": pa.array(["tpl_a", "tpl_b", "tpl_c", "wt"], type=pa.string()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "regulator_composition_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["design_regulator_composition"],
                    "derivations": {
                        "design_regulator_composition": _annotation_derivation(
                            "design_regulator_composition",
                            required_columns=[
                                "densegen__plan",
                                "densegen__required_regulators",
                                "usr_label__primary",
                            ],
                        )
                    },
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["design_regulator_composition"] for row in rows] == [
        "cpxR",
        "baeR+lexA",
        "background",
        "control",
    ]


def test_materialize_view_includes_source_scoped_metadata_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "usr_label__primary": pa.array(["row_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b"], type=pa.string()),
                "densegen__required_regulators": pa.array(["cpxR"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
                "anchor_logp": pa.array([-1.5], type=pa.float64()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "source_metadata_include_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "metadata_include": ["anchor_logp"],
                    }
                },
                "metadata": {
                    "include": ["sig35_variant"],
                    "derivations": {"sig35_variant": _sig35_derivation()},
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "row_a", "subject_id": "row_a", "anchor_logp": -1.5, "sig35_variant": "b"}]


def test_materialize_view_can_replace_workspace_metadata_per_source(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
                "anchor_logp": pa.array([-1.5], type=pa.float64()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "source_metadata_replace_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "reference_view": {
                        "kind": "parquet",
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "metadata_include": ["anchor_logp"],
                        "metadata_include_mode": "replace",
                    }
                },
                "metadata": {
                    "include": ["sig35_variant"],
                    "derivations": {"sig35_variant": _sig35_derivation()},
                },
                "views": {
                    "intermediate_embedding_reference": {
                        "source": "reference_view",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "reference"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_reference")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "row_a", "subject_id": "row_a", "anchor_logp": -1.5}]


def test_materialize_view_includes_explicit_annotation_metadata(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "usr_label__primary": pa.array(["row_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b"], type=pa.string()),
                "densegen__used_tfbs_detail": pa.array(
                    ['[{"part_kind":"fixed_element","spacer_length":17}]'], type=pa.string()
                ),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "promoter_cohort_auto_include_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["spacer_length"],
                    "derivations": {
                        "spacer_length": _annotation_derivation(
                            "spacer_length",
                            required_columns=["densegen__plan", "densegen__used_tfbs_detail", "usr_label__primary"],
                            value_type="int64",
                        )
                    },
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "row_a", "subject_id": "row_a", "spacer_length": 17}]


def test_promoter_metadata_spacer_length_array_uses_stable_int_type_when_batch_is_all_null(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace: {id: stable_promoter_metadata_type_demo, output_root: ./outputs}
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  anchor_60bp:
    kind: parquet
    path: inputs/records.parquet
    record_key: id
    subject_key: id
metadata:
  derivations:
    spacer_length:
      kind: annotation
      source: row
      handler: dnadesign.latentdna.src.views.promoter_metadata:derive_promoter_metadata_value
      derive: spacer_length
      required_columns: [densegen__plan, densegen__used_tfbs_detail, usr_label__primary]
      missing_policy: error
      value_type: int64
views: {}
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    context = load_workspace_config(workspace_dir)

    array = _derived_metadata_array(context, [{"densegen__used_tfbs_detail": None}], column_name="spacer_length")

    assert array.type == pa.int64()
    assert array.to_pylist() == [None]


def test_construct_template_id_derivation_uses_stable_string_type_for_all_null_batch(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace: {id: stable_construct_template_type_demo, output_root: ./outputs}
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources: {}
metadata:
  derivations:
    construct_template_id:
      kind: coalesce
      sources: [construct_template_id, template_id, construct__template_id]
      value_type: string
views: {}
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    context = load_workspace_config(workspace_dir)

    array = _derived_metadata_array(context, [{"construct__template_id": None}], column_name="construct_template_id")

    assert array.type == pa.string()
    assert array.to_pylist() == [None]


def test_materialize_view_does_not_auto_include_unrequested_annotation_derivations(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["anchor_a"], type=pa.string()),
                "subject_id": pa.array(["anchor_a"], type=pa.string()),
                "usr_label__primary": pa.array(["anchor_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b"], type=pa.string()),
                "densegen__used_tfbs_detail": pa.array(
                    ['[{"part_kind":"sigma35","variant_id":"b","spacer_length":17}]']
                ),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "anchor.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["control_a"], type=pa.string()),
                "subject_id": pa.array(["control_a"], type=pa.string()),
                "usr_label__primary": pa.array(["control_a"], type=pa.string()),
                "embedding": pa.array([[1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "control.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "source_scoped_cohort_demo", "output_root": "./outputs"},
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
                    },
                    "controls": {
                        "kind": "parquet",
                        "path": "inputs/control.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": [],
                    "derivations": {
                        "spacer_length": _annotation_derivation(
                            "spacer_length",
                            required_columns=["densegen__plan", "densegen__used_tfbs_detail", "usr_label__primary"],
                            value_type="int64",
                        )
                    },
                },
                "views": {
                    "control_embedding": {
                        "source": "controls",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="control_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "control_a", "subject_id": "control_a"}]


def test_spacer_length_accepts_numpy_object_array_entries() -> None:
    row = {
        "usr_label__primary": "synthetic_a",
        "densegen__plan": "ethanol__sig35=f",
        "densegen__used_tfbs_detail": np.array(
            [
                {
                    "part_kind": "tfbs",
                    "spacer_length": None,
                },
                {
                    "part_kind": "fixed_element",
                    "role": "upstream",
                    "spacer_length": 17,
                },
                {
                    "part_kind": "fixed_element",
                    "role": "downstream",
                    "spacer_length": 17,
                },
            ],
            dtype=object,
        ),
    }

    assert spacer_length(row) == 17


def test_spacer_length_returns_none_when_synthetic_detail_is_missing() -> None:
    row = {
        "usr_label__primary": "synthetic_a",
        "densegen__plan": "ethanol__sig35=f",
        "densegen__used_tfbs_detail": None,
    }

    assert spacer_length(row) is None


def test_materialize_view_populates_coalesced_metadata_from_source_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a", "row_b"], type=pa.string()),
                "subject_id": pa.array(["row_a", "row_b"], type=pa.string()),
                "usr_label__primary": pa.array(["row_a", "row_b"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
                "score_anchor": pa.array([-1.2, None], type=pa.float32()),
                "score_context": pa.array([None, -6.3], type=pa.float32()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "coalesce_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["llr_demo"],
                    "derivations": {
                        "llr_demo": {
                            "kind": "coalesce",
                            "sources": ["score_anchor", "score_context"],
                        }
                    },
                },
                "views": {
                    "demo_embedding": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="demo_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["llr_demo"] for row in rows] == pytest.approx([-1.2, -6.3])


def test_materialize_view_allows_coalesce_derivation_when_one_input_column_is_available(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
                "score_context": pa.array([-6.3], type=pa.float32()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "missing_derivation_input_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["llr_demo"],
                    "derivations": {
                        "llr_demo": {
                            "kind": "coalesce",
                            "sources": ["score_anchor", "score_context"],
                        }
                    },
                },
                "views": {
                    "demo_embedding": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="demo_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["llr_demo"] for row in rows] == pytest.approx([-6.3])


def test_materialize_view_fails_fast_when_all_coalesce_inputs_are_missing(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "missing_all_derivation_inputs_demo", "output_root": "./outputs"},
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
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["llr_demo"],
                    "derivations": {
                        "llr_demo": {
                            "kind": "coalesce",
                            "sources": ["score_anchor", "score_context"],
                        }
                    },
                },
                "views": {
                    "demo_embedding": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    with pytest.raises(ContractViolationError, match="metadata derivation inputs are missing"):
        materialize_view_artifact(context, view_id="demo_embedding")
