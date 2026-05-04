"""
Promoter snapshot LatentDNA seam tests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.families.promoter.analysis_surfaces import inspect_promoter_exploratory_analysis
from dnadesign.studies.families.promoter.latentdna_contract import _validate_binding, _validate_workspace_snapshot
from dnadesign.studies.families.promoter.latentdna_readiness import inspect_promoter_latentdna_readiness
from dnadesign.studies.tests.test_promoter_snapshot import _make_study_context


def _write_latentdna_binding_fixture(tmp_path: Path) -> Path:
    binding_path = tmp_path / "docs" / "studies" / "demo_study" / "latentdna_binding.yaml"
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(
        yaml.safe_dump(
            {
                "workspace_id": "stress_ethanol_cipro_growth",
                "workspace_ref": "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth",
                "snapshot_ref": (
                    "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/status/"
                    "workspace_snapshot.json"
                ),
                "source_datasets": {
                    "anchor_60bp": "promoter/demo_anchor_set",
                    "full_context_1kb": "promoter/demo_construct_contexts",
                    "reference_core60": "promoter/demo_reference_core60",
                    "reference_contexts": "promoter/demo_reference_contexts",
                },
                "supported_model_families": ["evo2_20b", "evo2_7b"],
                "default_model_family": "evo2_20b",
                "required_wildtype_references": ["spyp", "sulap", "j23105"],
                "decision_deliverables": [
                    "dataset_overview",
                    "representation_health_summary",
                    "design_structure_summary",
                    "sigma35_ordinal_audit",
                    "context_robustness_summary",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return binding_path


def _write_latentdna_snapshot_fixture(tmp_path: Path) -> Path:
    snapshot_path = (
        tmp_path
        / "src"
        / "dnadesign"
        / "latentdna"
        / "workspaces"
        / "stress_ethanol_cipro_growth"
        / "outputs"
        / "status"
        / "workspace_snapshot.json"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "latentdna.workspace_snapshot.v1",
                "workspace_id": "stress_ethanol_cipro_growth",
                "output_root": "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs",
                "sources": {
                    "anchor_60bp": {
                        "kind": "usr",
                        "path": "src/dnadesign/usr/datasets/promoter/demo_anchor_set",
                        "dataset_id": "promoter/demo_anchor_set",
                        "row_count": 8,
                        "columns": ["id", "design_family", "sig35_variant"],
                        "vector_columns": ["infer__evo2_20b__..."],
                    },
                    "full_context_1kb": {
                        "kind": "usr",
                        "path": "src/dnadesign/usr/datasets/promoter/demo_construct_contexts",
                        "dataset_id": "promoter/demo_construct_contexts",
                        "row_count": 8,
                        "columns": ["id", "construct_template_id", "source_class"],
                        "vector_columns": ["infer__evo2_20b__..."],
                    },
                },
                "model_families": ["evo2_20b", "evo2_7b"],
                "canonical_views": [
                    "intermediate_embedding_20b_anchor_60bp",
                    "intermediate_embedding_20b_full_context_1kb",
                    "intermediate_embedding_7b_anchor_60bp",
                    "intermediate_embedding_7b_full_context_1kb",
                    "pooled_logits_20b_anchor_60bp",
                    "pooled_logits_20b_full_context_1kb",
                    "pooled_logits_7b_anchor_60bp",
                    "pooled_logits_7b_full_context_1kb",
                ],
                "deliverables": {
                    "dataset_overview": {
                        "title": "Dataset overview",
                        "status": "ok",
                        "freshness": "ok",
                        "acceptance_checks": [],
                        "artifact_paths": ["plots/dataset_overview"],
                        "docs_refs": [],
                        "warnings": [],
                    },
                    "representation_health_summary": {
                        "title": "Representation health summary",
                        "status": "ok",
                        "freshness": "ok",
                        "acceptance_checks": [],
                        "artifact_paths": ["plots/representation_health_summary"],
                        "docs_refs": [],
                        "warnings": [],
                    },
                    "design_structure_summary": {
                        "title": "Design-structure summary",
                        "status": "attention",
                        "freshness": "attention",
                        "acceptance_checks": [],
                        "artifact_paths": [],
                        "docs_refs": [],
                        "warnings": ["pending refreshed artifact run"],
                    },
                    "sigma35_ordinal_audit": {
                        "title": "Sigma-35 ordinal audit",
                        "status": "missing",
                        "freshness": "missing",
                        "acceptance_checks": [],
                        "artifact_paths": [],
                        "docs_refs": [],
                        "warnings": [],
                    },
                    "context_robustness_summary": {
                        "title": "Context robustness summary",
                        "status": "missing",
                        "freshness": "missing",
                        "acceptance_checks": [],
                        "artifact_paths": [],
                        "docs_refs": [],
                        "warnings": [],
                    },
                    "appendix_umap_gallery": {
                        "title": "Appendix UMAP gallery",
                        "status": "ok",
                        "freshness": "ok",
                        "acceptance_checks": [],
                        "artifact_paths": ["plots/appendix_umap_gallery"],
                        "docs_refs": [],
                        "warnings": [],
                    },
                },
                "exports": {},
                "browser": {
                    "default_geometry_ids": [
                        "intermediate_embedding_20b_anchor_60bp",
                        "intermediate_embedding_20b_full_context_1kb",
                        "intermediate_embedding_7b_anchor_60bp",
                        "intermediate_embedding_7b_full_context_1kb",
                        "pooled_logits_20b_anchor_60bp",
                        "pooled_logits_20b_full_context_1kb",
                        "pooled_logits_7b_anchor_60bp",
                        "pooled_logits_7b_full_context_1kb",
                    ],
                    "preferred_hues": [
                        "design_family",
                        "design_regulator_composition",
                        "sig35_variant",
                        "source_class",
                        "is_control",
                        "wildtype_margin_ethanol_vs_control",
                    ],
                },
                "decision_ladder": [
                    "dataset_overview",
                    "representation_health_summary",
                    "design_structure_summary",
                    "sigma35_ordinal_audit",
                    "context_robustness_summary",
                ],
                "last_updated_at": "2026-04-15T12:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return snapshot_path


def test_inspect_promoter_latentdna_readiness_uses_binding_and_snapshot_contract(tmp_path: Path) -> None:
    binding_path = _write_latentdna_binding_fixture(tmp_path)
    snapshot_path = _write_latentdna_snapshot_fixture(tmp_path)
    study_context = _make_study_context(tmp_path)
    study_context = replace(
        study_context,
        study_pipeline={
            **study_context.study_pipeline,
            "latentdna": {
                "binding": binding_path.relative_to(tmp_path).as_posix(),
                "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md",
            },
        },
    )

    readiness = inspect_promoter_latentdna_readiness(study_context=study_context)

    assert readiness["configured"] is True
    assert readiness["state"] == "attention"
    assert readiness["binding_ref"] == str(binding_path)
    assert readiness["workspace_ref"] == str(snapshot_path.parents[2])
    assert readiness["snapshot_ref"] == str(snapshot_path)
    assert readiness["workspace_id"] == "stress_ethanol_cipro_growth"
    assert readiness["source_datasets"] == {
        "anchor_60bp": "promoter/demo_anchor_set",
        "full_context_1kb": "promoter/demo_construct_contexts",
        "reference_core60": "promoter/demo_reference_core60",
        "reference_contexts": "promoter/demo_reference_contexts",
    }
    assert readiness["missing_source_datasets"] == ["reference_contexts", "reference_core60"]
    assert readiness["ok_deliverables"] == [
        "dataset_overview",
        "representation_health_summary",
    ]
    assert "design_structure_summary" in readiness["pending_deliverables"]
    assert readiness["exports_ok"] is True
    assert readiness["browser_default_geometry_ids"] == [
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    ]


def test_promoter_latentdna_binding_accepts_generic_source_dataset_keys() -> None:
    binding = _validate_binding(
        {
            "workspace_id": "regulondb_native_promoter_panel",
            "workspace_ref": "src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel",
            "snapshot_ref": (
                "src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/outputs/status/"
                "workspace_snapshot.json"
            ),
            "source_datasets": {
                "native_source_records": "usr_regulondb_native_promoters",
                "native_source_record_features_7b": "usr_regulondb_native_promoters/_derived/infer",
                "core60_tss_upstream": "usr_regulondb_native_promoter_core60",
                "core60_tss_upstream_7b_core60_mean_features": ("usr_regulondb_native_promoter_core60/_derived/infer"),
                "core60_tss_upstream_7b_core60_mean_output_layer_features": (
                    "usr_regulondb_native_promoter_core60/_derived/infer"
                ),
                "core60_tss_upstream_7b_core60_mean_log_likelihood_mean": (
                    "usr_regulondb_native_promoter_core60/_derived/infer"
                ),
                "core60_tss_upstream_7b_core60_mean_log_likelihood_total": (
                    "usr_regulondb_native_promoter_core60/_derived/infer"
                ),
            },
            "supported_model_families": ["evo2_7b"],
            "default_model_family": "evo2_7b",
            "required_wildtype_references": [],
            "decision_deliverables": ["dataset_overview"],
        }
    )

    snapshot = _validate_workspace_snapshot(
        binding=binding,
        snapshot={
            "schema_version": "latentdna.workspace_snapshot.v1",
            "workspace_id": "regulondb_native_promoter_panel",
            "output_root": "src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/outputs",
            "sources": {
                "native_source_records": {
                    "kind": "usr",
                    "path": "src/dnadesign/usr/datasets/usr_regulondb_native_promoters/records.parquet",
                    "row_count": 3182,
                },
                "native_source_record_features_7b": {
                    "kind": "infer_feature_sidecar",
                    "path": "src/dnadesign/usr/datasets/usr_regulondb_native_promoters/_derived/infer",
                    "row_count": 0,
                },
                "core60_tss_upstream": {
                    "kind": "usr",
                    "path": "src/dnadesign/usr/datasets/usr_regulondb_native_promoter_core60/records.parquet",
                    "row_count": 3182,
                },
            },
            "model_families": ["evo2_7b"],
            "canonical_views": ["intermediate_embedding_7b_native_source_record_seq_mean"],
            "deliverables": {
                "dataset_overview": {
                    "title": "Dataset overview",
                    "status": "planned",
                    "freshness": "planned",
                    "acceptance_checks": [],
                    "artifact_paths": [],
                    "docs_refs": [],
                    "warnings": ["pending infer batch"],
                }
            },
            "exports": {},
            "browser": {
                "default_geometry_ids": [],
                "preferred_hues": ["regulondb__sigma_factor_set"],
            },
            "decision_ladder": ["dataset_overview"],
            "last_updated_at": "2026-04-29T00:00:00+00:00",
        },
    )

    assert binding["source_datasets"] == {
        "core60_tss_upstream": "usr_regulondb_native_promoter_core60",
        "core60_tss_upstream_7b_core60_mean_features": "usr_regulondb_native_promoter_core60/_derived/infer",
        "core60_tss_upstream_7b_core60_mean_log_likelihood_mean": (
            "usr_regulondb_native_promoter_core60/_derived/infer"
        ),
        "core60_tss_upstream_7b_core60_mean_log_likelihood_total": (
            "usr_regulondb_native_promoter_core60/_derived/infer"
        ),
        "core60_tss_upstream_7b_core60_mean_output_layer_features": (
            "usr_regulondb_native_promoter_core60/_derived/infer"
        ),
        "native_source_record_features_7b": "usr_regulondb_native_promoters/_derived/infer",
        "native_source_records": "usr_regulondb_native_promoters",
    }
    assert binding["required_wildtype_references"] == []
    assert sorted(snapshot["sources"]) == [
        "core60_tss_upstream",
        "native_source_record_features_7b",
        "native_source_records",
    ]
    assert snapshot["missing_binding_sources"] == [
        "core60_tss_upstream_7b_core60_mean_features",
        "core60_tss_upstream_7b_core60_mean_log_likelihood_mean",
        "core60_tss_upstream_7b_core60_mean_log_likelihood_total",
        "core60_tss_upstream_7b_core60_mean_output_layer_features",
    ]
    assert snapshot["missing_decision_deliverables"] == []


def test_promoter_latentdna_snapshot_validation_requires_study_binding(tmp_path: Path) -> None:
    snapshot_path = _write_latentdna_snapshot_fixture(tmp_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="validated study binding"):
        _validate_workspace_snapshot(binding=None, snapshot=snapshot)


def test_inspect_promoter_latentdna_readiness_rejects_snapshot_schema_mismatch(tmp_path: Path) -> None:
    binding_path = _write_latentdna_binding_fixture(tmp_path)
    snapshot_path = _write_latentdna_snapshot_fixture(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "latentdna.workspace_snapshot.v0"
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    study_context = _make_study_context(tmp_path)
    study_context = replace(
        study_context,
        study_pipeline={
            **study_context.study_pipeline,
            "latentdna": {
                "binding": binding_path.relative_to(tmp_path).as_posix(),
                "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md",
            },
        },
    )

    readiness = inspect_promoter_latentdna_readiness(study_context=study_context)

    assert readiness["configured"] is True
    assert readiness["state"] == "error"
    assert "schema_version" in str(readiness["error"])
    assert readiness["snapshot_ref"] is None


def test_inspect_promoter_latentdna_readiness_rejects_structurally_invalid_snapshot(tmp_path: Path) -> None:
    binding_path = _write_latentdna_binding_fixture(tmp_path)
    snapshot_path = _write_latentdna_snapshot_fixture(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    del payload["browser"]["preferred_hues"]
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    study_context = _make_study_context(tmp_path)
    study_context = replace(
        study_context,
        study_pipeline={
            **study_context.study_pipeline,
            "latentdna": {
                "binding": binding_path.relative_to(tmp_path).as_posix(),
                "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md",
            },
        },
    )

    readiness = inspect_promoter_latentdna_readiness(study_context=study_context)

    assert readiness["configured"] is True
    assert readiness["state"] == "error"
    assert "preferred_hues" in str(readiness["error"])
    assert readiness["snapshot_ref"] is None


def test_inspect_promoter_latentdna_readiness_rejects_malformed_binding(tmp_path: Path) -> None:
    binding_path = _write_latentdna_binding_fixture(tmp_path)
    _write_latentdna_snapshot_fixture(tmp_path)
    payload = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    del payload["workspace_id"]
    binding_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    study_context = _make_study_context(tmp_path)
    study_context = replace(
        study_context,
        study_pipeline={
            **study_context.study_pipeline,
            "latentdna": {
                "binding": binding_path.relative_to(tmp_path).as_posix(),
                "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md",
            },
        },
    )

    readiness = inspect_promoter_latentdna_readiness(study_context=study_context)

    assert readiness["configured"] is True
    assert readiness["state"] == "error"
    assert readiness["binding_ref"] == str(binding_path)
    assert readiness["workspace_id"] is None
    assert "missing required top-level fields" in str(readiness["error"])
    assert readiness["snapshot_ref"] is None


def test_inspect_promoter_exploratory_analysis_reports_snapshot_backed_latentdna_surface(tmp_path: Path) -> None:
    binding_path = _write_latentdna_binding_fixture(tmp_path)
    _write_latentdna_snapshot_fixture(tmp_path)
    study_context = _make_study_context(tmp_path)
    study_context = replace(
        study_context,
        study_pipeline={
            **study_context.study_pipeline,
            "latentdna": {
                "binding": binding_path.relative_to(tmp_path).as_posix(),
            },
        },
    )
    latentdna_state = inspect_promoter_latentdna_readiness(study_context=study_context)

    surfaces = inspect_promoter_exploratory_analysis(
        study_context=study_context,
        latentdna_state=latentdna_state,
        downstream_surfaces={"cluster": {"state": "planned", "doc": "cluster-doc", "entry_artifact": "demo"}},
    )

    latentdna_surface = surfaces["latentdna"]
    assert latentdna_surface["state"] == "attention"
    assert latentdna_surface["workspace_id"] == "stress_ethanol_cipro_growth"
    assert latentdna_surface["deliverable_ids"][0] == "dataset_overview"
    assert latentdna_surface["browser_default_geometry_ids"] == [
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    ]
    assert latentdna_surface["commands"]["snapshot"] == (
        "uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json"
    )


def test_promoter_latentdna_status_modules_do_not_import_latentdna_internals() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for relative_path in [
        "src/dnadesign/studies/families/promoter/latentdna_readiness.py",
        "src/dnadesign/studies/families/promoter/analysis_surfaces.py",
    ]:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert ".".join(["dnadesign", "latentdna", "src"]) not in source
