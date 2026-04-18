"""
Promoter snapshot LatentDNA seam tests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from dnadesign.studies.families.promoter.analysis_surfaces import inspect_promoter_exploratory_analysis
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
                },
                "supported_model_families": ["evo2_20b", "evo2_7b"],
                "default_model_family": "evo2_20b",
                "required_wildtype_references": ["spyp", "sulap", "j23105"],
                "decision_deliverables": [
                    "dataset_overview",
                    "reference_margin_analysis",
                    "context_geometry_audit",
                    "representation_comparison",
                    "representation_health_diagnostic",
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
                    "reference_margin_analysis": {
                        "title": "Reference margin analysis",
                        "status": "ok",
                        "freshness": "ok",
                        "acceptance_checks": [],
                        "artifact_paths": ["plots/reference_margin_gallery_wildtype"],
                        "docs_refs": [],
                        "warnings": [],
                    },
                    "context_geometry_audit": {
                        "title": "Context geometry audit",
                        "status": "attention",
                        "freshness": "attention",
                        "acceptance_checks": [],
                        "artifact_paths": [],
                        "docs_refs": [],
                        "warnings": ["pending refreshed artifact run"],
                    },
                    "representation_comparison": {
                        "title": "Representation comparison",
                        "status": "missing",
                        "freshness": "missing",
                        "acceptance_checks": [],
                        "artifact_paths": [],
                        "docs_refs": [],
                        "warnings": [],
                    },
                    "representation_health_diagnostic": {
                        "title": "Representation health diagnostic",
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
                    "reference_margin_analysis",
                    "context_geometry_audit",
                    "representation_comparison",
                    "representation_health_diagnostic",
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
    }
    assert readiness["ok_deliverables"] == [
        "dataset_overview",
        "reference_margin_analysis",
    ]
    assert "representation_comparison" in readiness["pending_deliverables"]
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
