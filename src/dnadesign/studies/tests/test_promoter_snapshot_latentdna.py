"""
Promoter snapshot latentdna readiness tests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from dnadesign.studies.core.models import StudyOpsContract, StudyPreflightContract, StudyStatusContext
from dnadesign.studies.families.promoter.adapter import STUDY_FAMILY_ADAPTER, PromoterFamilyContext
from dnadesign.studies.families.promoter.infer_runtime import (
    PromoterStudyInferRuntimeDependencies,
    PromoterStudyInferRuntimeResolvedContext,
)
from dnadesign.studies.families.promoter.latentdna_readiness import inspect_promoter_latentdna_readiness
from dnadesign.studies.families.promoter.snapshot import (
    PromoterStudyStatusDependencies,
    PromoterStudyStatusResolvedContext,
    build_promoter_study_status,
)
from dnadesign.studies.tests.test_promoter_snapshot import (
    _make_study_context,
    _string_list_or_empty,
    _string_or_none,
)


def _write_latentdna_workspace_fixture(tmp_path: Path) -> Path:
    workspace_dir = tmp_path / "src" / "dnadesign" / "latentdna" / "workspaces" / "stress_ethanol_cipro_growth"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "stress_ethanol_cipro_growth", "output_root": "./outputs/latentdna"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {},
                "deliverables": {
                    "atlas_2x2_intermediate_main": {
                        "kind": "projection_grid",
                        "description": "Primary atlas",
                        "question": "Do the design families separate in latent space at all?",
                        "section": "Atlas",
                        "recipe": "noop",
                        "outputs": {"plots": ["atlas_2x2_intermediate_main"], "notebooks": ["browser"]},
                    }
                },
                "recipes": {
                    "noop": {
                        "steps": [
                            {
                                "id": "render_atlas",
                                "op": "plot.render",
                                "params": {"plot_id": "atlas_2x2_intermediate_main"},
                            },
                            {
                                "id": "generate_browser",
                                "op": "notebook.generate",
                                "depends_on": ["render_atlas"],
                                "params": {"notebook_id": "browser"},
                            },
                        ]
                    }
                },
                "plots": {
                    "atlas_2x2_intermediate_main": {
                        "kind": "projection_grid",
                        "projections": ["umap_z20_60"],
                        "color_column": "design_family",
                    }
                },
                "notebooks": {
                    "browser": {
                        "kind": "workspace_browser",
                        "title": "Browser",
                        "description": "Read-only browser",
                        "default_deliverable": "atlas_2x2_intermediate_main",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    outputs_root = workspace_dir / "outputs" / "latentdna"
    (outputs_root / "plots" / "atlas_2x2_intermediate_main").mkdir(parents=True)
    (outputs_root / "plots" / "atlas_2x2_intermediate_main" / "plot.svg").write_text("<svg />", encoding="utf-8")
    (outputs_root / "plots" / "atlas_2x2_intermediate_main" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "plot",
                "artifact_id": "atlas_2x2_intermediate_main",
                "workspace_id": "stress_ethanol_cipro_growth",
                "command": "plot render",
                "status": "ok",
                "created_at": "2026-04-13T00:00:00Z",
                "outputs": [{"path": "plot.svg", "media_type": "image/svg+xml"}],
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "plots" / "index.json").write_text(
        json.dumps(
            {
                "workspace_id": "stress_ethanol_cipro_growth",
                "plots": [
                    {
                        "plot_id": "atlas_2x2_intermediate_main",
                        "deliverable_id": "atlas_2x2_intermediate_main",
                        "status": "ok",
                        "rendered_formats": ["svg"],
                        "output_paths": ["plots/atlas_2x2_intermediate_main/plot.svg"],
                        "input_artifact_ids": [],
                        "created_at": "2026-04-13T00:00:00Z",
                        "stale": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "notebooks" / "browser").mkdir(parents=True)
    (outputs_root / "notebooks" / "browser" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "notebook",
                "artifact_id": "browser",
                "workspace_id": "stress_ethanol_cipro_growth",
                "command": "notebook generate",
                "status": "ok",
                "created_at": "2026-04-13T00:00:00Z",
                "inputs": [
                    {
                        "kind": "plot",
                        "id": "atlas_2x2_intermediate_main",
                        "digest": "demo-digest",
                    }
                ],
                "outputs": [{"path": "../browser.py", "media_type": "text/x-python"}],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "notebooks").mkdir(parents=True, exist_ok=True)
    (outputs_root / "notebooks" / "browser.py").write_text("import marimo\n", encoding="utf-8")
    (outputs_root / "notebooks" / "health.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "checks": {
                    "notebook_exists": True,
                    "imports_resolve": True,
                    "plot_catalog_loads": True,
                    "default_deliverable_ready": True,
                    "static_links_resolve": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "clusters" / "leiden_z20_60").mkdir(parents=True)
    (outputs_root / "clusters" / "leiden_z20_60" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "cluster_set",
                "artifact_id": "leiden_z20_60",
                "workspace_id": "stress_ethanol_cipro_growth",
                "command": "cluster fit",
                "status": "ok",
                "created_at": "2026-04-13T00:00:00Z",
                "params": {"method": "leiden"},
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "clusters" / "leiden_z20_1k_anchor").mkdir(parents=True)
    (outputs_root / "clusters" / "leiden_z20_1k_anchor" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "cluster_set",
                "artifact_id": "leiden_z20_1k_anchor",
                "workspace_id": "stress_ethanol_cipro_growth",
                "command": "cluster fit",
                "status": "ok",
                "created_at": "2026-04-13T00:00:00Z",
                "params": {"method": "leiden"},
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "clusters" / "leiden_logits20_60").mkdir(parents=True)
    (outputs_root / "clusters" / "leiden_logits20_60" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "cluster_set",
                "artifact_id": "leiden_logits20_60",
                "workspace_id": "stress_ethanol_cipro_growth",
                "command": "cluster fit",
                "status": "ok",
                "created_at": "2026-04-13T00:00:00Z",
                "params": {"method": "leiden"},
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "exports" / "x2_primary_20b").mkdir(parents=True)
    (outputs_root / "exports" / "x2_primary_20b" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "export_bundle",
                "artifact_id": "x2_primary_20b",
                "workspace_id": "stress_ethanol_cipro_growth",
                "command": "export matrix",
                "status": "ok",
                "created_at": "2026-04-13T00:00:00Z",
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )
    return workspace_dir


def test_inspect_promoter_latentdna_readiness_reports_workspace_evidence(tmp_path: Path) -> None:
    workspace_dir = _write_latentdna_workspace_fixture(tmp_path)
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_pipeline={
            **dict(base_context.study_pipeline),
            "latentdna": {
                "workspace": workspace_dir.as_posix(),
                "expected_deliverables": [
                    "atlas_2x2_intermediate_main",
                    "cluster_correspondence_primary",
                ],
                "required_leiden_runs": [
                    "leiden_z20_60",
                    "leiden_z20_1k_anchor",
                    "leiden_logits20_60",
                ],
                "required_exports": ["x2_primary_20b"],
                "notebook": "browser",
            },
        },
    )

    readiness = inspect_promoter_latentdna_readiness(study_context=study_context)

    assert readiness == {
        "latentdna_workspace_id": "stress_ethanol_cipro_growth",
        "latentdna_state": "attention",
        "latentdna_expected_deliverables": [
            "atlas_2x2_intermediate_main",
            "cluster_correspondence_primary",
        ],
        "latentdna_ok_deliverables": [],
        "latentdna_missing_deliverables": [
            "atlas_2x2_intermediate_main",
            "cluster_correspondence_primary",
        ],
        "latentdna_rendered_plot_count": 1,
        "latentdna_notebook_generated": True,
        "latentdna_notebook_smoke_ok": True,
        "latentdna_leiden_runs_ok": True,
        "latentdna_exports_ok": True,
    }


def test_adapter_build_snapshot_includes_real_latentdna_readiness(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _write_latentdna_workspace_fixture(tmp_path)
    study_context = replace(
        _make_study_context(tmp_path),
        study_pipeline={
            "infer": {
                "preferred_model_family": "evo2_20b",
                "supported_model_families": ["evo2_20b", "evo2_7b"],
                "configs": {
                    "anchor_only_20b": tmp_path / "config.anchor_only.evo2_20b.yaml",
                    "anchor_only_7b": tmp_path / "config.anchor_only.evo2_7b.yaml",
                },
            },
            "latentdna": {
                "workspace": workspace_dir.as_posix(),
                "expected_deliverables": [
                    "atlas_2x2_intermediate_main",
                    "cluster_correspondence_primary",
                ],
                "required_leiden_runs": [
                    "leiden_z20_60",
                    "leiden_z20_1k_anchor",
                    "leiden_logits20_60",
                ],
                "required_exports": ["x2_primary_20b"],
                "notebook": "browser",
            },
        },
    )
    contract = StudyOpsContract(
        study_id="demo_study",
        family="promoter",
        phase_order=("infer_batch_preparation",),
        snapshot_summary_scope="repo",
        preflight=StudyPreflightContract(default_scope="next"),
        current_phase_id="infer_batch_preparation",
        phases=(),
        raw_payload={"study_id": "demo_study", "family": "promoter"},
    )
    adapter_context = StudyStatusContext(
        repo_root=tmp_path,
        study_root=study_context.resolved_study_dir,
        contract=contract,
        family_context=PromoterFamilyContext(study_context=study_context),
    )

    monkeypatch.setattr(
        "dnadesign.studies.families.promoter.adapter.build_promoter_study_infer_runtime_dependencies",
        lambda: PromoterStudyInferRuntimeDependencies(
            resolve_named_path_mapping=lambda value, *, repo_root, label, status_kind: {
                name: Path(path) for name, path in dict(value or {}).items()
            },
            resolve_infer_runtime_lane_contracts=lambda config_paths, *, preferred_model_family: (
                SimpleNamespace(
                    phase_id="infer_anchor_only_20b",
                    config_label="anchor_only_20b",
                    config_path=config_paths["anchor_only_20b"],
                    runtime_label="anchor_only_20b",
                ),
                SimpleNamespace(
                    phase_id="infer_anchor_only_7b",
                    config_label="anchor_only_7b",
                    config_path=config_paths["anchor_only_7b"],
                    runtime_label="anchor_only_7b",
                ),
            ),
            derive_infer_notify_profile_paths=lambda config_paths: (
                {label: path.parent / "notify" / f"{label}.json" for label, path in config_paths.items()},
                {},
            ),
            load_infer_model_summary=lambda config_path: {
                "model_id": f"model-{Path(config_path).stem}",
                "device": "cuda:0",
            },
            string_or_none=_string_or_none,
            string_list_or_empty=_string_list_or_empty,
        ),
    )

    state, _, evidence = STUDY_FAMILY_ADAPTER.build_snapshot(adapter_context)

    assert state == "attention"
    assert evidence["latentdna"]["latentdna_state"] == "attention"
    assert evidence["latentdna"]["latentdna_workspace_id"] == "stress_ethanol_cipro_growth"
    assert evidence["latentdna"]["latentdna_rendered_plot_count"] == 1
    assert evidence["latentdna"]["latentdna_leiden_runs_ok"] is True
    assert evidence["latentdna"]["latentdna_exports_ok"] is True


def test_build_promoter_study_status_exposes_latentdna_readiness_without_gating_snapshot(tmp_path: Path) -> None:
    workspace_dir = _write_latentdna_workspace_fixture(tmp_path)

    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_pipeline={
            **dict(base_context.study_pipeline),
            "latentdna": {"workspace": workspace_dir.as_posix()},
        },
    )

    state, _, evidence = build_promoter_study_status(
        study_context=study_context,
        status_context=PromoterStudyStatusResolvedContext(
            infer_runtime=PromoterStudyInferRuntimeResolvedContext(
                preferred_model_family="evo2_20b",
                supported_model_families=("evo2_20b", "evo2_7b"),
                infer_config_paths={},
                runtime_lane_contracts=(),
                runtime_config_paths={},
                phase_targets=(),
                phase_targets_by_id={},
                config_phase_ids={},
                runtime_phase_ids={},
                infer_notify_profile_paths={},
                infer_notify_profile_errors={},
                runtime_model_summaries=(
                    SimpleNamespace(
                        label="anchor_only_20b",
                        model_id="evo2_20b",
                        device="cuda:0",
                        as_dict=lambda: {
                            "label": "anchor_only_20b",
                            "model_id": "evo2_20b",
                            "device": "cuda:0",
                        },
                    ),
                ),
                gpu_required_runtime_labels=("anchor_only_20b",),
            ),
        ),
        dependencies=PromoterStudyStatusDependencies(
            infer_runtime=PromoterStudyInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
            inspect_latentdna_readiness=lambda **kwargs: {
                "latentdna_workspace_id": "stress_ethanol_cipro_growth",
                "latentdna_state": "ok",
                "latentdna_expected_deliverables": ["atlas_2x2_intermediate_main"],
                "latentdna_ok_deliverables": ["atlas_2x2_intermediate_main"],
                "latentdna_missing_deliverables": [],
                "latentdna_rendered_plot_count": 1,
                "latentdna_notebook_generated": True,
                "latentdna_notebook_smoke_ok": True,
                "latentdna_leiden_runs_ok": True,
                "latentdna_exports_ok": True,
            },
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert evidence["latentdna"] == {
        "latentdna_workspace_id": "stress_ethanol_cipro_growth",
        "latentdna_state": "ok",
        "latentdna_expected_deliverables": ["atlas_2x2_intermediate_main"],
        "latentdna_ok_deliverables": ["atlas_2x2_intermediate_main"],
        "latentdna_missing_deliverables": [],
        "latentdna_rendered_plot_count": 1,
        "latentdna_notebook_generated": True,
        "latentdna_notebook_smoke_ok": True,
        "latentdna_leiden_runs_ok": True,
        "latentdna_exports_ok": True,
    }
    assert evidence["attention_reasons"] == ["DenseGen source gate is still active"]
