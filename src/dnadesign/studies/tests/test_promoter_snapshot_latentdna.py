"""
Promoter snapshot latentdna readiness tests.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from dnadesign.studies.core.models import StudyOpsContract, StudyPreflightContract, StudyStatusContext
from dnadesign.studies.families.promoter.adapter import STUDY_FAMILY_ADAPTER, PromoterFamilyContext
from dnadesign.studies.families.promoter.analysis_surfaces import inspect_promoter_exploratory_analysis
from dnadesign.studies.families.promoter.downstream_surfaces import inspect_promoter_downstream_surfaces
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


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _write_latentdna_workspace_fixture(tmp_path: Path) -> Path:
    workspace_dir = tmp_path / "src" / "dnadesign" / "latentdna" / "workspaces" / "stress_ethanol_cipro_growth"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "stress_ethanol_cipro_growth", "output_root": "./outputs"},
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
                        "title": "Atlas 2x2 intermediate main",
                        "summary": "Primary atlas",
                        "question": "Do the design families separate in latent space at all?",
                        "section": "Atlas",
                        "recipe": "noop",
                        "requires": {"recipes": ["noop"]},
                        "outputs": {"plots": ["atlas_2x2_intermediate_main"], "notebooks": ["browser"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
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
                        "kind": "workspace",
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
    outputs_root = workspace_dir / "outputs"
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
                "outputs": [{"path": "notebook.py", "media_type": "text/x-python"}],
            }
        ),
        encoding="utf-8",
    )
    (outputs_root / "notebooks").mkdir(parents=True, exist_ok=True)
    (outputs_root / "notebooks" / "browser" / "notebook.py").write_text("import marimo\n", encoding="utf-8")
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


def _write_densegen_workspace_fixture(tmp_path: Path) -> Path:
    workspace_dir = tmp_path / "src" / "dnadesign" / "densegen" / "workspaces" / "study_stress_ethanol_cipro"
    (workspace_dir / "outputs" / "plots").mkdir(parents=True)
    (workspace_dir / "outputs" / "notebooks").mkdir(parents=True)
    (workspace_dir / "README.md").write_text("DenseGen workspace\n", encoding="utf-8")
    (workspace_dir / "config.yaml").write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    (workspace_dir / "outputs" / "plots" / "plot_manifest.json").write_text(
        json.dumps(
            {
                "plots": [
                    {
                        "plot_id": "dataset_source_inventory",
                        "variant": "source_inventory",
                        "path": "dataset/dataset_source_inventory.pdf",
                        "plan_name": "unscoped",
                    },
                    {
                        "plot_id": "run_health",
                        "variant": "run_health",
                        "path": "run_health/run_health.pdf",
                        "plan_name": "unscoped",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (workspace_dir / "outputs" / "notebooks" / "densegen_run_overview.py").write_text(
        "import marimo\n",
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
        "configured": True,
        "state": "attention",
        "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md",
        "surface_ref": str(workspace_dir),
        "workspace_id": "stress_ethanol_cipro_growth",
        "expected_deliverables": [
            "atlas_2x2_intermediate_main",
            "cluster_correspondence_primary",
        ],
        "ok_deliverables": [],
        "missing_deliverables": [
            "atlas_2x2_intermediate_main",
            "cluster_correspondence_primary",
        ],
        "rendered_plot_count": 1,
        "notebook_generated": True,
        "notebook_smoke_ok": True,
        "leiden_runs_ok": True,
        "exports_ok": True,
        "required_leiden_runs": [
            "leiden_z20_60",
            "leiden_z20_1k_anchor",
            "leiden_logits20_60",
        ],
        "required_exports": ["x2_primary_20b"],
    }


def test_inspect_promoter_latentdna_readiness_resolves_repo_relative_workspace_path(tmp_path: Path) -> None:
    workspace_dir = _write_latentdna_workspace_fixture(tmp_path)
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_pipeline={
            **dict(base_context.study_pipeline),
            "latentdna": {
                "workspace": workspace_dir.relative_to(tmp_path).as_posix(),
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

    assert readiness is not None
    assert readiness["workspace_id"] == "stress_ethanol_cipro_growth"
    assert readiness["state"] == "attention"
    assert readiness["notebook_smoke_ok"] is True
    assert readiness["leiden_runs_ok"] is True
    assert readiness["exports_ok"] is True


def test_inspect_promoter_downstream_surfaces_reports_cluster_and_opal_explicitly(tmp_path: Path) -> None:
    study_context = replace(
        _make_study_context(tmp_path),
        study_pipeline={
            "cluster": {
                "doc": "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md",
                "entry_artifact": "promoter/demo_feature_matrix",
                "results_root": "n/a",
                "state": "planned",
            },
            "opal": {
                "doc": "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md",
                "entry_artifact": "promoter/demo_feature_matrix",
                "config": "n/a",
                "state": "not_configured",
            },
        },
    )

    downstream = inspect_promoter_downstream_surfaces(study_context=study_context)

    assert downstream == {
        "cluster": {
            "configured": False,
            "state": "planned",
            "doc": "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md",
            "surface_ref": None,
            "entry_artifact": "promoter/demo_feature_matrix",
            "results_root": "n/a",
        },
        "opal": {
            "configured": False,
            "state": "not_configured",
            "doc": "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md",
            "surface_ref": None,
            "entry_artifact": "promoter/demo_feature_matrix",
            "config": "n/a",
        },
    }


def test_inspect_promoter_exploratory_analysis_reports_densegen_latentdna_and_cluster_routes(tmp_path: Path) -> None:
    densegen_workspace = _write_densegen_workspace_fixture(tmp_path)
    latentdna_workspace = _write_latentdna_workspace_fixture(tmp_path)
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_pipeline={
            **dict(base_context.study_pipeline),
            "densegen": {
                "workspace": densegen_workspace.as_posix(),
                "default_plot_ids": [
                    "dataset_source_inventory",
                    "dataset_metadata_heatmap",
                    "stage_a_summary",
                    "placement_map",
                    "run_health",
                    "tfbs_usage",
                ],
                "optional_plot_ids": ["dense_array_video_showcase"],
            },
            "latentdna": {
                "workspace": latentdna_workspace.as_posix(),
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
            "cluster": {
                "doc": "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md",
                "entry_artifact": "promoter/demo_feature_matrix",
                "results_root": "n/a",
                "state": "planned",
                "workspace_example": "src/dnadesign/cluster/workspaces/promoter_clusters_v1/config.yaml",
            },
        },
    )
    latentdna_state = inspect_promoter_latentdna_readiness(study_context=study_context)
    downstream = inspect_promoter_downstream_surfaces(study_context=study_context)

    analysis_surfaces = inspect_promoter_exploratory_analysis(
        study_context=study_context,
        latentdna_state=latentdna_state,
        downstream_surfaces=downstream,
    )

    assert analysis_surfaces["densegen"]["state"] == "ok"
    assert analysis_surfaces["densegen"]["rendered_plot_count"] == 2
    assert analysis_surfaces["densegen"]["notebook_generated"] is True
    assert analysis_surfaces["densegen"]["default_plot_ids"] == [
        "dataset_source_inventory",
        "dataset_metadata_heatmap",
        "stage_a_summary",
        "placement_map",
        "run_health",
        "tfbs_usage",
    ]
    assert analysis_surfaces["densegen"]["optional_plot_ids"] == ["dense_array_video_showcase"]
    assert analysis_surfaces["densegen"]["rendered_plots"] == [
        {
            "plot_id": "dataset_source_inventory",
            "variant": "source_inventory",
            "path": "dataset/dataset_source_inventory.pdf",
            "plan_name": "unscoped",
        },
        {
            "plot_id": "run_health",
            "variant": "run_health",
            "path": "run_health/run_health.pdf",
            "plan_name": "unscoped",
        },
    ]
    assert analysis_surfaces["latentdna"]["state"] == "attention"
    assert analysis_surfaces["latentdna"]["notebook_id"] == "browser"
    assert "atlas_2x2_intermediate_main" in analysis_surfaces["latentdna"]["plot_ids"]
    assert "atlas_2x2_intermediate_main" in analysis_surfaces["latentdna"]["deliverable_ids"]
    assert "x2_primary_20b" in analysis_surfaces["latentdna"]["export_ids"]
    assert analysis_surfaces["cluster"]["state"] == "planned"
    assert analysis_surfaces["cluster"]["commands"]["catalog_show"] == (
        "uv run ops catalog show cluster.downstream.exploratory-clustering"
    )


def test_promoter_study_docs_track_live_latentdna_snapshot() -> None:
    repo_root = _repo_root()
    status_path = repo_root / "docs/studies/stress_ethanol_cipro_growth/status.md"
    routes_path = repo_root / "docs/studies/stress_ethanol_cipro_growth/routes.md"
    status_doc = status_path.read_text(encoding="utf-8")
    routes_doc = routes_path.read_text(encoding="utf-8")
    result = subprocess.run(
        ["uv", "run", "ops", "progress", "show", "usr.data-plane.promoter-study-status", "--json"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    latentdna = payload["evidence"]["analysis_surfaces"]["latentdna"]

    assert "current readiness is `attention`" in status_doc
    assert "drag_qc" in routes_doc
    assert latentdna["state"] == "attention"
    assert "error" not in latentdna
    assert "drag_qc" in latentdna["deliverable_ids"]


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
            "densegen": {
                "workspace": "src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro",
                "default_plot_ids": [
                    "dataset_source_inventory",
                    "dataset_metadata_heatmap",
                    "stage_a_summary",
                    "placement_map",
                    "run_health",
                    "tfbs_usage",
                ],
                "optional_plot_ids": ["dense_array_video_showcase"],
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
    assert evidence["latentdna"]["state"] == "attention"
    assert evidence["latentdna"]["workspace_id"] == "stress_ethanol_cipro_growth"
    assert evidence["latentdna"]["rendered_plot_count"] == 1
    assert evidence["latentdna"]["leiden_runs_ok"] is True
    assert evidence["latentdna"]["exports_ok"] is True
    assert evidence["cluster"]["state"] == "planned"
    assert evidence["opal"]["state"] == "not_configured"
    assert sorted(evidence["analysis_surfaces"]) == ["cluster", "densegen", "latentdna"]
    assert evidence["analysis_surfaces"]["densegen"]["default_plot_ids"] == [
        "dataset_source_inventory",
        "dataset_metadata_heatmap",
        "stage_a_summary",
        "placement_map",
        "run_health",
        "tfbs_usage",
    ]
    assert evidence["analysis_surfaces"]["latentdna"]["notebook_id"] == "browser"
    assert evidence["analysis_surfaces"]["cluster"]["state"] == "planned"


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
                "configured": True,
                "state": "ok",
                "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md",
                "surface_ref": str(workspace_dir),
                "workspace_id": "stress_ethanol_cipro_growth",
                "expected_deliverables": ["atlas_2x2_intermediate_main"],
                "ok_deliverables": ["atlas_2x2_intermediate_main"],
                "missing_deliverables": [],
                "rendered_plot_count": 1,
                "notebook_generated": True,
                "notebook_smoke_ok": True,
                "leiden_runs_ok": True,
                "exports_ok": True,
                "required_leiden_runs": [],
                "required_exports": [],
            },
            inspect_additional_downstream_surfaces=lambda **kwargs: {
                "cluster": {
                    "configured": False,
                    "state": "planned",
                    "doc": "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md",
                    "surface_ref": None,
                    "results_root": None,
                },
                "opal": {
                    "configured": False,
                    "state": "not_configured",
                    "doc": "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md",
                    "surface_ref": None,
                    "config": None,
                },
            },
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert evidence["latentdna"] == {
        "configured": True,
        "state": "ok",
        "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md",
        "surface_ref": str(workspace_dir),
        "workspace_id": "stress_ethanol_cipro_growth",
        "expected_deliverables": ["atlas_2x2_intermediate_main"],
        "ok_deliverables": ["atlas_2x2_intermediate_main"],
        "missing_deliverables": [],
        "rendered_plot_count": 1,
        "notebook_generated": True,
        "notebook_smoke_ok": True,
        "leiden_runs_ok": True,
        "exports_ok": True,
        "required_leiden_runs": [],
        "required_exports": [],
    }
    assert evidence["cluster"]["state"] == "planned"
    assert evidence["opal"]["state"] == "not_configured"
    assert evidence["attention_reasons"] == ["DenseGen source gate is still active"]


def test_build_promoter_study_status_keeps_planned_downstream_surfaces_out_of_top_level_state(tmp_path: Path) -> None:
    study_context = replace(_make_study_context(tmp_path), densegen_row_target=8, densegen_row_gap=0)

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
                runtime_model_summaries=(),
                gpu_required_runtime_labels=(),
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
                "configured": True,
                "state": "missing",
                "doc": "src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md",
                "surface_ref": str(tmp_path / "src" / "dnadesign" / "latentdna" / "workspaces" / "demo"),
                "workspace_id": "demo",
                "expected_deliverables": [],
                "ok_deliverables": [],
                "missing_deliverables": [],
                "rendered_plot_count": 0,
                "notebook_generated": False,
                "notebook_smoke_ok": False,
                "leiden_runs_ok": True,
                "exports_ok": True,
                "required_leiden_runs": [],
                "required_exports": [],
            },
            inspect_additional_downstream_surfaces=lambda **kwargs: {
                "cluster": {
                    "configured": False,
                    "state": "planned",
                    "doc": "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md",
                    "surface_ref": None,
                    "results_root": None,
                },
                "opal": {
                    "configured": False,
                    "state": "not_configured",
                    "doc": "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md",
                    "surface_ref": None,
                    "config": None,
                },
            },
        ),
        summary_scope="repo",
    )

    assert state == "ok"
    assert evidence["latentdna"]["state"] == "missing"
    assert evidence["cluster"]["state"] == "planned"
    assert evidence["opal"]["state"] == "not_configured"
    assert "attention_reasons" not in evidence
