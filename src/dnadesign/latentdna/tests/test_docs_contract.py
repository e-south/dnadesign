"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_docs_contract.py

Documentation routing contracts for latentdna.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_latentdna_readme_routes_to_docs_and_root_index() -> None:
    readme = (_repo_root() / "src/dnadesign/latentdna/README.md").read_text(encoding="utf-8")

    assert "## Documentation" in readme
    assert "docs/README.md" in readme
    assert "workspaces/README.md" in readme
    assert "docs/reference/cli-contracts.md" in readme
    assert "docs/reference/workspace-schema.md" in readme
    assert "docs/reference/source-contract.md" in readme
    assert "docs/reference/view-contract.md" in readme
    assert "docs/reference/deliverable-contract.md" in readme
    assert "docs/reference/performance-budgets.md" in readme
    assert "docs/workflows/promoter-study-latent-atlas.md" in readme
    assert "docs/workflows/context-shift.md" in readme
    assert "docs/workflows/cross-view-agreement.md" in readme
    assert "docs/workflows/export-opal-x.md" in readme
    assert "../../../docs/README.md" in readme
    assert "artifact-first downstream latent analysis surface" in readme


def test_latentdna_docs_tree_exposes_workflow_reference_and_dev_surfaces() -> None:
    repo_root = _repo_root()
    workflow_index = (repo_root / "src/dnadesign/latentdna/docs/README.md").read_text(encoding="utf-8")
    by_type = (repo_root / "src/dnadesign/latentdna/docs/index.md").read_text(encoding="utf-8")
    workflow = (repo_root / "src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md").read_text(
        encoding="utf-8"
    )
    reference = (repo_root / "src/dnadesign/latentdna/docs/reference/cli-contracts.md").read_text(encoding="utf-8")
    workspace_schema = (repo_root / "src/dnadesign/latentdna/docs/reference/workspace-schema.md").read_text(
        encoding="utf-8"
    )
    reference_index = (repo_root / "src/dnadesign/latentdna/docs/reference/README.md").read_text(encoding="utf-8")
    workflow_router = (repo_root / "src/dnadesign/latentdna/docs/workflows/README.md").read_text(encoding="utf-8")
    export_workflow = (repo_root / "src/dnadesign/latentdna/docs/workflows/export-opal-x.md").read_text(
        encoding="utf-8"
    )
    performance = (repo_root / "src/dnadesign/latentdna/docs/reference/performance-budgets.md").read_text(
        encoding="utf-8"
    )
    dev = (repo_root / "src/dnadesign/latentdna/docs/dev/README.md").read_text(encoding="utf-8")
    journal = (repo_root / "src/dnadesign/latentdna/docs/dev/journal.md").read_text(encoding="utf-8")

    assert "### Start here" in workflow_index
    assert "### Route map" in workflow_index
    assert "### Adjacent handoffs" in workflow_index
    assert "workflows/promoter-study-latent-atlas.md" in workflow_index
    assert "workflows/context-shift.md" in workflow_index
    assert "workflows/cross-view-agreement.md" in workflow_index
    assert "workflows/export-opal-x.md" in workflow_index
    assert "reference/cli-contracts.md" in workflow_index
    assert "reference/workspace-schema.md" in workflow_index
    assert "reference/performance-budgets.md" in workflow_index
    assert "src/workspaces/__init__.py" in workflow_index
    assert "Execution-helper surface" in workflow_index
    assert "src/api.py" in workflow_index
    assert "src/workspaces/api.py" not in workflow_index
    assert "../api.py" not in workflow_index
    assert "../../infer/docs/README.md" in workflow_index
    assert "../../opal/README.md" in workflow_index
    assert "../../usr/README.md" in workflow_index

    assert "unified [latentdna docs](README.md) index" in by_type
    assert "Open the latentdna docs index" in by_type
    assert "promoter-study latent atlas workflow" in by_type
    assert "cross-view agreement" in by_type
    assert "performance budgets" in by_type

    assert "Source contract" in reference_index
    assert "Alignment contract" in reference_index
    assert "View contract" in reference_index
    assert "Scalar contract" in reference_index
    assert "Deliverable contract" in reference_index
    assert "Performance budgets" in reference_index

    assert "Landmark neighborhoods" in workflow_router
    assert "Control distances" in workflow_router
    assert "Context shift" in workflow_router
    assert "Cross-view agreement" in workflow_router
    assert "Export to OPAL X bundles" in workflow_router

    assert "**Type:** workflow" in workflow
    assert "**Plane:** downstream-tool" in workflow
    assert "**Owner-boundary:** latentdna" in workflow
    assert "**Registry-id:** latentdna.promoter-study.latent-atlas" in workflow
    assert "### View taxonomy for the active study" in workflow
    assert "z20_1k_seq" in workflow
    assert "logits7_60" in workflow
    assert "logits7_1k_anchor" in workflow
    assert "logits20_60" in workflow
    assert "logits20_1k_anchor" in workflow
    assert "pooled logits" in workflow
    assert "2 x 3" in workflow
    assert "### First tracer-bullet path" in workflow
    assert "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth" in workflow
    assert "uv run latentdna workspace refresh" in workflow
    assert "uv run latentdna validate workspace" in workflow
    assert "--deep" in workflow
    assert "uv run latentdna view materialize z20_60" in workflow
    assert "uv run latentdna sample build atlas_anchor_sample" in workflow
    assert "--group-column design_family" in workflow
    assert "--reference-set promoter_wt_core" in workflow
    assert "densegen__plan" not in workflow
    assert "uv run latentdna projection fit z20_60" in workflow
    assert "uv run latentdna deliverable status atlas_2x2_intermediate_main" in workflow
    assert "uv run latentdna deliverable run atlas_2x2_intermediate_main" in workflow
    assert "uv run latentdna deliverable run context_shift_primary" in workflow
    assert "uv run latentdna deliverable run agreement_7b_vs_20b" in workflow
    assert "### Context-audit and browser control-plane slice" in workflow
    assert "uv run latentdna deliverable status geometry_switchboard_20b" in workflow
    assert "uv run latentdna deliverable run geometry_switchboard_20b" in workflow
    assert "uv run latentdna deliverable status context_audit_primary_20b" in workflow
    assert "uv run latentdna deliverable run context_audit_primary_20b" in workflow
    assert "uv run latentdna view reduce z20_60" in workflow
    assert "uv run latentdna view reduce z20_1k_anchor" in workflow
    assert "--reduced-view-id z20_60_anchor_ctx_pc32" in workflow
    assert "--reduced-view-id z20_1k_anchor_anchor_ctx_pc32" in workflow
    assert "uv run latentdna neighbors fit leiden_z20_60_knn" in workflow
    assert "uv run latentdna neighbors fit leiden_z20_1k_anchor_knn" in workflow
    assert "uv run latentdna cluster fit leiden_z20_60" in workflow
    assert "uv run latentdna cluster fit leiden_z20_1k_anchor" in workflow
    assert "--method leiden" in workflow
    assert "kmeans" not in workflow
    assert "uv run latentdna deliverable run cluster_correspondence_primary" in workflow
    assert "uv run latentdna deliverable run control_pca_explained_variance_curve" in workflow
    assert "uv run latentdna export matrix x2_primary_20b" in workflow
    assert "uv run latentdna export matrix x3_ablation_7b" in workflow
    assert "uv run latentdna notebook generate browser" in workflow
    assert "controls.json" in workflow
    assert "uv run latentdna notebook smoke" in workflow
    assert "uv run marimo run" in workflow
    assert "outputs/notebooks/browser/notebook.py" in workflow
    assert "outputs/plots" in workflow
    assert "outputs/latentdna" in workflow
    assert "--backend exact" not in workflow

    assert "x2_primary_20b" in export_workflow
    assert "x3_ablation_7b" in export_workflow
    assert "feature names are stable" in export_workflow.lower()

    assert "### Common flags" in reference
    assert "### Primitive command groups" in reference
    assert "`latentdna workspace init`" in reference
    assert "`latentdna workspace refresh`" in reference
    assert "`latentdna validate workspace`" in reference
    assert "`latentdna workspace init --from-study-dir <path>`" in reference
    assert "`latentdna validate workspace --deep`" in reference
    assert "`latentdna inspect source`" in reference
    assert "`latentdna view materialize`" in reference
    assert "`latentdna view stats`" in reference
    assert "`latentdna sample build`" in reference
    assert "`latentdna projection fit`" in reference
    assert "`latentdna enrich score`" in reference
    assert "`latentdna plot render`" in reference
    assert "`latentdna notebook generate`" in reference
    assert "`latentdna notebook smoke`" in reference
    assert "`latentdna recipe validate`" in reference
    assert "`latentdna deliverable status`" in reference
    assert "`latentdna inspect notebook-health`" in reference
    assert "`latentdna.command_result.v1`" in reference
    assert "`latentdna.deliverable_status.v1`" in reference
    assert "`--quiet`" in reference
    assert "`--dry-run`" in reference
    assert "distance_scatter" in reference
    assert "xy_scatter" in reference
    assert "distribution" in reference
    assert "curve" in reference
    assert "correspondence_heatmap" in reference
    assert "agreement_summary" in reference
    assert "workspace-wide plot browser" in reference
    assert "controls.json" in reference
    assert "status=attention" in reference

    assert "outputs/plots" in workspace_schema
    assert "notebooks.<id>.default_deliverable" in workspace_schema
    assert "outputs/notebooks/<id>/controls.json" in workspace_schema
    assert "reference_set" in workspace_schema
    assert "xy_scatter" in workspace_schema
    assert "curve" in workspace_schema
    assert "correspondence_heatmap" in workspace_schema

    assert "bench_view_materialize" in performance
    assert "bench_export_x2" in performance
    assert "fixture-scale benchmarks" in performance.lower()

    assert "development journal" in dev.lower()
    assert "journal.md" in dev
    assert "Phase 1" in journal
    assert "Recipe and Deliverable Slice" in journal
    assert "Workspace Plot Browser Follow-On Slice" in journal
    assert "Next steps" in journal
