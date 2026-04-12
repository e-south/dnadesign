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
    assert "../api.py" in workflow_index or "api.py" in workflow_index
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
    assert "### First tracer-bullet path" in workflow
    assert "--from-study-dir docs/studies/stress_ethanol_cipro_growth" in workflow
    assert "uv run latentdna validate workspace" in workflow
    assert "--deep" in workflow
    assert "uv run latentdna view materialize z20_60" in workflow
    assert "uv run latentdna sample build atlas_sample" in workflow
    assert "densegen__plan" in workflow
    assert "densegen__base_plan" not in workflow
    assert "uv run latentdna projection fit z20_60" in workflow
    assert "uv run latentdna plot render atlas_2x2_main" in workflow
    assert "uv run latentdna plot render primary_landmark_scatter" in workflow
    assert "--kind distance_scatter" in workflow
    assert "uv run latentdna plot render agreement_20b_anchor_vs_context_summary" in workflow
    assert "--kind agreement_summary" in workflow
    assert "uv run latentdna enrich score control_plan_enrichment" in workflow
    assert "--kind heatmap" in workflow
    assert "uv run latentdna recipe validate control_plan_heatmap_recipe" in workflow
    assert "uv run latentdna deliverable run control_neighborhood_enrichment" in workflow
    assert "uv run latentdna notebook generate control_plan_review" in workflow
    assert "uv run marimo run" in workflow
    assert "outputs/latentdna/notebooks/control_plan_review/notebook.py" in workflow
    assert "outputs/latentdna/plots" in workflow
    assert "--backend exact" not in workflow

    assert "x2_primary_20b" in export_workflow
    assert "x3_ablation_7b" in export_workflow
    assert "feature names are stable" in export_workflow.lower()

    assert "### Common flags" in reference
    assert "### Primitive command groups" in reference
    assert "`latentdna workspace init`" in reference
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
    assert "`latentdna recipe validate`" in reference
    assert "`latentdna deliverable status`" in reference
    assert "`latentdna.command_result.v1`" in reference
    assert "`latentdna.deliverable_status.v1`" in reference
    assert "`--quiet`" in reference
    assert "`--dry-run`" in reference
    assert "distance_scatter" in reference
    assert "distribution" in reference
    assert "agreement_summary" in reference
    assert "workspace-wide plot browser" in reference

    assert "outputs/latentdna/plots" in workspace_schema
    assert "notebooks.<id>.artifacts" in workspace_schema

    assert "bench_view_materialize" in performance
    assert "bench_export_x2" in performance
    assert "fixture-scale benchmarks" in performance.lower()

    assert "development journal" in dev.lower()
    assert "journal.md" in dev
    assert "Phase 1" in journal
    assert "Recipe and Deliverable Slice" in journal
    assert "Workspace Plot Browser Follow-On Slice" in journal
    assert "Next steps" in journal
