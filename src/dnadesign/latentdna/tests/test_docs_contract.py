"""Documentation routing contracts for LatentDNA."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


_TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".svg", ".html"}


def _is_scan_text_file(path: Path) -> bool:
    return "outputs" not in path.parts and "tests" not in path.parts and path.suffix.lower() in _TEXT_SUFFIXES


def test_latentdna_readme_routes_to_reference_first_docs() -> None:
    repo_root = _repo_root()
    readme = (repo_root / "src/dnadesign/latentdna/README.md").read_text(encoding="utf-8")
    docs_index = (repo_root / "src/dnadesign/latentdna/docs/README.md").read_text(encoding="utf-8")
    workflow = (
        repo_root / "src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md"
    ).read_text(encoding="utf-8")
    cli_contracts = (repo_root / "src/dnadesign/latentdna/docs/reference/cli-contracts.md").read_text(encoding="utf-8")
    workspace_schema = (repo_root / "src/dnadesign/latentdna/docs/reference/workspace-schema.md").read_text(
        encoding="utf-8"
    )
    reference_index = (repo_root / "src/dnadesign/latentdna/docs/reference/README.md").read_text(encoding="utf-8")
    operations = (repo_root / "src/dnadesign/latentdna/docs/operations/README.md").read_text(encoding="utf-8")
    workspace_readme = (
        repo_root / "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/README.md"
    ).read_text(encoding="utf-8")
    study_routes = (repo_root / "docs/studies/stress_ethanol_cipro_growth/routes.md").read_text(encoding="utf-8")
    study_status = (repo_root / "docs/studies/stress_ethanol_cipro_growth/status.md").read_text(encoding="utf-8")

    assert "comparison layer for `dnadesign`" in readme
    assert "workspace snapshot contract" in readme.lower()
    assert "promoter-study representation comparison workflow" in readme.lower()
    assert "docs/workflows/promoter-study-representation-comparison.md" in readme

    assert "workflows/promoter-study-representation-comparison.md" in docs_index
    assert "reference/workspace-snapshot-contract.md" in docs_index
    assert "reference/artifact-naming.md" in docs_index
    assert "operations/README.md" in docs_index

    assert "**Type:** workflow" in workflow
    assert "**Plane:** data-plane" in workflow
    assert "**Surface role:** downstream-analysis" in workflow
    assert "**Owner-boundary:** latentdna" in workflow
    assert "**Entry artifact:** promoter/stress_ethanol_cipro_anchor_set" in workflow
    assert (
        "**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables" in workflow
    )
    assert "reference_margin_analysis" in workflow
    assert "representation_comparison" in workflow
    assert "appendix_umap_gallery" in workflow
    assert "workspace snapshot" in workflow

    assert "`latentdna workspace snapshot`" in cli_contracts
    assert "`latentdna.workspace_snapshot.v1`" in cli_contracts
    assert "--progress none|text|json" in cli_contracts
    assert "Nested output roots are rejected." in cli_contracts

    assert "<workspace>/outputs" in workspace_schema
    assert "promoter-study reference-margin template" in workspace_schema

    assert "Workspace snapshot contract" in reference_index
    assert "Artifact naming grammar" in reference_index
    assert "ops/status.registry.yaml" in operations

    assert "latentdna_binding.yaml" in workspace_readme
    assert "workspace_snapshot.json" in workspace_readme
    assert "UMAP role: appendix context only" in workspace_readme

    assert "Primary review path:" in study_routes
    assert "representation_health_diagnostic" in study_routes
    assert "appendix_umap_gallery" in study_routes
    assert (
        "Snapshot attention surfaces: `dataset_overview`, `reference_margin_analysis`, `representation_comparison`"
        in study_routes
    )
    assert "Plane: `data-plane`" in study_routes
    assert "Plane: `control-plane`" in study_routes
    assert "Surface role: `producer`" in study_routes
    assert "Surface role: `operator`" in study_routes
    assert "Surface role: `downstream-analysis`" in study_routes
    assert "Surface role: `decision`" in study_routes
    assert "Plane: `producer-analysis`" not in study_routes
    assert "Plane: `execution-surface`" not in study_routes
    assert "Plane: `downstream-analysis`" not in study_routes
    assert "Plane: `downstream-tool`" not in study_routes

    assert "LatentDNA is a downstream comparison surface" in study_status
    assert (
        "Current attention surfaces: `dataset_overview`, `reference_margin_analysis`, `representation_comparison`"
        in study_status
    )
    assert "Appendix surfaces remain secondary" in study_status


def test_latentdna_docs_remove_legacy_promoter_surface_names() -> None:
    repo_root = _repo_root()
    scan_roots = [
        repo_root / "src/dnadesign/latentdna",
        repo_root / "docs/studies/stress_ethanol_cipro_growth",
        repo_root / "src/dnadesign/studies/stress_ethanol_cipro_growth",
    ]
    forbidden_tokens = [
        "".join(["atlas", "_2x2_intermediate_main"]),
        "".join(["atlas", "_2x3_model_family"]),
        "".join(["context_shift", "_vs_drag_primary"]),
        "".join(["geometry", "_switchboard_20b"]),
        "".join(["x2", "_primary_20b"]),
        "/".join(["outputs", "latentdna"]),
        "benchmark_results_summary",
        "benchmark_feature_matrix",
        "representation_selection",
        "selection_state_code",
    ]

    checked_files: list[Path] = []
    for root in scan_roots:
        checked_files.extend(path for path in root.rglob("*") if path.is_file() and _is_scan_text_file(path))

    for forbidden in forbidden_tokens:
        hits = [
            path.as_posix() for path in checked_files if forbidden in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert hits == [], f"forbidden legacy token {forbidden!r} still present in: {hits}"
