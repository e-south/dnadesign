"""Documentation routing contracts for LatentDNA."""

from __future__ import annotations

from pathlib import Path

import yaml


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
        repo_root / "src/dnadesign/latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md"
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
    latentdna_binding = yaml.safe_load(
        (repo_root / "docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml").read_text(encoding="utf-8")
    )
    study_pipeline = yaml.safe_load(
        (repo_root / "docs/studies/stress_ethanol_cipro_growth/pipeline.yaml").read_text(encoding="utf-8")
    )
    study_routes = (repo_root / "docs/studies/stress_ethanol_cipro_growth/routes.md").read_text(encoding="utf-8")
    study_status = (repo_root / "docs/studies/stress_ethanol_cipro_growth/status.md").read_text(encoding="utf-8")

    assert "LatentDNA compares learned sequence representations" in readme
    assert "comparison layer for `dnadesign`" not in readme
    assert "workspace snapshot contract" in readme.lower()
    assert "stress ethanol/cipro representation comparison workflow" in readme.lower()
    assert "docs/workflows/stress-ethanol-cipro-representation-comparison.md" in readme

    assert "workflows/stress-ethanol-cipro-representation-comparison.md" in docs_index
    assert "reference/workspace-snapshot-contract.md" in docs_index
    assert "reference/artifact-naming.md" in docs_index
    assert "operations/README.md" in docs_index

    assert "**Type:** workflow" in workflow
    assert "**Plane:** data-plane" in workflow
    assert "**Surface role:** downstream-analysis" in workflow
    assert "**Owner-boundary:** latentdna" in workflow
    assert "**Entry artifact:** usr_prom_eth_cip_anchor" in workflow
    assert (
        "**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables" in workflow
    )
    assert "representation_health_summary" in workflow
    assert "design_structure_summary" in workflow
    assert "sigma35_ordinal_audit" in workflow
    assert "context_robustness_summary" in workflow
    assert "candidate_decision_frontier" in workflow
    assert "balanced_design_family_margin_gallery" in workflow
    assert "sigma35_margin_ladder_gallery" in workflow
    assert "sigma35_centroid_distance_gallery" in workflow
    assert "sigma35_stress_margin_gallery" in workflow
    assert "context_pair_summary" in workflow
    assert "appendix_umap_gallery" in workflow
    assert "### Gate" in workflow
    assert "pre-assay representation triage" in workflow
    assert "canonical Infer feature sidecars" in workflow
    assert "candidate_x_selection_scorecard" in workflow
    assert "controlled equal-block" in workflow
    assert "bidirectional forward/RC context `anchor_mean` concat" in workflow
    assert "prefix-conditioned causal mean-pooled span embedding" in workflow
    assert "not a native bidirectional Evo2 hidden state" in workflow
    assert "working pre-assay `X`" in workflow
    assert "eight canonical 7B+20B" not in workflow
    assert "Leave geodesic pilots in study notes" in workflow
    assert 'zero_variance_policy="drop_or_zero"' in workflow
    assert 'zero_row_policy="zero"' in workflow
    assert "workspace snapshot" in workflow
    assert latentdna_binding["supported_model_families"] == ["evo2_7b", "evo2_20b"]
    assert latentdna_binding["default_model_family"] == "evo2_7b"
    assert latentdna_binding["source_datasets"]["merged_anchor_insert"] == "usr_prom_eth_cip_anchor"
    assert latentdna_binding["source_datasets"]["reference_native"] == "usr_promoter_references"
    assert latentdna_binding["source_datasets"]["reference_core60"] == "construct_prom_eth_cip_reference_core60"
    assert latentdna_binding["source_datasets"]["reference_contexts"] == "construct_prom_eth_cip_reference_contexts"
    assert (
        latentdna_binding["default_geometry_inventory"]["working_candidate"]
        == "intermediate_embedding_7b_context_anchor_mean_bidir_concat"
    )
    assert "candidate_x_selection_scorecard" in latentdna_binding["decision_deliverables"]
    assert study_pipeline["study_pipeline"]["infer"]["preferred_model_family"] == "evo2_7b"
    assert study_pipeline["study_pipeline"]["infer"]["supported_model_families"] == ["evo2_7b", "evo2_20b"]
    assert (
        study_pipeline["study_pipeline"]["infer"]["infer_priority"]["working_candidate_family"]
        == "evo2_7b_context_anchor_mean_bidir_concat_intermediate"
    )

    assert "`latentdna workspace snapshot`" in cli_contracts
    assert "`latentdna.workspace_snapshot.v1`" in cli_contracts
    assert "--progress none|text|json" in cli_contracts
    assert "Nested output roots are rejected." in cli_contracts

    assert "<workspace>/outputs" in workspace_schema
    assert "promoter-study pre-assay template" in workspace_schema

    assert "Workspace snapshot contract" in reference_index
    assert "Artifact naming grammar" in reference_index
    assert "ops/status.registry.yaml" in operations

    assert "latentdna_binding.yaml" in workspace_readme
    assert "workspace_snapshot.json" in workspace_readme
    assert "UMAP role: appendix orientation only" in workspace_readme
    assert "Reference metadata sources:" in workspace_readme
    assert "causal and prefix-conditioned" in workspace_readme

    assert "Gate:" in study_routes
    assert "representation_health_summary" in study_routes
    assert "Primary review path:" in study_routes
    assert "sigma35_ordinal_audit" in study_routes
    assert "appendix_umap_gallery" in study_routes
    assert "Snapshot attention surfaces:" in study_routes
    assert "dataset_overview" in study_routes
    assert "design_structure_summary" in study_routes
    assert "context_robustness_summary" in study_routes
    assert "candidate_decision_frontier" in study_routes
    assert "candidate_x_selection_scorecard" in study_routes
    assert "balanced_design_family_margin_gallery" in study_routes
    assert "sigma35_margin_ladder_gallery" in study_routes
    assert "sigma35_centroid_distance_gallery" in study_routes
    assert "sigma35_stress_margin_gallery" in study_routes
    assert "context_pair_summary" in study_routes
    assert "Snapshot attention surfaces: none for LatentDNA decision deliverables" in study_routes
    assert "Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`" in study_routes
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
    assert "7B-first sidecar-backed browser posture" in study_routes
    assert "available 7B sequence-view sidecar geometries" in study_routes
    assert "preferred infer family is now `evo2_7b`" in study_routes
    assert "token-position" in study_routes
    assert "not as native bidirectional encodings" in study_routes
    assert "eight canonical 7B+20B" not in study_routes

    assert "The study phase is `infer_batch_preparation`" in study_status
    assert "Current LatentDNA decision surfaces:" in study_status
    assert "representation_health_summary" in study_status
    assert "candidate_decision_frontier" in study_status
    assert "candidate_x_selection_scorecard" in study_status
    assert "balanced_design_family_margin_gallery" in study_status
    assert "sigma35_margin_ladder_gallery" in study_status
    assert "sigma35_stress_margin_gallery" in study_status
    assert "Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`" in study_status
    assert "Preferred infer family: `evo2_7b`" in study_status
    assert "Supported infer families: `evo2_7b`, `evo2_20b`" in study_status
    assert "LatentDNA browser default family: `evo2_7b`" in study_status
    assert "LatentDNA gate:" in study_status
    assert "LatentDNA primary review path:" in study_status
    assert "LatentDNA companion visuals:" in study_status
    assert "LatentDNA appendix support:" in study_status
    assert "available 7B sequence-view feature" in study_status
    assert "sidecars" in study_status
    assert "eight canonical 7B+20B" not in study_status
    assert "secondary review material" in study_status
    assert "Pooling semantics guardrail" in study_status
    assert "not a native bidirectional Evo2 state" in study_status


def test_latentdna_docs_remove_legacy_promoter_surface_names() -> None:
    repo_root = _repo_root()
    scan_roots = [
        repo_root / "src/dnadesign/latentdna",
        repo_root / "docs/studies/stress_ethanol_cipro_growth",
        repo_root / "src/dnadesign/studies/studies/stress_ethanol_cipro_growth",
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
