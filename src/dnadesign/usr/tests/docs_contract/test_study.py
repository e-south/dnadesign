"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/docs_contract/test_study.py

Structural study and skill-doc contracts touched by USR runbook surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import subprocess

from dnadesign.studies.core.record_loader import load_study_ops_contract

from .helpers import assert_markdown_links_resolve, load_yaml, read_text, repo_root


def test_promoter_study_registry_and_snapshot_surfaces_have_expected_structure() -> None:
    studies_index = load_yaml("docs/studies/index.yaml")
    datasets = load_yaml("docs/studies/stress_ethanol_cipro_growth/record/datasets.yaml")
    pipeline = load_yaml("docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/pipeline.yaml")
    ops_study = load_yaml("docs/studies/stress_ethanol_cipro_growth/operations/ops.study.yaml")
    ops_snapshot = load_yaml("docs/studies/stress_ethanol_cipro_growth/operations/contract/status/snapshot.yaml")
    ops_contract = load_study_ops_contract(repo_root() / "docs" / "studies" / "stress_ethanol_cipro_growth")
    by_id = {row["study_id"]: row for row in studies_index["studies"]}

    assert studies_index["active_study_id"] == "stress_ethanol_cipro_growth"
    assert by_id["stress_ethanol_cipro_growth"]["record_root"] == "docs/studies/stress_ethanol_cipro_growth"

    roles = {row["role"] for row in datasets["datasets"]}
    assert "densegen_anchor" in roles
    assert "opal_candidate_feature_table" in roles

    study_pipeline = pipeline["study_pipeline"]
    assert "construct_workspace" in study_pipeline["execution_surfaces"]
    assert "densegen" in study_pipeline
    assert "latentdna" in study_pipeline
    assert "cluster" in study_pipeline
    assert "opal" in study_pipeline

    assert ops_study["version"] == 2
    assert ops_study["parts"]["snapshot"] == "contract/status/snapshot.yaml"
    assert ops_snapshot["summary_scope"] == "repo"
    assert ops_contract.preflight.check_specs
    assert "construct_workspace" in ops_contract.execution_surfaces


def test_promoter_study_docs_link_and_reference_owner_surfaces() -> None:
    for rel_path in (
        "docs/studies/README.md",
        "src/dnadesign/usr/docs/operations/promoter/evo2-journey.md",
        "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md",
    ):
        assert_markdown_links_resolve(rel_path)

    journey = read_text("src/dnadesign/usr/docs/operations/promoter/evo2-journey.md")
    status = read_text("docs/studies/stress_ethanol_cipro_growth/record/status.md")
    routes = read_text("docs/studies/stress_ethanol_cipro_growth/routes/README.md")
    densegen_route = read_text("docs/studies/stress_ethanol_cipro_growth/routes/source/densegen.md")
    infer_route = read_text("docs/studies/stress_ethanol_cipro_growth/routes/compute/infer.md")
    latentdna_route = read_text("docs/studies/stress_ethanol_cipro_growth/routes/analysis/latentdna.md")

    assert "assembly/multi-source-shared-dataset.md" in journey
    assert "construct-infer-shared-dataset-runbook.md" in journey
    assert "operations/catalog/contracts/status.md" in journey
    assert "usr-infer-x-active-learning.md" in journey
    assert "Route map: `../routes/README.md`" in status
    assert "Study execution map: `../operations/runtime/command-groups/pipeline.yaml`" in status
    assert "| DenseGen EDA |" in routes
    assert "## DenseGen EDA Route Detail" in densegen_route
    assert "## Infer Lanes Route Detail" in infer_route
    assert "## LatentDNA Route Detail" in latentdna_route


def test_stress_ethanol_cipro_contract_avoids_family_templates() -> None:
    contract = read_text("docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md")
    preflight = read_text("docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md")
    templates_index = read_text("docs/templates/README.md")

    assert "stress_ethanol_cipro_growth" in contract
    assert "studies.stress-ethanol-cipro-growth.status --json" in contract
    assert "Use this only for `stress_ethanol_cipro_growth`" in contract
    assert "studies.stress-ethanol-cipro-growth.preflight --scope next --json" in preflight
    assert "promoter-study-index.yaml" not in templates_index
    assert "promoter-study-datasets.yaml" not in templates_index
    assert "promoter-study-ops.study.yaml" not in templates_index
    assert "stress-ethanol-cipro-growth-status.md" not in templates_index


def test_repo_local_skill_audit_scripts_are_documented_and_present() -> None:
    dev_docs = read_text("docs/dev/README.md")
    repo = repo_root()
    promoter_skill_root = repo / ".agents" / "skills" / "stress-ethanol-cipro-growth-status"
    sync_skill_root = repo / ".agents" / "skills" / "bu-scc-usr-sync"

    assert (
        ".agents/skills/stress-ethanol-cipro-growth-status/scripts/audit-stress-ethanol-cipro-growth-status-skill.sh"
        in dev_docs
    )
    assert ".agents/skills/bu-scc-usr-sync/scripts/audit-bu-scc-usr-sync-skill.sh" in dev_docs

    assert (promoter_skill_root / "SKILL.md").exists()
    assert (promoter_skill_root / "scripts" / "audit-stress-ethanol-cipro-growth-status-skill.sh").exists()
    assert (promoter_skill_root / "references" / "route-matrix.md").exists()
    assert (promoter_skill_root / "references" / "refresh-loop.md").exists()
    assert (promoter_skill_root / "references" / "study-surfaces.md").exists()

    assert (sync_skill_root / "SKILL.md").exists()
    assert (sync_skill_root / "scripts" / "audit-bu-scc-usr-sync-skill.sh").exists()
    assert (sync_skill_root / "references" / "sync-loop.md").exists()


def test_repo_local_skill_audits_pass() -> None:
    repo = repo_root()
    commands = [
        repo
        / ".agents"
        / "skills"
        / "stress-ethanol-cipro-growth-status"
        / "scripts"
        / "audit-stress-ethanol-cipro-growth-status-skill.sh",
        repo / ".agents" / "skills" / "bu-scc-usr-sync" / "scripts" / "audit-bu-scc-usr-sync-skill.sh",
    ]
    for command in commands:
        result = subprocess.run(
            [shutil.which("bash") or "bash", str(command)],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
