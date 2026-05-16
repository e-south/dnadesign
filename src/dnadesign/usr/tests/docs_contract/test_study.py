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

from .helpers import assert_markdown_links_resolve, load_yaml, read_text, repo_root


def test_promoter_study_registry_and_snapshot_surfaces_have_expected_structure() -> None:
    studies_index = load_yaml("docs/studies/index.yaml")
    datasets = load_yaml("docs/studies/stress_ethanol_cipro_growth/datasets.yaml")
    pipeline = load_yaml("docs/studies/stress_ethanol_cipro_growth/pipeline.yaml")
    ops_study = load_yaml("docs/studies/stress_ethanol_cipro_growth/ops.study.yaml")
    by_id = {row["study_id"]: row for row in studies_index["studies"]}

    assert studies_index["active_study_id"] == "stress_ethanol_cipro_growth"
    assert by_id["stress_ethanol_cipro_growth"]["family"] == "promoter"
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
    assert ops_study["snapshot"]["summary_scope"] == "repo"
    assert "checks" in ops_study["preflight"]
    assert "execution_surfaces" in ops_study


def test_promoter_study_docs_link_and_reference_owner_surfaces() -> None:
    for rel_path in (
        "docs/studies/README.md",
        "src/dnadesign/usr/docs/operations/promoter-evo2-journey.md",
        "src/dnadesign/usr/docs/operations/promoter-study-status-contract.md",
    ):
        assert_markdown_links_resolve(rel_path)

    journey = read_text("src/dnadesign/usr/docs/operations/promoter-evo2-journey.md")
    status = read_text("docs/studies/stress_ethanol_cipro_growth/status.md")
    routes = read_text("docs/studies/stress_ethanol_cipro_growth/routes.md")

    assert "multi-source-shared-dataset-assembly.md" in journey
    assert "construct-infer-shared-dataset-runbook.md" in journey
    assert "promoter-study-status-contract.md" in journey
    assert "usr-infer-x-active-learning.md" in journey
    assert "Route map: `routes.md`" in status
    assert "Study execution map: `pipeline.yaml`" in status
    assert "### DenseGen EDA" in routes
    assert "### Infer lanes" in routes
    assert "### LatentDNA comparison surface" in routes


def test_promoter_study_contract_and_templates_reference_checked_in_record_surfaces() -> None:
    contract = read_text("src/dnadesign/usr/docs/operations/promoter-study-status-contract.md")
    preflight = read_text("src/dnadesign/usr/docs/operations/promoter-study-preflight.md")
    templates_index = read_text("docs/templates/README.md")
    status_template = read_text("docs/templates/promoter-study-status.md")
    datasets_template = read_text("docs/templates/promoter-study-datasets.yaml")

    assert "docs/studies/index.yaml" in contract
    assert "docs/studies/<study-id>/" in contract
    assert "promoter-study-preflight --scope next --json" in contract
    assert "Minimum blocker evidence" in preflight
    assert "promoter-study-index.yaml" in templates_index
    assert "promoter-study-datasets.yaml" in templates_index
    assert "promoter-study-status.md" in templates_index
    assert "### Current datasets" in status_template
    assert "### Current phase" in status_template
    assert "root_kind: shared|workspace_local_export|external_usr" in datasets_template
    assert "onboard_mode: existing_local|existing_remote|existing_both|create_new" in datasets_template


def test_repo_local_skill_audit_scripts_are_documented_and_present() -> None:
    dev_docs = read_text("docs/dev/README.md")
    repo = repo_root()
    promoter_skill_root = repo / ".agents" / "skills" / "promoter-study-status"
    sync_skill_root = repo / ".agents" / "skills" / "bu-scc-usr-sync"

    assert ".agents/skills/promoter-study-status/scripts/audit-promoter-study-status-skill.sh" in dev_docs
    assert ".agents/skills/bu-scc-usr-sync/scripts/audit-bu-scc-usr-sync-skill.sh" in dev_docs

    assert (promoter_skill_root / "SKILL.md").exists()
    assert (promoter_skill_root / "scripts" / "audit-promoter-study-status-skill.sh").exists()
    assert (promoter_skill_root / "references" / "route-matrix.md").exists()
    assert (promoter_skill_root / "references" / "refresh-loop.md").exists()
    assert (promoter_skill_root / "references" / "study-surfaces.md").exists()

    assert (sync_skill_root / "SKILL.md").exists()
    assert (sync_skill_root / "scripts" / "audit-bu-scc-usr-sync-skill.sh").exists()
    assert (sync_skill_root / "references" / "sync-loop.md").exists()


def test_repo_local_skill_audits_pass() -> None:
    repo = repo_root()
    commands = [
        repo / ".agents" / "skills" / "promoter-study-status" / "scripts" / "audit-promoter-study-status-skill.sh",
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
