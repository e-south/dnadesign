"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/docs_contract/test_sync.py

Structural sync and workflow-surface contracts for USR docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from .helpers import assert_markdown_links_resolve, heading_lines, metadata, read_text


def test_sync_docs_use_positional_remote_contract() -> None:
    combined = "\n".join(
        [
            read_text("src/dnadesign/usr/README.md"),
            read_text("src/dnadesign/usr/docs/operations/sync/README.md"),
            read_text("docs/bu-scc/runbooks/batch-notify.md"),
        ]
    )
    stale = re.compile(r"usr\s+(?:pull|push|diff|status)\s+[^\n]*--remote\b")
    assert stale.search(combined) is None


def test_sync_router_and_ops_indexes_have_resolving_links_and_route_metadata() -> None:
    assert_markdown_links_resolve("src/dnadesign/usr/docs/operations/README.md")
    assert_markdown_links_resolve("src/dnadesign/usr/docs/operations/sync/README.md")

    route_metadata = metadata("src/dnadesign/usr/docs/operations/sync/README.md")
    assert route_metadata["Type"] == "route"
    assert route_metadata["Plane"] == "data-plane"
    assert route_metadata["Owner-boundary"] == "usr"

    assert "# USR operations runbooks" in heading_lines("src/dnadesign/usr/docs/operations/README.md")


def test_workflow_map_and_harness_document_stable_sync_drill_entrypoint() -> None:
    workflow_map = read_text("src/dnadesign/usr/docs/operations/routes/workflow-map.md")
    ops_index = read_text("src/dnadesign/usr/docs/operations/README.md")
    harness = read_text("src/dnadesign/usr/scripts/run_usr_harness_cycle.sh")

    assert "uv run usr-sync-audit-drill" in workflow_map
    assert "run_usr_sync_audit_drill.py" not in workflow_map
    assert "uv run usr-sync-audit-drill" in ops_index
    assert "run_usr_sync_audit_drill.py" not in ops_index
    assert "uv run usr-sync-audit-drill" in harness
    assert "run_usr_sync_audit_drill.py" not in harness
    assert "USR_HARNESS_RUN_SYNC_AUDIT_DRILL" in workflow_map
    assert "USR_HARNESS_SYNC_AUDIT_REPORT_PATH" in workflow_map


def test_sync_runbooks_keep_hash_and_auxiliary_contract_terms() -> None:
    sync_router = read_text("src/dnadesign/usr/docs/operations/sync/README.md")
    quickstart = read_text("src/dnadesign/usr/docs/operations/sync/quickstart.md")
    hpc = read_text("src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.md")
    chained = read_text("src/dnadesign/usr/docs/operations/sync/chained-densegen-infer-runbook.md")
    fidelity = read_text("src/dnadesign/usr/docs/operations/sync/fidelity-drills.md")

    for text in (sync_router, quickstart, hpc, chained, fidelity):
        assert "_auxiliary" in text
    assert "--audit-json-out" in quickstart
    assert "--no-verify-derived-hashes" in hpc
    assert "--no-verify-derived-hashes" in chained
    assert "--verify auto" not in hpc
    assert "--verify parquet" not in hpc
    assert "--verify auto" not in chained
    assert "--verify parquet" not in chained


def test_sync_docs_avoid_agent_or_human_labeling_language() -> None:
    combined = "\n".join(
        [
            read_text("src/dnadesign/usr/docs/README.md"),
            read_text("src/dnadesign/usr/docs/operations/sync/README.md"),
            read_text("src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.md"),
            read_text("src/dnadesign/usr/docs/operations/sync/audit-loop.md"),
            read_text("src/dnadesign/usr/docs/operations/sync/fidelity-drills.md"),
        ]
    ).lower()
    for token in ("agent-oriented", "agent runbook", "agent checklist", "agentic", "for agents", "for humans"):
        assert token not in combined


def test_hpc_sync_runbook_covers_bootstrap_from_either_side() -> None:
    runbook = read_text("src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.md")
    assert "Bootstrap from either side" in runbook
    assert "HPC has dataset, local does not" in runbook
    assert "Local has dataset, HPC does not" in runbook
    assert 'uv run usr pull "$DATASET_ID" bu-scc -y' in runbook
    assert 'uv run usr push "$DATASET_ID" bu-scc -y' in runbook
