"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/docs_contract/test_sync.py

Structural sync and workflow-surface contracts for SeqLedger docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from .helpers import assert_markdown_links_resolve, heading_lines, metadata, read_text


def test_sync_docs_use_positional_remote_contract() -> None:
    combined = "\n".join(
        [
            read_text("src/dnadesign/seqledger/README.md"),
            read_text("src/dnadesign/seqledger/docs/operations/sync/README.md"),
            read_text("docs/bu-scc/runbooks/batch-notify.md"),
        ]
    )
    stale = re.compile(r"seqledger\s+(?:pull|push|diff|status)\s+[^\n]*--remote\b")
    assert stale.search(combined) is None


def test_sync_router_and_ops_indexes_have_resolving_links_and_route_metadata() -> None:
    assert_markdown_links_resolve("src/dnadesign/seqledger/docs/operations/README.md")
    assert_markdown_links_resolve("src/dnadesign/seqledger/docs/operations/sync/README.md")

    route_metadata = metadata("src/dnadesign/seqledger/docs/operations/sync/README.md")
    assert route_metadata["Type"] == "route"
    assert route_metadata["Plane"] == "data-plane"
    assert route_metadata["Owner-boundary"] == "seqledger"

    assert "# SeqLedger operations runbooks" in heading_lines("src/dnadesign/seqledger/docs/operations/README.md")


def test_navigation_routes_to_the_stable_sync_drill() -> None:
    workflow_map = read_text("src/dnadesign/seqledger/docs/operations/routes/workflow-map.md")
    ops_index = read_text("src/dnadesign/seqledger/docs/operations/README.md")
    drills = read_text("src/dnadesign/seqledger/docs/operations/sync/fidelity-drills.md")
    harness = read_text("src/dnadesign/seqledger/scripts/run_seqledger_harness_cycle.sh")

    assert "../sync/fidelity-drills.md" in workflow_map
    assert "sync/fidelity-drills.md" in ops_index
    assert_markdown_links_resolve("src/dnadesign/seqledger/docs/operations/routes/workflow-map.md")
    assert_markdown_links_resolve("src/dnadesign/seqledger/docs/operations/sync/fidelity-drills.md")
    for text in (drills, harness):
        assert "uv run seqledger-sync-audit-drill" in text
        assert "run_usr_sync_audit_drill.py" not in text
        assert "SEQLEDGER_HARNESS_RUN_SYNC_AUDIT_DRILL" in text
        assert "SEQLEDGER_HARNESS_SYNC_AUDIT_REPORT_PATH" in text


def test_sync_runbooks_keep_hash_and_auxiliary_contract_terms() -> None:
    sync_router = read_text("src/dnadesign/seqledger/docs/operations/sync/README.md")
    quickstart = read_text("src/dnadesign/seqledger/docs/operations/sync/quickstart.md")
    hpc = read_text("src/dnadesign/seqledger/docs/operations/sync/hpc-agent-flow.md")
    fidelity = read_text("src/dnadesign/seqledger/docs/operations/sync/fidelity-drills.md")

    for text in (sync_router, quickstart, hpc, fidelity):
        assert "_auxiliary" in text
    assert "--audit-json-out" in quickstart
    assert "--no-verify-derived-hashes" in hpc
    assert "--verify auto" not in hpc
    assert "--verify parquet" not in hpc


def test_sync_docs_avoid_agent_or_human_labeling_language() -> None:
    combined = "\n".join(
        [
            read_text("src/dnadesign/seqledger/docs/README.md"),
            read_text("src/dnadesign/seqledger/docs/operations/sync/README.md"),
            read_text("src/dnadesign/seqledger/docs/operations/sync/hpc-agent-flow.md"),
            read_text("src/dnadesign/seqledger/docs/operations/sync/audit-loop.md"),
            read_text("src/dnadesign/seqledger/docs/operations/sync/fidelity-drills.md"),
        ]
    ).lower()
    for token in ("agent-oriented", "agent runbook", "agent checklist", "agentic", "for agents", "for humans"):
        assert token not in combined


def test_hpc_sync_runbook_covers_bootstrap_from_either_side() -> None:
    runbook = read_text("src/dnadesign/seqledger/docs/operations/sync/hpc-agent-flow.md")
    assert "Bootstrap from either side" in runbook
    assert "HPC has dataset, local does not" in runbook
    assert "Local has dataset, HPC does not" in runbook
    assert 'uv run seqledger pull "$DATASET_ID" bu-scc -y' in runbook
    assert 'uv run seqledger push "$DATASET_ID" bu-scc -y' in runbook
