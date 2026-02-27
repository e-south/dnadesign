"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_usr_docs_contract.py

Contracts for USR sync syntax and DenseGen-to-Notify event-boundary docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(rel_path: str) -> str:
    path = _repo_root() / rel_path
    return path.read_text(encoding="utf-8")


def test_usr_sync_docs_use_positional_remote_for_sync_commands() -> None:
    readme = _read("src/dnadesign/usr/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    combined = f"{readme}\n{sync_ops}"

    stale = re.compile(r"usr\s+(?:pull|push|diff|status)\s+[^\n]*--remote\b")
    assert stale.search(combined) is None
    assert "usr diff densegen/60bp_dual_promoter_cpxR_LexA bu-scc" in readme
    assert "uv run usr pull densegen/my_dataset bu-scc -y" in sync_ops


def test_bu_scc_runbook_uses_positional_usr_pull_example() -> None:
    runbook = _read("docs/bu-scc/batch-notify.md")
    stale = re.compile(r"uv run usr pull [^\n]*--remote\b")
    assert stale.search(runbook) is None
    assert "uv run usr pull densegen/demo_hpc bu-scc -y" in runbook


def test_usr_notify_boundary_docs_keep_events_contract_explicit() -> None:
    notify_doc = _read("docs/notify/usr-events.md")
    usr_readme = _read("src/dnadesign/usr/README.md")

    assert ".events.log" in notify_doc
    assert "outputs/meta/events.jsonl" in notify_doc
    assert ".events.log" in usr_readme
    assert "not Notify input" in usr_readme


def test_usr_sync_docs_follow_progressive_disclosure_flow() -> None:
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    usr_readme = _read("src/dnadesign/usr/README.md")

    assert "Quick path" in sync_ops
    assert "Advanced path" in sync_ops
    assert "Failure diagnosis" in sync_ops
    assert "progressive disclosure" in usr_readme.lower()


def test_usr_sync_docs_cover_iterative_hpc_clone_safety_loop() -> None:
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")

    assert "Iterative batch loop (HPC clone -> local clone)" in sync_ops
    assert "uv run usr diff densegen/my_dataset bu-scc" in sync_ops
    assert "uv run usr pull densegen/my_dataset bu-scc -y" in sync_ops
    assert "uv run usr push densegen/my_dataset bu-scc -y" in sync_ops
    assert "fails fast when remote `records.parquet` is missing" in sync_ops
    assert "fails fast when local `records.parquet` is missing" in sync_ops
    assert "skip transfer when no changes are detected" in sync_ops
    assert "shared remote dataset lock (`.usr.lock`)" in sync_ops
    assert "--verify-sidecars" in sync_ops
    assert "--no-verify-sidecars" in sync_ops
    assert "--verify-derived-hashes" in sync_ops
    assert "defaults to `--verify hash` plus strict sidecar fidelity checks" in sync_ops
    assert "--strict-bootstrap-id" in sync_ops
    assert "USR_SYNC_STRICT_BOOTSTRAP_ID=1" in sync_ops
    assert "stage into a temporary directory and only promote after verification" in sync_ops
    assert "reject symlink and unsupported entry types before promotion" in sync_ops
    assert "post-action sync audit summary" in sync_ops


def test_docs_index_links_progressive_usr_sync_workflows() -> None:
    docs_index = _read("docs/README.md")
    assert "Progressive workflows" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-fidelity-drills.md" in docs_index


def test_usr_docs_index_exposes_agent_sync_runbook() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")

    assert "operations/README.md" in usr_docs
    assert "sync.md" in ops_index
    assert "hpc-agent-sync-flow.md" in ops_index
    assert "chained-densegen-infer-sync-demo.md" in ops_index
    assert "sync-fidelity-drills.md" in ops_index
    assert "hpc-agent-sync-flow.md" in usr_docs
    assert "chained-densegen-infer-sync-demo.md" in usr_docs
    assert "sync-fidelity-drills.md" in usr_docs
    assert "hpc-agent-sync-flow.md" in sync_ops
    assert "chained-densegen-infer-sync-demo.md" in sync_ops
    assert "sync-fidelity-drills.md" in sync_ops
    assert "Preflight" in runbook
    assert "Run loop" in runbook
    assert "Verify loop" in runbook
    assert "uv run usr diff" in runbook
    assert "uv run usr pull" in runbook
    assert "uv run usr push" in runbook
    assert "Full chained loop" in chained
    assert "uv run infer run --preset evo2/extract_logits_ll --usr" in chained
    assert "qsub -P <project>" in chained
    assert "_derived changed" in chained
    assert "Drill 1: Pull must fail when `_derived` payload is missing" in fidelity
    assert "Drill 2: Push must fail when remote misses local overlays" in fidelity
    assert "Drill 3: Overlay schema attack surface" in fidelity
    assert "--verify-sidecars" in fidelity
    assert "--no-verify-sidecars" in fidelity
    assert "--verify-derived-hashes" in fidelity
    assert "post-pull-sidecars" in fidelity
    assert "post-push-sidecars" in fidelity


def test_usr_sync_runbooks_avoid_agent_or_human_labeling_language() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")
    combined = "\n".join([usr_docs, sync_ops, runbook, fidelity]).lower()

    banned = [
        "agent-oriented",
        "agent runbook",
        "agent checklist",
        "agentic",
        "for agents",
        "for humans",
    ]
    for token in banned:
        assert token not in combined


def test_usr_hpc_and_chained_runbooks_use_default_hash_sync_contract_examples() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md")

    assert "--verify auto" not in runbook
    assert "--verify parquet" not in runbook
    assert "--verify auto" not in chained
    assert "--verify parquet" not in chained
    assert "defaults to `--verify hash` plus strict sidecar fidelity checks" in runbook
    assert "defaults to `--verify hash` plus strict sidecar fidelity checks" in chained


def test_usr_sync_docs_include_auxiliary_file_audit_contract() -> None:
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")

    assert "_auxiliary" in sync_ops
    assert "_auxiliary" in chained
    assert "_auxiliary" in fidelity
