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
