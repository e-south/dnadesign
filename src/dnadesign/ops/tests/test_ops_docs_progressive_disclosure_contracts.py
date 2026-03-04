"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_ops_docs_progressive_disclosure_contracts.py

Progressive-disclosure contract tests for Ops package and top-level Ops docs.

Module Author(s): Eric J. South
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


def _read(path: Path) -> str:
    assert path.exists(), f"Missing markdown file: {path}"
    return path.read_text(encoding="utf-8")


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


def test_ops_module_readme_has_banner_narrative_and_doc_map() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "ops" / "README.md")
    _assert_token_order(
        text,
        [
            "![Ops banner](assets/ops-banner.svg)",
            "## Documentation map",
            "## Entrypoint contract",
            "## Boundary reminder",
        ],
        label="src/dnadesign/ops/README.md",
    )
    assert "cross-tool orchestration control plane" in text
    assert "tool-specific workflow semantics" in text.lower()
    assert "docs/operations/README.md" in text
    assert "docs/operations/orchestration-runbooks.md" in text
    assert "../../../docs/README.md" in text
    assert "progressive disclosure" not in text.lower()


def test_ops_docs_index_has_progressive_disclosure_routes() -> None:
    text = _read(_repo_root() / "docs" / "operations" / "README.md")
    _assert_token_order(
        text,
        [
            "### What Ops is for",
            "### Start here",
            "### Workflow routes",
            "### Contracts",
            "### Verification loop",
            "### Operator quickstart",
        ],
        label="docs/operations/README.md",
    )
    assert "runbook init command contract" in text
    assert "runbook plan command contract" in text
    assert "runbook execute command contract" in text
    assert "ops runbook init --workflow" in text
    assert "orchestration-runbooks.md" in text
    assert "../README.md" in text
    assert "../../src/dnadesign/ops/README.md" in text
    assert "progressive disclosure" not in text.lower()


def test_orchestration_runbook_doc_keeps_run_order_and_contract_sections() -> None:
    text = _read(_repo_root() / "docs" / "operations" / "orchestration-runbooks.md")
    _assert_token_order(
        text,
        [
            "### Why this exists",
            "### Runbook bootstrap path",
            "### 2-minute dry-run path",
            "### Workflow routes",
            "### Runbook schema (v1)",
            "### Planner and executor commands",
            "### Contract rules",
        ],
        label="docs/operations/orchestration-runbooks.md",
    )
    assert "uv run ops runbook init" in text
    assert "uv run ops runbook precedents" in text
    assert "uv run ops runbook active-jobs" in text
    assert "default is `300`" in text
    assert "operator and agent review" not in text
    assert "--command-timeout-seconds" in text
    assert "mode=auto" in text
    assert "none -> fresh" in text
    assert "resume_ready -> resume" in text
    assert "partial -> contract error" in text
    assert "<workspace-root>/outputs/logs/ops/audit/<file>.json" in text
    assert "prune-ops-logs" in text
    assert "logging.retention.keep_last" in text
    assert "logging.retention.max_age_days" in text
    assert "outputs/logs/ops/runtime" in text
    assert "usr-overlay-guard" in text
    assert "usr-records-part-guard" in text
    assert "usr-archived-overlay-guard" in text
    assert "densegen-overlay-guard" not in text
    assert "densegen.overlay_guard.overlay_namespace" in text
    assert "densegen.overlay_guard.namespace" not in text
    assert "transient operational working directories at repo root" in text
    assert "/scratch" in text


def test_repo_docs_index_exposes_ops_tool_and_operations_route() -> None:
    text = _read(_repo_root() / "docs" / "README.md")
    assert "### Workflow routes" in text
    assert "### Workflow lanes" not in text
    assert "[Workflow routes](#workflow-routes)" in text
    assert "[Ops operations index](operations/README.md)" in text
    assert "| `ops` | `uv run ops --help` | [ops README](../src/dnadesign/ops/README.md) |" in text


def test_repo_root_readme_lists_ops_in_docs_and_tool_catalog() -> None:
    text = _read(_repo_root() / "README.md")
    assert "[Docs index](docs/README.md)" in text
    assert "[Ops operations](docs/operations/README.md)" not in text
    assert "[Notify operations](docs/notify/README.md)" not in text
    assert "[Workflow lanes](docs/README.md#workflow-lanes)" not in text
    assert (
        "[Cross-tool information architecture contract](ARCHITECTURE.md#cross-tool-information-architecture)"
        not in text
    )
    assert "[Boundary rules](DESIGN.md#toolpackage-boundaries)" not in text
    assert "| [**ops**](src/dnadesign/ops/README.md) |" in text
    assert "DenseGen/Infer + Notify batch workflows" not in text


def test_root_ops_row_is_tool_agnostic() -> None:
    text = _read(_repo_root() / "README.md")
    expected_row = (
        "| [**ops**](src/dnadesign/ops/README.md) | "
        "Runbook-driven orchestration for deterministic batch workflows across tools. |"
    )
    assert expected_row in text


def test_dev_docs_index_is_action_oriented() -> None:
    text = _read(_repo_root() / "docs" / "dev" / "README.md")
    _assert_token_order(
        text,
        [
            "## Developer Documentation",
            "### Start here",
            "### Day-to-day tasks",
            "### CI and quality checks",
            "### Planning and decisions",
        ],
        label="docs/dev/README.md",
    )
    assert "journal.md" in text
    assert "uv run python -m dnadesign.devtools.docs_checks --repo-root ." in text
    assert "for agents" not in text.lower()
    assert "for humans" not in text.lower()


def test_core_docs_avoid_contrived_doc_language() -> None:
    targets = [
        "README.md",
        "docs/README.md",
        "docs/dev/README.md",
        "docs/notify/README.md",
        "docs/operations/README.md",
        "src/dnadesign/ops/README.md",
        "src/dnadesign/notify/README.md",
        "src/dnadesign/notify/docs/README.md",
        "src/dnadesign/usr/README.md",
        "src/dnadesign/usr/docs/README.md",
        "ARCHITECTURE.md",
        "DESIGN.md",
        "RELIABILITY.md",
        "QUALITY_SCORE.md",
    ]
    banned_tokens = ("progressive disclosure", "canonical", "for agents", "for humans")
    repo_root = _repo_root()
    for rel in targets:
        text = _read(repo_root / rel).lower()
        for token in banned_tokens:
            assert token not in text, f"{rel}: contains banned token {token!r}"
