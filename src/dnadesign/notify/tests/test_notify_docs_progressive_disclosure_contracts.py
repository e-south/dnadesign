"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_notify_docs_progressive_disclosure_contracts.py

Progressive-disclosure contract tests for Notify operator and maintainer docs.

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


def test_notify_docs_readme_keeps_operator_progressive_disclosure() -> None:
    text = _read(_repo_root() / "docs" / "notify" / "README.md")
    _assert_token_order(
        text,
        [
            "### Before you start",
            "### Choose a workflow",
            "### Quick path",
            "### Troubleshooting",
            "### References",
        ],
        label="docs/notify/README.md",
    )
    assert "multi-source-source-of-truth-assembly.md" in text
    assert "promoter-characterization-feature-matrix.md" in text
    assert "--secret-source file" in text
    assert "--secret-ref file://" in text
    assert "--url-env" in text
    assert "--secret-source auto" not in text
    assert "Notify command contracts" in text
    assert "route map only" not in text
    assert "Start here for setup, watching, and recovery." in text
    assert "DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not Notify input." in text


def test_notify_usr_events_manual_keeps_setup_run_recover_flow() -> None:
    text = _read(_repo_root() / "docs" / "notify" / "usr-events.md")
    _assert_token_order(
        text,
        [
            "### Minimal operator quickstart",
            "### Command contract: setup vs watch",
            "### Setup flow",
            "### Run flow",
            "### Recover flow",
            "### Common mistakes",
        ],
        label="docs/notify/usr-events.md",
    )
    assert "--secret-source file" in text
    assert "--secret-ref file://" in text
    assert "--url-env" in text
    assert "chmod 600" in text
    assert "--secret-source auto" not in text
    assert "--only-actions merge_datasets,attach,materialize" in text
    assert "NOTIFY_ACTIONS" in text
    assert "Workspace shorthand for any tool is repo-rooted" in text
    assert "DNADESIGN_REPO_ROOT=<repo-root>" in text
    assert "Multi-destination infer configs must use explicit `--events <path>`" in text


def test_notify_module_readme_is_lightweight_router_and_links_top_level_runbook() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "notify" / "README.md")
    _assert_token_order(
        text,
        [
            "## Documentation",
        ],
        label="src/dnadesign/notify/README.md",
    )
    assert "docs/README.md" in text
    assert "docs/notify/usr-events.md" in text
    assert "../../../docs/notify/README.md" in text
    assert "docs/reference/README.md" in text
    assert "docs/dev/architecture.md" in text
    assert "../../../docs/README.md" in text
    assert text.find("docs/README.md") < text.find("docs/reference/README.md")
    assert "## Start here in 3 commands" not in text
    assert "## Entrypoint contract" not in text
    assert "## Boundary reminder" not in text


def test_notify_module_docs_index_has_progressive_disclosure_workflow_and_type_maps() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "notify" / "docs" / "README.md")
    _assert_token_order(
        text,
        [
            "### Ownership boundary",
            "### Start here",
            "### Prerequisites",
            "### Package docs by task",
            "### Documentation by type",
        ],
        label="src/dnadesign/notify/docs/README.md",
    )
    assert "notify send contract" in text
    assert "Runtime evidence pointers" in text
    assert "Shared watcher and scheduler docs" in text
    assert "Run cross-tool or cluster workflows" not in text
    assert "../../../../docs/notify/usr-events.md" in text
    assert "Use this page for Notify command contracts and maintainer docs." in text
    assert "Use [Notify USR events runbook]" in text
    assert "notify setup list-workspaces" not in text
    assert "notify usr-events watch --tool <tool> --workspace <workspace-name> --follow" not in text


def test_notify_reference_index_keeps_reference_first_and_routes_operator_steps_outward() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "notify" / "docs" / "reference" / "README.md")
    _assert_token_order(
        text,
        [
            "### Read order",
            "[Command contracts](command-contracts.md)",
            "[USR event schema reference](../../../usr/docs/reference/event-log.md)",
            "### Need operator steps instead?",
            "../../../../docs/notify/usr-events.md",
            "### Coverage",
            "### Verify next",
        ],
        label="src/dnadesign/notify/docs/reference/README.md",
    )
    assert "strict command, profile, and boundary contracts" in text


def test_notify_command_contracts_cover_setup_helpers_and_send() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "notify" / "docs" / "reference" / "command-contracts.md")
    _assert_token_order(
        text,
        [
            "### notify setup webhook",
            "### notify setup list-workspaces",
            "### notify setup resolve-events",
            "### notify setup slack",
            "### notify send",
            "### notify usr-events watch",
            "### notify profile doctor",
            "### notify spool drain",
            "### profile schema contract",
            "### observer boundary",
            "### no-silent-fallback contract",
            "### Runtime evidence pointers",
        ],
        label="src/dnadesign/notify/docs/reference/command-contracts.md",
    )
    assert "Workspace shorthand is repo-rooted for all resolver-mode tools." in text
    assert "ingest.root" in text
    assert "requires exactly one USR write-back destination and explicit `ingest.root`" in text
    assert "Multi-destination infer configs must use explicit `--events <path>`" in text


def test_notify_maintainer_docs_use_deps_package_paths_not_removed_monolith() -> None:
    docs_readme = _read(_repo_root() / "docs" / "notify" / "README.md")

    assert "src/dnadesign/notify/docs/README.md" in docs_readme
    architecture = _read(_repo_root() / "src" / "dnadesign" / "notify" / "docs" / "dev" / "architecture.md")
    assert "src/dnadesign/notify/cli/bindings/deps/" in architecture
    assert "src/dnadesign/notify/cli/bindings/deps.py" not in docs_readme
    assert "src/dnadesign/notify/cli/bindings/deps.py" not in architecture
