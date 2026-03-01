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
            "### Choose a workflow",
            "### Progressive disclosure path",
            "### Prompt-to-command router",
            "### 2-minute operator path",
            "### Interface contract summary",
            "### Command surface map",
            "### Troubleshooting and recovery",
            "### Canonical runbooks",
        ],
        label="docs/notify/README.md",
    )
    assert "start a densegen workspace watcher and send to slack" in text
    assert "start an infer_evo2 workspace watcher and send to slack" in text
    assert "i already have a profile, just validate wiring" in text
    assert "resume failed deliveries from spool" in text
    assert "`notify setup slack` mode contract" in text
    assert "`notify usr-events watch` mode contract" in text


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


def test_notify_module_readme_is_lightweight_router_and_links_top_level_runbook() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "notify" / "README.md")
    _assert_token_order(
        text,
        [
            "## Documentation map",
            "## Entrypoint contract",
            "## Boundary reminder",
        ],
        label="src/dnadesign/notify/README.md",
    )
    assert "docs/README.md" in text
    assert "docs/notify/usr-events.md" in text
    assert "Universal Sequence Record `<dataset>/.events.log`" in text


def test_notify_module_docs_index_has_progressive_disclosure_workflow_and_type_maps() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "notify" / "docs" / "README.md")
    _assert_token_order(
        text,
        [
            "### Progressive disclosure route",
            "### Audience and prerequisites",
            "### Documentation by workflow",
            "### Documentation by type",
        ],
        label="src/dnadesign/notify/docs/README.md",
    )
    assert "notify send contract" in text
    assert "Runtime evidence pointers" in text
    assert "../../../../docs/notify/usr-events.md" in text


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


def test_notify_maintainer_docs_use_deps_package_paths_not_removed_monolith() -> None:
    docs_readme = _read(_repo_root() / "docs" / "notify" / "README.md")

    assert "src/dnadesign/notify/docs/README.md" in docs_readme
    architecture = _read(_repo_root() / "src" / "dnadesign" / "notify" / "docs" / "dev" / "architecture.md")
    assert "src/dnadesign/notify/cli/bindings/deps/" in architecture
    assert "src/dnadesign/notify/cli/bindings/deps.py" not in docs_readme
    assert "src/dnadesign/notify/cli/bindings/deps.py" not in architecture
