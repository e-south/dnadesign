"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_docs_information_architecture_contracts.py

Information-architecture contract tests for infer docs progressive disclosure.

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


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


def test_infer_top_readme_is_lightweight_router() -> None:
    readme = _read("src/dnadesign/infer/README.md")

    _assert_token_order(
        readme,
        [
            "## Documentation map",
            "## Entrypoint contract",
            "## Boundary reminder",
        ],
        label="src/dnadesign/infer/README.md",
    )
    assert "docs/README.md" in readme
    assert "docs/index.md" in readme
    assert "docs/getting-started/cli-quickstart.md" in readme
    assert "workspaces/README.md" in readme
    assert "docs/operations/pressure-test-agnostic-models.md" in readme
    assert "docs/tutorials/demo_pressure_test_usr_ops_notify.md" in readme
    assert "docs/reference/README.md" in readme
    assert "docs/dev/README.md" in readme
    assert "### CLI Quick Reference" not in readme
    assert "### Python API" not in readme
    assert "### Extending Presets" not in readme


def test_infer_docs_readme_keeps_workflow_then_type_progressive_disclosure() -> None:
    docs_readme = _read("src/dnadesign/infer/docs/README.md")

    _assert_token_order(
        docs_readme,
        [
            "### Read order",
            "### Documentation by workflow",
            "### Documentation by type",
        ],
        label="src/dnadesign/infer/docs/README.md",
    )
    assert "getting-started/README.md" in docs_readme
    assert "getting-started/cli-quickstart.md" in docs_readme
    assert "operations/README.md" in docs_readme
    assert "../workspaces/README.md" in docs_readme
    assert "operations/pressure-test-agnostic-models.md" in docs_readme
    assert "tutorials/demo_pressure_test_usr_ops_notify.md" in docs_readme
    assert "reference/README.md" in docs_readme
    assert "architecture/README.md" in docs_readme
    assert "dev/README.md" in docs_readme
    assert "dev/journal.md" in docs_readme


def test_infer_docs_index_exists_and_points_back_to_docs_readme() -> None:
    docs_index = _read("src/dnadesign/infer/docs/index.md")

    assert "docs/README.md" in docs_index or "README.md" in docs_index
    assert "### Getting started" in docs_index
    assert "### Tutorials" in docs_index
    assert "### Operations" in docs_index
    assert "### Reference" in docs_index
    assert "### Developer notes" in docs_index


def test_infer_operations_index_links_pressure_test_demo_and_runbook() -> None:
    ops_index = _read("src/dnadesign/infer/docs/operations/README.md")
    assert "pressure-test-agnostic-models.md" in ops_index
    assert "../tutorials/demo_pressure_test_usr_ops_notify.md" in ops_index


def test_infer_pressure_test_tutorial_covers_local_and_ops_paths() -> None:
    tutorial = _read("src/dnadesign/infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md")
    assert "uv run infer validate config --config" in tutorial
    assert "uv run infer workspace init --id test_stress_ethanol --profile usr-pressure" in tutorial
    assert "uv run infer run --config" in tutorial
    assert "uv run ops runbook init" in tutorial
    assert "uv run ops runbook execute" in tutorial
    assert "--no-submit" in tutorial
    assert "--submit" in tutorial
    assert "uv run usr --root" in tutorial


def test_infer_docs_excluding_journal_avoid_legacy_flat_module_paths() -> None:
    docs_root = _repo_root() / "src" / "dnadesign" / "infer" / "docs"
    legacy_tokens = [
        "src/dnadesign/infer/adapter_dispatch.py",
        "src/dnadesign/infer/adapter_runtime.py",
        "src/dnadesign/infer/cli_builders.py",
        "src/dnadesign/infer/cli_ingest.py",
        "src/dnadesign/infer/cli_requests.py",
        "src/dnadesign/infer/tests/test_",
    ]
    offenders: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        if path.resolve() == (docs_root / "dev" / "journal.md").resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in legacy_tokens):
            offenders.append(str(path.relative_to(_repo_root())))
    assert offenders == []
