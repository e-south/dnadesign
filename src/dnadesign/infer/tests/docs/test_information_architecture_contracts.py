"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py

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


def _markdown_targets(text: str) -> list[str]:
    import re

    return re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)


def test_infer_top_readme_routes_to_guide_and_first_job() -> None:
    readme = _read("src/dnadesign/infer/README.md")
    links = _markdown_targets(readme)
    assert links[0] == "docs/README.md"
    assert "docs/getting-started/cli-quickstart.md" in links
    assert "docs/reference/README.md" in links
    assert "../../../docs/README.md" in links


def test_infer_guide_prioritizes_first_job_before_runtime_and_maintenance() -> None:
    guide = _read("src/dnadesign/infer/docs/README.md")
    links = _markdown_targets(guide)
    quickstart = "getting-started/cli-quickstart.md"
    for destination in (
        "operations/evo2-sequence-features.md",
        "operations/pressure-test-agnostic-models.md",
        "operations/scc-evo2-gpu-uv-runbook.md",
        "dev/README.md",
        "architecture/README.md",
    ):
        assert links.index(quickstart) < links.index(destination)
    assert "../workspaces/README.md" in links
    assert "reference/command-contracts.md" in links
    assert "reference/feature-schema.md" in links
    assert "../../cluster/docs/workflows/exploratory-clustering.md" in links
    assert "../../opal/docs/workflows/seqledger-infer-x-active-learning.md" in links
    assert "dev/journal.md" not in guide


def test_infer_section_indexes_offer_return_routes() -> None:
    for relative in (
        "index.md",
        "getting-started/README.md",
        "operations/README.md",
        "reference/README.md",
        "tutorials/README.md",
        "dev/README.md",
        "architecture/README.md",
    ):
        path = _repo_root() / "src/dnadesign/infer/docs" / relative
        targets = _markdown_targets(path.read_text())
        guide = _repo_root() / "src/dnadesign/infer/docs/README.md"
        assert any((path.parent / target.split("#")[0]).resolve() == guide for target in targets)


def test_infer_tutorial_routes_execution_to_one_procedure() -> None:
    tutorial = _read("src/dnadesign/infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md")
    assert "../operations/pressure-test-agnostic-models.md" in _markdown_targets(tutorial)
    assert "--no-submit" in tutorial
    assert "--submit" in tutorial
    # Keep executable setup in the procedure rather than two drifting copies.
    assert "infer workspace init" not in tutorial
    assert "ops runbook execute" not in tutorial


def test_infer_workspace_guide_links_validation_and_lists_inventory_command() -> None:
    workspaces = _read("src/dnadesign/infer/workspaces/README.md")
    assert "infer workspace list" in workspaces
    assert "../docs/getting-started/cli-quickstart.md" in _markdown_targets(workspaces)


def test_infer_command_reference_uses_supported_ingest_discriminators() -> None:
    from dnadesign.infer.src.config import IngestConfig

    reference = _read("src/dnadesign/infer/docs/reference/command-contracts.md")
    sources = IngestConfig.model_json_schema()["properties"]["source"]["enum"]
    for source in sources:
        assert f"| `{source}` |" in reference
    assert "| `seqledger` |" not in reference


def test_infer_feature_schema_documents_causal_pooling_semantics() -> None:
    feature_schema = _read("src/dnadesign/infer/docs/reference/feature-schema.md")

    assert "token-position" in feature_schema
    assert "prefix-conditioned on the emitted sequence" in feature_schema
    assert "not a native bidirectional Evo2 hidden state" in feature_schema


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


def test_infer_docs_examples_and_workspaces_avoid_hardcoded_personal_usr_roots() -> None:
    repo_root = _repo_root()
    targets = [
        repo_root / "src/dnadesign/infer/docs",
        repo_root / "src/dnadesign/infer/workspaces",
    ]
    offenders: list[str] = []
    for target in targets:
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == (repo_root / "src/dnadesign/infer/docs/dev/journal.md").resolve():
                continue
            if path.suffix not in {".md", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "/projectnb/private-user/outputs/usr_datasets" in text:
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []
