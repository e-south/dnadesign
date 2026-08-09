"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_docs_contract.py

Documentation routing contracts for LatentDNA.

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


_TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".svg", ".html"}


def _is_scan_text_file(path: Path) -> bool:
    return "outputs" not in path.parts and "tests" not in path.parts and path.suffix.lower() in _TEXT_SUFFIXES


def test_latentdna_readme_routes_to_reference_first_docs() -> None:
    repo_root = _repo_root()
    root = repo_root / "src/dnadesign/latentdna"
    readme = (root / "README.md").read_text(encoding="utf-8")
    docs_index = (root / "docs/README.md").read_text(encoding="utf-8")
    cli_contracts = (root / "docs/reference/cli-contracts.md").read_text(encoding="utf-8")
    workspace_schema = (root / "docs/reference/workspace-schema.md").read_text(encoding="utf-8")
    reference_index = (root / "docs/reference/README.md").read_text(encoding="utf-8")
    operations = (root / "docs/operations/README.md").read_text(encoding="utf-8")

    assert "LatentDNA compares learned sequence representations" in readme
    assert "docs/reference/cli-contracts.md" in readme
    assert "docs/reference/workspace-schema.md" in readme
    assert "docs/integrations/study-workspaces.md" in readme

    assert "workflows/context-shift.md" in docs_index
    assert "workflows/cross-view-agreement.md" in docs_index
    assert "workflows/export-opal-x.md" in docs_index
    assert "reference/workspace-snapshot-contract.md" in docs_index
    assert "reference/artifact-naming.md" in docs_index

    assert "`latentdna workspace snapshot`" in cli_contracts
    assert "`latentdna.workspace_snapshot.v1`" in cli_contracts
    assert "Nested output roots are rejected." in cli_contracts

    assert "<workspace>/outputs" in workspace_schema
    assert "study_id" in workspace_schema
    assert "record_root" in workspace_schema
    assert "docs/studies/" not in workspace_schema

    assert "Workspace snapshot contract" in reference_index
    assert "Artifact naming grammar" in reference_index
    assert "ops/status.registry.yaml" in operations


def test_latentdna_docs_do_not_embed_private_study_surfaces() -> None:
    root = _repo_root() / "src/dnadesign/latentdna"
    forbidden_tokens = {
        "docs/studies/",
        "src/dnadesign/studies/",
    }
    checked_files = [path for path in root.rglob("*") if path.is_file() and _is_scan_text_file(path)]

    for forbidden in forbidden_tokens:
        hits = [
            path.as_posix() for path in checked_files if forbidden in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert hits == [], f"private study token {forbidden!r} present in: {hits}"
