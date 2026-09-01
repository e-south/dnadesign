"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/check_test_support.py

Shared fixture builders for documentation-check tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from dnadesign.ops.catalog import (
    load_runbook_catalog,
    render_catalog_procedure_section,
    render_catalog_tool_source_section,
)

VALID_TOOL_BANNER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="180" viewBox="0 0 1200 180"></svg>\n'
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_changed_files(repo_root: Path, *relative_paths: str) -> Path:
    path = repo_root / "changed-files.txt"
    path.write_text("".join(f"{relative_path}\n" for relative_path in relative_paths), encoding="utf-8")
    return path


def _git_init(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)


def _git_add(repo_root: Path, *paths: str) -> None:
    subprocess.run(["git", "add", *paths], cwd=repo_root, check=True, capture_output=True, text=True)


def _write_registry_metadata(
    doc_path: Path,
    *,
    catalog_order: int,
    registry_id: str,
    entry_type: str,
    plane: str,
    owner_boundary: str,
    entry_artifact: str,
    exit_artifact: str,
    summary: str,
    execution_kind: str,
    status_kind: str,
    relations: list[dict[str, str]] | None = None,
) -> None:
    metadata_path = doc_path.with_name(f"{doc_path.stem}.registry.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "catalog_order": catalog_order,
                "registry_id": registry_id,
                "type": entry_type,
                "plane": plane,
                "owner_boundary": owner_boundary,
                "entry_artifact": entry_artifact,
                "exit_artifact": exit_artifact,
                "summary": summary,
                "execution_kind": execution_kind,
                "status_kind": status_kind,
                "relations": relations or [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_tool_source_metadata(
    doc_path: Path,
    *,
    catalog_order: int,
    tool: str,
    summary: str,
    keywords: list[str] | None = None,
) -> None:
    metadata_path = doc_path.with_name(f"{doc_path.stem}.tool-source.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "catalog_order": catalog_order,
                "tool": tool,
                "summary": summary,
                "keywords": keywords or [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_runbook_catalog_readme(
    repo_root: Path,
    *,
    procedure_section: str,
    tool_source_section: str,
    glossary_rows: list[str],
) -> None:
    _write(
        repo_root / "docs" / "runbooks" / "README.md",
        "\n".join(
            [
                "## Runbook Catalog",
                "",
                "### Cross-tool procedures",
                "",
                procedure_section,
                "",
                "### Tool docs",
                "",
                tool_source_section,
                "",
                "### Status views",
                "",
                "| Status kind | Meaning | Check next |",
                "| --- | --- | --- |",
                *glossary_rows,
            ]
        )
        + "\n",
    )


def _write_generated_runbook_catalog_readme(repo_root: Path, *, glossary_rows: list[str]) -> None:
    _write_runbook_catalog_readme(
        repo_root,
        procedure_section="_placeholder_",
        tool_source_section="_placeholder_",
        glossary_rows=glossary_rows,
    )
    catalog = load_runbook_catalog(repo_root=repo_root)
    _write_runbook_catalog_readme(
        repo_root,
        procedure_section=render_catalog_procedure_section(catalog),
        tool_source_section=render_catalog_tool_source_section(catalog),
        glossary_rows=glossary_rows,
    )


def _empty_tool_source_section() -> str:
    return "\n".join(
        [
            (
                "This table is generated from owner-local `*.tool-source.yaml` metadata sidecars. "
                "Edit those files instead of hand-editing rows here."
            ),
            "",
            "| Tool | Docs entrypoint | What you will find |",
            "| --- | --- | --- |",
        ]
    )
