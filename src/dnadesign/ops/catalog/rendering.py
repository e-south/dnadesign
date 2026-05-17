"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/rendering.py

Markdown rendering helpers for the Ops runbook catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .constants import (
    PROCEDURES_SECTION_HEADING,
    PROCEDURES_SECTION_INTRO,
    TOOL_SOURCES_SECTION_HEADING,
    TOOL_SOURCES_SECTION_INTRO,
)
from .loader import load_runbook_catalog
from .models import RunbookCatalog


def render_catalog_procedure_section(catalog: RunbookCatalog) -> str:
    lines = [
        PROCEDURES_SECTION_INTRO,
        "",
        "| Registry id | Procedure | Type | Plane | Execution kind | Status kind | Summary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in catalog.procedures:
        lines.append(
            "| "
            f"`{entry.registry_id}` | "
            f"[{entry.title}]({entry.doc_path}) | "
            f"`{entry.entry_type}` | "
            f"`{entry.plane}` | "
            f"`{entry.execution_kind}` | "
            f"`{entry.status_kind}` | "
            f"{entry.summary} |"
        )
    return "\n".join(lines)


def render_catalog_tool_source_section(catalog: RunbookCatalog) -> str:
    lines = [
        TOOL_SOURCES_SECTION_INTRO,
        "",
        "| Tool | Docs entrypoint | What you will find |",
        "| --- | --- | --- |",
    ]
    for entry in catalog.tool_sources:
        lines.append(f"| `{entry.tool}` | [{entry.title}]({entry.doc_path}) | {entry.summary} |")
    return "\n".join(lines)


def rewrite_runbook_catalog_sections(*, repo_root: Path | None = None) -> Path:
    catalog = load_runbook_catalog(repo_root=repo_root)
    catalog_path = catalog.catalog_path
    catalog_text = catalog_path.read_text(encoding="utf-8")
    rendered_procedure_section = render_catalog_procedure_section(catalog)
    rendered_tool_source_section = render_catalog_tool_source_section(catalog)
    updated_text = _replace_markdown_section(
        text=catalog_text,
        heading=PROCEDURES_SECTION_HEADING,
        body=rendered_procedure_section,
    )
    updated_text = _replace_markdown_section(
        text=updated_text,
        heading=TOOL_SOURCES_SECTION_HEADING,
        body=rendered_tool_source_section,
    )
    catalog_path.write_text(updated_text.rstrip() + "\n", encoding="utf-8")
    return catalog_path


def rewrite_runbook_catalog_procedure_section(*, repo_root: Path | None = None) -> Path:
    return rewrite_runbook_catalog_sections(repo_root=repo_root)


def _replace_markdown_section(*, text: str, heading: str, body: str) -> str:
    lines = text.splitlines()
    try:
        heading_index = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as exc:
        raise ValueError(f"runbook catalog missing section: {heading}") from exc

    end_index = len(lines)
    for index in range(heading_index + 1, len(lines)):
        if lines[index].strip().startswith("### "):
            end_index = index
            break
    replacement = [lines[heading_index], "", *body.rstrip().splitlines()]
    if end_index < len(lines):
        replacement.append("")
    updated_lines = [*lines[:heading_index], *replacement, *lines[end_index:]]
    return "\n".join(updated_lines)
