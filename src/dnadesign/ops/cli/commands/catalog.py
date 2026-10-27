"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/commands/catalog.py

Direct catalog command implementation for the OPS discovery plane.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Sequence

import typer
import typer.main

from dnadesign.ops.catalog import (
    CatalogProcedureDetails,
    CatalogProcedureEntry,
    CatalogQuery,
    CatalogToolSourceEntry,
    RunbookCatalog,
    filter_runbook_catalog,
    load_catalog_procedure_details,
    load_catalog_related_tool_routes,
    load_catalog_related_tool_sources,
    load_runbook_catalog,
    repo_relative_catalog_doc_path,
)
from dnadesign.ops.cli.common import (
    append_registry_suggestions,
    normalize_optional_filter,
    raise_contract_error,
    render_command,
)
from dnadesign.ops.cli.dynamic_inputs import render_progress_show_command

if TYPE_CHECKING:
    from dnadesign.ops.status import InputFieldSpec, StatusKindSpec


app = typer.Typer(
    help=(
        "Discovery commands for the shared runbook catalog. "
        "Start with `ops catalog list --simple`, `ops catalog list`, or `ops catalog list --query <term>`."
    )
)


def get_click_command():
    return typer.main.get_command(app)


def _load_status_kind_spec(status_kind: str) -> StatusKindSpec:
    from dnadesign.ops.status.registry_loader import load_status_kind_spec

    return load_status_kind_spec(status_kind)


def _status_required_inputs(status_kind: str) -> tuple[InputFieldSpec, ...]:
    return _load_status_kind_spec(status_kind).required_inputs


def _catalog_counts(
    *,
    procedures: Sequence[CatalogProcedureEntry],
    tool_sources: Sequence[CatalogToolSourceEntry],
    section: Literal["all", "procedures", "tool-sources"],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if section in {"all", "procedures"}:
        counts["procedures"] = len(procedures)
    if section in {"all", "tool-sources"}:
        counts["tool_sources"] = len(tool_sources)
    return counts


def _render_catalog_counts(
    *,
    procedures: Sequence[CatalogProcedureEntry],
    tool_sources: Sequence[CatalogToolSourceEntry],
    section: Literal["all", "procedures", "tool-sources"],
) -> str:
    counts = _catalog_counts(procedures=procedures, tool_sources=tool_sources, section=section)
    parts: list[str] = []
    if "procedures" in counts:
        noun = "procedure" if counts["procedures"] == 1 else "procedures"
        parts.append(f"{counts['procedures']} cross-tool {noun}")
    if "tool_sources" in counts:
        noun = "source" if counts["tool_sources"] == 1 else "sources"
        parts.append(f"{counts['tool_sources']} tool-local {noun}")
    return "Counts: " + ", ".join(parts)


def _render_catalog_filters(filters: CatalogQuery) -> str | None:
    rendered = ", ".join(f"{name}={value}" for name, value in filters.as_dict().items())
    if not rendered:
        return None
    return "Filters: " + rendered


def _catalog_query_is_broad(filters: CatalogQuery) -> bool:
    return all(
        value is None
        for value in (
            filters.query,
            filters.entry_type,
            filters.plane,
            filters.execution_kind,
            filters.status_kind,
            filters.related_to,
            filters.tool,
        )
    )


def _catalog_list_next_steps(
    *,
    repo_root: Path,
    catalog_path: Path,
    procedures: Sequence[CatalogProcedureEntry],
    tool_sources: Sequence[CatalogToolSourceEntry],
    section: Literal["all", "procedures", "tool-sources"],
    filters: CatalogQuery,
) -> tuple[tuple[str, str], ...]:
    next_steps: list[tuple[str, str]] = []
    visible_procedures = procedures if section in {"all", "procedures"} else ()
    visible_tool_sources = tool_sources if section in {"all", "tool-sources"} else ()
    first_procedure = visible_procedures[0] if visible_procedures else None
    first_tool_source = visible_tool_sources[0] if visible_tool_sources else None
    broad_inventory = _catalog_query_is_broad(filters)
    multiple_visible_matches = len(visible_procedures) + len(visible_tool_sources) > 1

    if first_procedure is not None:
        if broad_inventory and multiple_visible_matches:
            next_steps.append(
                (
                    "Narrow the inventory by topic",
                    render_command(["uv", "run", "ops", "catalog", "list", "--query", "<term>"]),
                )
            )
            next_steps.append(
                (
                    "Use the task-first view",
                    render_command(["uv", "run", "ops", "catalog", "list", "--simple"]),
                )
            )
        next_steps.append(
            (
                "Inspect the first matching procedure",
                render_command(["uv", "run", "ops", "catalog", "show", first_procedure.registry_id]),
            )
        )
        next_steps.append(
            (
                "See the required status inputs",
                render_command(["uv", "run", "ops", "progress", "explain", first_procedure.registry_id]),
            )
        )
        if filters.related_to:
            next_steps.append(
                (
                    "Start a manifest from this related route set",
                    render_command(["uv", "run", "ops", "progress", "scaffold", "--related-to", filters.related_to]),
                )
            )
        else:
            next_steps.append(
                (
                    "Emit a manifest skeleton for the first match",
                    render_command(["uv", "run", "ops", "progress", "scaffold", first_procedure.registry_id]),
                )
            )
        return tuple(next_steps)

    if first_tool_source is not None:
        if broad_inventory and len(visible_tool_sources) > 1:
            next_steps.append(
                (
                    "Narrow the docs by topic",
                    render_command(
                        ["uv", "run", "ops", "catalog", "list", "--section", "tool-sources", "--query", "<term>"]
                    ),
                )
            )
        if filters.related_to:
            next_steps.append(
                (
                    "Inspect the route behind these related tool docs",
                    render_command(["uv", "run", "ops", "catalog", "show", filters.related_to]),
                )
            )
        else:
            next_steps.append(
                (
                    "Browse all registered procedures",
                    render_command(["uv", "run", "ops", "catalog", "list", "--section", "procedures"]),
                )
            )
        next_steps.append(
            (
                "Read the first matching owner docs",
                repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=first_tool_source.doc_path,
                ),
            )
        )
        return tuple(next_steps)

    next_steps.append(("Browse the full inventory", render_command(["uv", "run", "ops", "catalog", "list"])))
    if section == "tool-sources":
        next_steps.append(
            (
                "Browse all registered procedures",
                render_command(["uv", "run", "ops", "catalog", "list", "--section", "procedures"]),
            )
        )
    else:
        next_steps.append(
            (
                "Browse tool docs only",
                render_command(["uv", "run", "ops", "catalog", "list", "--section", "tool-sources"]),
            )
        )
    return tuple(next_steps)


def _catalog_next_commands(
    *,
    entry: CatalogProcedureEntry,
    details: CatalogProcedureDetails,
    owner_tool_source: CatalogToolSourceEntry | None,
    related_tool_sources: Sequence[CatalogToolSourceEntry],
) -> tuple[tuple[str, str], ...]:
    required_inputs = _status_required_inputs(entry.status_kind)
    commands: list[tuple[str, str]] = [
        (
            "progress_explain",
            render_command(["uv", "run", "ops", "progress", "explain", entry.registry_id]),
        ),
        (
            "progress_show",
            render_progress_show_command(registry_id=entry.registry_id, required_inputs=required_inputs),
        ),
        (
            "progress_scaffold",
            render_command(["uv", "run", "ops", "progress", "scaffold", entry.registry_id]),
        ),
    ]
    if owner_tool_source is not None:
        commands.append(
            (
                "catalog_owner_tool_source",
                render_command(
                    [
                        "uv",
                        "run",
                        "ops",
                        "catalog",
                        "list",
                        "--section",
                        "tool-sources",
                        "--tool",
                        owner_tool_source.tool,
                    ]
                ),
            )
        )
    if related_tool_sources:
        commands.append(
            (
                "catalog_related_tool_sources",
                render_command(
                    [
                        "uv",
                        "run",
                        "ops",
                        "catalog",
                        "list",
                        "--section",
                        "tool-sources",
                        "--related-to",
                        entry.registry_id,
                    ]
                ),
            )
        )
    if details.related_registry_ids:
        commands.extend(
            (
                (
                    "catalog_related",
                    render_command(
                        [
                            "uv",
                            "run",
                            "ops",
                            "catalog",
                            "list",
                            "--section",
                            "procedures",
                            "--related-to",
                            entry.registry_id,
                        ]
                    ),
                ),
                (
                    "progress_scaffold_related",
                    render_command(
                        [
                            "uv",
                            "run",
                            "ops",
                            "progress",
                            "scaffold",
                            "--related-to",
                            entry.registry_id,
                        ]
                    ),
                ),
            )
        )
    return tuple(commands)


def _emit_catalog_list_simple_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    procedures: Sequence[CatalogProcedureEntry],
    tool_sources: Sequence[CatalogToolSourceEntry],
    section: Literal["all", "procedures", "tool-sources"],
    filters: CatalogQuery,
) -> None:
    lines: list[str] = [
        "Catalog inventory",
        _render_catalog_counts(procedures=procedures, tool_sources=tool_sources, section=section),
    ]
    rendered_filters = _render_catalog_filters(filters)
    if rendered_filters is not None:
        lines.append(rendered_filters)
    lines.append("")
    if section in {"all", "procedures"}:
        lines.append("Task-first procedures")
        for entry in procedures:
            lines.append(f"- {entry.summary}")
            lines.append(f"  Registry id: {entry.registry_id}")
            lines.append(f"  Inspect: {render_command(['uv', 'run', 'ops', 'catalog', 'show', entry.registry_id])}")
            lines.append(
                "  Doc: "
                + repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                )
            )
        if not procedures:
            lines.append("- none")
    if section == "all":
        lines.append("")
    if section in {"all", "tool-sources"}:
        lines.append("Tool docs")
        for entry in tool_sources:
            lines.append(f"- {entry.tool}: {entry.summary}")
            lines.append(
                "  Doc: "
                + repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                )
            )
        if not tool_sources:
            lines.append("- none")
    next_steps = _catalog_list_next_steps(
        repo_root=repo_root,
        catalog_path=catalog_path,
        procedures=procedures,
        tool_sources=tool_sources,
        section=section,
        filters=filters,
    )
    if next_steps:
        lines.append("")
        lines.append(
            "No matching catalog entries. Try:" if not procedures and not tool_sources else "Suggested next steps"
        )
        for label, value in next_steps:
            lines.append(f"- {label}: {value}")
    typer.echo("\n".join(lines))


def _emit_catalog_list_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    procedures: Sequence[CatalogProcedureEntry],
    tool_sources: Sequence[CatalogToolSourceEntry],
    section: Literal["all", "procedures", "tool-sources"],
    filters: CatalogQuery,
) -> None:
    lines: list[str] = [
        "Catalog inventory",
        _render_catalog_counts(procedures=procedures, tool_sources=tool_sources, section=section),
    ]
    rendered_filters = _render_catalog_filters(filters)
    if rendered_filters is not None:
        lines.append(rendered_filters)
    lines.append("")
    if section in {"all", "procedures"}:
        lines.append("Cross-tool procedures")
        for entry in procedures:
            status_summary = f"{entry.entry_type} | {entry.plane} | {entry.execution_kind} | {entry.status_kind}"
            lines.append(f"- {entry.registry_id} [{status_summary}]")
            lines.append(f"  {entry.summary}")
            lines.append(
                "  Doc: "
                + repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                )
            )
        if not procedures:
            lines.append("- none")
    if section == "all":
        lines.append("")
    if section in {"all", "tool-sources"}:
        lines.append("Tool docs")
        for entry in tool_sources:
            lines.append(f"- {entry.tool}: {entry.title}")
            lines.append(f"  {entry.summary}")
            lines.append(
                "  Doc: "
                + repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                )
            )
        if not tool_sources:
            lines.append("- none")
    next_steps = _catalog_list_next_steps(
        repo_root=repo_root,
        catalog_path=catalog_path,
        procedures=procedures,
        tool_sources=tool_sources,
        section=section,
        filters=filters,
    )
    if next_steps:
        lines.append("")
        lines.append(
            "No matching catalog entries. Try:" if not procedures and not tool_sources else "Suggested next steps"
        )
        for label, value in next_steps:
            lines.append(f"- {label}: {value}")
    typer.echo("\n".join(lines))


def _emit_catalog_list_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    procedures: Sequence[CatalogProcedureEntry],
    tool_sources: Sequence[CatalogToolSourceEntry],
    section: Literal["all", "procedures", "tool-sources"],
    filters: CatalogQuery,
    simple: bool,
) -> None:
    payload: dict[str, object] = {
        "section": section,
        "view": "simple" if simple else "full",
        "filters": filters.as_dict(),
        "counts": _catalog_counts(procedures=procedures, tool_sources=tool_sources, section=section),
        "next_steps": [
            {"label": label, "value": value}
            for label, value in _catalog_list_next_steps(
                repo_root=repo_root,
                catalog_path=catalog_path,
                procedures=procedures,
                tool_sources=tool_sources,
                section=section,
                filters=filters,
            )
        ],
    }
    if section in {"all", "procedures"}:
        payload["procedures"] = [
            {
                "registry_id": entry.registry_id,
                "title": entry.title,
                "doc_path": repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                ),
                "type": entry.entry_type,
                "plane": entry.plane,
                "execution_kind": entry.execution_kind,
                "status_kind": entry.status_kind,
                "summary": entry.summary,
            }
            for entry in procedures
        ]
    if section in {"all", "tool-sources"}:
        payload["tool_sources"] = [
            {
                "tool": entry.tool,
                "title": entry.title,
                "doc_path": repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                ),
                "summary": entry.summary,
                "keywords": list(entry.keywords),
            }
            for entry in tool_sources
        ]
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _emit_catalog_show_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    details: CatalogProcedureDetails,
    catalog: RunbookCatalog,
) -> None:
    entry = details.entry
    status_inputs = _status_required_inputs(entry.status_kind)
    owner_tool_source = catalog.find_tool_source(details.owner_boundary)
    related_tool_sources = load_catalog_related_tool_sources(catalog, entry.registry_id)
    related_tool_routes = load_catalog_related_tool_routes(catalog, entry.registry_id)
    next_commands = _catalog_next_commands(
        entry=entry,
        details=details,
        owner_tool_source=owner_tool_source,
        related_tool_sources=related_tool_sources,
    )
    lines = [
        f"Registry id: {entry.registry_id}",
        f"Procedure: {entry.title}",
        "Doc: "
        + repo_relative_catalog_doc_path(
            repo_root=repo_root,
            catalog_path=catalog_path,
            doc_path=entry.doc_path,
        ),
        f"Type: {entry.entry_type}",
        f"Plane: {entry.plane}",
        f"Owner boundary: {details.owner_boundary}",
        f"Entry artifact: {details.entry_artifact}",
        f"Exit artifact: {details.exit_artifact}",
        f"Execution kind: {entry.execution_kind}",
        f"Status kind: {entry.status_kind}",
        f"Summary: {entry.summary}",
    ]
    if owner_tool_source is not None:
        lines.extend(
            [
                "Owner docs:",
                f"- {owner_tool_source.tool}: {owner_tool_source.title}",
                f"  {owner_tool_source.summary}",
                "  Doc: "
                + repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=owner_tool_source.doc_path,
                ),
            ]
        )
    if related_tool_sources:
        lines.append("Related tool docs:")
        for related_tool in related_tool_sources:
            lines.extend(
                [
                    f"- {related_tool.tool}: {related_tool.title}",
                    f"  {related_tool.summary}",
                    "  Doc: "
                    + repo_relative_catalog_doc_path(
                        repo_root=repo_root,
                        catalog_path=catalog_path,
                        doc_path=related_tool.doc_path,
                    ),
                ]
            )
    if related_tool_routes:
        lines.append("Related deep docs:")
        for route in related_tool_routes:
            lines.extend(
                [
                    f"- {route.tool}/{route.route_id}: {route.title}",
                    f"  {route.summary}",
                    "  Doc: "
                    + repo_relative_catalog_doc_path(
                        repo_root=repo_root,
                        catalog_path=catalog_path,
                        doc_path=route.doc_path,
                    ),
                ]
            )
    lines.append("Required status inputs:")
    if status_inputs:
        for field in status_inputs:
            lines.append(f"- {field.cli_flag} {field.placeholder}: {field.summary}")
    else:
        lines.append("- none")
    if details.relations:
        lines.append("Related procedures:")
        for relation in details.relations:
            related_entry = catalog.find_procedure(relation.target_registry_id)
            if related_entry is None:
                continue
            status_summary = (
                f"{related_entry.entry_type} | {related_entry.plane} | "
                f"{related_entry.execution_kind} | {related_entry.status_kind}"
            )
            lines.append(f"- {relation.relation_type}: {related_entry.registry_id} [{status_summary}]")
            lines.append(f"  {related_entry.summary}")
    lines.append("Next commands:")
    for label, command in next_commands:
        lines.append(f"- {label}: {command}")
    typer.echo("\n".join(lines))


def _emit_catalog_show_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    details: CatalogProcedureDetails,
    catalog: RunbookCatalog,
) -> None:
    entry = details.entry
    status_inputs = _status_required_inputs(entry.status_kind)
    owner_tool_source = catalog.find_tool_source(details.owner_boundary)
    related_tool_sources = load_catalog_related_tool_sources(catalog, entry.registry_id)
    related_tool_routes = load_catalog_related_tool_routes(catalog, entry.registry_id)
    next_commands = _catalog_next_commands(
        entry=entry,
        details=details,
        owner_tool_source=owner_tool_source,
        related_tool_sources=related_tool_sources,
    )
    typer.echo(
        json.dumps(
            {
                "registry_id": entry.registry_id,
                "title": entry.title,
                "doc_path": repo_relative_catalog_doc_path(
                    repo_root=repo_root,
                    catalog_path=catalog_path,
                    doc_path=entry.doc_path,
                ),
                "type": entry.entry_type,
                "plane": entry.plane,
                "owner_boundary": details.owner_boundary,
                "entry_artifact": details.entry_artifact,
                "exit_artifact": details.exit_artifact,
                "execution_kind": entry.execution_kind,
                "status_kind": entry.status_kind,
                "required_status_inputs": [field.as_dict() for field in status_inputs],
                "summary": entry.summary,
                "owner_tool_source": (
                    {
                        "tool": owner_tool_source.tool,
                        "title": owner_tool_source.title,
                        "doc_path": repo_relative_catalog_doc_path(
                            repo_root=repo_root,
                            catalog_path=catalog_path,
                            doc_path=owner_tool_source.doc_path,
                        ),
                        "summary": owner_tool_source.summary,
                        "keywords": list(owner_tool_source.keywords),
                    }
                    if owner_tool_source is not None
                    else None
                ),
                "related_tool_sources": [
                    {
                        "tool": related_tool.tool,
                        "title": related_tool.title,
                        "doc_path": repo_relative_catalog_doc_path(
                            repo_root=repo_root,
                            catalog_path=catalog_path,
                            doc_path=related_tool.doc_path,
                        ),
                        "summary": related_tool.summary,
                        "keywords": list(related_tool.keywords),
                    }
                    for related_tool in related_tool_sources
                ],
                "related_tool_routes": [
                    {
                        "tool": route.tool,
                        "route_id": route.route_id,
                        "title": route.title,
                        "doc_path": repo_relative_catalog_doc_path(
                            repo_root=repo_root,
                            catalog_path=catalog_path,
                            doc_path=route.doc_path,
                        ),
                        "summary": route.summary,
                    }
                    for route in related_tool_routes
                ],
                "next_commands": {name: command for name, command in next_commands},
                "related_procedures": [
                    {
                        "relation_type": relation.relation_type,
                        "registry_id": related_entry.registry_id,
                        "title": related_entry.title,
                        "doc_path": repo_relative_catalog_doc_path(
                            repo_root=repo_root,
                            catalog_path=catalog_path,
                            doc_path=related_entry.doc_path,
                        ),
                        "type": related_entry.entry_type,
                        "plane": related_entry.plane,
                        "execution_kind": related_entry.execution_kind,
                        "status_kind": related_entry.status_kind,
                        "summary": related_entry.summary,
                    }
                    for relation in details.relations
                    for related_entry in [catalog.find_procedure(relation.target_registry_id)]
                    if related_entry is not None
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("list")
def catalog_list(
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root containing docs/runbooks/README.md when invoking outside the repository.",
        ),
    ] = None,
    section: Annotated[
        Literal["all", "procedures", "tool-sources"],
        typer.Option(
            "--section",
            help="Which catalog section to show: cross-tool procedures, tool-local sources, or both.",
        ),
    ] = "all",
    entry_type: Annotated[
        str | None,
        typer.Option("--type", help="Exact Type filter for cross-tool procedures."),
    ] = None,
    plane: Annotated[
        str | None,
        typer.Option("--plane", help="Exact Plane filter for cross-tool procedures."),
    ] = None,
    execution_kind: Annotated[
        str | None,
        typer.Option("--execution-kind", help="Exact Execution-kind filter for cross-tool procedures."),
    ] = None,
    status_kind: Annotated[
        str | None,
        typer.Option("--status-kind", help="Exact status-kind filter for cross-tool procedures."),
    ] = None,
    related_to: Annotated[
        str | None,
        typer.Option(
            "--related-to",
            help="List only typed related procedures or typed related tool docs for one registered procedure.",
        ),
    ] = None,
    tool: Annotated[
        str | None,
        typer.Option("--tool", help="Exact tool filter for tool-local runbook sources."),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option(
            "--query", help="Case-insensitive token query across registry ids, titles, summaries, and doc paths."
        ),
    ] = None,
    simple: Annotated[
        bool,
        typer.Option(
            "--simple/--no-simple",
            help="Show a task-first plain-text view that hides type/plane taxonomy on first contact.",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json/--no-json", help="Emit machine-readable JSON instead of plain text."),
    ] = False,
) -> None:
    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        raise_contract_error(f"Catalog contract error: {exc}")

    normalized_related_to = normalize_optional_filter(related_to)
    if normalized_related_to is not None and catalog.find_procedure(normalized_related_to) is None:
        message = f"Catalog contract error: unknown --related-to registry id: {normalized_related_to}"
        message = append_registry_suggestions(message=message, catalog=catalog, registry_id=normalized_related_to)
        raise_contract_error(message)

    filters = CatalogQuery(
        query=normalize_optional_filter(query),
        entry_type=normalize_optional_filter(entry_type),
        plane=normalize_optional_filter(plane),
        execution_kind=normalize_optional_filter(execution_kind),
        status_kind=normalize_optional_filter(status_kind),
        related_to=normalized_related_to,
        tool=normalize_optional_filter(tool),
    )
    procedures, tool_sources = filter_runbook_catalog(catalog, query=filters)

    if as_json:
        _emit_catalog_list_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            procedures=procedures,
            tool_sources=tool_sources,
            section=section,
            filters=filters,
            simple=simple,
        )
        return
    if simple:
        _emit_catalog_list_simple_text(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            procedures=procedures,
            tool_sources=tool_sources,
            section=section,
            filters=filters,
        )
        return
    _emit_catalog_list_text(
        repo_root=catalog.repo_root,
        catalog_path=catalog.catalog_path,
        procedures=procedures,
        tool_sources=tool_sources,
        section=section,
        filters=filters,
    )


@app.command("show")
def catalog_show(
    registry_id: Annotated[str, typer.Argument(help="Cross-tool runbook or workflow registry id.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root containing docs/runbooks/README.md when invoking outside the repository.",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json/--no-json", help="Emit machine-readable JSON instead of plain text."),
    ] = False,
) -> None:
    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        raise_contract_error(f"Catalog contract error: {exc}")

    entry = catalog.find_procedure(registry_id)
    if entry is None:
        message = append_registry_suggestions(
            message=f"Catalog contract error: unknown registry id: {registry_id}",
            catalog=catalog,
            registry_id=registry_id,
        )
        raise_contract_error(message)
    try:
        details = load_catalog_procedure_details(catalog, entry)
    except ValueError as exc:
        raise_contract_error(f"Catalog contract error: {exc}")

    if as_json:
        _emit_catalog_show_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            details=details,
            catalog=catalog,
        )
        return
    _emit_catalog_show_text(
        repo_root=catalog.repo_root,
        catalog_path=catalog.catalog_path,
        details=details,
        catalog=catalog,
    )


__all__ = ["app", "get_click_command"]
