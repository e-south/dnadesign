"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/commands/progress_render.py

Text, JSON, and YAML renderers for OPS progress commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml

from dnadesign.ops.catalog import CatalogProcedureEntry, repo_relative_catalog_doc_path
from dnadesign.ops.cli.common import render_command
from dnadesign.ops.cli.dynamic_inputs import render_progress_show_command

from .progress_status_specs import (
    list_status_kind_specs,
    load_status_kind_spec,
    status_notes,
    status_optional_inputs,
    status_required_inputs,
)

if TYPE_CHECKING:
    from dnadesign.ops.status import CampaignScaffold, CampaignStatus, ProcedureStatus


def emit_progress_show_text(*, repo_root: Path, catalog_path: Path, result: ProcedureStatus) -> None:
    doc_path = repo_relative_catalog_doc_path(
        repo_root=repo_root,
        catalog_path=catalog_path,
        doc_path=result.doc_path,
    )
    lines = [
        f"Registry id: {result.registry_id}",
        f"Procedure: {result.title}",
        f"Doc: {doc_path}",
        f"Owner boundary: {result.owner_boundary}",
        f"Status kind: {result.status_kind}",
        f"Observes plane: {result.observes_plane}",
        f"Surface type: {result.surface_type}",
        f"Cost class: {result.cost_class}",
        f"Summary scope: {result.summary_scope}",
        f"State: {result.state}",
        f"Summary: {result.summary}",
        "Evidence:",
    ]
    for key, value in result.evidence.items():
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        lines.append(f"- {key}: {rendered}")
    typer.echo("\n".join(lines))


def emit_progress_show_json(*, repo_root: Path, catalog_path: Path, result: ProcedureStatus) -> None:
    payload = result.as_dict()
    payload["doc_path"] = repo_relative_catalog_doc_path(
        repo_root=repo_root,
        catalog_path=catalog_path,
        doc_path=result.doc_path,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def emit_progress_explain_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    entry: CatalogProcedureEntry,
    owner_boundary: str,
    has_related_routes: bool,
) -> None:
    spec = load_status_kind_spec(entry.status_kind)
    required_inputs = status_required_inputs(entry.status_kind)
    optional_inputs = status_optional_inputs(entry.status_kind)
    lines = [
        f"Registry id: {entry.registry_id}",
        f"Procedure: {entry.title}",
        "Doc: "
        + repo_relative_catalog_doc_path(
            repo_root=repo_root,
            catalog_path=catalog_path,
            doc_path=entry.doc_path,
        ),
        f"Owner boundary: {owner_boundary}",
        f"Status kind: {entry.status_kind}",
        f"Observes plane: {spec.observes_plane}",
        f"Surface type: {spec.surface_type}",
        f"Cost class: {spec.cost_class}",
        f"Summary scope: {spec.summary_scope}",
        f"Provider: {spec.provider_id}",
        f"What this status reads: {spec.description}",
        "Required inputs:",
    ]
    if required_inputs:
        for field in required_inputs:
            lines.append(f"- {field.cli_flag} {field.placeholder}: {field.summary}")
    else:
        lines.append("- none")
    if optional_inputs:
        lines.append("Also accepted:")
        for flag, summary in optional_inputs:
            lines.append(f"- {flag}: {summary}")
    lines.append("Next commands:")
    lines.append(f"- catalog_show: {render_command(['uv', 'run', 'ops', 'catalog', 'show', entry.registry_id])}")
    progress_show_command = render_progress_show_command(
        registry_id=entry.registry_id,
        required_inputs=required_inputs,
    )
    lines.append(f"- progress_show: {progress_show_command}")
    lines.append(
        f"- progress_scaffold: {render_command(['uv', 'run', 'ops', 'progress', 'scaffold', entry.registry_id])}"
    )
    if has_related_routes:
        lines.append(
            "- progress_scaffold_related: "
            + render_command(["uv", "run", "ops", "progress", "scaffold", "--related-to", entry.registry_id])
        )
    notes = status_notes(entry)
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
    typer.echo("\n".join(lines))


def emit_progress_explain_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    entry: CatalogProcedureEntry,
    owner_boundary: str,
    has_related_routes: bool,
) -> None:
    spec = load_status_kind_spec(entry.status_kind)
    required_inputs = status_required_inputs(entry.status_kind)
    optional_inputs = status_optional_inputs(entry.status_kind)
    payload = {
        "registry_id": entry.registry_id,
        "title": entry.title,
        "doc_path": repo_relative_catalog_doc_path(
            repo_root=repo_root,
            catalog_path=catalog_path,
            doc_path=entry.doc_path,
        ),
        "owner_boundary": owner_boundary,
        "status_kind": entry.status_kind,
        "observes_plane": spec.observes_plane,
        "surface_type": spec.surface_type,
        "cost_class": spec.cost_class,
        "summary_scope": spec.summary_scope,
        "provider_id": spec.provider_id,
        "description": spec.description,
        "required_inputs": [field.as_dict() for field in required_inputs],
        "optional_inputs": [{"cli_flag": flag, "summary": summary} for flag, summary in optional_inputs],
        "next_commands": {
            "catalog_show": render_command(["uv", "run", "ops", "catalog", "show", entry.registry_id]),
            "progress_show": render_progress_show_command(
                registry_id=entry.registry_id,
                required_inputs=required_inputs,
            ),
            "progress_scaffold": render_command(["uv", "run", "ops", "progress", "scaffold", entry.registry_id]),
        },
        "notes": list(status_notes(entry)),
    }
    if has_related_routes:
        payload["next_commands"]["progress_scaffold_related"] = render_command(
            ["uv", "run", "ops", "progress", "scaffold", "--related-to", entry.registry_id]
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def emit_status_kinds_text() -> None:
    lines = [
        "Status kinds",
        "Use `ops catalog list --simple` for the public registry routes that rely on these underlying status kinds.",
    ]
    for spec in list_status_kind_specs():
        lines.append(f"- {spec.status_kind} [{spec.provider_id}]")
        lines.append(f"  {spec.description}")
        lines.append(
            "  Ontology: "
            f"plane={spec.observes_plane} "
            f"surface={spec.surface_type} "
            f"scope={spec.summary_scope} "
            f"cost={spec.cost_class}"
        )
        if spec.required_inputs:
            rendered_required = ", ".join(f"{field.cli_flag} {field.placeholder}" for field in spec.required_inputs)
            lines.append(f"  Required inputs: {rendered_required}")
        else:
            lines.append("  Required inputs: none")
        if spec.optional_inputs:
            rendered_optional = ", ".join(field.cli_flag for field in spec.optional_inputs)
            lines.append(f"  Optional inputs: {rendered_optional}")
    typer.echo("\n".join(lines))


def emit_status_kinds_json() -> None:
    payload = {"status_kinds": [spec.as_inventory_dict() for spec in list_status_kind_specs()]}
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def emit_campaign_progress_text(*, repo_root: Path, catalog_path: Path, result: CampaignStatus) -> None:
    counts = result.counts()
    lines = [
        "Campaign status",
        f"Campaign id: {result.campaign_id}",
        f"Manifest: {result.manifest_path}",
        f"Overall state: {result.overall_state()}",
        f"Counts: ok={counts['ok']} attention={counts['attention']} missing={counts['missing']}",
        "",
        "Steps",
    ]
    for step in result.steps:
        heading = f"- {step.label}: {step.registry_id}" if step.label else f"- {step.registry_id}"
        lines.append(f"{heading} [{step.state} | {step.status_kind}]")
        lines.append(f"  {step.summary}")
        lines.append(
            "  Doc: "
            + repo_relative_catalog_doc_path(
                repo_root=repo_root,
                catalog_path=catalog_path,
                doc_path=step.doc_path,
            )
        )
    typer.echo("\n".join(lines))


def emit_campaign_progress_json(*, repo_root: Path, catalog_path: Path, result: CampaignStatus) -> None:
    payload = result.as_dict()
    payload["steps"] = [
        {
            **step.as_dict(),
            "doc_path": repo_relative_catalog_doc_path(
                repo_root=repo_root,
                catalog_path=catalog_path,
                doc_path=step.doc_path,
            ),
        }
        for step in result.steps
    ]
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def render_progress_scaffold_yaml(*, result: CampaignScaffold) -> str:
    return yaml.safe_dump(result.as_manifest_dict(), sort_keys=False)


def emit_progress_scaffold_json(*, repo_root: Path, catalog_path: Path, result: CampaignScaffold) -> None:
    payload = result.as_dict()
    payload["steps"] = [
        {
            **step.as_dict(),
            "doc_path": repo_relative_catalog_doc_path(
                repo_root=repo_root,
                catalog_path=catalog_path,
                doc_path=step.doc_path,
            ),
        }
        for step in result.steps
    ]
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "emit_campaign_progress_json",
    "emit_campaign_progress_text",
    "emit_progress_explain_json",
    "emit_progress_explain_text",
    "emit_progress_scaffold_json",
    "emit_progress_show_json",
    "emit_progress_show_text",
    "emit_status_kinds_json",
    "emit_status_kinds_text",
    "render_progress_scaffold_yaml",
]
