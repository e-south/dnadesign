"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli.py

CLI for rendering deterministic batch orchestration plans from machine runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated, Literal, Sequence

import typer
import yaml
from pydantic import ValidationError

from .catalog import (
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
    suggest_procedure_registry_ids,
)
from .orchestrator.execute import execute_batch_plan
from .orchestrator.plan import build_batch_plan
from .orchestrator.state import discover_active_job_ids_for_runbook
from .progress import (
    CampaignProgress,
    CampaignScaffold,
    ProcedureProgress,
    ProgressFieldSpec,
    ProgressInputs,
    build_campaign_scaffold,
    build_procedure_progress,
    load_campaign_progress,
    load_progress_required_inputs,
)
from .runbooks.path_policy import (
    REPO_TRANSIENT_OPERATIONAL_DIR_NAMES,
    WORKSPACE_AUDIT_RELATIVE_DIR,
    WORKSPACE_RUNBOOKS_RELATIVE_DIR,
    WORKSPACE_SGE_STDOUT_RELATIVE_DIR,
)
from .runbooks.schema import load_orchestration_runbook
from .runbooks.workflow_metadata import resolve_workflow_id, resolve_workflow_tool

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help=(
        "Cross-tool orchestration commands for deterministic batch plans. "
        "Start with `uv run ops catalog list --simple` to browse routes from the terminal."
    ),
)

runbook_app = typer.Typer(help="Runbook contract commands.")
app.add_typer(runbook_app, name="runbook")
catalog_app = typer.Typer(
    help=(
        "Discovery commands for the shared runbook catalog. "
        "Start with `ops catalog list --simple`, `ops catalog list`, or `ops catalog list --query <term>`."
    )
)
app.add_typer(catalog_app, name="catalog")
progress_app = typer.Typer(
    help=(
        "Status inspection, status explanation, and manifest scaffold commands "
        "for registered runbooks and explicit campaigns. "
        "`show` and `campaign` are read-only; `scaffold` prints YAML unless `--out` is used."
    )
)
app.add_typer(progress_app, name="progress")


def _load_runbook_or_exit(runbook_path: Path):
    try:
        return load_orchestration_runbook(runbook_path.expanduser())
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _workspace_runbook_path_hint() -> str:
    return f"<workspace-root>/{WORKSPACE_RUNBOOKS_RELATIVE_DIR.as_posix()}/<runbook-id>.yaml"


def _contract_path(path: Path, *, runbook_parent: Path) -> str:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        return str(expanded)
    resolved = expanded.resolve()
    try:
        return str(resolved.relative_to(runbook_parent.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_workspace_root_for_init(workspace_root: Path, *, repo_base: Path) -> Path:
    expanded = workspace_root.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (repo_base / expanded).resolve()


def _resolve_repo_base(repo_root: Path | None) -> Path:
    if repo_root is None:
        return Path.cwd().resolve()
    return repo_root.expanduser().resolve()


def _render_notify_contract_warning(*, workspace_root: Path, notify_tool: str) -> str:
    profile_path = (workspace_root / "outputs" / "notify" / notify_tool / "profile.json").resolve()
    return (
        "Notify contract required before planning.\n"
        "Set NOTIFY_WEBHOOK_FILE to a readable file path, or configure "
        f"{profile_path} with webhook.source=secret_ref and a file:// secret reference."
    )


def _validate_runbook_output_path_for_init(*, runbook_path: Path, repo_base: Path) -> None:
    resolved_repo_base = repo_base.resolve()
    resolved_runbook = runbook_path.resolve()
    try:
        relative_to_repo = resolved_runbook.relative_to(resolved_repo_base)
    except ValueError:
        return
    if relative_to_repo.parent == Path("."):
        raise ValueError(f"runbook path must not be at repository root; use {_workspace_runbook_path_hint()}")
    for segment in REPO_TRANSIENT_OPERATIONAL_DIR_NAMES:
        if segment in relative_to_repo.parts:
            raise ValueError(f"runbook path must not use '{segment}'; use {_workspace_runbook_path_hint()}")


def _discover_repo_base_for_path(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    anchor = resolved if resolved.is_dir() else resolved.parent
    for parent in (anchor, *anchor.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "dnadesign").exists():
            return parent.resolve()
    return None


def _validate_runbook_input_path_for_runtime(*, runbook_path: Path, repo_base: Path) -> None:
    resolved_runbook = runbook_path.expanduser().resolve()
    discovered_repo_base = _discover_repo_base_for_path(resolved_runbook)
    resolved_repo_base = discovered_repo_base if discovered_repo_base is not None else repo_base
    _validate_runbook_output_path_for_init(runbook_path=resolved_runbook, repo_base=resolved_repo_base)


def _validate_audit_json_path_for_execute(*, audit_json_path: Path, workspace_root: Path) -> Path:
    resolved_audit_json = audit_json_path.expanduser().resolve()
    expected_audit_dir = (workspace_root / WORKSPACE_AUDIT_RELATIVE_DIR).resolve()
    if resolved_audit_json.parent != expected_audit_dir:
        raise ValueError(
            f"audit-json path must be exactly <workspace-root>/{WORKSPACE_AUDIT_RELATIVE_DIR.as_posix()}/<file>.json"
        )
    if resolved_audit_json.suffix.lower() != ".json":
        raise ValueError("audit-json file extension must be .json")
    return resolved_audit_json


def _build_init_payload(
    *,
    workflow: Literal["densegen", "infer"],
    with_notify: bool,
    runbook_id: str,
    project: str,
    workspace_root: Path,
    runbook_parent: Path,
    cuda_module: str,
    gcc_module: str,
    pe_omp: int | None,
    h_rt: str | None,
    mem_per_core: str | None,
    notify_qsub_template: str,
    densegen_qsub_template: str,
    densegen_post_run_qsub_template: str,
    infer_qsub_template: str,
) -> dict[str, object]:
    workspace_contract = Path(_contract_path(workspace_root, runbook_parent=runbook_parent))
    workflow_id = resolve_workflow_id(tool=workflow, with_notify=with_notify)
    payload: dict[str, object] = {
        "runbook": {
            "schema_version": 1,
            "id": runbook_id,
            "workflow_id": workflow_id,
            "project": project,
            "workspace_root": str(workspace_contract),
            "logging": {
                "stdout_dir": str(workspace_contract / WORKSPACE_SGE_STDOUT_RELATIVE_DIR / runbook_id),
                "retention": {
                    "keep_last": 20,
                    "max_age_days": 14,
                },
            },
            "mode_policy": {
                "default": "auto",
                "on_active_job": "hold_jid",
            },
        }
    }
    if with_notify:
        notify_tool = resolve_workflow_tool(workflow_id)
        notify_policy = "infer" if notify_tool == "infer" else "generic"
        payload["runbook"]["notify"] = {
            "tool": notify_tool,
            "policy": notify_policy,
            "profile": str(workspace_contract / "outputs" / "notify" / notify_tool / "profile.json"),
            "cursor": str(workspace_contract / "outputs" / "notify" / notify_tool / "cursor"),
            "spool_dir": str(workspace_contract / "outputs" / "notify" / notify_tool / "spool"),
            "webhook_env": "NOTIFY_WEBHOOK",
            "orchestration_events": True,
            "qsub_template": notify_qsub_template,
            "smoke": "dry",
        }
    if workflow == "densegen":
        payload["runbook"]["densegen"] = {
            "config": str(workspace_contract / "config.yaml"),
            "qsub_template": densegen_qsub_template,
            "run_args": {
                "fresh": "--fresh --no-plot",
                "resume": "--resume --no-plot",
            },
            "post_run": {
                "qsub_template": densegen_post_run_qsub_template,
            },
            "overlay_guard": {
                "max_projected_overlay_parts": 10000,
                "max_existing_overlay_parts": 1000,
                "auto_compact_existing_overlay_parts": True,
                "overlay_namespace": "densegen",
            },
            "records_part_guard": {
                "max_projected_records_parts": 10000,
                "max_existing_records_parts": 1000,
                "max_existing_records_part_age_days": 14,
                "auto_compact_existing_records_parts": True,
            },
            "archived_overlay_guard": {
                "max_archived_entries": 1000,
                "max_archived_bytes": 2147483648,
            },
        }
        payload["runbook"]["resources"] = {
            "pe_omp": pe_omp if pe_omp is not None else 12,
            "h_rt": h_rt or "08:00:00",
            "mem_per_core": mem_per_core or "8G",
        }
    else:
        payload["runbook"]["infer"] = {
            "config": str(workspace_contract / "config.yaml"),
            "qsub_template": infer_qsub_template,
            "cuda_module": cuda_module,
            "gcc_module": gcc_module,
        }
        payload["runbook"]["resources"] = {
            "pe_omp": pe_omp if pe_omp is not None else 4,
            "h_rt": h_rt or "04:00:00",
            "mem_per_core": mem_per_core or "8G",
            "gpus": 1,
            "gpu_capability": "8.9",
            "gpu_memory_gib": 45.0,
        }
    return payload


def _resolve_active_job_ids(
    *,
    runbook,
    active_job_ids: list[str],
    discover_active_jobs: bool,
    max_discovery_jobs: int,
) -> tuple[str, ...]:
    resolved_job_ids = _split_active_job_id_tokens(active_job_ids)
    if not discover_active_jobs:
        return tuple(dict.fromkeys(resolved_job_ids))

    try:
        discovered_job_ids = discover_active_job_ids_for_runbook(runbook, max_jobs=max_discovery_jobs)
    except RuntimeError as exc:
        typer.echo(f"Active-job discovery warning: {exc}", err=True)
        discovered_job_ids = ()

    for discovered in discovered_job_ids:
        if discovered not in resolved_job_ids:
            resolved_job_ids.append(discovered)
    return tuple(resolved_job_ids)


def _split_active_job_id_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for item in str(value).split(","):
            token = item.strip()
            if token:
                tokens.append(token)
    return tokens


def _render_active_job_hints(*, runbook_path: Path, active_job_ids: Sequence[str]) -> dict[str, object]:
    deduped_job_ids = tuple(dict.fromkeys(_split_active_job_id_tokens(active_job_ids)))
    csv_value = ",".join(deduped_job_ids)
    repeat_args = " ".join(f"--active-job-id {shlex.quote(job_id)}" for job_id in deduped_job_ids)
    runbook_arg = shlex.quote(str(runbook_path.expanduser()))
    if repeat_args:
        plan_hint = f"uv run ops runbook plan --runbook {runbook_arg} --no-discover-active-jobs {repeat_args}"
    else:
        plan_hint = f"uv run ops runbook plan --runbook {runbook_arg}"
    return {
        "active_job_count": len(deduped_job_ids),
        "active_job_ids_csv": csv_value,
        "active_job_id_args": repeat_args,
        "plan_command_hint": plan_hint,
    }


def _packaged_preset_paths() -> list[Path]:
    preset_dir = Path(__file__).resolve().parent / "runbooks" / "presets"
    if not preset_dir.exists():
        return []
    return sorted(path.resolve() for path in preset_dir.glob("*.yaml"))


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
            lines.append(f"  Inspect: {_render_command(['uv', 'run', 'ops', 'catalog', 'show', entry.registry_id])}")
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
        if not procedures and not tool_sources:
            lines.append("No matching catalog entries. Try:")
        else:
            lines.append("Suggested next steps")
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
            status_summary = f"{entry.entry_type} | {entry.plane} | {entry.execution_kind} | {entry.progress_kind}"
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
        if not procedures and not tool_sources:
            lines.append("No matching catalog entries. Try:")
        else:
            lines.append("Suggested next steps")
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
                "progress_kind": entry.progress_kind,
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
    progress_inputs = load_progress_required_inputs(entry.progress_kind)
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
        f"Progress kind: {entry.progress_kind}",
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
    lines.append("Required progress inputs:")
    if progress_inputs:
        for field in progress_inputs:
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
                f"{related_entry.execution_kind} | {related_entry.progress_kind}"
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
    progress_inputs = load_progress_required_inputs(entry.progress_kind)
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
                "progress_kind": entry.progress_kind,
                "progress_required_inputs": [field.as_dict() for field in progress_inputs],
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
                        "progress_kind": related_entry.progress_kind,
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


def _emit_progress_show_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    result: ProcedureProgress,
) -> None:
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
        f"Progress kind: {result.progress_kind}",
        f"State: {result.state}",
        f"Summary: {result.summary}",
        "Evidence:",
    ]
    for key, value in result.evidence.items():
        lines.append(f"- {key}: {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}")
    typer.echo("\n".join(lines))


def _emit_progress_show_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    result: ProcedureProgress,
) -> None:
    payload = result.as_dict()
    payload["doc_path"] = repo_relative_catalog_doc_path(
        repo_root=repo_root,
        catalog_path=catalog_path,
        doc_path=result.doc_path,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _progress_kind_description(progress_kind: str) -> str:
    descriptions = {
        "ops-audit-json": "Read one workspace-scoped orchestration audit JSON emitted by `ops runbook execute`.",
        "usr-sync-audit": "Read one USR sync audit JSON emitted by `usr diff`, `usr pull`, or `usr push`.",
        "usr-dataset-state": "Read one USR dataset directory, its records.parquet, and related overlay sidecars.",
        "cluster-run-index": "Read one cluster results root and summarize the run index for that workspace.",
        "opal-campaign-state": "Read one OPAL campaign workdir and summarize state.json plus round ledgers.",
    }
    return descriptions.get(progress_kind, "Read one explicit, artifact-backed status surface.")


def _progress_optional_inputs(progress_kind: str) -> tuple[tuple[str, str], ...]:
    if progress_kind == "opal-campaign-state":
        return (
            (
                "--opal-workdir",
                "Use when you want to point directly at the OPAL campaign workdir instead of resolving it from config.",
            ),
        )
    return ()


def _progress_notes(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    notes: list[str] = []
    if entry.progress_kind == "ops-audit-json":
        notes.append(
            "Smallest positive control-plane demo: run "
            "`uv run ops runbook execute ... --no-submit "
            "--audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json`, "
            "then pass the same audit path to `ops progress show`."
        )
        notes.append(
            "On workstations without `qstat`, add `--allow-missing-qstat` so the queue probe stays explicit "
            "but non-fatal during a dry-run demo."
        )
    if entry.progress_kind == "opal-campaign-state":
        notes.append(
            "Prefer `--opal-config` so Ops resolves `campaign.workdir` relative "
            "to the campaign root, matching OPAL's config contract."
        )
    return tuple(notes)


def _emit_progress_explain_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    entry: CatalogProcedureEntry,
    owner_boundary: str,
    has_related_routes: bool,
) -> None:
    required_inputs = load_progress_required_inputs(entry.progress_kind)
    optional_inputs = _progress_optional_inputs(entry.progress_kind)
    progress_show_command = _catalog_progress_show_command(
        registry_id=entry.registry_id,
        required_inputs=required_inputs,
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
        f"Owner boundary: {owner_boundary}",
        f"Progress kind: {entry.progress_kind}",
        f"What this status reads: {_progress_kind_description(entry.progress_kind)}",
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
    lines.append(f"- catalog_show: {_render_command(['uv', 'run', 'ops', 'catalog', 'show', entry.registry_id])}")
    lines.append(f"- progress_show: {progress_show_command}")
    lines.append(
        f"- progress_scaffold: {_render_command(['uv', 'run', 'ops', 'progress', 'scaffold', entry.registry_id])}"
    )
    if has_related_routes:
        lines.append(
            "- progress_scaffold_related: "
            + _render_command(["uv", "run", "ops", "progress", "scaffold", "--related-to", entry.registry_id])
        )
    notes = _progress_notes(entry)
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
    typer.echo("\n".join(lines))


def _emit_progress_explain_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    entry: CatalogProcedureEntry,
    owner_boundary: str,
    has_related_routes: bool,
) -> None:
    required_inputs = load_progress_required_inputs(entry.progress_kind)
    optional_inputs = _progress_optional_inputs(entry.progress_kind)
    payload = {
        "registry_id": entry.registry_id,
        "title": entry.title,
        "doc_path": repo_relative_catalog_doc_path(
            repo_root=repo_root,
            catalog_path=catalog_path,
            doc_path=entry.doc_path,
        ),
        "owner_boundary": owner_boundary,
        "progress_kind": entry.progress_kind,
        "description": _progress_kind_description(entry.progress_kind),
        "required_inputs": [field.as_dict() for field in required_inputs],
        "optional_inputs": [{"cli_flag": flag, "summary": summary} for flag, summary in optional_inputs],
        "next_commands": {
            "catalog_show": _render_command(["uv", "run", "ops", "catalog", "show", entry.registry_id]),
            "progress_show": _catalog_progress_show_command(
                registry_id=entry.registry_id,
                required_inputs=required_inputs,
            ),
            "progress_scaffold": _render_command(["uv", "run", "ops", "progress", "scaffold", entry.registry_id]),
        },
        "notes": list(_progress_notes(entry)),
    }
    if has_related_routes:
        payload["next_commands"]["progress_scaffold_related"] = _render_command(
            ["uv", "run", "ops", "progress", "scaffold", "--related-to", entry.registry_id]
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _emit_campaign_progress_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    result: CampaignProgress,
) -> None:
    counts = result.counts()
    lines = [
        "Campaign progress",
        f"Campaign id: {result.campaign_id}",
        f"Manifest: {result.manifest_path}",
        f"Overall state: {result.overall_state()}",
        f"Counts: ok={counts['ok']} attention={counts['attention']} missing={counts['missing']}",
        "",
        "Steps",
    ]
    for step in result.steps:
        label = step.label
        heading = f"- {label}: {step.registry_id}" if label else f"- {step.registry_id}"
        lines.append(f"{heading} [{step.state} | {step.progress_kind}]")
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


def _emit_campaign_progress_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    result: CampaignProgress,
) -> None:
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


def _emit_progress_scaffold_yaml(*, result: CampaignScaffold) -> str:
    return yaml.safe_dump(result.as_manifest_dict(), sort_keys=False)


def _emit_progress_scaffold_json(
    *,
    repo_root: Path,
    catalog_path: Path,
    result: CampaignScaffold,
) -> None:
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
    items = filters.as_dict().items()
    rendered = ", ".join(f"{name}={value}" for name, value in items)
    if not rendered:
        return None
    return "Filters: " + rendered


def _render_command(parts: Sequence[str]) -> str:
    return " ".join(parts)


def _catalog_query_is_broad(filters: CatalogQuery) -> bool:
    return all(
        value is None
        for value in (
            filters.query,
            filters.entry_type,
            filters.plane,
            filters.execution_kind,
            filters.progress_kind,
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
                    _render_command(["uv", "run", "ops", "catalog", "list", "--query", "<term>"]),
                )
            )
            next_steps.append(
                (
                    "Use the task-first view",
                    _render_command(["uv", "run", "ops", "catalog", "list", "--simple"]),
                )
            )
        next_steps.append(
            (
                "Inspect the first matching procedure",
                _render_command(["uv", "run", "ops", "catalog", "show", first_procedure.registry_id]),
            )
        )
        next_steps.append(
            (
                "See the required status inputs",
                _render_command(["uv", "run", "ops", "progress", "explain", first_procedure.registry_id]),
            )
        )
        if filters.related_to:
            next_steps.append(
                (
                    "Start a manifest from this related route set",
                    _render_command(["uv", "run", "ops", "progress", "scaffold", "--related-to", filters.related_to]),
                )
            )
        else:
            next_steps.append(
                (
                    "Emit a manifest skeleton for the first match",
                    _render_command(["uv", "run", "ops", "progress", "scaffold", first_procedure.registry_id]),
                )
            )
        return tuple(next_steps)

    if first_tool_source is not None:
        if broad_inventory and len(visible_tool_sources) > 1:
            next_steps.append(
                (
                    "Narrow the docs by topic",
                    _render_command(
                        ["uv", "run", "ops", "catalog", "list", "--section", "tool-sources", "--query", "<term>"]
                    ),
                )
            )
        if filters.related_to:
            next_steps.append(
                (
                    "Inspect the route behind these related tool docs",
                    _render_command(["uv", "run", "ops", "catalog", "show", filters.related_to]),
                )
            )
        else:
            next_steps.append(
                (
                    "Browse all registered procedures",
                    _render_command(["uv", "run", "ops", "catalog", "list", "--section", "procedures"]),
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

    next_steps.append(("Browse the full inventory", _render_command(["uv", "run", "ops", "catalog", "list"])))
    if section == "tool-sources":
        next_steps.append(
            (
                "Browse all registered procedures",
                _render_command(["uv", "run", "ops", "catalog", "list", "--section", "procedures"]),
            )
        )
    else:
        next_steps.append(
            (
                "Browse tool docs only",
                _render_command(["uv", "run", "ops", "catalog", "list", "--section", "tool-sources"]),
            )
        )
    return tuple(next_steps)


def _catalog_progress_show_command(
    *,
    registry_id: str,
    required_inputs: Sequence[ProgressFieldSpec],
) -> str:
    parts = ["uv", "run", "ops", "progress", "show", registry_id]
    for field in required_inputs:
        parts.extend((field.cli_flag, field.placeholder))
    return _render_command(parts)


def _catalog_next_commands(
    *,
    entry: CatalogProcedureEntry,
    details: CatalogProcedureDetails,
    owner_tool_source: CatalogToolSourceEntry | None,
    related_tool_sources: Sequence[CatalogToolSourceEntry],
) -> tuple[tuple[str, str], ...]:
    required_inputs = load_progress_required_inputs(entry.progress_kind)
    commands: list[tuple[str, str]] = [
        (
            "progress_explain",
            _render_command(["uv", "run", "ops", "progress", "explain", entry.registry_id]),
        ),
        (
            "progress_show",
            _catalog_progress_show_command(registry_id=entry.registry_id, required_inputs=required_inputs),
        ),
        (
            "progress_scaffold",
            _render_command(["uv", "run", "ops", "progress", "scaffold", entry.registry_id]),
        ),
    ]
    if owner_tool_source is not None:
        commands.append(
            (
                "catalog_owner_tool_source",
                _render_command(
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
                _render_command(
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
                    _render_command(
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
                    _render_command(
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


def _normalize_optional_filter(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _append_registry_suggestions(*, message: str, catalog: RunbookCatalog, registry_id: str) -> str:
    suggestions = suggest_procedure_registry_ids(catalog, registry_id)
    if suggestions:
        message += "\nDid you mean:\n" + "\n".join(f"- {candidate}" for candidate in suggestions)
    return message


def _progress_campaign_recovery_hint() -> str:
    return (
        "Hint: use `uv run ops progress scaffold <registry-id> ...` to emit a manifest skeleton, "
        "or `uv run ops progress scaffold --related-to <registry-id>` to start from one registered route."
    )


def _progress_campaign_path_hint() -> str:
    return "Hint: check the manifest path from `pwd` or pass an absolute path."


def _progress_required_input_lines(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    required_inputs = load_progress_required_inputs(entry.progress_kind)
    if not required_inputs:
        return ()
    lines = [f"Required inputs for {entry.registry_id}:"]
    for field in required_inputs:
        lines.append(f"- {field.cli_flag} {field.placeholder}: {field.summary}")
    return tuple(lines)


def _progress_optional_input_lines(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    optional_inputs = _progress_optional_inputs(entry.progress_kind)
    if not optional_inputs:
        return ()
    lines = ["Also accepted:"]
    for flag, summary in optional_inputs:
        lines.append(f"- {flag}: {summary}")
    return tuple(lines)


def _progress_scaffold_recovery_hint() -> str:
    return (
        "Hint: start with `uv run ops catalog list --simple`, inspect a route with "
        "`uv run ops catalog show <registry-id>`, or bootstrap a related manifest with "
        "`uv run ops progress scaffold --related-to <registry-id>`."
    )


def _first_unknown_registry_id(
    catalog: RunbookCatalog,
    *,
    registry_ids: Sequence[str],
    related_to: str | None = None,
) -> str | None:
    normalized_related_to = _normalize_optional_filter(related_to)
    if normalized_related_to is not None and catalog.find_procedure(normalized_related_to) is None:
        return normalized_related_to
    for registry_id in registry_ids:
        normalized_registry_id = registry_id.strip()
        if normalized_registry_id and catalog.find_procedure(normalized_registry_id) is None:
            return normalized_registry_id
    return None


@runbook_app.command("init")
def runbook_init(
    runbook: Annotated[Path, typer.Option("--runbook", help="Output path for orchestration runbook yaml.")],
    workflow: Annotated[
        Literal["densegen", "infer"],
        typer.Option("--workflow", help="Workflow family for scaffolded runbook."),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Workspace root path used to derive config and notify paths."),
    ],
    project: Annotated[str, typer.Option("--project", help="Scheduler project/account id.")] = "dunlop",
    runbook_id: Annotated[str, typer.Option("--id", help="Runbook id slug.")] = "batch_demo",
    cuda_module: Annotated[
        str,
        typer.Option("--cuda-module", help="Infer workflow CUDA module name."),
    ] = "cuda/12.4",
    gcc_module: Annotated[
        str,
        typer.Option("--gcc-module", help="Infer workflow GCC module name."),
    ] = "gcc/13.2.0",
    pe_omp: Annotated[
        int | None,
        typer.Option("--pe-omp", help="Override resources.pe_omp in the scaffolded runbook."),
    ] = None,
    h_rt: Annotated[
        str | None,
        typer.Option("--h-rt", help="Override resources.h_rt in HH:MM:SS format."),
    ] = None,
    mem_per_core: Annotated[
        str | None,
        typer.Option("--mem-per-core", help="Override resources.mem_per_core."),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root used to resolve default qsub template paths."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite runbook path when it already exists."),
    ] = False,
    with_notify: Annotated[
        bool,
        typer.Option(
            "--with-notify/--no-notify",
            help="Include notify smoke and watcher submit contracts in the scaffold (default: on).",
        ),
    ] = True,
) -> None:
    runbook_path = runbook.expanduser()
    repo_base = _resolve_repo_base(repo_root)
    if pe_omp is not None and pe_omp <= 0:
        typer.echo("Runbook contract error: --pe-omp must be > 0", err=True)
        raise typer.Exit(code=2)
    try:
        _validate_runbook_output_path_for_init(runbook_path=runbook_path, repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if runbook_path.exists() and not force:
        typer.echo(f"Runbook contract error: file exists: {runbook_path}", err=True)
        raise typer.Exit(code=2)

    def _template_or_default(relative_path: str) -> Path:
        candidate = repo_base / relative_path
        if candidate.exists():
            return candidate
        return Path(relative_path)

    notify_template = _template_or_default("docs/bu-scc/jobs/notify-watch.qsub")
    densegen_template = _template_or_default("docs/bu-scc/jobs/densegen-cpu.qsub")
    densegen_post_run_template = _template_or_default("docs/bu-scc/jobs/densegen-analysis.qsub")
    infer_template = _template_or_default("docs/bu-scc/jobs/evo2-gpu-infer.qsub")
    resolved_workspace_root = _resolve_workspace_root_for_init(workspace_root, repo_base=repo_base)
    payload = _build_init_payload(
        workflow=workflow,
        with_notify=with_notify,
        runbook_id=runbook_id,
        project=project,
        workspace_root=resolved_workspace_root,
        runbook_parent=runbook_path.parent,
        cuda_module=cuda_module,
        gcc_module=gcc_module,
        pe_omp=pe_omp,
        h_rt=h_rt,
        mem_per_core=mem_per_core,
        notify_qsub_template=_contract_path(notify_template, runbook_parent=runbook_path.parent),
        densegen_qsub_template=_contract_path(densegen_template, runbook_parent=runbook_path.parent),
        densegen_post_run_qsub_template=_contract_path(
            densegen_post_run_template,
            runbook_parent=runbook_path.parent,
        ),
        infer_qsub_template=_contract_path(infer_template, runbook_parent=runbook_path.parent),
    )
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    typer.echo(str(runbook_path.resolve()))
    if with_notify:
        typer.echo(
            _render_notify_contract_warning(
                workspace_root=resolved_workspace_root,
                notify_tool=resolve_workflow_tool(workflow_id=payload["runbook"]["workflow_id"]),
            ),
            err=True,
        )


def _emit_packaged_runbook_presets() -> None:
    presets = [{"name": path.stem, "path": str(path)} for path in _packaged_preset_paths()]
    typer.echo(json.dumps({"presets": presets}, indent=2, sort_keys=True))


@runbook_app.command("presets")
def runbook_presets() -> None:
    _emit_packaged_runbook_presets()


@catalog_app.command("list")
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
    progress_kind: Annotated[
        str | None,
        typer.Option("--progress-kind", help="Exact Progress-kind filter for cross-tool procedures."),
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
        typer.echo(f"Catalog contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    normalized_related_to = _normalize_optional_filter(related_to)
    if normalized_related_to is not None and catalog.find_procedure(normalized_related_to) is None:
        suggestions = suggest_procedure_registry_ids(catalog, normalized_related_to)
        message = f"Catalog contract error: unknown --related-to registry id: {normalized_related_to}"
        if suggestions:
            message += "\nDid you mean:\n" + "\n".join(f"- {candidate}" for candidate in suggestions)
        typer.echo(message, err=True)
        raise typer.Exit(code=2)

    filters = CatalogQuery(
        query=_normalize_optional_filter(query),
        entry_type=_normalize_optional_filter(entry_type),
        plane=_normalize_optional_filter(plane),
        execution_kind=_normalize_optional_filter(execution_kind),
        progress_kind=_normalize_optional_filter(progress_kind),
        related_to=normalized_related_to,
        tool=_normalize_optional_filter(tool),
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


@catalog_app.command("show")
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
        typer.echo(f"Catalog contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    entry = catalog.find_procedure(registry_id)
    if entry is None:
        message = f"Catalog contract error: unknown registry id: {registry_id}"
        message = _append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        typer.echo(message, err=True)
        raise typer.Exit(code=2)
    try:
        details = load_catalog_procedure_details(catalog, entry)
    except ValueError as exc:
        typer.echo(f"Catalog contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

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


@progress_app.command("show")
def progress_show(
    registry_id: Annotated[str, typer.Argument(help="Registered runbook or workflow registry id.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root containing docs/runbooks/README.md when invoking outside the repository.",
        ),
    ] = None,
    audit_json: Annotated[
        Path | None,
        typer.Option("--audit-json", help="Audit JSON artifact for ops-audit-json surfaces."),
    ] = None,
    sync_audit_json: Annotated[
        Path | None,
        typer.Option("--sync-audit-json", help="USR sync audit JSON artifact for usr-sync-audit surfaces."),
    ] = None,
    usr_root: Annotated[
        Path | None,
        typer.Option("--usr-root", help="USR root for usr-dataset-state surfaces."),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="USR dataset id for usr-dataset-state surfaces."),
    ] = None,
    cluster_results_root: Annotated[
        Path | None,
        typer.Option("--cluster-results-root", help="Cluster results root containing index.parquet."),
    ] = None,
    opal_config: Annotated[
        Path | None,
        typer.Option("--opal-config", help="OPAL campaign config path for opal-campaign-state surfaces."),
    ] = None,
    opal_workdir: Annotated[
        Path | None,
        typer.Option("--opal-workdir", help="OPAL campaign workdir when config resolution is not desired."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json/--no-json", help="Emit machine-readable JSON instead of plain text."),
    ] = False,
) -> None:
    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        message = f"Progress contract error: {exc}"
        typer.echo(message, err=True)
        raise typer.Exit(code=2) from exc

    entry = catalog.find_procedure(registry_id)
    if entry is None:
        message = f"Progress contract error: unknown registry id: {registry_id}"
        message = _append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        typer.echo(message, err=True)
        raise typer.Exit(code=2)

    try:
        result = build_procedure_progress(
            catalog,
            registry_id,
            inputs=ProgressInputs(
                audit_json=audit_json,
                sync_audit_json=sync_audit_json,
                usr_root=usr_root,
                dataset=dataset,
                cluster_results_root=cluster_results_root,
                opal_config=opal_config,
                opal_workdir=opal_workdir,
            ),
        )
    except ValueError as exc:
        message = f"Progress contract error: {exc}"
        if "requires --" in str(exc):
            message += "\n" + "\n".join(_progress_required_input_lines(entry))
            optional_lines = _progress_optional_input_lines(entry)
            if optional_lines:
                message += "\n" + "\n".join(optional_lines)
            message += (
                f"\nHint: use `uv run ops progress explain {registry_id}` to see the required flags and next commands."
            )
            message += (
                f"\nHint: use `uv run ops progress scaffold {registry_id}` to emit a manifest step "
                "with the required fields."
            )
        typer.echo(message, err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        _emit_progress_show_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            result=result,
        )
        return

    _emit_progress_show_text(
        repo_root=catalog.repo_root,
        catalog_path=catalog.catalog_path,
        result=result,
    )


@progress_app.command("explain")
def progress_explain(
    registry_id: Annotated[str, typer.Argument(help="Registered runbook or workflow registry id.")],
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
        typer.echo(f"Progress contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    entry = catalog.find_procedure(registry_id)
    if entry is None:
        message = f"Progress contract error: unknown registry id: {registry_id}"
        message = _append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        typer.echo(message, err=True)
        raise typer.Exit(code=2)

    details = load_catalog_procedure_details(catalog, entry)
    owner_boundary = details.owner_boundary
    if as_json:
        _emit_progress_explain_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            entry=entry,
            owner_boundary=owner_boundary,
            has_related_routes=bool(details.related_registry_ids),
        )
        return

    _emit_progress_explain_text(
        repo_root=catalog.repo_root,
        catalog_path=catalog.catalog_path,
        entry=entry,
        owner_boundary=owner_boundary,
        has_related_routes=bool(details.related_registry_ids),
    )


@progress_app.command("campaign")
def progress_campaign(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="YAML manifest listing explicit campaign progress steps."),
    ],
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
        typer.echo(f"Progress contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        result = load_campaign_progress(catalog, manifest_path=manifest)
    except (FileNotFoundError, ValueError) as exc:
        error_text = str(exc)
        message = f"Progress contract error: {error_text}"
        unknown_registry_prefix = "unknown registry id: "
        if error_text.startswith(unknown_registry_prefix):
            registry_id = error_text.removeprefix(unknown_registry_prefix).strip()
            message = _append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        if error_text.startswith("campaign manifest not found: "):
            message += "\n" + _progress_campaign_path_hint()
        if (
            "campaign manifest" in error_text
            or "missing 'registry_id'" in error_text
            or "must define a non-empty 'steps' list" in error_text
        ):
            message += "\n" + _progress_campaign_recovery_hint()
        typer.echo(message, err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        _emit_campaign_progress_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            result=result,
        )
        return

    _emit_campaign_progress_text(
        repo_root=catalog.repo_root,
        catalog_path=catalog.catalog_path,
        result=result,
    )


@progress_app.command("scaffold")
def progress_scaffold(
    registry_ids: Annotated[
        list[str] | None,
        typer.Argument(help="Zero or more registered runbook or workflow registry ids."),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root containing docs/runbooks/README.md when invoking outside the repository.",
        ),
    ] = None,
    campaign_id: Annotated[
        str | None,
        typer.Option("--campaign-id", help="Campaign id for the scaffolded manifest."),
    ] = None,
    related_to: Annotated[
        str | None,
        typer.Option(
            "--related-to",
            help=(
                "Expand one registered procedure into a manifest starting point: the named procedure first, "
                "then its typed related procedures."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write scaffolded campaign manifest YAML to this path."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite --out when the file already exists."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json/--no-json", help="Emit scaffold metadata as JSON instead of YAML."),
    ] = False,
) -> None:
    if as_json and out is not None:
        typer.echo("Progress contract error: --json cannot be combined with --out", err=True)
        raise typer.Exit(code=2)

    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        typer.echo(f"Progress contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    normalized_campaign_id = _normalize_optional_filter(campaign_id)
    normalized_related_to = _normalize_optional_filter(related_to)
    requested_registry_ids = registry_ids or []
    try:
        result = build_campaign_scaffold(
            catalog,
            registry_ids=requested_registry_ids,
            campaign_id=normalized_campaign_id,
            related_to=normalized_related_to,
        )
    except ValueError as exc:
        missing_registry_id = _first_unknown_registry_id(
            catalog,
            registry_ids=requested_registry_ids,
            related_to=normalized_related_to,
        )
        message = f"Progress contract error: {exc}"
        if missing_registry_id is not None:
            suggestions = suggest_procedure_registry_ids(catalog, missing_registry_id)
            if suggestions:
                message += "\nDid you mean:\n" + "\n".join(f"- {candidate}" for candidate in suggestions)
        if str(exc) == "progress scaffold requires at least one registry id or --related-to":
            message += "\n" + _progress_scaffold_recovery_hint()
        typer.echo(message, err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        _emit_progress_scaffold_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            result=result,
        )
        return

    rendered_yaml = _emit_progress_scaffold_yaml(result=result)
    if out is None:
        typer.echo(rendered_yaml.rstrip())
        return

    out_path = out.expanduser()
    if out_path.exists() and not force:
        typer.echo(f"Progress contract error: file exists: {out_path}", err=True)
        raise typer.Exit(code=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered_yaml, encoding="utf-8")
    typer.echo(str(out_path.resolve()))


@runbook_app.command("plan")
def runbook_plan(
    runbook: Annotated[Path, typer.Option("--runbook", help="Path to orchestration runbook yaml.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for runtime path contract checks when invoking outside the repository.",
        ),
    ] = None,
    mode: Annotated[
        Literal["auto", "fresh", "resume"] | None,
        typer.Option("--mode", help="Run mode policy override."),
    ] = None,
    smoke: Annotated[
        Literal["dry", "live"] | None,
        typer.Option("--notify-smoke", help="Notify smoke override."),
    ] = None,
    active_job_id: Annotated[
        list[str],
        typer.Option(
            "--active-job-id",
            help=(
                "Existing active job id(s) for hold_jid policy decisions; repeat option or pass a comma-delimited list."
            ),
        ),
    ] = [],
    discover_active_jobs: Annotated[
        bool,
        typer.Option(
            "--discover-active-jobs/--no-discover-active-jobs",
            help="Auto-discover active matching jobs from qstat/qstat -j and merge into hold_jid decisions.",
        ),
    ] = True,
    max_discovery_jobs: Annotated[
        int,
        typer.Option("--max-discovery-jobs", help="Maximum qstat jobs inspected during active-job discovery."),
    ] = 24,
    allow_fresh_reset: Annotated[
        bool,
        typer.Option(
            "--allow-fresh-reset/--no-allow-fresh-reset",
            help="Allow --mode fresh when resume artifacts already exist in the workspace.",
        ),
    ] = False,
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help=(
                "Render preflight gate commands with explicit degraded queue-probe mode when `qstat` is unavailable. "
                "Useful for workstation dry-run demos."
            ),
        ),
    ] = False,
) -> None:
    if max_discovery_jobs <= 0:
        typer.echo("Runbook contract error: --max-discovery-jobs must be > 0", err=True)
        raise typer.Exit(code=2)
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    loaded = _load_runbook_or_exit(runbook)
    resolved_active_job_ids = _resolve_active_job_ids(
        runbook=loaded,
        active_job_ids=active_job_id,
        discover_active_jobs=discover_active_jobs,
        max_discovery_jobs=max_discovery_jobs,
    )
    try:
        plan = build_batch_plan(
            runbook=loaded,
            requested_mode=mode,
            requested_smoke=smoke,
            active_job_ids=resolved_active_job_ids,
            allow_fresh_reset=allow_fresh_reset,
            allow_missing_qstat=allow_missing_qstat,
        )
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(plan.as_dict(), indent=2, sort_keys=True))


@runbook_app.command("active-jobs")
def runbook_active_jobs(
    runbook: Annotated[Path, typer.Option("--runbook", help="Path to orchestration runbook yaml.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for runtime path contract checks when invoking outside the repository.",
        ),
    ] = None,
    max_discovery_jobs: Annotated[
        int,
        typer.Option("--max-discovery-jobs", help="Maximum qstat jobs inspected during active-job discovery."),
    ] = 24,
) -> None:
    if max_discovery_jobs <= 0:
        typer.echo("Runbook contract error: --max-discovery-jobs must be > 0", err=True)
        raise typer.Exit(code=2)
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    loaded = _load_runbook_or_exit(runbook)
    try:
        active_job_ids = discover_active_job_ids_for_runbook(loaded, max_jobs=max_discovery_jobs)
    except RuntimeError as exc:
        typer.echo(f"Runbook contract error: active-job discovery failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    hints = _render_active_job_hints(runbook_path=runbook, active_job_ids=active_job_ids)
    payload = {
        "runbook_id": loaded.id,
        "workflow_id": loaded.workflow_id,
        "active_job_ids": list(active_job_ids),
        **hints,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runbook_app.command("execute")
def runbook_execute(
    runbook: Annotated[Path, typer.Option("--runbook", help="Path to orchestration runbook yaml.")],
    audit_json: Annotated[Path, typer.Option("--audit-json", help="Output path for audit artifact json.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for runtime path contract checks when invoking outside the repository.",
        ),
    ] = None,
    mode: Annotated[
        Literal["auto", "fresh", "resume"] | None,
        typer.Option("--mode", help="Run mode policy override."),
    ] = None,
    smoke: Annotated[
        Literal["dry", "live"] | None,
        typer.Option("--notify-smoke", help="Notify smoke override."),
    ] = None,
    active_job_id: Annotated[
        list[str],
        typer.Option(
            "--active-job-id",
            help=(
                "Existing active job id(s) for hold_jid policy decisions; repeat option or pass a comma-delimited list."
            ),
        ),
    ] = [],
    discover_active_jobs: Annotated[
        bool,
        typer.Option(
            "--discover-active-jobs/--no-discover-active-jobs",
            help="Auto-discover active matching jobs from qstat/qstat -j and merge into hold_jid decisions.",
        ),
    ] = True,
    max_discovery_jobs: Annotated[
        int,
        typer.Option("--max-discovery-jobs", help="Maximum qstat jobs inspected during active-job discovery."),
    ] = 24,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit/--no-submit",
            help="Run submit-phase qsub commands after preflight/smoke pass. Default is no-submit.",
        ),
    ] = False,
    command_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--command-timeout-seconds",
            help="Per-command timeout in seconds for execute phases.",
        ),
    ] = 300.0,
    allow_fresh_reset: Annotated[
        bool,
        typer.Option(
            "--allow-fresh-reset/--no-allow-fresh-reset",
            help="Allow --mode fresh when resume artifacts already exist in the workspace.",
        ),
    ] = False,
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help=(
                "Allow qstat-dependent preflight gates to emit explicit degraded advisory records instead of failing "
                "when `qstat` is unavailable. Intended for workstation dry-run demos."
            ),
        ),
    ] = False,
) -> None:
    if command_timeout_seconds is not None and command_timeout_seconds <= 0:
        typer.echo("Runbook contract error: --command-timeout-seconds must be > 0", err=True)
        raise typer.Exit(code=2)
    if max_discovery_jobs <= 0:
        typer.echo("Runbook contract error: --max-discovery-jobs must be > 0", err=True)
        raise typer.Exit(code=2)
    if submit and allow_missing_qstat:
        typer.echo(
            "Runbook contract error: --allow-missing-qstat is only allowed with --no-submit dry-run demos.",
            err=True,
        )
        raise typer.Exit(code=2)
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    loaded = _load_runbook_or_exit(runbook)
    try:
        resolved_audit_json = _validate_audit_json_path_for_execute(
            audit_json_path=audit_json,
            workspace_root=loaded.workspace_root,
        )
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    resolved_active_job_ids = _resolve_active_job_ids(
        runbook=loaded,
        active_job_ids=active_job_id,
        discover_active_jobs=discover_active_jobs,
        max_discovery_jobs=max_discovery_jobs,
    )
    try:
        plan = build_batch_plan(
            runbook=loaded,
            requested_mode=mode,
            requested_smoke=smoke,
            active_job_ids=resolved_active_job_ids,
            allow_fresh_reset=allow_fresh_reset,
            allow_missing_qstat=allow_missing_qstat,
        )
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    result = execute_batch_plan(
        plan=plan,
        audit_json_path=resolved_audit_json,
        submit=submit,
        command_timeout_seconds=command_timeout_seconds,
    )
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    if not result.ok:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
