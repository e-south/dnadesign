"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/commands/progress.py

Direct progress command implementation backed by the neutral OPS status service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Sequence

import click
import typer
import typer.main
import yaml

from dnadesign.ops.catalog import (
    CatalogProcedureEntry,
    RunbookCatalog,
    load_catalog_procedure_details,
    load_runbook_catalog,
    repo_relative_catalog_doc_path,
)
from dnadesign.ops.cli.common import (
    append_registry_suggestions,
    normalize_optional_filter,
    raise_contract_error,
    render_command,
)
from dnadesign.ops.cli.dynamic_inputs import (
    build_dynamic_input_options,
    merge_status_input_values,
    optional_input_lines,
    render_progress_show_command,
    required_input_lines,
)

if TYPE_CHECKING:
    from dnadesign.ops.status import CampaignScaffold, CampaignStatus, InputFieldSpec, ProcedureStatus
    from dnadesign.ops.status.models import StatusKindSpec

app = typer.Typer(
    help=(
        "Status inspection, status explanation, and manifest scaffold commands "
        "for registered runbooks and explicit campaigns. "
        "Start with `ops catalog list --simple` for public routes, then use "
        "`ops progress explain <registry-id>` or `ops progress kinds` for the "
        "underlying status-kind inventory. "
        "`show` and `campaign` are read-only; `scaffold` prints YAML unless `--out` is used."
    )
)


def get_click_command():
    command = typer.main.get_command(app)
    if isinstance(command, click.Group):
        command.add_command(_build_progress_show_click_command(), "show")
    return command


def _build_campaign_scaffold(*args, **kwargs) -> CampaignScaffold:
    from dnadesign.ops.status.campaign import build_campaign_scaffold

    return build_campaign_scaffold(*args, **kwargs)


def _build_procedure_status(*args, **kwargs) -> ProcedureStatus:
    from dnadesign.ops.status.campaign import build_procedure_status

    return build_procedure_status(*args, **kwargs)


def _list_status_kind_specs() -> tuple[StatusKindSpec, ...]:
    from dnadesign.ops.status.registry_loader import list_status_kind_specs

    return list_status_kind_specs()


def _load_campaign_status(*args, **kwargs) -> CampaignStatus:
    from dnadesign.ops.status.campaign import load_campaign_status

    return load_campaign_status(*args, **kwargs)


def _load_status_kind_spec(status_kind: str) -> StatusKindSpec:
    from dnadesign.ops.status.registry_loader import load_status_kind_spec

    return load_status_kind_spec(status_kind)


def _status_required_inputs(status_kind: str) -> tuple[InputFieldSpec, ...]:
    return _load_status_kind_spec(status_kind).required_inputs


def _status_optional_inputs(status_kind: str) -> tuple[tuple[str, str], ...]:
    spec = _load_status_kind_spec(status_kind)
    return tuple((field.cli_flag, field.summary) for field in spec.optional_inputs)


def _status_notes(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    return _load_status_kind_spec(entry.status_kind).notes


def _progress_campaign_recovery_hint() -> str:
    return (
        "Hint: use `uv run ops progress scaffold <registry-id> ...` to emit a manifest skeleton, "
        "or `uv run ops progress scaffold --related-to <registry-id>` to start from one registered route."
    )


def _progress_campaign_path_hint() -> str:
    return "Hint: check the manifest path from `pwd` or pass an absolute path."


def _progress_required_input_lines(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    return required_input_lines(
        label=entry.registry_id,
        required_inputs=_status_required_inputs(entry.status_kind),
    )


def _progress_optional_input_lines(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    return optional_input_lines(_load_status_kind_spec(entry.status_kind).optional_inputs)


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
    normalized_related_to = normalize_optional_filter(related_to)
    if normalized_related_to is not None and catalog.find_procedure(normalized_related_to) is None:
        return normalized_related_to
    for registry_id in registry_ids:
        normalized_registry_id = registry_id.strip()
        if normalized_registry_id and catalog.find_procedure(normalized_registry_id) is None:
            return normalized_registry_id
    return None


def _emit_progress_show_text(*, repo_root: Path, catalog_path: Path, result: ProcedureStatus) -> None:
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


def _emit_progress_show_json(*, repo_root: Path, catalog_path: Path, result: ProcedureStatus) -> None:
    payload = result.as_dict()
    payload["doc_path"] = repo_relative_catalog_doc_path(
        repo_root=repo_root,
        catalog_path=catalog_path,
        doc_path=result.doc_path,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _emit_progress_explain_text(
    *,
    repo_root: Path,
    catalog_path: Path,
    entry: CatalogProcedureEntry,
    owner_boundary: str,
    has_related_routes: bool,
) -> None:
    spec = _load_status_kind_spec(entry.status_kind)
    required_inputs = _status_required_inputs(entry.status_kind)
    optional_inputs = _status_optional_inputs(entry.status_kind)
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
    notes = _status_notes(entry)
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
    spec = _load_status_kind_spec(entry.status_kind)
    required_inputs = _status_required_inputs(entry.status_kind)
    optional_inputs = _status_optional_inputs(entry.status_kind)
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
        "notes": list(_status_notes(entry)),
    }
    if has_related_routes:
        payload["next_commands"]["progress_scaffold_related"] = render_command(
            ["uv", "run", "ops", "progress", "scaffold", "--related-to", entry.registry_id]
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _emit_status_kinds_text() -> None:
    lines = [
        "Status kinds",
        "Use `ops catalog list --simple` for the public registry routes that rely on these underlying status kinds.",
    ]
    for spec in _list_status_kind_specs():
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


def _emit_status_kinds_json() -> None:
    payload = {"status_kinds": [spec.as_inventory_dict() for spec in _list_status_kind_specs()]}
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _emit_campaign_progress_text(*, repo_root: Path, catalog_path: Path, result: CampaignStatus) -> None:
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


def _emit_campaign_progress_json(*, repo_root: Path, catalog_path: Path, result: CampaignStatus) -> None:
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


def _emit_progress_scaffold_json(*, repo_root: Path, catalog_path: Path, result: CampaignScaffold) -> None:
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


class DynamicProgressShowCommand(click.Command):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dynamic_param_names: tuple[str, ...] = ()

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        self._configure_dynamic_input_options(args)
        return super().parse_args(ctx, args)

    def _configure_dynamic_input_options(self, args: Sequence[str]) -> None:
        spec = _resolve_progress_show_spec_from_args(args)
        next_param_names = tuple(field.name for field in spec.input_schema) if spec is not None else ()
        if next_param_names == self._dynamic_param_names:
            return

        if self._dynamic_param_names:
            self.params = [
                param for param in self.params if getattr(param, "name", None) not in self._dynamic_param_names
            ]
            self._dynamic_param_names = ()

        if spec is None or not spec.input_schema:
            return

        insert_at = next(
            (index for index, param in enumerate(self.params) if getattr(param, "name", None) == "input_items"),
            len(self.params),
        )
        dynamic_params = list(build_dynamic_input_options(spec.input_schema))
        self.params = [*self.params[:insert_at], *dynamic_params, *self.params[insert_at:]]
        self._dynamic_param_names = next_param_names


def _build_progress_show_click_command() -> click.Command:
    return DynamicProgressShowCommand(
        name="show",
        help="Show one registered runbook or workflow status surface.",
        params=[
            click.Argument(["registry_id"], metavar="REGISTRY_ID", required=True),
            click.Option(
                ["repo_root", "--repo-root"],
                type=click.Path(path_type=Path),
                help="Repository root containing docs/runbooks/README.md when invoking outside the repository.",
            ),
            click.Option(
                ["input_items", "--input"],
                multiple=True,
                help="Provider-owned input override as key=value. May be passed more than once.",
            ),
            click.Option(
                ["as_json", "--json/--no-json"],
                default=False,
                help="Emit machine-readable JSON instead of plain text.",
            ),
        ],
        callback=_progress_show_callback,
    )


def _progress_show_callback(
    registry_id: str,
    repo_root: Path | None,
    input_items: tuple[str, ...],
    as_json: bool,
    **dynamic_values: object,
) -> None:
    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        raise_contract_error(f"Progress contract error: {exc}")

    entry = catalog.find_procedure(registry_id)
    if entry is None:
        message = f"Progress contract error: unknown registry id: {registry_id}"
        message = append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        raise_contract_error(message)

    try:
        spec = _load_status_kind_spec(entry.status_kind)
        raw_inputs = merge_status_input_values(
            flag_values=dynamic_values,
            input_items=input_items,
            input_schema=spec.input_schema,
        )
        result = _build_procedure_status(catalog, registry_id, raw_inputs=raw_inputs)
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
        raise_contract_error(message)

    if as_json:
        _emit_progress_show_json(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)
        return
    _emit_progress_show_text(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)


def _resolve_progress_show_spec_from_args(args: Sequence[str]) -> StatusKindSpec | None:
    registry_id, repo_root = _scan_progress_show_args(args)
    if registry_id is None:
        return None
    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError:
        return None
    entry = catalog.find_procedure(registry_id)
    if entry is None:
        return None
    return _load_status_kind_spec(entry.status_kind)


def _scan_progress_show_args(args: Sequence[str]) -> tuple[str | None, Path | None]:
    registry_id: str | None = None
    repo_root: Path | None = None
    index = 0
    while index < len(args):
        token = str(args[index]).strip()
        if not token:
            index += 1
            continue
        if token == "--":
            break
        if token == "--repo-root":
            if index + 1 < len(args):
                repo_root = Path(args[index + 1])
            index += 2
            continue
        if token.startswith("--repo-root="):
            _, _, raw_path = token.partition("=")
            if raw_path.strip():
                repo_root = Path(raw_path)
            index += 1
            continue
        if token == "--input":
            index += 2
            continue
        if token.startswith("--input=") or token in {"--json", "--no-json", "--help"}:
            index += 1
            continue
        if token.startswith("-"):
            if registry_id is None:
                return None, repo_root
            index += 1
            continue
        if registry_id is None:
            registry_id = token
        index += 1
    return registry_id, repo_root


@app.command("kinds")
def status_kinds(
    as_json: Annotated[
        bool,
        typer.Option("--json/--no-json", help="Emit machine-readable JSON instead of plain text."),
    ] = False,
) -> None:
    if as_json:
        _emit_status_kinds_json()
        return
    _emit_status_kinds_text()


@app.command("explain")
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
        raise_contract_error(f"Progress contract error: {exc}")

    entry = catalog.find_procedure(registry_id)
    if entry is None:
        message = f"Progress contract error: unknown registry id: {registry_id}"
        message = append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        raise_contract_error(message)

    details = load_catalog_procedure_details(catalog, entry)
    if as_json:
        _emit_progress_explain_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            entry=entry,
            owner_boundary=details.owner_boundary,
            has_related_routes=bool(details.related_registry_ids),
        )
        return
    _emit_progress_explain_text(
        repo_root=catalog.repo_root,
        catalog_path=catalog.catalog_path,
        entry=entry,
        owner_boundary=details.owner_boundary,
        has_related_routes=bool(details.related_registry_ids),
    )


@app.command("campaign")
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
        raise_contract_error(f"Progress contract error: {exc}")

    try:
        result = _load_campaign_status(catalog, manifest_path=manifest)
    except (FileNotFoundError, ValueError) as exc:
        error_text = str(exc)
        message = f"Progress contract error: {error_text}"
        unknown_registry_prefix = "unknown registry id: "
        if error_text.startswith(unknown_registry_prefix):
            registry_id = error_text.removeprefix(unknown_registry_prefix).strip()
            message = append_registry_suggestions(message=message, catalog=catalog, registry_id=registry_id)
        if error_text.startswith("campaign manifest not found: "):
            message += "\n" + _progress_campaign_path_hint()
        if (
            "campaign manifest" in error_text
            or "missing 'registry_id'" in error_text
            or "must define a non-empty 'steps' list" in error_text
        ):
            message += "\n" + _progress_campaign_recovery_hint()
        raise_contract_error(message)

    if as_json:
        _emit_campaign_progress_json(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)
        return
    _emit_campaign_progress_text(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)


@app.command("scaffold")
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
        raise_contract_error("Progress contract error: --json cannot be combined with --out")

    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        raise_contract_error(f"Progress contract error: {exc}")

    normalized_campaign_id = normalize_optional_filter(campaign_id)
    normalized_related_to = normalize_optional_filter(related_to)
    requested_registry_ids = registry_ids or []
    try:
        result = _build_campaign_scaffold(
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
            from dnadesign.ops.catalog import suggest_procedure_registry_ids

            suggestions = suggest_procedure_registry_ids(catalog, missing_registry_id)
            if suggestions:
                message += "\nDid you mean:\n" + "\n".join(f"- {candidate}" for candidate in suggestions)
        if str(exc) == "campaign scaffold requires at least one registry id or --related-to":
            message += "\n" + _progress_scaffold_recovery_hint()
        raise_contract_error(message)

    if as_json:
        _emit_progress_scaffold_json(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)
        return

    rendered_yaml = _emit_progress_scaffold_yaml(result=result)
    if out is None:
        typer.echo(rendered_yaml.rstrip())
        return
    out_path = out.expanduser()
    if out_path.exists() and not force:
        raise_contract_error(f"Progress contract error: file exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered_yaml, encoding="utf-8")
    typer.echo(str(out_path.resolve()))


__all__ = ["app", "get_click_command"]
