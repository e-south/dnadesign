"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/commands/progress.py

Direct progress command implementation backed by the neutral OPS status service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Sequence

import click
import typer
import typer.main

from dnadesign.ops.catalog import (
    CatalogProcedureEntry,
    RunbookCatalog,
    load_catalog_procedure_details,
    load_runbook_catalog,
)
from dnadesign.ops.cli.commands.progress_render import (
    emit_campaign_progress_json,
    emit_campaign_progress_text,
    emit_progress_explain_json,
    emit_progress_explain_text,
    emit_progress_scaffold_json,
    emit_progress_show_json,
    emit_progress_show_text,
    emit_status_kinds_json,
    emit_status_kinds_text,
    render_progress_scaffold_yaml,
)
from dnadesign.ops.cli.commands.progress_status_specs import (
    build_campaign_scaffold,
    build_procedure_status,
    load_campaign_status,
    load_status_kind_spec,
    status_required_inputs,
)
from dnadesign.ops.cli.common import (
    append_registry_suggestions,
    normalize_optional_filter,
    raise_contract_error,
)
from dnadesign.ops.cli.dynamic_inputs import (
    build_dynamic_input_options,
    merge_status_input_values,
    optional_input_lines,
    required_input_lines,
)

if TYPE_CHECKING:
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
    add_command = getattr(command, "add_command", None)
    if not callable(add_command):
        raise RuntimeError("OPS progress command must expose group command registration")
    add_command(_build_progress_show_click_command(), "show")
    return command


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
        required_inputs=status_required_inputs(entry.status_kind),
    )


def _progress_optional_input_lines(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    return optional_input_lines(load_status_kind_spec(entry.status_kind).optional_inputs)


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
        spec = load_status_kind_spec(entry.status_kind)
        raw_inputs = merge_status_input_values(
            flag_values=dynamic_values,
            input_items=input_items,
            input_schema=spec.input_schema,
        )
        result = build_procedure_status(catalog, registry_id, raw_inputs=raw_inputs)
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
        emit_progress_show_json(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)
        return
    emit_progress_show_text(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)


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
    return load_status_kind_spec(entry.status_kind)


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
        emit_status_kinds_json()
        return
    emit_status_kinds_text()


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
        emit_progress_explain_json(
            repo_root=catalog.repo_root,
            catalog_path=catalog.catalog_path,
            entry=entry,
            owner_boundary=details.owner_boundary,
            has_related_routes=bool(details.related_registry_ids),
        )
        return
    emit_progress_explain_text(
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
        result = load_campaign_status(catalog, manifest_path=manifest)
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
        emit_campaign_progress_json(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)
        return
    emit_campaign_progress_text(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)


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
            from dnadesign.ops.catalog import suggest_procedure_registry_ids

            suggestions = suggest_procedure_registry_ids(catalog, missing_registry_id)
            if suggestions:
                message += "\nDid you mean:\n" + "\n".join(f"- {candidate}" for candidate in suggestions)
        if str(exc) == "campaign scaffold requires at least one registry id or --related-to":
            message += "\n" + _progress_scaffold_recovery_hint()
        raise_contract_error(message)

    if as_json:
        emit_progress_scaffold_json(repo_root=catalog.repo_root, catalog_path=catalog.catalog_path, result=result)
        return

    rendered_yaml = render_progress_scaffold_yaml(result=result)
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
