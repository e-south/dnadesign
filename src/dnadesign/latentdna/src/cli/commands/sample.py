"""
Sample CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.sample_service import build_sample
from ..common import emit, fail, resolve_format
from ..previews import preview_sample_build

app = typer.Typer(help="Sample commands for latentdna.")


@app.command("build")
def build(
    sample_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    view: str | None = typer.Option(None, "--view"),
    strategy: str = typer.Option("all", "--strategy"),
    group_column: str | None = typer.Option(None, "--group-column"),
    n: int | None = typer.Option(None, "--n"),
    reference_set: str | None = typer.Option(None, "--reference-set"),
    record_id: list[str] = typer.Option([], "--record-id"),
    input_sample: list[str] = typer.Option([], "--input-sample"),
    seed: int = typer.Option(17, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_sample_build(
                workspace,
                sample_id,
                view_id=view,
                strategy=strategy,
                reference_set_id=reference_set,
                explicit_ids=list(record_id) or None,
                input_sample_ids=list(input_sample) or None,
                force=force,
            )
            if dry_run
            else build_sample(
                workspace,
                sample_id,
                view_id=view,
                strategy=strategy,
                n=n,
                group_column=group_column,
                seed=seed,
                reference_set_id=reference_set,
                explicit_ids=list(record_id) or None,
                input_sample_ids=list(input_sample) or None,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
