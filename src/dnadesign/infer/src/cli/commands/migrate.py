"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/migrate.py

Registration for infer data cleanup utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ...features.overlay_pruning import prune_stale_infer_overlay_columns
from ..common import raise_cli_error


def register(app: typer.Typer) -> None:
    migrate_app = typer.Typer(no_args_is_help=True, help="Data cleanup utilities.")
    app.add_typer(migrate_app, name="migrate")

    @migrate_app.command(
        "prune-stale-overlay-columns",
        help="Drop explicitly approved stale Infer overlay columns by prefix/name without reading payloads.",
    )
    def prune_stale_overlay_columns(
        usr_root: Path = typer.Option(..., "--usr-root", help="USR datasets root."),
        dataset: str = typer.Option(..., "--dataset", help="Dataset id containing _derived/infer parts."),
        namespace: str = typer.Option("infer", "--namespace", help="Overlay namespace. Only infer is supported."),
        column_prefix: list[str] | None = typer.Option(
            None,
            "--column-prefix",
            help="Column prefix to prune. May be repeated.",
        ),
        column_name: list[str] | None = typer.Option(
            None,
            "--column-name",
            help="Exact column name to prune. May be repeated.",
        ),
        reason: str = typer.Option("", "--reason", help="Short audit reason recorded on write."),
        write: bool = typer.Option(False, "--write", help="Rewrite/delete selected columns. Default is dry-run."),
        keep_empty_parts: bool = typer.Option(
            False,
            "--keep-empty-parts",
            help="Keep part files that contain only id after stale-column pruning.",
        ),
        fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    ) -> None:
        try:
            result = prune_stale_infer_overlay_columns(
                dataset_root=usr_root,
                dataset_id=dataset,
                namespace=namespace,
                column_prefixes=column_prefix or (),
                column_names=column_name or (),
                reason=reason,
                write=write,
                delete_empty_parts=not keep_empty_parts,
            ).to_dict()
            if fmt == "json":
                typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return
            if fmt != "text":
                raise ValueError("format must be one of: text, json.")
            typer.echo(
                "prune-stale-overlay-columns "
                f"mode={result['mode']} dataset={dataset} namespace={namespace} "
                f"parts_scanned={result['parts_scanned']} "
                f"parts_with_columns={result['parts_with_columns']} "
                f"columns_removed={len(result['removed_columns'])} "
                f"bytes_reclaimable={result['bytes_reclaimable']} "
                f"bytes_reclaimed={result['bytes_reclaimed']} "
                f"files_rewritten={result['files_rewritten']} files_deleted={result['files_deleted']}"
            )
        except Exception as error:
            raise_cli_error(error)
