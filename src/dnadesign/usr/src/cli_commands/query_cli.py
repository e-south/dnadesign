"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/query_cli.py

Typer registration helpers for USR dataset query commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_query_commands(
    app: typer.Typer,
    events_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_ls: Callable[[object], None],
    cmd_info: Callable[[object], None],
    cmd_schema: Callable[[object], None],
    cmd_head: Callable[[object], None],
    cmd_cols: Callable[[object], None],
    cmd_describe: Callable[[object], None],
    cmd_cell: Callable[[object], None],
    cmd_validate: Callable[[object], None],
    cmd_events_tail: Callable[[object], None],
    cmd_get: Callable[[object], None],
    cmd_grep: Callable[[object], None],
    cmd_export: Callable[[object], None],
) -> None:
    @app.command("ls")
    def cli_ls(
        ctx: typer.Context,
        format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
    ) -> None:
        cmd_ls(ctx_args_builder(ctx, format=format))

    @app.command("info")
    def cli_info(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
    ) -> None:
        cmd_info(ctx_args_builder(ctx, dataset=dataset, format=format))

    @app.command("schema")
    def cli_schema(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        tree: bool = typer.Option(False, "--tree"),
        format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
    ) -> None:
        cmd_schema(ctx_args_builder(ctx, dataset=dataset, tree=tree, format=format))

    @app.command("head")
    def cli_head(
        ctx: typer.Context,
        target: str = typer.Argument(".", help="Dataset name or file/directory path"),
        n: int = typer.Option(10, "-n"),
        columns: str = typer.Option("", "--columns"),
        include_deleted: bool = typer.Option(False, "--include-deleted"),
        raw: bool = typer.Option(False, "--raw"),
        max_colwidth: int = typer.Option(80, "--max-colwidth"),
        max_list_items: int = typer.Option(6, "--max-list-items"),
        precision: int = typer.Option(4, "--precision"),
    ) -> None:
        cmd_head(
            ctx_args_builder(
                ctx,
                target=target,
                n=n,
                columns=columns,
                include_deleted=include_deleted,
                raw=raw,
                max_colwidth=max_colwidth,
                max_list_items=max_list_items,
                precision=precision,
            )
        )

    @app.command("cols")
    def cli_cols(
        ctx: typer.Context,
        target: str = typer.Argument(".", help="Dataset name or file/directory path"),
        glob: str | None = typer.Option(None, "--glob"),
    ) -> None:
        cmd_cols(ctx_args_builder(ctx, target=target, glob=glob))

    @app.command("describe")
    def cli_describe(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        columns: str = typer.Option("", "--columns"),
        sample: int = typer.Option(1024, "--sample"),
        include_deleted: bool = typer.Option(False, "--include-deleted"),
        max_colwidth: int = typer.Option(80, "--max-colwidth"),
        max_list_items: int = typer.Option(6, "--max-list-items"),
        precision: int = typer.Option(4, "--precision"),
    ) -> None:
        cmd_describe(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                columns=columns,
                sample=sample,
                include_deleted=include_deleted,
                max_colwidth=max_colwidth,
                max_list_items=max_list_items,
                precision=precision,
            )
        )

    @app.command("cell")
    def cli_cell(
        ctx: typer.Context,
        target: str = typer.Argument(".", help="Dataset name or file/directory path"),
        row: int = typer.Option(0, "--row"),
        col: str = typer.Option("", "--col"),
        glob: str | None = typer.Option(None, "--glob"),
    ) -> None:
        cmd_cell(ctx_args_builder(ctx, target=target, row=row, col=col, glob=glob))

    @app.command("validate")
    def cli_validate(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        strict: bool = typer.Option(False, "--strict"),
        registry_mode: str = typer.Option(
            "current",
            "--registry-mode",
            help="Registry mode: current|frozen|either",
        ),
    ) -> None:
        cmd_validate(ctx_args_builder(ctx, dataset=dataset, strict=strict, registry_mode=registry_mode))

    @events_app.command("tail")
    def cli_events_tail(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        format: str = typer.Option("json", "--format", help="Output format: json|raw"),
        n: int = typer.Option(0, "--n", help="Show only the last N events (0 = all)."),
        follow: bool = typer.Option(False, "--follow", help="Follow the events log for new entries."),
    ) -> None:
        cmd_events_tail(ctx_args_builder(ctx, dataset=dataset, format=format, n=n, follow=follow))

    @app.command("get")
    def cli_get(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        record_id: str = typer.Option(..., "--id"),
        columns: str = typer.Option("", "--columns"),
        include_deleted: bool = typer.Option(False, "--include-deleted"),
    ) -> None:
        cmd_get(ctx_args_builder(ctx, dataset=dataset, id=record_id, columns=columns, include_deleted=include_deleted))

    @app.command("grep")
    def cli_grep(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        pattern: str = typer.Option(..., "--pattern"),
        limit: int = typer.Option(20, "--limit"),
        batch_size: int = typer.Option(65536, "--batch-size"),
        include_deleted: bool = typer.Option(False, "--include-deleted"),
    ) -> None:
        cmd_grep(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                pattern=pattern,
                limit=limit,
                batch_size=batch_size,
                include_deleted=include_deleted,
            )
        )

    @app.command("export")
    def cli_export(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        fmt: str = typer.Option(..., "--fmt", help="Export format: csv|jsonl|parquet."),
        out: Path = typer.Option(
            ...,
            "--out",
            path_type=Path,
            help="Output file path, or an existing directory for auto-named export output.",
        ),
        columns: str = typer.Option("", "--columns"),
        include_deleted: bool = typer.Option(False, "--include-deleted"),
    ) -> None:
        cmd_export(
            ctx_args_builder(ctx, dataset=dataset, fmt=fmt, out=out, columns=columns, include_deleted=include_deleted)
        )
