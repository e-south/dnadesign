"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/cli.py

Typer registration helpers for USR maintenance commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_maintenance_commands(
    maintenance_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_dedupe_sequences: Callable[[object], None],
    cmd_registry_freeze: Callable[[object], None],
    cmd_overlay_compact: Callable[[object], None],
    cmd_overlay_project: Callable[[object], None],
    cmd_overlay_remove: Callable[[object], None],
    cmd_merge_datasets: Callable[[object], None],
) -> None:
    @maintenance_app.command("dedupe")
    def cli_dedupe_sequences(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        key: str = typer.Option(..., "--key", help="Dedupe key: id|sequence|sequence_norm|sequence_ci"),
        keep: str = typer.Option("keep-first", "--keep", help="Which occurrence to keep: keep-first|keep-last"),
        batch_size: int = typer.Option(65536, "--batch-size", help="Parquet batch size for streaming dedupe"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        yes: bool = typer.Option(False, "--yes"),
    ) -> None:
        cmd_dedupe_sequences(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                key=key,
                keep=keep,
                batch_size=batch_size,
                dry_run=dry_run,
                yes=yes,
            )
        )

    @maintenance_app.command("registry-freeze")
    def cli_registry_freeze(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
    ) -> None:
        cmd_registry_freeze(ctx_args_builder(ctx, dataset=dataset))

    @maintenance_app.command("overlay-compact")
    def cli_overlay_compact(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        namespace: str = typer.Option(..., "--namespace"),
    ) -> None:
        cmd_overlay_compact(ctx_args_builder(ctx, dataset=dataset, namespace=namespace))

    @maintenance_app.command("overlay-remove")
    def cli_overlay_remove(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        namespace: str = typer.Option(..., "--namespace"),
        mode: str = typer.Option("error", "--mode", help="Removal mode: error|delete|archive"),
    ) -> None:
        cmd_overlay_remove(ctx_args_builder(ctx, dataset=dataset, namespace=namespace, mode=mode))

    @maintenance_app.command("overlay-project")
    def cli_overlay_project(
        ctx: typer.Context,
        src: str = typer.Option(..., "--src", help="Source dataset that already exposes the namespace live."),
        dest: str = typer.Option(..., "--dest", help="Destination dataset to receive the projected overlay."),
        namespace: str = typer.Option(..., "--namespace", help="Namespace to project from source onto destination."),
        src_join: str = typer.Option("id", "--src-join", help="Join column in the source live view."),
        dest_join: str = typer.Option("id", "--dest-join", help="Join column in the destination live view."),
        columns: str | None = typer.Option(
            None,
            "--columns",
            help="Comma-separated source columns. Default: all columns in the namespace.",
        ),
        overwrite: bool = typer.Option(
            True,
            "--overwrite/--no-overwrite",
            help="Overwrite existing overlay values for matched destination ids.",
        ),
        allow_missing: bool = typer.Option(
            False,
            "--allow-missing",
            help="Allow destination rows whose join values do not resolve in the source dataset.",
        ),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        cmd_overlay_project(
            ctx_args_builder(
                ctx,
                src=src,
                dest=dest,
                namespace=namespace,
                src_join=src_join,
                dest_join=dest_join,
                columns=columns,
                overwrite=overwrite,
                allow_missing=allow_missing,
                dry_run=dry_run,
            )
        )

    @maintenance_app.command("merge")
    def cli_merge_datasets(
        ctx: typer.Context,
        dest: str = typer.Option(..., "--dest"),
        src: str = typer.Option(..., "--src"),
        require_same_columns: bool = typer.Option(False, "--require-same-columns"),
        union_columns: bool = typer.Option(False, "--union-columns"),
        dup_policy: str = typer.Option("error", "--if-duplicate"),
        coerce_overlap: str = typer.Option("none", "--coerce-overlap"),
        carry_namespaces: list[str] | None = typer.Option(None, "--carry-namespace"),
        no_avoid_casefold_dups: bool = typer.Option(False, "--no-avoid-casefold-dups"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        cmd_merge_datasets(
            ctx_args_builder(
                ctx,
                dest=dest,
                src=src,
                require_same=require_same_columns,
                union_columns=union_columns,
                dup_policy=dup_policy,
                coerce_overlap=coerce_overlap,
                carry_namespaces=carry_namespaces,
                no_avoid_casefold_dups=no_avoid_casefold_dups,
                dry_run=dry_run,
            )
        )
