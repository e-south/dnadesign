"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/lifecycle_cli.py

Typer registration helpers for USR dataset lifecycle commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_lifecycle_commands(
    app: typer.Typer,
    state_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_init: Callable[[object], None],
    cmd_import: Callable[[object], None],
    cmd_attach: Callable[[object], None],
    cmd_delete: Callable[[object], None],
    cmd_restore: Callable[[object], None],
    cmd_state_set: Callable[[object], None],
    cmd_state_clear: Callable[[object], None],
    cmd_state_get: Callable[[object], None],
    cmd_materialize: Callable[[object], None],
    cmd_snapshot: Callable[[object], None],
) -> None:
    @app.command("init", help="Initialize dataset metadata. Requires registry.yaml under --root.")
    def cli_init(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        source: str = typer.Option("", "--source"),
        notes: str = typer.Option("", "--notes"),
    ) -> None:
        cmd_init(ctx_args_builder(ctx, dataset=dataset, source=source, notes=notes))

    @app.command("import")
    def cli_import(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        source_format: str = typer.Option(..., "--from", help="Source format", case_sensitive=False),
        path: Path = typer.Option(..., "--path", exists=True, readable=True, path_type=Path),
        bio_type: str = typer.Option("dna", "--bio-type"),
        alphabet: str = typer.Option("dna_4", "--alphabet"),
    ) -> None:
        cmd_import(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                source_format=source_format,
                path=path,
                bio_type=bio_type,
                alphabet=alphabet,
            )
        )

    @app.command("attach")
    def cli_attach(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        path: Path = typer.Option(..., "--path", exists=True, readable=True, path_type=Path),
        namespace: str = typer.Option(..., "--namespace"),
        key: str = typer.Option(..., "--key"),
        key_col: str = typer.Option("", "--key-col"),
        columns: str = typer.Option("", "--columns"),
        allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
        allow_missing: bool = typer.Option(False, "--allow-missing"),
        parse_json: bool = typer.Option(True, "--parse-json/--no-parse-json"),
        backend: str = typer.Option("pyarrow", "--backend"),
        note: str = typer.Option("", "--note"),
    ) -> None:
        cmd_attach(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                path=path,
                namespace=namespace,
                key=key,
                key_col=key_col,
                columns=columns,
                allow_overwrite=allow_overwrite,
                allow_missing=allow_missing,
                parse_json=parse_json,
                backend=backend,
                note=note,
            )
        )

    @app.command("delete")
    def cli_delete(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        id: list[str] = typer.Option(None, "--id"),
        id_file: Path | None = typer.Option(None, "--id-file"),
        reason: str = typer.Option("", "--reason"),
        allow_missing: bool = typer.Option(False, "--allow-missing"),
    ) -> None:
        cmd_delete(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                id=id,
                id_file=id_file,
                reason=reason,
                allow_missing=allow_missing,
            )
        )

    @app.command("restore")
    def cli_restore(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        id: list[str] = typer.Option(None, "--id"),
        id_file: Path | None = typer.Option(None, "--id-file"),
        allow_missing: bool = typer.Option(False, "--allow-missing"),
    ) -> None:
        cmd_restore(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                id=id,
                id_file=id_file,
                allow_missing=allow_missing,
            )
        )

    @state_app.command("set")
    def cli_state_set(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        id: list[str] = typer.Option(None, "--id"),
        id_file: Path | None = typer.Option(None, "--id-file"),
        masked: bool | None = typer.Option(None, "--masked/--unmasked"),
        qc_status: str = typer.Option("", "--qc-status"),
        split: str = typer.Option("", "--split"),
        supersedes: str = typer.Option("", "--supersedes"),
        lineage: list[str] = typer.Option(None, "--lineage"),
        allow_missing: bool = typer.Option(False, "--allow-missing"),
    ) -> None:
        cmd_state_set(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                id=id,
                id_file=id_file,
                masked=masked,
                qc_status=qc_status,
                split=split,
                supersedes=supersedes,
                lineage=lineage,
                allow_missing=allow_missing,
            )
        )

    @state_app.command("clear")
    def cli_state_clear(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        id: list[str] = typer.Option(None, "--id"),
        id_file: Path | None = typer.Option(None, "--id-file"),
        allow_missing: bool = typer.Option(False, "--allow-missing"),
    ) -> None:
        cmd_state_clear(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                id=id,
                id_file=id_file,
                allow_missing=allow_missing,
            )
        )

    @state_app.command("get")
    def cli_state_get(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        id: list[str] = typer.Option(None, "--id"),
        id_file: Path | None = typer.Option(None, "--id-file"),
        allow_missing: bool = typer.Option(False, "--allow-missing"),
        format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
    ) -> None:
        cmd_state_get(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                id=id,
                id_file=id_file,
                allow_missing=allow_missing,
                format=format,
            )
        )

    @app.command("materialize")
    def cli_materialize(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        yes: bool = typer.Option(False, "--yes", "-y"),
        snapshot_before: bool = typer.Option(False, "--snapshot-before"),
        namespaces: str = typer.Option("", "--namespaces", help="Comma-separated overlay namespaces to materialize"),
        drop_overlays: bool = typer.Option(False, "--drop-overlays"),
        archive_overlays: bool = typer.Option(False, "--archive-overlays"),
        drop_deleted: bool = typer.Option(False, "--drop-deleted"),
    ) -> None:
        cmd_materialize(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                yes=yes,
                snapshot_before=snapshot_before,
                namespaces=namespaces,
                drop_overlays=drop_overlays,
                archive_overlays=archive_overlays,
                drop_deleted=drop_deleted,
            )
        )

    @app.command("snapshot")
    def cli_snapshot(ctx: typer.Context, dataset: str = typer.Argument(None)) -> None:
        cmd_snapshot(ctx_args_builder(ctx, dataset=dataset))
