"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/genbank/cli.py

Typer registration helpers for GenBank import commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ....genbank import import_genbank_manifest


def register_genbank_commands(genbank_app: typer.Typer, *, ctx_args_builder) -> None:
    @genbank_app.command("import")
    def _import(
        ctx: typer.Context,
        manifest: Path = typer.Option(
            ...,
            "--manifest",
            exists=True,
            dir_okay=False,
            file_okay=True,
            readable=True,
            path_type=Path,
            help="Path to a usr.genbank_import manifest.",
        ),
    ) -> None:
        args = ctx_args_builder(ctx)
        result = import_genbank_manifest(
            root=args.root,
            manifest_path=manifest,
            actor=None,
        )
        typer.echo(
            "Imported "
            f"{result.native_records} native record(s), "
            f"{result.extracted_records} extracted record(s), "
            f"and {result.sequence_views_written} sequence view row(s) into {result.dataset}."
        )


__all__ = ["register_genbank_commands"]
