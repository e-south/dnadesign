"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/src/cli/commands/tooling/cli.py

Typer registration helpers for SeqLedger tooling subcommands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_ops_commands(
    dev_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_make_mock: Callable[[object], None],
    cmd_add_demo: Callable[[object], None],
) -> None:
    @dev_app.command("make-mock")
    def cli_make_mock(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        n: int = typer.Option(100, "--n"),
        length: int = typer.Option(60, "--length"),
        x_dim: int = typer.Option(512, "--x-dim"),
        y_dim: int = typer.Option(8, "--y-dim"),
        seed: int = typer.Option(7, "--seed"),
        namespace: str = typer.Option("demo", "--namespace"),
        from_csv: str = typer.Option("", "--from-csv"),
        force: bool = typer.Option(False, "--force"),
    ) -> None:
        cmd_make_mock(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                n=n,
                length=length,
                x_dim=x_dim,
                y_dim=y_dim,
                seed=seed,
                namespace=namespace,
                from_csv=from_csv,
                force=force,
            )
        )

    @dev_app.command("add-demo-cols")
    def cli_add_demo_cols(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        x_dim: int = typer.Option(512, "--x-dim"),
        y_dim: int = typer.Option(8, "--y-dim"),
        seed: int = typer.Option(7, "--seed"),
        namespace: str = typer.Option("demo", "--namespace"),
        allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
    ) -> None:
        cmd_add_demo(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                x_dim=x_dim,
                y_dim=y_dim,
                seed=seed,
                namespace=namespace,
                allow_overwrite=allow_overwrite,
            )
        )
