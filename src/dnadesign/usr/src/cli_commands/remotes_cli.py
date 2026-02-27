"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/remotes_cli.py

Typer registration helpers for USR remotes commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_remotes_commands(
    remotes_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_remotes_list: Callable[[object], None],
    cmd_remotes_show: Callable[[object], None],
    cmd_remotes_add: Callable[[object], None],
    cmd_remotes_wizard: Callable[[object], None],
    cmd_remotes_doctor: Callable[[object], None],
) -> None:
    @remotes_app.command("list")
    def cli_remotes_list(ctx: typer.Context) -> None:
        cmd_remotes_list(ctx_args_builder(ctx))

    @remotes_app.command("show")
    def cli_remotes_show(ctx: typer.Context, name: str = typer.Argument(...)) -> None:
        cmd_remotes_show(ctx_args_builder(ctx, name=name))

    @remotes_app.command("add")
    def cli_remotes_add(
        ctx: typer.Context,
        name: str = typer.Argument(...),
        type: str = typer.Option("ssh", "--type"),
        host: str = typer.Option(..., "--host"),
        user: str = typer.Option(..., "--user"),
        base_dir: str = typer.Option(..., "--base-dir"),
        ssh_key_env: str | None = typer.Option(None, "--ssh-key-env"),
    ) -> None:
        cmd_remotes_add(
            ctx_args_builder(
                ctx,
                name=name,
                type=type,
                host=host,
                user=user,
                base_dir=base_dir,
                ssh_key_env=ssh_key_env,
            )
        )

    @remotes_app.command("wizard")
    def cli_remotes_wizard(
        ctx: typer.Context,
        preset: str = typer.Option("bu-scc", "--preset", help="Wizard preset: bu-scc."),
        name: str = typer.Option("bu-scc", "--name", help="Remote config name."),
        user: str = typer.Option(..., "--user", help="Remote SSH username."),
        base_dir: str = typer.Option(..., "--base-dir", help="Remote dataset root path."),
        host: str = typer.Option(
            "",
            "--host",
            help="Remote host override. Defaults to scc1.bu.edu, or scc-globus.bu.edu with --transfer-node.",
        ),
        transfer_node: bool = typer.Option(
            False,
            "--transfer-node",
            help="Use BU SCC transfer host default (scc-globus.bu.edu).",
        ),
        ssh_key_env: str | None = typer.Option(None, "--ssh-key-env"),
    ) -> None:
        cmd_remotes_wizard(
            ctx_args_builder(
                ctx,
                preset=preset,
                name=name,
                user=user,
                base_dir=base_dir,
                host=host,
                transfer_node=transfer_node,
                ssh_key_env=ssh_key_env,
            )
        )

    @remotes_app.command("doctor")
    def cli_remotes_doctor(
        ctx: typer.Context,
        remote: str = typer.Option(..., "--remote", help="Configured remote name."),
        check_base_dir: bool = typer.Option(True, "--check-base-dir/--no-check-base-dir"),
    ) -> None:
        cmd_remotes_doctor(ctx_args_builder(ctx, remote=remote, check_base_dir=check_base_dir))
