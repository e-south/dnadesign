"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/sync_cli.py

Typer registration helpers for USR sync commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_sync_commands(
    app: typer.Typer,
    *,
    sync_args_builder: Callable[..., object],
    cmd_diff: Callable[[object], None],
    cmd_pull: Callable[[object], None],
    cmd_push: Callable[[object], None],
) -> None:
    @app.command("diff")
    def cli_diff(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        remote: str = typer.Argument(...),
        primary_only: bool = typer.Option(False, "--primary-only"),
        skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        yes: bool = typer.Option(False, "--yes"),
        verify: str = typer.Option("hash", "--verify", help="Verification mode: hash|auto|size|parquet"),
        format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
        repo_root: str | None = typer.Option(None, "--repo-root"),
        remote_path: str | None = typer.Option(None, "--remote-path"),
    ) -> None:
        cmd_diff(
            sync_args_builder(
                ctx,
                dataset=dataset,
                remote=remote,
                primary_only=primary_only,
                skip_snapshots=skip_snapshots,
                dry_run=dry_run,
                yes=yes,
                verify=verify,
                format=format,
                repo_root=repo_root,
                remote_path=remote_path,
            )
        )

    @app.command("status")
    def cli_status(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        remote: str = typer.Argument(...),
        primary_only: bool = typer.Option(False, "--primary-only"),
        skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        yes: bool = typer.Option(False, "--yes"),
        verify: str = typer.Option("hash", "--verify", help="Verification mode: hash|auto|size|parquet"),
        repo_root: str | None = typer.Option(None, "--repo-root"),
        remote_path: str | None = typer.Option(None, "--remote-path"),
    ) -> None:
        cmd_diff(
            sync_args_builder(
                ctx,
                dataset=dataset,
                remote=remote,
                primary_only=primary_only,
                skip_snapshots=skip_snapshots,
                dry_run=dry_run,
                yes=yes,
                verify=verify,
                format=None,
                repo_root=repo_root,
                remote_path=remote_path,
            )
        )

    @app.command("pull")
    def cli_pull(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        remote: str = typer.Argument(...),
        primary_only: bool = typer.Option(False, "--primary-only"),
        skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        yes: bool = typer.Option(False, "--yes"),
        verify: str = typer.Option("hash", "--verify", help="Verification mode: hash|auto|size|parquet"),
        verify_sidecars: bool = typer.Option(
            False,
            "--verify-sidecars",
            help="Enable strict sidecar fidelity checks for dataset sync (already default for datasets).",
        ),
        no_verify_sidecars: bool = typer.Option(
            False,
            "--no-verify-sidecars",
            help="Disable strict sidecar fidelity checks for dataset sync.",
        ),
        verify_derived_hashes: bool = typer.Option(
            False,
            "--verify-derived-hashes",
            help="Also verify _derived file content hashes (high assurance, slower).",
        ),
        repo_root: str | None = typer.Option(None, "--repo-root"),
        remote_path: str | None = typer.Option(None, "--remote-path"),
        strict_bootstrap_id: bool = typer.Option(
            False,
            "--strict-bootstrap-id",
            help="Require namespace-qualified dataset ids (<namespace>/<dataset>) for bootstrap pulls.",
        ),
    ) -> None:
        cmd_pull(
            sync_args_builder(
                ctx,
                dataset=dataset,
                remote=remote,
                primary_only=primary_only,
                skip_snapshots=skip_snapshots,
                dry_run=dry_run,
                yes=yes,
                verify=verify,
                format=None,
                repo_root=repo_root,
                remote_path=remote_path,
                strict_bootstrap_id=strict_bootstrap_id,
                verify_sidecars=verify_sidecars,
                no_verify_sidecars=no_verify_sidecars,
                verify_derived_hashes=verify_derived_hashes,
            )
        )

    @app.command("push")
    def cli_push(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        remote: str = typer.Argument(...),
        primary_only: bool = typer.Option(False, "--primary-only"),
        skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        yes: bool = typer.Option(False, "--yes"),
        verify: str = typer.Option("hash", "--verify", help="Verification mode: hash|auto|size|parquet"),
        verify_sidecars: bool = typer.Option(
            False,
            "--verify-sidecars",
            help="Enable strict sidecar fidelity checks for dataset sync (already default for datasets).",
        ),
        no_verify_sidecars: bool = typer.Option(
            False,
            "--no-verify-sidecars",
            help="Disable strict sidecar fidelity checks for dataset sync.",
        ),
        verify_derived_hashes: bool = typer.Option(
            False,
            "--verify-derived-hashes",
            help="Also verify _derived file content hashes (high assurance, slower).",
        ),
        repo_root: str | None = typer.Option(None, "--repo-root"),
        remote_path: str | None = typer.Option(None, "--remote-path"),
    ) -> None:
        cmd_push(
            sync_args_builder(
                ctx,
                dataset=dataset,
                remote=remote,
                primary_only=primary_only,
                skip_snapshots=skip_snapshots,
                dry_run=dry_run,
                yes=yes,
                verify=verify,
                format=None,
                repo_root=repo_root,
                remote_path=remote_path,
                strict_bootstrap_id=False,
                verify_sidecars=verify_sidecars,
                no_verify_sidecars=no_verify_sidecars,
                verify_derived_hashes=verify_derived_hashes,
            )
        )
