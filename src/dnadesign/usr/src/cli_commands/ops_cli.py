"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/ops_cli.py

Typer registration helpers for USR operations subcommands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_ops_commands(
    maintenance_app: typer.Typer,
    densegen_app: typer.Typer,
    dev_app: typer.Typer,
    legacy_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_dedupe_sequences: Callable[[object], None],
    cmd_registry_freeze: Callable[[object], None],
    cmd_overlay_compact: Callable[[object], None],
    cmd_repair_densegen: Callable[[object], None],
    cmd_make_mock: Callable[[object], None],
    cmd_add_demo: Callable[[object], None],
    cmd_convert_legacy: Callable[[object], None],
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

    @densegen_app.command("repair")
    def cli_repair_densegen(
        ctx: typer.Context,
        dataset: str = typer.Argument(None),
        min_tfbs_len: int = typer.Option(6, "--min-tfbs-len"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        yes: bool = typer.Option(False, "--yes"),
        dedupe: str = typer.Option("off", "--dedupe"),
        drop_missing_used_tfbs: bool = typer.Option(False, "--drop-missing-used-tfbs"),
        drop_single_tf: bool = typer.Option(False, "--drop-single-tf"),
        drop_id_seq_only: bool = typer.Option(False, "--drop-id-seq-only"),
        filter_single_tf: bool = typer.Option(False, "--filter-single-tf"),
    ) -> None:
        cmd_repair_densegen(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                min_tfbs_len=min_tfbs_len,
                dry_run=dry_run,
                yes=yes,
                dedupe=dedupe,
                drop_missing_used_tfbs=drop_missing_used_tfbs,
                drop_single_tf=drop_single_tf,
                drop_id_seq_only=drop_id_seq_only,
                filter_single_tf=filter_single_tf,
            )
        )

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

    @legacy_app.command("convert")
    def cli_convert_legacy(
        ctx: typer.Context,
        dataset: str = typer.Argument(...),
        paths: list[Path] = typer.Argument(...),
        expected_length: int | None = typer.Option(None, "--expected-length"),
        plan: str | None = typer.Option(None, "--plan"),
        force: bool = typer.Option(False, "--force"),
        profile_60bp: bool = typer.Option(True, "--profile-60bp/--no-profile-60bp"),
    ) -> None:
        cmd_convert_legacy(
            ctx_args_builder(
                ctx,
                dataset=dataset,
                paths=paths,
                expected_length=expected_length,
                plan=plan,
                force=force,
                profile_60bp=profile_60bp,
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
                no_avoid_casefold_dups=no_avoid_casefold_dups,
                dry_run=dry_run,
            )
        )
