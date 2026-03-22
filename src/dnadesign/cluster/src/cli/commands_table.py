"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/commands_table.py

Table-mutation cluster CLI command registration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console


def register_delete_columns_command(app: typer.Typer, *, console: Console) -> None:
    @app.command(
        "delete-columns",
        help="Delete cluster__*-namespaced columns from a dataset/file with a safety preview and confirmation.",
    )
    def cmd_delete_columns(
        dataset: Optional[str] = typer.Option(None, help="USR dataset name"),
        file: Optional[str] = typer.Option(None, help="Parquet/CSV path"),
        usr_root: Optional[str] = typer.Option(None, help="USR root directory"),
        all_: bool = typer.Option(False, "--all", help="Delete ALL cluster__* columns"),
        name: List[str] = typer.Option(
            [],
            "--name",
            help="Delete columns for this fit alias (repeatable). Matches cluster__<name> and cluster__<name>__*",
        ),
        column: List[str] = typer.Option(
            [],
            "--column",
            help="Delete this fully-qualified column (repeatable). Must start with cluster__",
        ),
        write: bool = typer.Option(False, help="Apply changes (default is dry-run)"),
        yes: bool = typer.Option(False, "-y", "--yes", help="Skip interactive confirmation"),
        inplace: bool = typer.Option(
            False,
            help="For generic files: rewrite the input file in place (backs up to .bak)",
        ),
        out: Optional[str] = typer.Option(
            None,
            help="For generic files: write to this output path instead of --inplace",
        ),
    ) -> None:
        from ..execution import run_delete_columns

        run_delete_columns(
            dataset=dataset,
            file=file,
            usr_root=usr_root,
            all_=all_,
            names=name,
            columns=column,
            write=write,
            yes=yes,
            inplace=inplace,
            out=out,
            console=console,
        )


def register_intra_similarity_command(app: typer.Typer, *, console: Console) -> None:
    @app.command("intra-sim")
    def cmd_intra_sim(
        dataset: Optional[str] = typer.Option(None),
        file: Optional[str] = typer.Option(None),
        usr_root: Optional[str] = typer.Option(None),
        cluster_col: str = typer.Option(...),
        match: int = typer.Option(2),
        mismatch: int = typer.Option(-1),
        gap_open: int = typer.Option(10),
        gap_extend: int = typer.Option(1),
        max_per_cluster: int = typer.Option(2000),
        sample_if_larger: bool = typer.Option(True),
        write: bool = typer.Option(False),
        yes: bool = typer.Option(False, "-y", "--allow-overwrite"),
        inplace: bool = typer.Option(False),
        out: Optional[str] = typer.Option(None),
    ) -> None:
        from ..execution import run_intra_similarity

        run_intra_similarity(
            dataset=dataset,
            file=file,
            usr_root=usr_root,
            cluster_col=cluster_col,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
            max_per_cluster=max_per_cluster,
            sample_if_larger=sample_if_larger,
            write=write,
            allow_overwrite=yes,
            inplace=inplace,
            out=out,
            console=console,
        )


__all__ = ["register_delete_columns_command", "register_intra_similarity_command"]
