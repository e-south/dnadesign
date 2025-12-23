"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/app.py

Typer-based CLI surface for USR. Wraps existing argparse command implementations
in dnadesign.usr.src.cli to keep logic centralized and decoupled.

Key commands gain a --rich/--no-rich toggle (default: rich on).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import typer

from .cli import (
    _pkg_usr_root,
    cmd_describe,
    cmd_diff,
    cmd_head,
    cmd_ls,
    cmd_schema,
    cmd_validate,
)

app = typer.Typer(add_completion=True, no_args_is_help=True, help="USR datasets CLI (Typer)")


@app.command()
def ls(ctx: typer.Context) -> None:
    """List datasets under --root (pretty by default)."""
    args = NS(root=ctx.obj["root"], rich=ctx.obj["rich"])
    cmd_ls(args)


@app.callback()
def _root(
    ctx: typer.Context,
    root: Path = typer.Option(
        (_pkg_usr_root() / "datasets").resolve(),
        "--root",
        help="Datasets root folder",
        readable=True,
        exists=True,
        dir_okay=True,
        file_okay=False,
        path_type=Path,
    ),
    rich: bool = typer.Option(True, "--rich/--no-rich", help="Use Rich formatting for supported commands"),
) -> None:
    ctx.obj = {"root": root, "rich": rich}


@app.command()
def head(
    ctx: typer.Context,
    target: str = typer.Argument(".", help="Dataset name OR a filesystem path"),
    n: int = typer.Option(10, "-n", help="Rows to show"),
    raw: bool = typer.Option(False, "--raw", help="Disable compact value formatting"),
    max_colwidth: int = typer.Option(80, help="Max cell width"),
    max_list_items: int = typer.Option(6, help="Items to show for lists"),
    precision: int = typer.Option(4, help="Numeric precision"),
) -> None:
    args = NS(
        root=ctx.obj["root"],
        rich=ctx.obj["rich"],
        target=target,
        n=n,
        raw=raw,
        max_colwidth=max_colwidth,
        max_list_items=max_list_items,
        precision=precision,
    )
    cmd_head(args)


@app.command()
def describe(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    columns: str = typer.Option("", help="CSV list of columns to include"),
    sample: int = typer.Option(1024, help="Sample size for examples/stats"),
    max_colwidth: int = typer.Option(80, help="Max cell width"),
    max_list_items: int = typer.Option(6, help="Items to show for lists"),
    precision: int = typer.Option(4, help="Numeric precision"),
) -> None:
    args = NS(
        root=ctx.obj["root"],
        rich=ctx.obj["rich"],
        dataset=dataset,
        columns=columns,
        sample=sample,
        max_colwidth=max_colwidth,
        max_list_items=max_list_items,
        precision=precision,
    )
    cmd_describe(args)


@app.command()
def schema(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    tree: bool = typer.Option(False, "--tree", help="Pretty tree view"),
) -> None:
    args = NS(root=ctx.obj["root"], rich=ctx.obj["rich"], dataset=dataset, tree=tree)
    cmd_schema(args)


@app.command()
def validate(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    strict: bool = typer.Option(False),
) -> None:
    args = NS(root=ctx.obj["root"], dataset=dataset, strict=strict)
    cmd_validate(args)


@app.command()
def diff(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    remote: str = typer.Argument(...),
) -> None:
    args = NS(root=ctx.obj["root"], dataset=dataset, remote=remote, rich=ctx.obj["rich"])
    cmd_diff(args)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("Aborted by user (Ctrl-C).", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
