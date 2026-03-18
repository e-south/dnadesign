"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_table.py

Table mutation execution runtime for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .execution_support import (
    CommandExecution,
    _log,
    _rule,
    attach_columns_schema_preserving,
    context_and_df,
    intra_sim_overlay_col,
    progress_scope,
)
from .io.read import peek_columns
from .io.write import attach_usr, drop_usr_columns, write_generic
from .util.slug import slugify


def run_delete_columns(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    all_: bool,
    names: list[str],
    columns: list[str],
    write: bool,
    yes: bool,
    inplace: bool,
    out: str | None,
    console: Console | None = None,
) -> None:
    ictx, _ = context_and_df(dataset, file, usr_root, columns=None)
    _rule(console, "[bold]cluster delete-columns[/]")
    _log(console, "log", f"Input: kind={ictx['kind']} ref={ictx.get('dataset') or ictx.get('file')}")
    if sum(bool(x) for x in [all_, bool(names), bool(columns)]) != 1:
        raise typer.BadParameter("Choose exactly one: --all OR --name ... OR --column ...")

    cols = peek_columns(ictx)
    cluster_cols = [c for c in cols if c.startswith("cluster__")]

    if all_:
        to_delete = cluster_cols
        reason = "all cluster__*"
    elif names:
        aliases = [slugify(name) for name in names]
        prefixes = [f"cluster__{alias}" for alias in aliases]
        to_delete = [c for c in cluster_cols if any(c == p or c.startswith(p + "__") for p in prefixes)]
        reason = "name=" + ",".join(aliases)
    else:
        normalized_requested = [column.split(".", 1)[0] for column in columns]
        bad = [column for column in normalized_requested if not column.startswith("cluster__")]
        if bad:
            raise typer.BadParameter("Only 'cluster__*' columns can be deleted; offending: " + ", ".join(bad[:6]))
        to_delete = [column for column in normalized_requested if column in cluster_cols]
        missing = [column for column in normalized_requested if column not in cols]
        if missing:
            _log(
                console,
                "print",
                "[yellow]Note[/yellow]: the following columns were not found and will be ignored: "
                + ", ".join(missing[:8])
                + (" ..." if len(missing) > 8 else ""),
            )
        reason = "explicit columns"

    if not to_delete:
        _log(console, "print", "[green]Nothing to delete[/green]: no matching cluster__ columns found.")
        return

    if console is not None:
        table = Table(title=f"Columns to delete ({len(to_delete)}) — scope: {reason}", header_style="bold cyan")
        table.add_column("Column")
        for column in sorted(to_delete):
            table.add_row(column)
        console.print(table)

    if not yes and console is not None:
        if not typer.confirm(f"Are you sure you want to permanently delete {len(to_delete)} column(s)?"):
            raise typer.Abort()

    if not write:
        _log(console, "print", "[yellow]Dry-run[/yellow]: no changes applied. Re-run with --write to proceed.")
        return

    if ictx["kind"] == "usr":
        drop_usr_columns(ictx["usr_root"], ictx["dataset"], to_delete)
        _log(
            console,
            "print",
            f"[green]Removed[/green] {len(to_delete)} column(s) from USR dataset '{ictx['dataset']}'.",
        )
        return

    df = pd.read_parquet(ictx["file"]) if ictx["kind"] == "parquet" else pd.read_csv(ictx["file"])
    missing_at_exec = [column for column in to_delete if column not in df.columns]
    if missing_at_exec:
        _log(
            console,
            "print",
            "[yellow]Note[/yellow]: some columns disappeared during load and were skipped: "
            + ", ".join(missing_at_exec[:8])
            + (" ..." if len(missing_at_exec) > 8 else ""),
        )
    kept = [column for column in to_delete if column in df.columns]
    if not kept:
        _log(console, "print", "[green]Nothing left to delete[/green].")
        return
    df = df.drop(columns=kept)
    write_generic(ictx["file"], df, inplace=inplace, out=(Path(out) if out else None), backup_suffix=".bak")
    _log(console, "print", f"[green]Wrote[/green] updated file ({'inplace' if inplace else 'out=' + str(out)}).")
    if console is not None:
        recap = Table(title="Deleted columns recap", header_style="bold cyan")
        recap.add_column("Count", justify="right")
        recap.add_column("Preview")
        preview = ", ".join(sorted(to_delete)[:6]) + (" ..." if len(to_delete) > 6 else "")
        recap.add_row(str(len(to_delete)), preview)
        console.print(recap)


def run_intra_similarity(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    cluster_col: str,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    max_per_cluster: int,
    sample_if_larger: bool,
    write: bool,
    allow_overwrite: bool,
    inplace: bool,
    out: str | None,
    console: Console | None = None,
) -> CommandExecution:
    from .analysis.intra_similarity import intra_cluster_similarity

    ictx, df = context_and_df(dataset, file, usr_root)
    if cluster_col not in df.columns:
        raise typer.BadParameter(f"Cluster column '{cluster_col}' not found.")
    out_col = intra_sim_overlay_col(cluster_col)
    with progress_scope(console) as progress:
        task = progress.add_task("Computing intra-cluster similarity...", total=None)
        scores = intra_cluster_similarity(
            df,
            cluster_col=cluster_col,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
            max_per_cluster=max_per_cluster,
            sample_if_larger=sample_if_larger,
        )
        progress.update(task, completed=1)
    if not write:
        _log(
            console,
            "print",
            f"[yellow]Dry-run[/yellow]: computed intra-sim but did not write. Use --write to attach {out_col}.",
        )
        return CommandExecution(command="intra-sim", subject=out_col, artifact_path=Path(ictx["file"]))
    cols = pd.DataFrame({"id": df["id"].astype(str), out_col: scores})
    if ictx["kind"] == "usr":
        try:
            attach_usr(ictx["usr_root"], ictx["dataset"], cols, allow_overwrite=allow_overwrite)
        except Exception as exc:
            if "Columns already exist" in str(exc) and not allow_overwrite:
                raise RuntimeError("Columns already exist. Re-run with `-y/--allow-overwrite`.") from exc
            raise
        _log(console, "print", "[green]Attached[/green] intra-sim to USR dataset.")
    else:
        merged = attach_columns_schema_preserving(df, cols, "id", allow_overwrite=allow_overwrite)
        write_generic(ictx["file"], merged, inplace=inplace, out=(Path(out) if out else None), backup_suffix=".bak")
        _log(console, "print", "[green]Wrote[/green] updated file with intra-sim column.")
    return CommandExecution(command="intra-sim", subject=out_col, artifact_path=Path(ictx["file"]))


__all__ = ["run_delete_columns", "run_intra_similarity"]
