"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_support.py

Shared helpers for cluster execution command modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from .io.detect import detect_context
from .io.read import load_table
from .runs.contracts import fit_alias_from_cluster_col
from .runs.recorder import CommandRecord, append_command_record_entry
from .util.checks import assert_no_duplicate_ids


@dataclass(frozen=True, slots=True)
class CommandExecution:
    command: str
    subject: str
    artifact_path: Path
    run_record_subject: str | None = None


def _log(console: Console | None, method: str, message: str) -> None:
    if console is None:
        return
    getattr(console, method)(message)


def _rule(console: Console | None, message: str) -> None:
    if console is None:
        return
    console.rule(message)


def progress_scope(console: Console | None) -> Progress:
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        transient=console is not None,
        console=console,
        disable=console is None,
    )


def _apply_dedupe(df: pd.DataFrame, key_col: str, policy: str) -> pd.DataFrame:
    return assert_no_duplicate_ids(df, key_col=key_col, policy=policy)


def context_and_df(
    dataset: Optional[str],
    file: Optional[str],
    usr_root: Optional[str],
    columns: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ctx = detect_context(dataset, file, usr_root)
    return ctx, load_table(ctx, columns=columns)


def _rows_ids(df: pd.DataFrame, key_col: str) -> list[str]:
    return list(map(str, df[key_col].tolist()))


def _collect_existing_meta_sig(df: pd.DataFrame, name: str) -> Optional[str]:
    col = f"cluster__{name}__meta"
    if col in df.columns:
        try:
            obj = json.loads(df[col].dropna().iloc[0])
            return obj.get("sig")
        except Exception:
            return None
    return None


def attach_columns_schema_preserving(
    full_df: pd.DataFrame,
    cols_df: pd.DataFrame,
    key_col: str,
    *,
    allow_overwrite: bool,
) -> pd.DataFrame:
    if key_col not in full_df.columns:
        raise KeyError(f"Left table is missing key column '{key_col}'.")
    left = full_df.reset_index(drop=True) if full_df.index.name == key_col else full_df
    right = cols_df
    if right.index.name == key_col and key_col in right.columns:
        right = right.reset_index(drop=True)
    elif right.index.name == key_col and key_col not in right.columns:
        right = right.reset_index()
    if key_col not in right.columns:
        raise KeyError(f"Right table is missing key column '{key_col}'.")
    if right[key_col].duplicated().any():
        dupes = right.loc[right[key_col].duplicated(), key_col].astype(str).head(8).tolist()
        raise RuntimeError(f"Right table has duplicate '{key_col}' values (e.g., {dupes}).")
    try:
        right = right.copy()
        right[key_col] = right[key_col].astype(left[key_col].dtype)
    except Exception:
        left = left.copy()
        left[key_col] = left[key_col].astype(str)
        right[key_col] = right[key_col].astype(str)
    li = left.set_index(key_col, drop=False)
    ri = right.set_index(key_col, drop=False)
    to_attach = [c for c in ri.columns if c != key_col]
    if not to_attach:
        return left
    existing = [c for c in to_attach if c in li.columns]
    if existing and not allow_overwrite:
        raise RuntimeError(
            "Columns already exist: "
            + ", ".join(existing[:8])
            + (" ..." if len(existing) > 8 else "")
            + ". Re-run with `-y/--allow-overwrite` or use a new --name."
        )
    for column in to_attach:
        li[column] = ri[column].reindex(li.index).values
    return li.reset_index(drop=True) if full_df.index.name != key_col else li


def assert_preserve_columns(before: list[str], after: list[str]) -> None:
    missing = [c for c in before if c not in after]
    if missing:
        raise RuntimeError(
            "Refusing to write: detected potential column drop.\n"
            "Columns that would be lost: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")
        )


def resolve_color_by(
    cli_val: list[str],
    config_params: dict[str, Any],
    config_plot_cfg: dict[str, Any],
    preset_plot_cfg: dict[str, Any],
) -> list[str]:
    if cli_val and not (len(cli_val) == 1 and cli_val[0] == "cluster"):
        return list(cli_val)
    if isinstance(config_plot_cfg.get("color_by"), (list, tuple)):
        return list(config_plot_cfg["color_by"])
    if isinstance(config_params.get("color_by"), (list, tuple)):
        return list(config_params["color_by"])
    if isinstance(preset_plot_cfg.get("color_by"), (list, tuple)):
        return list(preset_plot_cfg["color_by"])
    return ["cluster"]


def resolve_scoped_out_dir(*, requested: str | None, root: Path) -> Path:
    if requested is None:
        return root
    requested_path = Path(requested).expanduser()
    resolved = (root / requested_path).resolve() if not requested_path.is_absolute() else requested_path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise typer.BadParameter(f"Output path '{resolved}' must stay inside '{root}'.")
    return resolved


def append_command_record_or_warn(run_dir: Path, record: CommandRecord, *, console: Console | None = None) -> None:
    try:
        append_command_record_entry(run_dir, record)
    except Exception as exc:
        _log(console, "print", f"[yellow]Warning[/yellow]: failed to append records.md entry under {run_dir}: {exc}")


def cluster_overlay_col(run_alias: str, suffix: str | None = None) -> str:
    base = f"cluster__{run_alias}"
    return base if suffix is None else f"{base}__{suffix}"


def intra_sim_overlay_col(cluster_col: str) -> str:
    run_alias = fit_alias_from_cluster_col(cluster_col)
    if run_alias is None:
        raise typer.BadParameter(
            "--cluster-col must be a fit label column of the form 'cluster__<NAME>' so intra-sim stays namespaced."
        )
    return cluster_overlay_col(run_alias, "intra_sim")


def load_highlight_ids_from_file(
    path_str: str,
    df: pd.DataFrame,
    key_col: str,
    warn_fn: Optional[Callable[[str], None]] = None,
    groupby_col: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise typer.BadParameter(f"--highlight path not found: {path}")
    tab = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    column = "id" if "id" in tab.columns else tab.columns[0]
    raw_ids = set(map(str, tab[column].tolist()))
    left = df if df.index.name == key_col else df.set_index(key_col, drop=False)
    present = raw_ids & set(map(str, left.index.astype(str).tolist()))
    missing = raw_ids - present
    if missing and warn_fn:
        warn_fn(f"{len(missing)} id(s) in highlight were not found in the dataset. They will be ignored.")
    out: dict[str, Any] = {"ids": list(present)}
    if groupby_col is not None:
        if groupby_col not in tab.columns:
            raise typer.BadParameter(f"--highlight-hue-col='{groupby_col}' not found in {path.name}.")
        sub = tab[[column, groupby_col]].copy()
        sub[column] = sub[column].astype(str)
        sub[groupby_col] = sub[groupby_col].astype(str)
        labels = {rid: cat for rid, cat in zip(sub[column].tolist(), sub[groupby_col].tolist()) if rid in present}
        categories = sorted(set(labels.values()))
        out.update({"labels": labels, "by": groupby_col, "categories": categories})
    return out


def print_fit_summary(
    labels: np.ndarray,
    name: str,
    size_counts: dict[Any, Any],
    *,
    console: Console | None = None,
) -> None:
    if console is None:
        return
    table = Table(title=f"Fit summary — {name}", show_lines=False, header_style="bold cyan")
    table.add_column("Cluster", justify="right")
    table.add_column("Count", justify="right")
    for cluster, count in sorted(size_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        table.add_row(str(cluster), str(count))
    console.print(table)


__all__ = [
    "CommandExecution",
    "_apply_dedupe",
    "_collect_existing_meta_sig",
    "_log",
    "_rows_ids",
    "_rule",
    "append_command_record_or_warn",
    "assert_preserve_columns",
    "attach_columns_schema_preserving",
    "cluster_overlay_col",
    "context_and_df",
    "intra_sim_overlay_col",
    "load_highlight_ids_from_file",
    "print_fit_summary",
    "progress_scope",
    "resolve_color_by",
    "resolve_scoped_out_dir",
]
