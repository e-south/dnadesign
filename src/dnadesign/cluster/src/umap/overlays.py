"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/umap/overlays.py

UMAP overlay attachment helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from ..execution_support import _log, attach_columns_schema_preserving, cluster_overlay_col
from ..io.write import attach_usr, write_generic, write_generic_attached_columns


def write_umap_overlays(
    *,
    ictx: dict[str, Any],
    attach_base_df: pd.DataFrame,
    df: pd.DataFrame,
    name: str,
    key_col: str,
    coords: np.ndarray,
    derived_cols: list[str],
    attach_coords: bool,
    write: bool,
    allow_overwrite: bool,
    inplace: bool,
    out: str | None,
    console: Console | None,
) -> None:
    to_attach: dict[str, Any] = {"id": df[key_col].astype(str)}
    if attach_coords:
        to_attach.update(
            {
                cluster_overlay_col(name, "umap_x"): coords[:, 0],
                cluster_overlay_col(name, "umap_y"): coords[:, 1],
            }
        )
    if derived_cols:
        for column in derived_cols:
            to_attach[cluster_overlay_col(name, column)] = df[column].astype(float).to_numpy()

    if not (attach_coords or derived_cols):
        return
    if not write:
        _log(console, "print", "Dry-run: computed artifacts. Use --write to attach.")
        return

    cols = pd.DataFrame(to_attach)
    if ictx["kind"] == "usr":
        try:
            attach_usr(ictx["usr_root"], ictx["dataset"], cols, allow_overwrite=allow_overwrite)
        except Exception as exc:
            if "Columns already exist" in str(exc) and not allow_overwrite:
                raise RuntimeError("Columns already exist for attachment. Re-run with `-y/--allow-overwrite`.") from exc
            raise
        _log(console, "print", "[green]Attached[/green] columns to USR dataset.")
        return

    if ictx["kind"] == "parquet":
        write_generic_attached_columns(
            src_file=ictx["file"],
            kind=ictx["kind"],
            key_col=key_col,
            cols_df=cols,
            allow_overwrite=allow_overwrite,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
            base_df=attach_base_df,
        )
    else:
        merged = attach_columns_schema_preserving(attach_base_df, cols, key_col, allow_overwrite=allow_overwrite)
        write_generic(
            ictx["file"],
            merged,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
        )
    _log(console, "print", "[green]Wrote[/green] updated file with attachments.")
