"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_fit_support.py

Shared load and attachment helpers for fit execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from .execution_support import (
    _log,
    assert_preserve_columns,
    cluster_overlay_col,
)
from .io.detect import detect_context
from .io.read import load_table, peek_columns
from .io.write import attach_usr, write_generic, write_generic_attached_columns
from .runtime_contracts import FeatureSpec
from .util.meta import compact_meta


@dataclass(frozen=True, slots=True)
class LoadedFitInput:
    context: dict[str, Any]
    fit_df: pd.DataFrame
    attach_base_df: pd.DataFrame | None


def _fit_projection_columns(*, ctx: dict[str, Any], key_col: str, feature_spec: FeatureSpec) -> list[str]:
    columns = [key_col, *feature_spec.columns]
    available_cols = set(peek_columns(ctx))
    if "sequence" in available_cols:
        columns.append("sequence")
    return list(dict.fromkeys(columns))


def load_fit_input(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    key_col: str,
    feature_spec: FeatureSpec,
    write: bool,
) -> LoadedFitInput:
    ctx = detect_context(dataset, file, usr_root)
    projection = _fit_projection_columns(ctx=ctx, key_col=key_col, feature_spec=feature_spec)
    if ctx["kind"] != "usr" and write:
        attach_base_df = load_table(ctx)
        fit_df = attach_base_df.loc[:, projection].copy(deep=False)
        return LoadedFitInput(context=ctx, fit_df=fit_df, attach_base_df=attach_base_df)
    return LoadedFitInput(
        context=ctx,
        fit_df=load_table(ctx, columns=projection),
        attach_base_df=None,
    )


def build_fit_attach_columns(
    *,
    df: pd.DataFrame,
    key_col: str,
    run_alias: str,
    labels: np.ndarray,
    meta_json: str,
    quality: np.ndarray | None,
) -> pd.DataFrame:
    attach_cols = pd.DataFrame(
        {
            "id": df[key_col].astype(str),
            cluster_overlay_col(run_alias): labels,
            cluster_overlay_col(run_alias, "meta"): meta_json,
        }
    )
    if quality is not None:
        attach_cols[cluster_overlay_col(run_alias, "quality")] = quality
    return attach_cols


def build_reused_fit_attach_columns(
    *,
    df: pd.DataFrame,
    key_col: str,
    run_alias: str,
    labels_path: str | Path,
    meta_json: str,
) -> pd.DataFrame:
    labels_df = pd.read_parquet(labels_path)
    attach_cols = pd.merge(
        df[[key_col]].astype(str).rename(columns={key_col: "id"}),
        labels_df,
        on="id",
        how="left",
    )
    attach_cols = attach_cols.rename(columns={"cluster_label": cluster_overlay_col(run_alias)})
    attach_cols[cluster_overlay_col(run_alias, "meta")] = meta_json
    return attach_cols


def apply_fit_attachment(
    *,
    ctx: dict[str, Any],
    attach_cols: pd.DataFrame,
    key_col: str,
    allow_overwrite: bool,
    inplace: bool,
    out: str | None,
    attach_base_df: pd.DataFrame | None,
    console: Console | None,
) -> None:
    if ctx["kind"] == "usr":
        attach_usr(ctx["usr_root"], ctx["dataset"], attach_cols, allow_overwrite=allow_overwrite)
        _log(console, "print", f"[green]Attached[/green] columns to USR dataset '{ctx['dataset']}'.")
        return

    if ctx["kind"] == "parquet":
        write_generic_attached_columns(
            src_file=ctx["file"],
            kind=ctx["kind"],
            key_col=key_col,
            cols_df=attach_cols,
            allow_overwrite=allow_overwrite,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
            base_df=attach_base_df,
        )
    else:
        base_df = attach_base_df if attach_base_df is not None else load_table(ctx)
        from .execution_support import attach_columns_schema_preserving

        merged = attach_columns_schema_preserving(base_df, attach_cols, key_col, allow_overwrite=allow_overwrite)
        assert_preserve_columns(list(base_df.columns), list(merged.columns))
        write_generic(
            ctx["file"],
            merged,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
        )
    _log(console, "print", "[green]Wrote[/green] updated file.")


def fit_meta_json(
    *,
    method_id: str,
    feature_label: str,
    n_rows: int,
    method_params: dict[str, Any],
    source_clause: str,
    method_sig: str,
) -> str:
    return compact_meta(
        "2.0.0",
        method_id,
        feature_label,
        n_rows,
        method_params,
        source_clause,
        sig_hash=method_sig,
    )


__all__ = [
    "LoadedFitInput",
    "apply_fit_attachment",
    "build_fit_attach_columns",
    "build_reused_fit_attach_columns",
    "fit_meta_json",
    "load_fit_input",
]
