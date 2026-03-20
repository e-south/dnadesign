"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/umap/frame.py

UMAP input-frame preparation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from ..execution_support import _log
from ..opal.join import join_fields as opal_join_fields
from ..opal.join import list_available_fields as opal_list_available_fields
from ..opal.join import resolve_campaign_dir as resolve_opal_campaign_dir


def prepare_umap_frame(
    df: pd.DataFrame,
    *,
    name: str,
    key_col: str,
    color_by: tuple[str, ...],
    highlight_payload: dict[str, Any] | None,
    opal_campaign: str | None,
    opal_run: str | None,
    opal_as_of_round: int | None,
    opal_fields: str | None,
    derive_ratio: list[str],
    resolve_hue_fn: Callable[..., Any],
    console: Console | None,
) -> tuple[pd.DataFrame, list[str]]:
    opal_needed_fields: set[str] = set()
    for spec in color_by:
        if spec.startswith(("numeric:", "categorical:")):
            column = spec.split(":", 1)[1]
            if column.startswith(("obj__", "pred__", "sel__")) and column not in df.columns:
                opal_needed_fields.add(column)
    if opal_fields:
        opal_needed_fields |= {column.strip() for column in opal_fields.split(",") if column.strip()}

    if opal_needed_fields:
        if not opal_campaign:
            raise typer.BadParameter(
                "The selected hues require OPAL predictions "
                f"(missing {', '.join(sorted(opal_needed_fields))}). "
                "Provide --opal-campaign and either --opal-run or --opal-as-of-round."
            )
        if opal_run and opal_as_of_round is not None:
            raise typer.BadParameter("Use only one of --opal-run or --opal-as-of-round, not both.")
        try:
            campaign_dir = resolve_opal_campaign_dir(opal_campaign)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if df.index.name != key_col:
            df = df.set_index(key_col, drop=False)
        df = opal_join_fields(
            df,
            campaign_dir=campaign_dir,
            run_selector=(opal_run or "latest"),
            fields=sorted(opal_needed_fields),
            as_of_round=opal_as_of_round,
            log_fn=lambda message: _log(console, "log", message),
        )
        missing_after = [column for column in opal_needed_fields if column not in df.columns]
        if missing_after:
            available = opal_list_available_fields(
                campaign_dir,
                run_selector=(opal_run or "latest"),
                as_of_round=opal_as_of_round,
            )[:60]
            raise RuntimeError(
                "OPAL join did not provide the requested column(s): "
                + ", ".join(missing_after)
                + ". Available columns include: "
                + ", ".join(available)
                + (" ..." if len(available) == 60 else "")
            )

    derived_cols: list[str] = []
    for spec in derive_ratio:
        parts = [part.strip() for part in spec.split(":", 2)]
        if len(parts) != 3 or any(not part for part in parts):
            raise typer.BadParameter(
                f"--derive-ratio expects '<new_col>:<numerator_col>:<denominator_col>'; got '{spec}'."
            )
        new_col, num_col, den_col = parts
        for column in (num_col, den_col):
            if column not in df.columns:
                raise typer.BadParameter(f"--derive-ratio: column '{column}' not found.")
        try:
            numerator = pd.to_numeric(df[num_col], errors="raise")
            denominator = pd.to_numeric(df[den_col], errors="raise")
        except Exception as exc:
            raise typer.BadParameter(f"--derive-ratio: numeric coercion failed: {exc}") from exc
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator
        nonfinite = ~np.isfinite(ratio.to_numpy(dtype="float64", copy=False))
        if nonfinite.any():
            _log(
                console,
                "print",
                f"[yellow]Note[/yellow]: derived '{new_col}' has {int(nonfinite.sum())} "
                "non-finite value(s) (NaN/Inf). These rows will be skipped for numeric hues.",
            )
        df[new_col] = ratio.astype(float)
        derived_cols.append(new_col)

    if df.index.name != "id":
        df = df.set_index(key_col, drop=False)
    try:
        resolve_hue_fn(
            df,
            color_specs=list(color_by),
            name=name,
            missing_policy="drop_and_log",
            log_fn=lambda message: _log(console, "print", f"[yellow]Note[/yellow]: {message}"),
            highlight=highlight_payload,
        )
    except Exception as exc:
        raise RuntimeError(f"Hue validation failed: {exc}") from exc
    return df, derived_cols
