"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/_run_resolution.py

Metric-neutral run and round resolution for single-run plot plugins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import polars as pl

from ..analysis.ledger import latest_round
from ..core.utils import ExitCodes, OpalError


def resolve_single_round(
    runs_df: pl.DataFrame,
    *,
    round_selector: str | int | list[int] | None,
) -> int:
    """Resolve one explicit or latest round for a single-round plot."""

    if runs_df.is_empty():
        raise OpalError("No runs available. Run `opal run ...` first.", ExitCodes.BAD_ARGS)
    if round_selector in (None, "unspecified", "latest"):
        return latest_round(runs_df)
    if round_selector == "all":
        raise OpalError("Select a single round for this plot (e.g., --round latest or --round 3).", ExitCodes.BAD_ARGS)
    if isinstance(round_selector, list):
        if len(round_selector) != 1:
            raise OpalError("Select a single round for this plot.", ExitCodes.BAD_ARGS)
        return int(round_selector[0])
    return int(round_selector)


def resolve_run_id(
    runs_df: pl.DataFrame,
    *,
    round_k: int,
    run_id: str | None,
) -> str | None:
    """Resolve one run for a round, rejecting ambiguous implicit selection."""

    if run_id is not None:
        return str(run_id)
    if "run_id" not in runs_df.columns:
        return None
    run_ids = (
        runs_df.filter(pl.col("as_of_round") == int(round_k))
        .select(pl.col("run_id").drop_nulls().unique())
        .to_series()
        .to_list()
    )
    run_ids = sorted({str(value) for value in run_ids if value is not None})
    if len(run_ids) > 1:
        raise OpalError(
            f"Multiple run_ids exist for round {round_k}; pass --run-id to disambiguate.",
            ExitCodes.BAD_ARGS,
        )
    return run_ids[0] if run_ids else None


def parse_run_view_definitions(raw: object, *, field_label: str) -> tuple[Mapping[str, object], ...]:
    """Parse one run-ledger JSON list of view-definition mappings."""

    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OpalError(f"{field_label} is invalid JSON: {exc}", ExitCodes.CONTRACT_VIOLATION) from exc
    if not isinstance(parsed, list) or any(not isinstance(item, Mapping) for item in parsed):
        raise OpalError(
            f"{field_label} must be a JSON list of mappings.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return tuple(parsed)


__all__ = ["parse_run_view_definitions", "resolve_run_id", "resolve_single_round"]
