"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_diagnostics.py

Progressive-disclosure diagnostics for unavailable BaseRender evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def render_notebook_baserender_diagnostic_panel(
    *,
    has_renderable_records: bool,
    selected_record_ids: Iterable[str],
    status_rows: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    build_contract_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
) -> Any | None:
    """Show adapter or selection failures without advertising a dead visual."""

    statuses = [dict(row) for row in status_rows]
    ledger_unavailable = any("unavailable:" in str(row.get("value") or "") for row in statuses)
    selected_but_unrenderable = bool(list(selected_record_ids)) and not has_renderable_records
    if not ledger_unavailable and not selected_but_unrenderable:
        return None
    reason = (
        "Selection evidence could not be resolved."
        if ledger_unavailable
        else "No selected sequence satisfies the declared public BaseRender adapter."
    )
    rows = [
        *statuses,
        {"field": "renderability", "value": reason},
        *build_contract_rows(contract),
    ]
    return mo.accordion(
        {"Sequence render diagnostics": opal_table(pl.DataFrame(rows), page_size=12)},
        multiple=True,
    )


__all__ = ["render_notebook_baserender_diagnostic_panel"]
