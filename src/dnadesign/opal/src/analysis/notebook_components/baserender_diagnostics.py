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
    candidate_record_ids: Iterable[str],
    unrenderable_record_ids: Iterable[str],
    status_rows: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    build_contract_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
) -> Any | None:
    """Show candidate-ledger and public-adapter failures beside BaseRender."""

    statuses = [dict(row) for row in status_rows]
    ledger_unavailable = any("unavailable:" in str(row.get("value") or "") for row in statuses)
    candidate_ids = [str(value) for value in candidate_record_ids]
    unrenderable_ids = [str(value) for value in unrenderable_record_ids]
    candidates_but_none_renderable = bool(candidate_ids) and not has_renderable_records
    if not ledger_unavailable and not candidates_but_none_renderable and not unrenderable_ids:
        return None
    if ledger_unavailable:
        reason = "Selection evidence could not be resolved."
    elif candidates_but_none_renderable:
        reason = "No campaign candidate satisfies the declared public BaseRender adapter."
    else:
        preview = ", ".join(unrenderable_ids[:8])
        suffix = "" if len(unrenderable_ids) <= 8 else f" and {len(unrenderable_ids) - 8} more"
        reason = f"{len(unrenderable_ids)} campaign candidates are missing or fail the adapter: {preview}{suffix}."
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
