"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/evaluation/round_dynamics.py

Round-level dynamics diagnostics for DenseGen axis probe metrics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.constants import NULL_ORACLE_ID


def round_dynamics_from_metrics(
    round_metrics: Sequence[Mapping[str, Any]],
    *,
    null_lift_threshold: float,
) -> list[dict[str, Any]]:
    """Summarize first/final/max lift per scored run across OPAL rounds."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in round_metrics:
        run_key = str(row.get("run_key") or "")
        if run_key:
            grouped[run_key].append(row)

    out: list[dict[str, Any]] = []
    for run_key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: _round_sort_key(row.get("as_of_round")))
        finite_rows = [
            (row, lift)
            for row in ordered
            for lift in [_finite_metric(row, "target_lift_at_k_true")]
            if lift is not None
        ]
        if not finite_rows:
            continue

        first_row, first_lift = finite_rows[0]
        final_row, final_lift = finite_rows[-1]
        max_row, max_lift = max(finite_rows, key=lambda item: item[1])
        oracle_id = str(final_row.get("oracle_id") or first_row.get("oracle_id") or "")
        is_null = oracle_id == NULL_ORACLE_ID
        exceeded = max_lift > null_lift_threshold
        final_exceeded = final_lift > null_lift_threshold
        out.append(
            {
                "run_key": run_key,
                "campaign": final_row.get("campaign") or first_row.get("campaign"),
                "split_id": final_row.get("split_id") or first_row.get("split_id"),
                "label_family_id": final_row.get("label_family_id") or first_row.get("label_family_id"),
                "oracle_id": oracle_id,
                "first_round": first_row.get("as_of_round"),
                "final_round": final_row.get("as_of_round"),
                "max_round": max_row.get("as_of_round"),
                "first_lift": first_lift,
                "final_lift": final_lift,
                "max_lift": max_lift,
                "final_minus_first_lift": float(final_lift - first_lift),
                "max_minus_final_lift": float(max_lift - final_lift),
                "null_lift_threshold": null_lift_threshold,
                "null_threshold_exceeded": bool(is_null and exceeded),
                "null_final_threshold_exceeded": bool(is_null and final_exceeded),
                "null_transient_spike": bool(is_null and exceeded and not final_exceeded),
                "round_count": len(finite_rows),
            }
        )
    return out


def round_gate_results_from_metrics(
    round_metrics: Sequence[Mapping[str, Any]],
    *,
    null_lift_threshold: float,
) -> list[dict[str, Any]]:
    """Build attention gates for round-level null-control behavior."""
    rows: list[dict[str, Any]] = []
    for summary in round_dynamics_from_metrics(round_metrics, null_lift_threshold=null_lift_threshold):
        if summary.get("oracle_id") != NULL_ORACLE_ID:
            continue
        final_exceeded = bool(summary.get("null_final_threshold_exceeded"))
        transient = bool(summary.get("null_transient_spike"))
        status = "attention" if final_exceeded or transient else "pass"
        reason = (
            f"null lift exceeds random-baseline lift {null_lift_threshold:g} at final round; diagnostic only"
            if final_exceeded
            else f"null lift spiked above random-baseline lift {null_lift_threshold:g} before returning below baseline"
            if transient
            else "null lift never exceeded random-baseline lift across rounds"
        )
        rows.append(
            {
                "gate": "H-NULL-ROUND-DYNAMICS",
                "status": status,
                "campaign": summary.get("campaign"),
                "split_id": summary.get("split_id"),
                "label_family_id": summary.get("label_family_id"),
                "run_key": summary.get("run_key"),
                "observed": summary.get("max_lift"),
                "threshold": null_lift_threshold,
                "reason": reason,
                "first_round": summary.get("first_round"),
                "final_round": summary.get("final_round"),
                "max_round": summary.get("max_round"),
                "first_lift": summary.get("first_lift"),
                "final_lift": summary.get("final_lift"),
                "max_lift": summary.get("max_lift"),
                "final_minus_first_lift": summary.get("final_minus_first_lift"),
                "max_minus_final_lift": summary.get("max_minus_final_lift"),
            }
        )
    return rows


def has_null_round_dynamics_attention(
    round_metrics: Sequence[Mapping[str, Any]],
    *,
    null_lift_threshold: float,
) -> bool:
    """Return true when null round dynamics should be called out as diagnostic attention."""
    return any(
        row.get("status") == "attention"
        for row in round_gate_results_from_metrics(round_metrics, null_lift_threshold=null_lift_threshold)
    )


def _round_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    return (0, number)


def _finite_metric(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
