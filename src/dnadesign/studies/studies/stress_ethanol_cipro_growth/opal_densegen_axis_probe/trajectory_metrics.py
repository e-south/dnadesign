"""Trajectory-based QA metrics for DenseGen motif probe runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import NULL_ORACLE_ID


def trajectory_metric_payload(
    *,
    run_metrics: Sequence[Mapping[str, Any]],
    round_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trajectories = trajectory_summaries_from_metrics(run_metrics=run_metrics, round_metrics=round_metrics)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_trajectory_qa.v1",
        "pairs": trajectories,
        "seed_summaries": seed_summaries_from_trajectories(trajectories),
    }


def trajectory_summaries_from_metrics(
    *,
    run_metrics: Sequence[Mapping[str, Any]],
    round_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [row for row in round_metrics if _finite_metric(row, "target_lift_at_k_true") is not None]
    metric_source = "round_metrics"
    if not rows:
        rows = [row for row in run_metrics if _finite_metric(row, "target_lift_at_k_true") is not None]
        metric_source = "final_metrics"
    grouped: dict[tuple[str, str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        campaign = str(row.get("campaign") or "")
        split_id = str(row.get("split_id") or "")
        seed = str(row.get("seed") if row.get("seed") is not None else "unknown")
        if not campaign or not split_id:
            continue
        oracle_kind = "null" if row.get("oracle_id") == NULL_ORACLE_ID else "positive"
        grouped[(seed, campaign, split_id)][oracle_kind].append(row)

    out: list[dict[str, Any]] = []
    for (seed, campaign, split_id), pair in sorted(grouped.items()):
        positive_curve = _curve(pair.get("positive") or [], metric_source=metric_source)
        null_curve = _curve(pair.get("null") or [], metric_source=metric_source)
        positive_auc = _normalized_auc(positive_curve)
        null_auc = _normalized_auc(null_curve)
        positive_final = positive_curve[-1]["lift"] if positive_curve else None
        null_final = null_curve[-1]["lift"] if null_curve else None
        auc_delta = _delta(positive_auc, null_auc)
        final_delta = _delta(positive_final, null_final)
        status = _trajectory_status(
            positive_curve=positive_curve,
            null_curve=null_curve,
            auc_delta=auc_delta,
            final_delta=final_delta,
        )
        out.append(
            {
                "seed": None if seed == "unknown" else int(seed),
                "campaign": campaign,
                "split_id": split_id,
                "metric_source": metric_source,
                "positive_run_key": _run_key(pair.get("positive") or []),
                "null_run_key": _run_key(pair.get("null") or []),
                "positive_lift_auc": positive_auc,
                "null_lift_auc": null_auc,
                "paired_auc_delta": auc_delta,
                "final_positive_lift": positive_final,
                "final_null_lift": null_final,
                "final_positive_minus_null_lift": final_delta,
                "positive_round_count": len(positive_curve),
                "null_round_count": len(null_curve),
                "positive_curve": positive_curve,
                "null_curve": null_curve,
                "status": status,
                "reason": _trajectory_reason(status=status, auc_delta=auc_delta, final_delta=final_delta),
            }
        )
    return out


def trajectory_gate_results_from_metrics(
    *,
    run_metrics: Sequence[Mapping[str, Any]],
    round_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in trajectory_summaries_from_metrics(run_metrics=run_metrics, round_metrics=round_metrics):
        rows.append(
            {
                "gate": "H-TRAJECTORY-SEPARATION",
                "status": summary["status"],
                "campaign": summary.get("campaign"),
                "split_id": summary.get("split_id"),
                "seed": summary.get("seed"),
                "observed": summary.get("paired_auc_delta"),
                "threshold": 0.0,
                "positive_lift_auc": summary.get("positive_lift_auc"),
                "null_lift_auc": summary.get("null_lift_auc"),
                "paired_auc_delta": summary.get("paired_auc_delta"),
                "final_positive_minus_null_lift": summary.get("final_positive_minus_null_lift"),
                "reason": summary.get("reason"),
            }
        )
    return rows


def has_trajectory_separation_failure(
    *,
    run_metrics: Sequence[Mapping[str, Any]],
    round_metrics: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        row.get("status") in {"debug", "pending"}
        for row in trajectory_gate_results_from_metrics(run_metrics=run_metrics, round_metrics=round_metrics)
    )


def seed_summaries_from_trajectories(trajectories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectories:
        seed = str(row.get("seed") if row.get("seed") is not None else "unknown")
        grouped[seed].append(row)
    out: list[dict[str, Any]] = []
    for seed, rows in sorted(grouped.items()):
        auc_deltas = [_finite_value(row.get("paired_auc_delta")) for row in rows]
        final_deltas = [_finite_value(row.get("final_positive_minus_null_lift")) for row in rows]
        auc_deltas = [value for value in auc_deltas if value is not None]
        final_deltas = [value for value in final_deltas if value is not None]
        statuses = {str(row.get("status")) for row in rows}
        out.append(
            {
                "seed": None if seed == "unknown" else int(seed),
                "pair_count": len(rows),
                "status": "pass" if statuses == {"pass"} else "debug" if "debug" in statuses else "pending",
                "paired_auc_delta_mean": _mean(auc_deltas),
                "paired_auc_delta_min": min(auc_deltas) if auc_deltas else None,
                "paired_auc_delta_max": max(auc_deltas) if auc_deltas else None,
                "final_delta_mean": _mean(final_deltas),
                "final_delta_min": min(final_deltas) if final_deltas else None,
                "final_delta_max": max(final_deltas) if final_deltas else None,
            }
        )
    return out


def _curve(rows: Sequence[Mapping[str, Any]], *, metric_source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for position, row in enumerate(sorted(rows, key=lambda item: _round_sort_key(item.get("as_of_round")))):
        lift = _finite_metric(row, "target_lift_at_k_true")
        if lift is None:
            continue
        round_value = row.get("as_of_round")
        if round_value is None:
            round_value = position
        out.append(
            {
                "round": int(round_value) if _is_int_like(round_value) else round_value,
                "lift": lift,
                "run_key": row.get("run_key"),
                "metric_source": metric_source,
            }
        )
    return out


def _normalized_auc(curve: Sequence[Mapping[str, Any]]) -> float | None:
    if not curve:
        return None
    y = np.asarray([float(row["lift"]) for row in curve], dtype=float)
    if len(curve) == 1:
        return float(y[0])
    x = np.asarray([float(row["round"]) for row in curve], dtype=float)
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return float(np.nanmean(y))
    return float(np.trapezoid(y, x) / span)


def _trajectory_status(
    *,
    positive_curve: Sequence[Mapping[str, Any]],
    null_curve: Sequence[Mapping[str, Any]],
    auc_delta: float | None,
    final_delta: float | None,
) -> str:
    if not positive_curve or not null_curve:
        return "pending"
    if auc_delta is None or final_delta is None:
        return "pending"
    return "pass" if auc_delta > 0.0 and final_delta > 0.0 else "debug"


def _trajectory_reason(*, status: str, auc_delta: float | None, final_delta: float | None) -> str:
    if status == "pending":
        return "positive/null trajectory pair incomplete"
    if status == "debug":
        return "positive trajectory does not exceed paired null by both AUC and final lift"
    return "positive trajectory exceeds paired null by AUC and final lift"


def _run_key(rows: Sequence[Mapping[str, Any]]) -> Any:
    return rows[-1].get("run_key") if rows else None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _round_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    return (0, number)


def _finite_metric(row: Mapping[str, Any], key: str) -> float | None:
    return _finite_value(row.get(key))


def _finite_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _is_int_like(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False
