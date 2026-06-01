"""Seed-replicate summaries for DenseGen OPAL probe suites."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

REPLICATE_METRICS = (
    "paired_auc_delta",
    "final_positive_minus_null_lift",
)


def replicate_summary(pairs: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for pair in pairs:
        label_family_id = str(pair.get("label_family_id") or "")
        campaign = str(pair.get("campaign") or "")
        split_id = str(pair.get("split_id") or "")
        seed = pair.get("seed")
        if not label_family_id or not campaign or not split_id or seed is None:
            continue
        grouped.setdefault((label_family_id, campaign, split_id), []).append(pair)

    rows: list[dict[str, Any]] = []
    for (label_family_id, campaign, split_id), group_pairs in sorted(grouped.items()):
        seeds = sorted({int(pair["seed"]) for pair in group_pairs if pair.get("seed") is not None})
        row: dict[str, Any] = {
            "label_family_id": label_family_id,
            "campaign": campaign,
            "split_id": split_id,
            "replicate_unit": "seed",
            "seeds": seeds,
            "seed_count": len(seeds),
        }
        for metric in REPLICATE_METRICS:
            values = [
                float(pair[metric])
                for pair in group_pairs
                if pair.get(metric) is not None and _is_finite_number(pair.get(metric))
            ]
            row[metric] = numeric_mean_ci_summary(values)
        rows.append(row)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.replicate_summary.v1",
        "replicate_unit": "seed",
        "confidence_level": 0.95,
        "interval_kind": "student_t_mean_ci",
        "group_count": len(rows),
        "metrics": list(REPLICATE_METRICS),
        "groups": rows,
    }


def numeric_mean_ci_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "std": None,
            "sem": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    sem = (std / math.sqrt(len(values))) if std is not None else None
    if sem is not None:
        half_width = _student_t_975(len(values) - 1) * sem
        ci_low = mean - half_width
        ci_high = mean + half_width
    else:
        ci_low = None
        ci_high = None
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean,
        "max": max(values),
        "std": std,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def write_replicate_summary_csv(payload: Mapping[str, Any], path: Path) -> None:
    rows = []
    for group in payload.get("groups") or []:
        if not isinstance(group, Mapping):
            continue
        for metric in REPLICATE_METRICS:
            summary = group.get(metric) if isinstance(group.get(metric), Mapping) else {}
            rows.append(
                {
                    "label_family_id": group.get("label_family_id"),
                    "campaign": group.get("campaign"),
                    "split_id": group.get("split_id"),
                    "metric": metric,
                    "replicate_unit": group.get("replicate_unit"),
                    "seed_count": group.get("seed_count"),
                    "seeds": ",".join(map(str, group.get("seeds") or [])),
                    "mean": summary.get("mean"),
                    "std": summary.get("std"),
                    "sem": summary.get("sem"),
                    "ci95_low": summary.get("ci95_low"),
                    "ci95_high": summary.get("ci95_high"),
                    "min": summary.get("min"),
                    "max": summary.get("max"),
                }
            )
    header = [
        "label_family_id",
        "campaign",
        "split_id",
        "metric",
        "replicate_unit",
        "seed_count",
        "seeds",
        "mean",
        "std",
        "sem",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_replicate_ci_plot(
    payload: Mapping[str, Any],
    *,
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for group in payload.get("groups") or []:
        if not isinstance(group, Mapping):
            continue
        summary = group.get(metric) if isinstance(group.get(metric), Mapping) else {}
        if not _is_finite_number(summary.get("mean")):
            continue
        label = "{family}\n{campaign}/{split}".format(
            family=_pretty_label(group.get("label_family_id")),
            campaign=_pretty_label(group.get("campaign")),
            split=_pretty_label(group.get("split_id")),
        )
        rows.append((group, summary, label))
    if not rows:
        return
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig_width = max(8.5, 0.62 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    xs = np.arange(len(rows), dtype=float)
    means = np.asarray([float(summary["mean"]) for _group, summary, _label in rows], dtype=float)
    lower = []
    upper = []
    for _group, summary, _label in rows:
        if _is_finite_number(summary.get("ci95_low")) and _is_finite_number(summary.get("ci95_high")):
            lower.append(float(summary["mean"]) - float(summary["ci95_low"]))
            upper.append(float(summary["ci95_high"]) - float(summary["mean"]))
        else:
            lower.append(0.0)
            upper.append(0.0)
    colors = [
        "#0072B2",
        "#009E73",
        "#E69F00",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]
    colors = [colors[index % len(colors)] for index, _row in enumerate(rows)]
    ax.bar(xs, means, color=colors, alpha=0.72, width=0.72, zorder=2)
    ax.errorbar(
        xs,
        means,
        yerr=np.asarray([lower, upper], dtype=float),
        fmt="none",
        ecolor="#202020",
        elinewidth=1.4,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )
    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.65)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _group, _summary, label in rows], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nmean with 95% Student-t CI across seed replicates")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _student_t_975(degrees_of_freedom: int) -> float:
    table = {
        1: 12.706204736432095,
        2: 4.302652729749464,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636314,
        6: 2.4469118511449692,
        7: 2.3646242510102993,
        8: 2.306004135033371,
        9: 2.2621571627409915,
        10: 2.2281388519649385,
        11: 2.200985160082949,
        12: 2.1788128296634177,
        13: 2.1603686564610127,
        14: 2.1447866879169273,
        15: 2.131449545559323,
        16: 2.1199052992210112,
        17: 2.1098155778331806,
        18: 2.10092204024096,
        19: 2.093024054408263,
        20: 2.0859634472658364,
        21: 2.079613844727662,
        22: 2.0738730679040147,
        23: 2.0686576104190406,
        24: 2.0638985616280205,
        25: 2.059538552753294,
        26: 2.055529438642871,
        27: 2.0518305164802833,
        28: 2.048407141795244,
        29: 2.045229642132703,
        30: 2.042272456301238,
    }
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive for a t confidence interval.")
    return table.get(int(degrees_of_freedom), 1.959963984540054)


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _pretty_label(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").strip().title()
