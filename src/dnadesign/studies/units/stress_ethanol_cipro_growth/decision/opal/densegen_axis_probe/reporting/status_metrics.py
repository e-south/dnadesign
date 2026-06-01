"""Metrics artifact validation for DenseGen probe run-root audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def _metrics_problems(metrics_path: Path) -> list[str]:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["metrics_json_malformed"]
    if not isinstance(metrics, Mapping):
        return ["metrics_json_not_mapping"]
    problems: list[str] = []
    safety = metrics.get("safety")
    if not isinstance(safety, Mapping):
        problems.append("metrics_json_missing_safety")
    else:
        for key in ("path_safety_pass", "forbidden_input_pass", "x_surface_pass", "quality_counts"):
            if key not in safety:
                problems.append(f"metrics_json_safety_missing_{key}")
    runs = metrics.get("runs")
    if not isinstance(runs, list):
        problems.append("metrics_json_missing_runs")
        return problems
    required_run_keys = ("run_key", "campaign", "oracle_id", "split_id", "target_class", "train_count", "eval_count")
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            problems.append(f"metrics_json_runs_{index}_not_mapping")
            continue
        for key in required_run_keys:
            if key not in run:
                problems.append(f"metrics_json_runs_{index}_missing_{key}")
    return problems
