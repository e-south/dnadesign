"""Artifact-contract checks for DenseGen axis probe status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ProbeArtifactLayout

_SPLIT_COLUMNS = ("id",)


def scored_plan_completion_problems(
    layout: ProbeArtifactLayout,
    *,
    metrics_present: bool,
    decision_present: bool,
    decision: str | None,
) -> list[str]:
    """Reject scored probe roots that were only materialized or partially scored."""

    if not layout.probe_plan_path.exists():
        return []
    try:
        payload = json.loads(layout.probe_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["probe_plan_json_malformed"]
    if not isinstance(payload, Mapping):
        return ["probe_plan_json_not_mapping"]
    plan = payload.get("plan")
    if not isinstance(plan, Mapping):
        return ["probe_plan_missing_plan"]
    if not _plan_requires_scored_outputs(plan):
        return []

    problems: list[str] = []
    if not metrics_present:
        problems.append("metrics_missing_for_scored_plan")
    if not decision_present:
        problems.append("decision_missing_for_scored_plan")
    if decision == "PENDING":
        problems.append("decision_pending_for_scored_plan")
    if not metrics_present:
        return problems
    try:
        metrics = json.loads(layout.metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return problems
    if not isinstance(metrics, Mapping):
        return problems
    return _metrics_count_problems(metrics, plan, problems)


def parquet_schema_problems(path: Path, *, required_columns: tuple[str, ...], problem_prefix: str) -> list[str]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
    except Exception:
        return [f"{problem_prefix}_parquet_unreadable"]
    names = set(parquet.schema_arrow.names)
    problems = [f"{problem_prefix}_missing_column_{column}" for column in required_columns if column not in names]
    if int(parquet.metadata.num_rows) <= 0:
        problems.append(f"{problem_prefix}_empty")
    return problems


def split_metadata_problems(layout: ProbeArtifactLayout) -> list[str]:
    try:
        metadata = json.loads(layout.split_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["split_metadata_json_malformed"]
    if not isinstance(metadata, Mapping):
        return ["split_metadata_json_not_mapping"]

    problems: list[str] = []
    for split_id, payload in metadata.items():
        if not isinstance(payload, Mapping):
            problems.append(f"split_metadata_{split_id}_not_mapping")
            continue
        for key in ("split_id", "train_ids_path", "eval_ids_path"):
            if key not in payload:
                problems.append(f"split_metadata_{split_id}_missing_{key}")
        _extend_split_path_problems(layout, payload, split_id=str(split_id), problems=problems)
    return problems


def split_ids_from_metadata(layout: ProbeArtifactLayout) -> list[str]:
    try:
        metadata = json.loads(layout.split_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(metadata, Mapping):
        return []
    split_ids: list[str] = []
    for split_id, payload in metadata.items():
        if not isinstance(payload, Mapping):
            continue
        value = payload.get("split_id", split_id)
        if isinstance(value, str) and value:
            split_ids.append(value)
    return sorted(dict.fromkeys(split_ids))


def _metrics_count_problems(
    metrics: Mapping[str, Any],
    plan: Mapping[str, Any],
    problems: list[str],
) -> list[str]:
    expected_runs = _positive_int(plan.get("planned_runs"))
    expected_rounds = _positive_int(plan.get("rounds"))
    run_rows = metrics.get("runs")
    if expected_runs is not None and isinstance(run_rows, list):
        observed_run_keys = {
            str(row.get("run_key"))
            for row in run_rows
            if isinstance(row, Mapping) and str(row.get("run_key") or "").strip()
        }
        if len(observed_run_keys) != expected_runs:
            problems.append(f"metrics_run_count_{len(observed_run_keys)}_expected_{expected_runs}")
        if expected_rounds is not None:
            _extend_final_round_problems(run_rows, expected_rounds=expected_rounds, problems=problems)
    round_rows = metrics.get("rounds")
    if expected_runs is not None and expected_rounds is not None and isinstance(round_rows, list):
        expected_round_row_count = expected_runs * expected_rounds
        if len(round_rows) != expected_round_row_count:
            problems.append(f"metrics_round_count_{len(round_rows)}_expected_{expected_round_row_count}")
    return problems


def _extend_final_round_problems(
    run_rows: list[Any],
    *,
    expected_rounds: int,
    problems: list[str],
) -> None:
    final_round = expected_rounds - 1
    for run in run_rows:
        if isinstance(run, Mapping) and int(run.get("as_of_round", -1)) != final_round:
            problems.append(f"metrics_run_{run.get('run_key', 'unknown')}_final_round_not_{final_round}")


def _extend_split_path_problems(
    layout: ProbeArtifactLayout,
    payload: Mapping[str, Any],
    *,
    split_id: str,
    problems: list[str],
) -> None:
    for key in ("train_ids_path", "eval_ids_path"):
        rel_path = payload.get(key)
        if not isinstance(rel_path, str) or not rel_path:
            problems.append(f"split_metadata_{split_id}_{key}_invalid")
            continue
        path = Path(rel_path)
        if path.is_absolute() or ".." in path.parts:
            problems.append(f"split_metadata_{split_id}_{key}_outside_splits_dir")
            continue
        path = layout.splits_dir / path
        if not path.exists():
            problems.append(f"split_metadata_{split_id}_{key}_missing_file")
            continue
        problems.extend(
            parquet_schema_problems(
                path,
                required_columns=_SPLIT_COLUMNS,
                problem_prefix=f"split_metadata_{split_id}_{key}",
            )
        )


def _plan_requires_scored_outputs(plan: Mapping[str, Any]) -> bool:
    if _positive_int(plan.get("planned_runs")) in (None, 0):
        return False
    stop_after = str(plan.get("stop_after") or "").strip().lower()
    if stop_after in {"", "materialize", "validate", "init", "ingest"}:
        return False
    gate = str(plan.get("gate") or "").strip().lower()
    return gate != "source"


def _positive_int(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None
