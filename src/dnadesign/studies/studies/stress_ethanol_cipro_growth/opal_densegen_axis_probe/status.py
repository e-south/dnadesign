"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .artifacts import ProbeArtifactLayout, RunRootAudit
from .constants import CANDIDATE_RECORDS, SFXI_INTENSITY_COLUMNS, SFXI_STATE_COLUMNS, SHARED_OBSERVED_LABEL_SIDECAR
from .paths import _repo_root_from, _resolve_repo_path
from .records_manifest import records_manifest_problems

_LABEL_COLUMNS = ("oracle_id", "id", "sequence", "axis_class", "quality_flag", "vec8")
_SPLIT_COLUMNS = ("id",)
_PASS_DECISIONS = {
    "PASS_CIPRO_RANDOM_GATE",
    "PASS_RANDOM_ALL_GATE",
    "PASS_LEAVE_SIGMA35_GATE",
    "PASS_FULL_MATRIX_GATE",
    "PASS_SCOPED_GATE",
}
_VALID_DECISIONS = {*_PASS_DECISIONS, "DEBUG", "STOP", "PENDING"}


def audit_run_root(run_root: Path) -> RunRootAudit:
    root = run_root.resolve()
    layout = ProbeArtifactLayout(root)
    repo_root = _repo_root_from(Path.cwd())
    labels_present = layout.densegen_labels_path.exists() and layout.null_labels_path.exists()
    splits_present = layout.split_metadata_path.exists()
    metrics_path = layout.metrics_path
    decision_path = layout.decision_path
    metrics_present = metrics_path.exists()
    decision_present = decision_path.exists()
    split_ids = _split_ids_from_metadata(layout) if splits_present else []
    split_records_paths = [layout.split_records_path(split_id) for split_id in split_ids]
    scratch_records_present = bool(split_records_paths) and all(path.exists() for path in split_records_paths)
    shared_sidecar_present = _resolve_repo_path(repo_root, SHARED_OBSERVED_LABEL_SIDECAR).exists()
    planned_campaign_count = (
        len(list(layout.scratch_campaigns_dir.glob("*"))) if layout.scratch_campaigns_dir.exists() else 0
    )

    decision: str | None = None
    if decision_present:
        text = decision_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(text):
            if line.strip() == "## Decision" and idx + 2 < len(text):
                decision = text[idx + 2].strip() or None
                break

    problems: list[str] = []
    if not root.exists():
        problems.append("run_root_missing")
    if root.exists() and not labels_present:
        problems.append("labels_missing")
    if labels_present:
        problems.extend(
            _parquet_schema_problems(
                layout.densegen_labels_path,
                required_columns=(*_LABEL_COLUMNS, *SFXI_STATE_COLUMNS, *SFXI_INTENSITY_COLUMNS),
                problem_prefix="densegen_labels",
            )
        )
        problems.extend(
            _parquet_schema_problems(
                layout.null_labels_path,
                required_columns=(*_LABEL_COLUMNS, *SFXI_STATE_COLUMNS, *SFXI_INTENSITY_COLUMNS),
                problem_prefix="null_labels",
            )
        )
    if root.exists() and not splits_present:
        problems.append("split_metadata_missing")
    if splits_present:
        problems.extend(_split_metadata_problems(layout))
    if decision_present and decision is None:
        problems.append("decision_value_missing")
    elif decision is not None and decision not in _VALID_DECISIONS:
        problems.append("decision_value_invalid")
    if metrics_present:
        problems.extend(_metrics_problems(metrics_path))
    source_records = _resolve_repo_path(repo_root, CANDIDATE_RECORDS)
    if split_records_paths:
        for path in split_records_paths:
            problems.extend(records_manifest_problems(path, source_records))
    if planned_campaign_count > 0 and not scratch_records_present:
        problems.append("scratch_records_missing_for_planned_campaigns")
    if not root.exists():
        status = "missing"
    elif problems:
        status = "attention"
    elif decision in {"STOP", "DEBUG"}:
        status = "attention"
    elif decision in _PASS_DECISIONS or decision == "PENDING" or metrics_present:
        status = "ok"
    else:
        status = "materialized"

    return RunRootAudit(
        run_root=root,
        exists=root.exists(),
        decision=decision,
        status=status,
        labels_present=labels_present,
        splits_present=splits_present,
        metrics_present=metrics_present,
        decision_present=decision_present,
        scratch_records_present=scratch_records_present,
        planned_campaign_count=int(planned_campaign_count),
        shared_sidecar_present=shared_sidecar_present,
        problems=tuple(problems),
    )


def _format_status_text(audit: RunRootAudit) -> str:
    data = audit.to_dict()
    lines = [
        "opal_densegen_axis_probe_v0 status",
        f"run_root: {data['run_root']}",
        f"status: {data['status']}",
        f"decision: {data['decision'] or 'n/a'}",
        f"labels_present: {data['labels_present']}",
        f"splits_present: {data['splits_present']}",
        f"metrics_present: {data['metrics_present']}",
        f"decision_present: {data['decision_present']}",
        f"scratch_records_present: {data['scratch_records_present']}",
        f"planned_campaign_count: {data['planned_campaign_count']}",
        f"shared_sidecar_present: {data['shared_sidecar_present']}",
    ]
    if data["problems"]:
        lines.append("problems:")
        lines.extend(f"  - {problem}" for problem in data["problems"])
    return "\n".join(lines) + "\n"


def _parquet_schema_problems(path: Path, *, required_columns: tuple[str, ...], problem_prefix: str) -> list[str]:
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


def _split_metadata_problems(layout: ProbeArtifactLayout) -> list[str]:
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
                _parquet_schema_problems(
                    path,
                    required_columns=_SPLIT_COLUMNS,
                    problem_prefix=f"split_metadata_{split_id}_{key}",
                )
            )
    return problems


def _split_ids_from_metadata(layout: ProbeArtifactLayout) -> list[str]:
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
