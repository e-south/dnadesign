"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

from pathlib import Path

from .artifacts import ProbeArtifactLayout, RunRootAudit
from .constants import DENSEGEN_PLAN_LOGIC4_COLUMNS, SHARED_OBSERVED_LABEL_SIDECAR
from .paths import _repo_root_from, _resolve_repo_path
from .status_contracts import (
    parquet_schema_problems,
    scored_plan_completion_problems,
    split_ids_from_metadata,
    split_metadata_problems,
)
from .status_metrics import _metrics_problems

_LABEL_COLUMNS = ("oracle_id", "id", "sequence", "axis_class", "quality_flag", "logic4")
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
    split_ids = split_ids_from_metadata(layout) if splits_present else []
    split_records_paths = [layout.split_records_path(split_id) for split_id in split_ids]
    scratch_records_present = bool(split_records_paths) and all(path.exists() for path in split_records_paths)
    candidate_scope_paths = [layout.split_candidate_scope_path(split_id) for split_id in split_ids]
    candidate_scope_present = bool(candidate_scope_paths) and all(path.exists() for path in candidate_scope_paths)
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
            parquet_schema_problems(
                layout.densegen_labels_path,
                required_columns=(*_LABEL_COLUMNS, *DENSEGEN_PLAN_LOGIC4_COLUMNS),
                problem_prefix="densegen_labels",
            )
        )
        problems.extend(
            parquet_schema_problems(
                layout.null_labels_path,
                required_columns=(*_LABEL_COLUMNS, *DENSEGEN_PLAN_LOGIC4_COLUMNS),
                problem_prefix="null_labels",
            )
        )
    if root.exists() and not splits_present:
        problems.append("split_metadata_missing")
    if splits_present:
        problems.extend(split_metadata_problems(layout))
    if decision_present and decision is None:
        problems.append("decision_value_missing")
    elif decision is not None and decision not in _VALID_DECISIONS:
        problems.append("decision_value_invalid")
    if metrics_present:
        problems.extend(_metrics_problems(metrics_path))
    problems.extend(
        scored_plan_completion_problems(
            layout,
            metrics_present=metrics_present,
            decision_present=decision_present,
            decision=decision,
        )
    )
    if split_records_paths:
        for path in split_records_paths:
            if path.exists() and not path.is_symlink():
                problems.append(f"split_records_{path.parent.name}_not_symlink")
    if planned_campaign_count > 0 and not scratch_records_present:
        problems.append("scratch_record_symlink_missing_for_planned_campaigns")
    if planned_campaign_count > 0 and not candidate_scope_present:
        problems.append("candidate_scope_missing_for_planned_campaigns")
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
        candidate_scope_present=candidate_scope_present,
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
        f"candidate_scope_present: {data['candidate_scope_present']}",
        f"planned_campaign_count: {data['planned_campaign_count']}",
        f"shared_sidecar_present: {data['shared_sidecar_present']}",
    ]
    if data["problems"]:
        lines.append("problems:")
        lines.extend(f"  - {problem}" for problem in data["problems"])
    return "\n".join(lines) + "\n"
