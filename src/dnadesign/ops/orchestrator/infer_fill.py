"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/infer_fill.py

Study-aware Infer completion planning for Ops runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from dnadesign.infer import plan_sequence_view_feature_inventory_completion_from_config

from ..runbooks.path_policy import WORKSPACE_AUDIT_RELATIVE_DIR
from ..runbooks.schema import OrchestrationRunbookV1, is_infer_workflow_id, load_orchestration_runbook
from .execute import BatchExecutionResult, execute_batch_plan
from .plan import BatchPlan, build_batch_plan
from .state import ActiveJobResolution, resolve_active_job_resolution

InferFillLaneAction = Literal["run", "skip_complete", "skip_unsupported", "blocked"]


@dataclass(frozen=True)
class InferFillLane:
    runbook_path: Path
    runbook_id: str
    workflow_id: str
    config_path: Path
    action: InferFillLaneAction
    reasons: tuple[str, ...]
    completion: tuple[Mapping[str, object], ...] = ()
    required_views: int = 0
    required_vectors: int = 0
    required_scalars: int = 0
    missing_products: int = 0
    missing_vectors: int = 0
    missing_scalars: int = 0
    stale_vectors: int = 0
    stale_scalars: int = 0
    plan: BatchPlan | None = None
    active_job_resolution: ActiveJobResolution | None = None
    audit_json_path: Path | None = None
    execution_result: BatchExecutionResult | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "runbook_path": str(self.runbook_path),
            "runbook_id": self.runbook_id,
            "workflow_id": self.workflow_id,
            "config_path": str(self.config_path),
            "action": self.action,
            "reasons": list(self.reasons),
            "completion": [dict(plan) for plan in self.completion],
            "required_views": self.required_views,
            "required_vectors": self.required_vectors,
            "required_scalars": self.required_scalars,
            "missing_products": self.missing_products,
            "missing_vectors": self.missing_vectors,
            "missing_scalars": self.missing_scalars,
            "stale_vectors": self.stale_vectors,
            "stale_scalars": self.stale_scalars,
            "plan": self.plan.as_dict() if self.plan is not None else None,
            "active_job_resolution": (
                self.active_job_resolution.as_dict() if self.active_job_resolution is not None else None
            ),
            "audit_json_path": str(self.audit_json_path) if self.audit_json_path is not None else None,
            "execution_result": self.execution_result.as_dict() if self.execution_result is not None else None,
        }


@dataclass(frozen=True)
class InferFillPlan:
    study_dir: Path | None
    runbook_paths: tuple[Path, ...]
    lanes: tuple[InferFillLane, ...]
    submit: bool = False
    executed: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    aggregate_submit_commands: int = 0

    @property
    def runnable_lanes(self) -> tuple[InferFillLane, ...]:
        return tuple(lane for lane in self.lanes if lane.action == "run")

    @property
    def blocked_lanes(self) -> tuple[InferFillLane, ...]:
        return tuple(lane for lane in self.lanes if lane.action == "blocked")

    @property
    def ok(self) -> bool:
        if self.errors:
            return False
        if self.executed:
            return all(
                lane.execution_result is None or lane.execution_result.ok for lane in self.lanes if lane.action == "run"
            )
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "study_dir": str(self.study_dir) if self.study_dir is not None else None,
            "runbook_paths": [str(path) for path in self.runbook_paths],
            "summary": {
                "lanes_total": len(self.lanes),
                "runnable_lanes": len(self.runnable_lanes),
                "blocked_lanes": len(self.blocked_lanes),
                "skip_complete_lanes": sum(1 for lane in self.lanes if lane.action == "skip_complete"),
                "skip_unsupported_lanes": sum(1 for lane in self.lanes if lane.action == "skip_unsupported"),
                "aggregate_submit_commands": self.aggregate_submit_commands,
                "missing_vectors": sum(lane.missing_vectors for lane in self.lanes),
                "missing_scalars": sum(lane.missing_scalars for lane in self.lanes),
                "missing_products": sum(lane.missing_products for lane in self.lanes),
                "stale_vectors": sum(lane.stale_vectors for lane in self.lanes),
                "stale_scalars": sum(lane.stale_scalars for lane in self.lanes),
            },
            "submit": self.submit,
            "executed": self.executed,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "lanes": [lane.as_dict() for lane in self.lanes],
        }


def resolve_active_study_dir(*, repo_root: Path) -> Path:
    index_path = repo_root / "docs" / "studies" / "index.yaml"
    payload = _read_yaml_mapping(index_path)
    active_study_id = str(payload.get("active_study_id") or "").strip()
    if not active_study_id:
        raise ValueError(f"active_study_id is required in {index_path}")
    studies = payload.get("studies") or ()
    if not isinstance(studies, Sequence) or isinstance(studies, (str, bytes)):
        raise ValueError(f"studies must be a list in {index_path}")
    for entry in studies:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("study_id") or "").strip() != active_study_id:
            continue
        record_root = str(entry.get("record_root") or "").strip()
        if not record_root:
            raise ValueError(f"active study {active_study_id} is missing record_root in {index_path}")
        return (repo_root / record_root).resolve()
    raise ValueError(f"active study {active_study_id} is not declared in {index_path}")


def discover_infer_runbook_paths_for_study(*, study_dir: Path, repo_root: Path) -> tuple[Path, ...]:
    ops_study_path = study_dir / "operations" / "ops.study.yaml"
    payload = _read_yaml_mapping(ops_study_path)
    surfaces = payload.get("execution_surfaces") or {}
    if not isinstance(surfaces, Mapping):
        raise ValueError(f"execution_surfaces must be a mapping in {ops_study_path}")
    paths: list[Path] = []
    for surface in surfaces.values():
        if not isinstance(surface, Mapping):
            continue
        if str(surface.get("surface_type") or "").strip() != "runbook":
            continue
        runbook_ref = str(surface.get("runbook_ref") or "").strip()
        if not runbook_ref:
            continue
        paths.append(_resolve_path_ref(runbook_ref, repo_root=repo_root, manifest_dir=study_dir))
    return _dedupe_paths(paths)


def build_infer_fill_plan(
    *,
    repo_root: Path,
    study_dir: Path | None = None,
    runbook_paths: Sequence[Path] = (),
    requested_mode: str | None = None,
    requested_smoke: str | None = None,
    active_job_ids: Sequence[str] = (),
    discover_active_jobs: bool = True,
    max_discovery_jobs: int = 24,
    allow_fresh_reset: bool = False,
    allow_missing_qstat: bool = False,
    allow_unknown_active_jobs: bool = False,
) -> InferFillPlan:
    selected_study_dir = study_dir.resolve() if study_dir is not None else None
    if selected_study_dir is None and not runbook_paths:
        selected_study_dir = resolve_active_study_dir(repo_root=repo_root)
    discovered_paths = (
        discover_infer_runbook_paths_for_study(study_dir=selected_study_dir, repo_root=repo_root)
        if selected_study_dir is not None
        else ()
    )
    selected_runbook_paths = _dedupe_paths(
        (*discovered_paths, *tuple(path.expanduser().resolve() for path in runbook_paths))
    )
    if not selected_runbook_paths:
        raise ValueError("no runbook paths were provided or discovered")

    lanes: list[InferFillLane] = []
    warnings: list[str] = []
    aggregate_submit_commands = 0
    for runbook_path in selected_runbook_paths:
        runbook = load_orchestration_runbook(runbook_path)
        if not is_infer_workflow_id(runbook.workflow_id):
            lanes.append(_skip_lane(runbook_path=runbook_path, runbook=runbook, reason="not an Infer workflow"))
            continue
        lane = _classify_infer_lane(runbook_path=runbook_path, runbook=runbook)
        if lane.action != "run":
            lanes.append(lane)
            continue
        active_job_resolution = resolve_active_job_resolution(
            runbook=runbook,
            explicit_job_ids=active_job_ids,
            discover_active_jobs=discover_active_jobs,
            max_jobs=max_discovery_jobs,
        )
        try:
            plan = build_batch_plan(
                runbook=runbook,
                requested_mode=requested_mode,
                requested_smoke=requested_smoke,
                active_job_ids=active_job_resolution.effective_job_ids,
                runtime_visibility=active_job_resolution.runtime_visibility,
                allow_fresh_reset=allow_fresh_reset,
                allow_missing_qstat=allow_missing_qstat,
                allow_unknown_active_jobs=allow_unknown_active_jobs,
            )
        except ValueError as exc:
            lanes.append(
                _replace_lane(
                    lane,
                    action="blocked",
                    reasons=(*lane.reasons, f"runbook plan blocked: {exc}"),
                    active_job_resolution=active_job_resolution,
                )
            )
            continue
        audit_json_path = _default_lane_audit_json_path(runbook)
        aggregate_submit_commands += len(plan.submit_commands)
        lanes.append(
            _replace_lane(
                lane,
                plan=plan,
                active_job_resolution=active_job_resolution,
                audit_json_path=audit_json_path,
            )
        )

    if aggregate_submit_commands > 3:
        warnings.append(
            "aggregate submit fanout exceeds 3 qsub commands; runbook gates still probe queue pressure per lane"
        )
    return InferFillPlan(
        study_dir=selected_study_dir,
        runbook_paths=selected_runbook_paths,
        lanes=tuple(lanes),
        warnings=tuple(warnings),
        aggregate_submit_commands=aggregate_submit_commands,
    )


def execute_infer_fill_plan(
    *,
    fill_plan: InferFillPlan,
    submit: bool,
    command_timeout_seconds: float | None = 300.0,
) -> InferFillPlan:
    blocked_errors = tuple(
        f"{lane.runbook_id}: lane blocked: {'; '.join(lane.reasons)}" for lane in fill_plan.blocked_lanes
    )
    if blocked_errors:
        return InferFillPlan(
            study_dir=fill_plan.study_dir,
            runbook_paths=fill_plan.runbook_paths,
            lanes=fill_plan.lanes,
            submit=submit,
            executed=False,
            errors=(*fill_plan.errors, *blocked_errors),
            warnings=fill_plan.warnings,
            aggregate_submit_commands=fill_plan.aggregate_submit_commands,
        )

    lanes: list[InferFillLane] = []
    errors: list[str] = []
    halted = False
    for lane in fill_plan.lanes:
        if halted:
            lanes.append(lane)
            continue
        if lane.action != "run":
            lanes.append(lane)
            continue
        if lane.plan is None or lane.audit_json_path is None:
            errors.append(f"{lane.runbook_id}: runnable lane has no batch plan or audit path")
            lanes.append(lane)
            continue
        result = execute_batch_plan(
            plan=lane.plan,
            audit_json_path=lane.audit_json_path,
            submit=submit,
            command_timeout_seconds=command_timeout_seconds,
        )
        lanes.append(_replace_lane(lane, execution_result=result))
        if not result.ok:
            errors.append(f"{lane.runbook_id}: execution failed in phase {result.failed_phase or 'unknown'}")
            halted = True
    return InferFillPlan(
        study_dir=fill_plan.study_dir,
        runbook_paths=fill_plan.runbook_paths,
        lanes=tuple(lanes),
        submit=submit,
        executed=True,
        errors=(*fill_plan.errors, *tuple(errors)),
        warnings=fill_plan.warnings,
        aggregate_submit_commands=fill_plan.aggregate_submit_commands,
    )


def _classify_infer_lane(*, runbook_path: Path, runbook: OrchestrationRunbookV1) -> InferFillLane:
    if runbook.infer is None:
        return _skip_lane(runbook_path=runbook_path, runbook=runbook, reason="runbook has no Infer block")
    config_path = runbook.infer.config
    try:
        completion = tuple(plan_sequence_view_feature_inventory_completion_from_config(config_path))
    except ValueError as exc:
        if str(exc) == "No selected jobs use feature_bundle.sequence_view_inputs.":
            return InferFillLane(
                runbook_path=runbook_path,
                runbook_id=runbook.id,
                workflow_id=runbook.workflow_id,
                config_path=config_path,
                action="blocked",
                reasons=("unsupported Infer config: selected jobs must define feature_bundle.sequence_view_inputs",),
            )
        raise
    if not completion:
        return _skip_lane(
            runbook_path=runbook_path,
            runbook=runbook,
            reason="Infer config has no sequence-view completion inventory contract",
        )
    totals = _completion_totals(completion)
    reasons: list[str] = []
    action: InferFillLaneAction = "run"
    if totals["missing_products"] > 0:
        action = "blocked"
        reasons.append("missing sequence products block submit")
    if action == "run" and _completion_requires_repair_without_durable_plan(completion):
        action = "blocked"
        reasons.append("Infer lane requires shard-level durability plan before submit")
    if (
        action == "run"
        and totals["missing_vectors"] == 0
        and totals["missing_scalars"] == 0
        and totals["stale_vectors"] == 0
        and totals["stale_scalars"] == 0
    ):
        action = "skip_complete"
        reasons.append("feature sidecars already complete")
    if action == "run":
        reasons.append("missing or stale vectors/scalars remain")
    return InferFillLane(
        runbook_path=runbook_path,
        runbook_id=runbook.id,
        workflow_id=runbook.workflow_id,
        config_path=config_path,
        action=action,
        reasons=tuple(reasons),
        completion=completion,
        required_views=totals["required_views"],
        required_vectors=totals["required_vectors"],
        required_scalars=totals["required_scalars"],
        missing_products=totals["missing_products"],
        missing_vectors=totals["missing_vectors"],
        missing_scalars=totals["missing_scalars"],
        stale_vectors=totals["stale_vectors"],
        stale_scalars=totals["stale_scalars"],
    )


def _completion_requires_repair_without_durable_plan(completion: tuple[Mapping[str, object], ...]) -> bool:
    for plan in completion:
        pending = (
            _mapping_int(plan, "missing_vectors")
            + _mapping_int(plan, "missing_scalars")
            + _mapping_int(plan, "stale_vectors")
            + _mapping_int(plan, "stale_scalars")
        )
        if pending <= 0:
            continue
        shard_plan = plan.get("shard_plan")
        if not isinstance(shard_plan, Mapping):
            return True
        if str(shard_plan.get("commit_policy") or "") != "temp_validate_promote":
            return True
        if str(shard_plan.get("resume_policy") or "") != "skip_committed_retry_failed":
            return True
    return False


def _skip_lane(*, runbook_path: Path, runbook: OrchestrationRunbookV1, reason: str) -> InferFillLane:
    config_path = runbook.infer.config if runbook.infer is not None else Path("")
    return InferFillLane(
        runbook_path=runbook_path,
        runbook_id=runbook.id,
        workflow_id=runbook.workflow_id,
        config_path=config_path,
        action="skip_unsupported",
        reasons=(reason,),
    )


def _replace_lane(
    lane: InferFillLane,
    *,
    action: InferFillLaneAction | None = None,
    reasons: tuple[str, ...] | None = None,
    plan: BatchPlan | None = None,
    active_job_resolution: ActiveJobResolution | None = None,
    audit_json_path: Path | None = None,
    execution_result: BatchExecutionResult | None = None,
) -> InferFillLane:
    return InferFillLane(
        runbook_path=lane.runbook_path,
        runbook_id=lane.runbook_id,
        workflow_id=lane.workflow_id,
        config_path=lane.config_path,
        action=action or lane.action,
        reasons=reasons or lane.reasons,
        completion=lane.completion,
        required_views=lane.required_views,
        required_vectors=lane.required_vectors,
        required_scalars=lane.required_scalars,
        missing_products=lane.missing_products,
        missing_vectors=lane.missing_vectors,
        missing_scalars=lane.missing_scalars,
        stale_vectors=lane.stale_vectors,
        stale_scalars=lane.stale_scalars,
        plan=plan if plan is not None else lane.plan,
        active_job_resolution=(
            active_job_resolution if active_job_resolution is not None else lane.active_job_resolution
        ),
        audit_json_path=audit_json_path if audit_json_path is not None else lane.audit_json_path,
        execution_result=execution_result if execution_result is not None else lane.execution_result,
    )


def _default_lane_audit_json_path(runbook: OrchestrationRunbookV1) -> Path:
    return (runbook.workspace_root / WORKSPACE_AUDIT_RELATIVE_DIR / f"{runbook.id}.fill-infer.json").resolve()


def _completion_totals(completion: Sequence[Mapping[str, object]]) -> dict[str, int]:
    fields = (
        "required_views",
        "required_vectors",
        "required_scalars",
        "missing_products",
        "missing_vectors",
        "missing_scalars",
        "stale_vectors",
        "stale_scalars",
    )
    return {field: sum(_mapping_int(plan, field) for plan in completion) for field in fields}


def _mapping_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field, 0)
    if isinstance(value, bool):
        raise ValueError(f"completion field {field} must be an integer, not boolean")
    return int(value or 0)


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"yaml root must be a mapping: {path}")
    return payload


def _resolve_path_ref(ref: str, *, repo_root: Path, manifest_dir: Path) -> Path:
    text = str(ref or "").strip()
    if not text:
        raise ValueError("path ref must be non-empty")
    if text.startswith("repo:"):
        return (repo_root / text.removeprefix("repo:")).resolve()
    if text.startswith("manifest:"):
        return (manifest_dir / text.removeprefix("manifest:")).resolve()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _dedupe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)
