"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/record_normalizer.py

Study-owned checked-in record normalization for the promoter family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dnadesign.studies.core.models import StudyOpsContract
from dnadesign.studies.core.record_loader import load_study_ops_contract


@dataclass(frozen=True)
class PromoterStudyContextDependencies:
    discover_active_study_dir: Callable[..., tuple[Path, Path, str]]
    required_path: Callable[..., Path]
    discover_repo_root: Callable[[Path], Path | None]
    load_yaml_mapping: Callable[..., dict[str, object]]
    resolve_repo_relative_path: Callable[..., Path]
    resolve_named_path_mapping: Callable[..., dict[str, Path]]
    string_or_none: Callable[[object], str | None]
    optional_positive_int: Callable[[object], int | None]
    required_metadata_text: Callable[..., str]
    parquet_row_count: Callable[[Path], int]


@dataclass(frozen=True)
class PromoterStudyResolvedContext:
    study_dir_exists: bool
    requested_study_dir: str | None
    resolved_study_dir: Path
    study_repo_root: Path | None
    study_id: str
    selection_source: str
    registry_path: Path | None
    active_study: str | None
    required_paths: dict[str, Path]
    missing_required_files: tuple[str, ...]
    pipeline_path: Path
    pipeline_present: bool
    datasets_entries: tuple[dict[str, object], ...]
    study_pipeline: dict[str, object]
    canonical_usr_root_path: Path | None
    dataset_states: tuple[dict[str, object], ...]
    dataset_index: dict[str, dict[str, object]]
    missing_declared_present: tuple[str, ...]
    present_but_planned: tuple[str, ...]
    execution_surface_states: tuple[dict[str, object], ...]
    execution_surface_index: dict[str, Path]
    missing_execution_surfaces: tuple[str, ...]
    phase_states: tuple[dict[str, object], ...]
    current_phase: str | None
    current_phase_is_known: bool
    next_ready_phase: dict[str, object] | None
    next_in_progress_phase: dict[str, object] | None
    next_planned_phase: dict[str, object] | None
    blocked_phases: tuple[dict[str, object], ...]
    densegen_dataset_id: str | None
    densegen_rows: int | None
    densegen_row_target: int | None
    densegen_row_gap: int | None
    merged_anchor_dataset_id: str | None
    merged_anchor_rows: int | None
    construct_context_dataset_id: str | None
    construct_context_rows: int | None
    dataset_refresh_states: tuple[dict[str, object], ...]
    stale_dataset_ids: tuple[str, ...]
    evidence: dict[str, object]
    derived_execution_surface_states: tuple[dict[str, object], ...] = ()
    derived_execution_surface_index: dict[str, Path] = field(default_factory=dict)
    missing_derived_execution_surfaces: tuple[str, ...] = ()
    ops_contract: StudyOpsContract | None = None


def resolve_promoter_study_context(
    study_dir: Path | None,
    *,
    repo_root: Path | None = None,
    status_kind: str,
    dependencies: PromoterStudyContextDependencies,
) -> PromoterStudyResolvedContext:
    resolved_input_repo_root = repo_root.expanduser().resolve() if repo_root is not None else None
    selection_source = "explicit"
    requested_study_dir = str(study_dir) if study_dir is not None else None
    if study_dir is None:
        resolved_study_dir, registry_path, active_registry_study = dependencies.discover_active_study_dir(
            repo_root=resolved_input_repo_root,
            status_kind=status_kind,
        )
        selection_source = "active_registry"
    else:
        resolved_study_dir = dependencies.required_path(
            study_dir,
            flag_name="--study-dir",
            status_kind=status_kind,
            base_dir=resolved_input_repo_root,
        )
        registry_path = None
        active_registry_study = None
    if not resolved_study_dir.exists():
        return PromoterStudyResolvedContext(
            study_dir_exists=False,
            requested_study_dir=requested_study_dir,
            resolved_study_dir=resolved_study_dir,
            study_repo_root=resolved_input_repo_root,
            study_id=resolved_study_dir.name,
            ops_contract=None,
            selection_source=selection_source,
            registry_path=registry_path,
            active_study=None,
            required_paths={},
            missing_required_files=(),
            pipeline_path=resolved_study_dir / "pipeline.yaml",
            pipeline_present=False,
            datasets_entries=(),
            study_pipeline={},
            canonical_usr_root_path=None,
            dataset_states=(),
            dataset_index={},
            missing_declared_present=(),
            present_but_planned=(),
            execution_surface_states=(),
            execution_surface_index={},
            missing_execution_surfaces=(),
            derived_execution_surface_states=(),
            derived_execution_surface_index={},
            missing_derived_execution_surfaces=(),
            phase_states=(),
            current_phase=None,
            current_phase_is_known=False,
            next_ready_phase=None,
            next_in_progress_phase=None,
            next_planned_phase=None,
            blocked_phases=(),
            densegen_dataset_id=None,
            densegen_rows=None,
            densegen_row_target=None,
            densegen_row_gap=None,
            merged_anchor_dataset_id=None,
            merged_anchor_rows=None,
            construct_context_dataset_id=None,
            construct_context_rows=None,
            dataset_refresh_states=(),
            stale_dataset_ids=(),
            evidence={
                "requested_study_dir": requested_study_dir,
                "study_dir": str(resolved_study_dir),
                "study_selection_source": selection_source,
            },
        )
    if not resolved_study_dir.is_dir():
        raise ValueError(f"study_dir must be a directory: {resolved_study_dir}")

    study_repo_root = dependencies.discover_repo_root(resolved_study_dir)
    if study_repo_root is None:
        raise ValueError(f"study_dir must live inside a dnadesign repository checkout: {resolved_study_dir}")
    if registry_path is None:
        registry_path = (study_repo_root / "docs" / "studies" / "index.yaml").resolve()

    required_paths = {
        "campaign.yaml": resolved_study_dir / "campaign.yaml",
        "datasets.yaml": resolved_study_dir / "datasets.yaml",
        "ops.study.yaml": resolved_study_dir / "ops.study.yaml",
        "status.md": resolved_study_dir / "status.md",
    }
    missing_required_files = tuple(name for name, path in required_paths.items() if not path.exists())
    pipeline_path = resolved_study_dir / "pipeline.yaml"
    evidence: dict[str, object] = {
        "requested_study_dir": requested_study_dir,
        "study_dir": str(resolved_study_dir),
        "repo_root": str(study_repo_root),
        "study_id": resolved_study_dir.name,
        "ops_study_contract_path": str(required_paths["ops.study.yaml"]),
        "study_selection_source": selection_source,
        "active_study_registry_path": str(registry_path),
        "required_files": {name: str(path) for name, path in required_paths.items()},
        "pipeline_path": str(pipeline_path),
        "pipeline_present": pipeline_path.exists(),
        "missing_required_files": list(missing_required_files),
    }
    if missing_required_files:
        return PromoterStudyResolvedContext(
            study_dir_exists=True,
            requested_study_dir=requested_study_dir,
            resolved_study_dir=resolved_study_dir,
            study_repo_root=study_repo_root,
            study_id=resolved_study_dir.name,
            ops_contract=None,
            selection_source=selection_source,
            registry_path=registry_path,
            active_study=active_registry_study,
            required_paths=required_paths,
            missing_required_files=missing_required_files,
            pipeline_path=pipeline_path,
            pipeline_present=pipeline_path.exists(),
            datasets_entries=(),
            study_pipeline={},
            canonical_usr_root_path=None,
            dataset_states=(),
            dataset_index={},
            missing_declared_present=(),
            present_but_planned=(),
            execution_surface_states=(),
            execution_surface_index={},
            missing_execution_surfaces=(),
            derived_execution_surface_states=(),
            derived_execution_surface_index={},
            missing_derived_execution_surfaces=(),
            phase_states=(),
            current_phase=None,
            current_phase_is_known=False,
            next_ready_phase=None,
            next_in_progress_phase=None,
            next_planned_phase=None,
            blocked_phases=(),
            densegen_dataset_id=None,
            densegen_rows=None,
            densegen_row_target=None,
            densegen_row_gap=None,
            merged_anchor_dataset_id=None,
            merged_anchor_rows=None,
            construct_context_dataset_id=None,
            construct_context_rows=None,
            dataset_refresh_states=(),
            stale_dataset_ids=(),
            evidence=evidence,
        )

    ops_contract = load_study_ops_contract(resolved_study_dir)
    datasets_payload = dependencies.load_yaml_mapping(required_paths["datasets.yaml"], label="datasets.yaml")
    datasets_entries = datasets_payload.get("datasets") or []
    if not isinstance(datasets_entries, list):
        raise ValueError(f"datasets.yaml must define a 'datasets' list: {required_paths['datasets.yaml']}")

    pipeline_payload = (
        dependencies.load_yaml_mapping(pipeline_path, label="pipeline.yaml") if pipeline_path.exists() else {}
    )
    study_pipeline = pipeline_payload.get("study_pipeline") or {}
    if study_pipeline and not isinstance(study_pipeline, dict):
        raise ValueError(f"pipeline.yaml must define a 'study_pipeline' mapping: {pipeline_path}")

    study_index_path = registry_path
    active_study = None
    if study_index_path is not None and study_index_path.exists():
        study_index = dependencies.load_yaml_mapping(study_index_path, label="study index")
        active_study = dependencies.string_or_none(study_index.get("active_study_id"))
    if active_study is None:
        active_study = active_registry_study

    canonical_usr_root_text = dependencies.string_or_none(study_pipeline.get("canonical_usr_root"))
    if canonical_usr_root_text is None:
        for entry in datasets_entries:
            if isinstance(entry, dict):
                canonical_usr_root_text = dependencies.string_or_none(entry.get("usr_root"))
                if canonical_usr_root_text is not None:
                    break
    canonical_usr_root_path = (
        dependencies.resolve_repo_relative_path(
            repo_root=study_repo_root,
            raw_path=canonical_usr_root_text,
            status_kind=status_kind,
        )
        if canonical_usr_root_text is not None
        else None
    )

    dataset_states: list[dict[str, object]] = []
    missing_declared_present: list[str] = []
    present_but_planned: list[str] = []
    dataset_index: dict[str, dict[str, object]] = {}
    for entry in datasets_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry must be a mapping: {required_paths['datasets.yaml']}")
        dataset_id = dependencies.required_metadata_text(
            entry.get("dataset"),
            label="dataset id",
            source=required_paths["datasets.yaml"],
        )
        role = dependencies.string_or_none(entry.get("role")) or dataset_id
        declared_status = dependencies.string_or_none(entry.get("status")) or "unknown"
        entry_usr_root_text = dependencies.string_or_none(entry.get("usr_root")) or canonical_usr_root_text
        entry_usr_root = dependencies.resolve_repo_relative_path(
            repo_root=study_repo_root,
            raw_path=entry_usr_root_text,
            status_kind=status_kind,
        )
        records_path = (entry_usr_root / dataset_id / "records.parquet").resolve()
        exists = records_path.exists()
        rows = dependencies.parquet_row_count(records_path) if exists else None
        dataset_state = {
            "role": role,
            "dataset": dataset_id,
            "declared_status": declared_status,
            "usr_root": str(entry_usr_root),
            "records_path": str(records_path),
            "exists": exists,
            "rows": rows,
        }
        dataset_states.append(dataset_state)
        dataset_index[dataset_id] = dataset_state
        if declared_status == "present" and not exists:
            missing_declared_present.append(dataset_id)
        if declared_status == "planned" and exists:
            present_but_planned.append(dataset_id)

    execution_surface_payload = _declared_execution_surface_payload(
        contract_payload=ops_contract.execution_surfaces,
    )
    execution_surface_index = dependencies.resolve_named_path_mapping(
        execution_surface_payload,
        repo_root=study_repo_root,
        label="execution_surfaces",
        status_kind=status_kind,
    )
    derived_execution_surface_index = _derived_execution_surface_index(
        pipeline_payload=study_pipeline.get("execution_surfaces"),
        execution_surface_index=execution_surface_index,
        dependencies=dependencies,
        repo_root=study_repo_root,
        status_kind=status_kind,
    )
    execution_surface_states, missing_execution_surfaces = _build_execution_surface_states(execution_surface_index)
    derived_execution_surface_states, missing_derived_execution_surfaces = _build_execution_surface_states(
        derived_execution_surface_index
    )

    phase_states = [phase.as_dict() for phase in ops_contract.phases]
    phase_index = {phase.id: phase.as_dict() for phase in ops_contract.phases}
    current_phase = ops_contract.current_phase_id
    current_phase_is_known = current_phase in phase_index if current_phase is not None else False
    next_ready_phase = _first_phase_by_status(phase_states, status="ready")
    next_in_progress_phase = _first_phase_by_status(phase_states, status="in_progress")
    next_planned_phase = _first_phase_by_status(phase_states, status="planned")
    blocked_phases = tuple(phase for phase in phase_states if phase["status"] == "blocked_gpu")

    densegen_dataset_id = dependencies.string_or_none(
        (study_pipeline.get("datasets") or {}).get("densegen_anchor_source")
    ) or dependencies.string_or_none((ops_contract.artifacts.get("densegen_anchor_source") or {}).get("dataset_id"))
    densegen_dataset_state = dataset_index.get(densegen_dataset_id or "") if densegen_dataset_id else None
    row_target = dependencies.optional_positive_int(
        ((study_pipeline.get("row_targets") or {}).get("densegen_anchor_minimum_before_first_full_lane_infer"))
        or _snapshot_target_rows(ops_contract=ops_contract, artifact_id="densegen_anchor_source")
    )
    densegen_rows = densegen_dataset_state.get("rows") if densegen_dataset_state is not None else None
    densegen_row_gap = (
        max(int(row_target) - int(densegen_rows), 0) if row_target is not None and densegen_rows is not None else None
    )
    merged_anchor_dataset_id = dependencies.string_or_none(
        (study_pipeline.get("datasets") or {}).get("merged_anchor_dataset")
    ) or dependencies.string_or_none((ops_contract.artifacts.get("merged_anchor_dataset") or {}).get("dataset_id"))
    merged_anchor_dataset_state = (
        dataset_index.get(merged_anchor_dataset_id or "") if merged_anchor_dataset_id else None
    )
    merged_anchor_rows = merged_anchor_dataset_state.get("rows") if merged_anchor_dataset_state is not None else None
    construct_context_dataset_id = dependencies.string_or_none(
        (study_pipeline.get("datasets") or {}).get("construct_context_dataset")
    ) or dependencies.string_or_none((ops_contract.artifacts.get("construct_context_dataset") or {}).get("dataset_id"))
    construct_context_dataset_state = (
        dataset_index.get(construct_context_dataset_id or "") if construct_context_dataset_id else None
    )
    construct_context_rows = (
        construct_context_dataset_state.get("rows") if construct_context_dataset_state is not None else None
    )
    dataset_refresh_states = _build_dataset_refresh_states(
        densegen_dataset_id=densegen_dataset_id,
        densegen_rows=densegen_rows,
        merged_anchor_dataset_id=merged_anchor_dataset_id,
        merged_anchor_rows=merged_anchor_rows,
        construct_context_dataset_id=construct_context_dataset_id,
        construct_context_rows=construct_context_rows,
    )
    stale_dataset_ids = tuple(
        str(state.get("downstream_dataset") or "").strip()
        for state in dataset_refresh_states
        if str(state.get("state") or "").strip() == "attention" and str(state.get("downstream_dataset") or "").strip()
    )

    evidence.update(
        {
            "active_study": active_study,
            "is_active_study": active_study == resolved_study_dir.name if active_study is not None else None,
            "ops_study_contract": dict(ops_contract.raw_payload),
            "canonical_usr_root": str(canonical_usr_root_path) if canonical_usr_root_path is not None else None,
            "datasets": dataset_states,
            "missing_declared_present": missing_declared_present,
            "present_but_planned": present_but_planned,
            "execution_surfaces": execution_surface_states,
            "missing_execution_surfaces": missing_execution_surfaces,
            "derived_execution_surfaces": derived_execution_surface_states,
            "missing_derived_execution_surfaces": missing_derived_execution_surfaces,
            "current_phase": current_phase,
            "current_phase_is_known": current_phase_is_known,
            "phase_states": phase_states,
            "next_ready_phase": next_ready_phase,
            "next_in_progress_phase": next_in_progress_phase,
            "next_planned_phase": next_planned_phase,
            "blocked_phases": list(blocked_phases),
            "densegen_dataset": densegen_dataset_id,
            "densegen_rows": densegen_rows,
            "densegen_row_target": row_target,
            "densegen_row_gap": densegen_row_gap,
            "merged_anchor_dataset": merged_anchor_dataset_id,
            "merged_anchor_rows": merged_anchor_rows,
            "construct_context_dataset": construct_context_dataset_id,
            "construct_context_rows": construct_context_rows,
            "dataset_refresh_states": dataset_refresh_states,
            "stale_dataset_ids": list(stale_dataset_ids),
        }
    )

    return PromoterStudyResolvedContext(
        study_dir_exists=True,
        requested_study_dir=requested_study_dir,
        resolved_study_dir=resolved_study_dir,
        study_repo_root=study_repo_root,
        study_id=resolved_study_dir.name,
        ops_contract=ops_contract,
        selection_source=selection_source,
        registry_path=registry_path,
        active_study=active_study,
        required_paths=required_paths,
        missing_required_files=missing_required_files,
        pipeline_path=pipeline_path,
        pipeline_present=pipeline_path.exists(),
        datasets_entries=tuple(dict(entry) for entry in datasets_entries if isinstance(entry, dict)),
        study_pipeline=dict(study_pipeline),
        canonical_usr_root_path=canonical_usr_root_path,
        dataset_states=tuple(dataset_states),
        dataset_index=dict(dataset_index),
        missing_declared_present=tuple(missing_declared_present),
        present_but_planned=tuple(present_but_planned),
        execution_surface_states=tuple(execution_surface_states),
        execution_surface_index=dict(execution_surface_index),
        missing_execution_surfaces=tuple(missing_execution_surfaces),
        derived_execution_surface_states=tuple(derived_execution_surface_states),
        derived_execution_surface_index=dict(derived_execution_surface_index),
        missing_derived_execution_surfaces=tuple(missing_derived_execution_surfaces),
        phase_states=tuple(phase_states),
        current_phase=current_phase,
        current_phase_is_known=current_phase_is_known,
        next_ready_phase=next_ready_phase,
        next_in_progress_phase=next_in_progress_phase,
        next_planned_phase=next_planned_phase,
        blocked_phases=blocked_phases,
        densegen_dataset_id=densegen_dataset_id,
        densegen_rows=densegen_rows,
        densegen_row_target=row_target,
        densegen_row_gap=densegen_row_gap,
        merged_anchor_dataset_id=merged_anchor_dataset_id,
        merged_anchor_rows=merged_anchor_rows,
        construct_context_dataset_id=construct_context_dataset_id,
        construct_context_rows=construct_context_rows,
        dataset_refresh_states=tuple(dataset_refresh_states),
        stale_dataset_ids=stale_dataset_ids,
        evidence=evidence,
    )


def _build_execution_surface_states(
    execution_surface_index: dict[str, Path],
) -> tuple[list[dict[str, object]], list[str]]:
    execution_surface_states: list[dict[str, object]] = []
    missing_execution_surfaces: list[str] = []
    for label, resolved_path in execution_surface_index.items():
        exists = resolved_path.exists()
        execution_surface_states.append({"label": label, "path": str(resolved_path), "exists": exists})
        if not exists:
            missing_execution_surfaces.append(label)
    return execution_surface_states, missing_execution_surfaces


def _first_phase_by_status(phases: list[dict[str, object]], *, status: str) -> dict[str, object] | None:
    for phase in phases:
        if str(phase.get("status") or "").strip() == status:
            return dict(phase)
    return None


def _declared_execution_surface_payload(*, contract_payload: dict[str, dict[str, object]]) -> dict[str, object]:
    declared: dict[str, object] = {}
    for label, surface_payload in contract_payload.items():
        if not isinstance(surface_payload, dict):
            continue
        path_ref = str(
            surface_payload.get("runbook_ref")
            or surface_payload.get("workspace_ref")
            or surface_payload.get("path_ref")
            or ""
        ).strip()
        if path_ref:
            declared[label] = path_ref
    return declared


def _derived_execution_surface_index(
    *,
    pipeline_payload: object,
    execution_surface_index: dict[str, Path],
    dependencies: PromoterStudyContextDependencies,
    repo_root: Path,
    status_kind: str,
) -> dict[str, Path]:
    pipeline_index = dependencies.resolve_named_path_mapping(
        pipeline_payload,
        repo_root=repo_root,
        label="execution_surfaces",
        status_kind=status_kind,
    )
    return {
        label: resolved_path for label, resolved_path in pipeline_index.items() if label not in execution_surface_index
    }


def _snapshot_target_rows(*, ops_contract: StudyOpsContract, artifact_id: str) -> int | None:
    snapshot_payload = ops_contract.raw_payload.get("snapshot")
    if not isinstance(snapshot_payload, dict):
        return None
    summary_inputs = snapshot_payload.get("summary_inputs")
    if not isinstance(summary_inputs, list):
        return None
    for item in summary_inputs:
        if not isinstance(item, dict):
            continue
        if str(item.get("artifact") or "").strip() != artifact_id:
            continue
        target_rows = item.get("target_rows")
        if isinstance(target_rows, bool):
            return None
        if isinstance(target_rows, int):
            return target_rows
        text = str(target_rows or "").strip()
        if text.isdigit():
            return int(text)
        return None
    return None


def _build_dataset_refresh_states(
    *,
    densegen_dataset_id: str | None,
    densegen_rows: int | None,
    merged_anchor_dataset_id: str | None,
    merged_anchor_rows: int | None,
    construct_context_dataset_id: str | None,
    construct_context_rows: int | None,
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    merged_anchor_state = _dataset_refresh_state(
        check_id="merged_anchor_from_densegen",
        upstream_dataset=densegen_dataset_id,
        upstream_rows=densegen_rows,
        downstream_dataset=merged_anchor_dataset_id,
        downstream_rows=merged_anchor_rows,
        ok_summary="Merged anchor dataset is at least as current as the DenseGen source.",
        attention_summary="Merged anchor dataset trails DenseGen source rows.",
    )
    if merged_anchor_state is not None:
        states.append(merged_anchor_state)
    construct_context_state = _dataset_refresh_state(
        check_id="construct_contexts_from_merged_anchor",
        upstream_dataset=merged_anchor_dataset_id,
        upstream_rows=merged_anchor_rows,
        downstream_dataset=construct_context_dataset_id,
        downstream_rows=construct_context_rows,
        ok_summary="Construct context dataset is at least as current as the merged anchor dataset.",
        attention_summary="Construct context dataset trails merged anchor rows.",
    )
    if construct_context_state is not None:
        if merged_anchor_state is not None and str(merged_anchor_state.get("state") or "").strip() == "attention":
            construct_context_state = {
                **construct_context_state,
                "state": "attention",
                "summary": (
                    "Construct context dataset is downstream of a stale merged anchor dataset. "
                    f"{construct_context_dataset_id}={construct_context_rows} still trails "
                    f"{densegen_dataset_id}={densegen_rows} through {merged_anchor_dataset_id}={merged_anchor_rows}."
                ),
                "upstream_refresh_blocked_by": str(merged_anchor_state.get("id") or "").strip(),
            }
        states.append(construct_context_state)
    return states


def _dataset_refresh_state(
    *,
    check_id: str,
    upstream_dataset: str | None,
    upstream_rows: int | None,
    downstream_dataset: str | None,
    downstream_rows: int | None,
    ok_summary: str,
    attention_summary: str,
) -> dict[str, object] | None:
    if upstream_rows is None or downstream_rows is None:
        return None
    lag_rows = max(int(upstream_rows) - int(downstream_rows), 0)
    state = "ok" if lag_rows == 0 else "attention"
    summary = ok_summary
    if lag_rows:
        summary = (
            f"{attention_summary} "
            f"{downstream_dataset}={downstream_rows} < {upstream_dataset}={upstream_rows} (lag={lag_rows})."
        )
    return {
        "id": check_id,
        "state": state,
        "summary": summary,
        "upstream_dataset": upstream_dataset,
        "upstream_rows": upstream_rows,
        "downstream_dataset": downstream_dataset,
        "downstream_rows": downstream_rows,
        "lag_rows": lag_rows,
    }
