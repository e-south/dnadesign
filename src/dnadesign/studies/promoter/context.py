"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/promoter/context.py

Study-owned checked-in record resolution for the promoter
status surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    evidence: dict[str, object]
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
        registry_path = resolved_study_dir.parent / "index.yaml"
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

    promoter_index_path = registry_path
    active_study = None
    if promoter_index_path.exists():
        promoter_index = dependencies.load_yaml_mapping(promoter_index_path, label="promoter index")
        active_study = dependencies.string_or_none(promoter_index.get("active_study"))
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

    execution_surface_index = dependencies.resolve_named_path_mapping(
        study_pipeline.get("execution_surfaces"),
        repo_root=study_repo_root,
        label="execution_surfaces",
        status_kind=status_kind,
    )
    execution_surface_states: list[dict[str, object]] = []
    missing_execution_surfaces: list[str] = []
    for label, resolved_path in execution_surface_index.items():
        exists = resolved_path.exists()
        execution_surface_states.append({"label": label, "path": str(resolved_path), "exists": exists})
        if not exists:
            missing_execution_surfaces.append(label)

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
    )
    densegen_dataset_state = dataset_index.get(densegen_dataset_id or "") if densegen_dataset_id else None
    row_target = dependencies.optional_positive_int(
        ((study_pipeline.get("row_targets") or {}).get("densegen_anchor_minimum_before_first_full_lane_infer"))
    )
    densegen_rows = densegen_dataset_state.get("rows") if densegen_dataset_state is not None else None
    densegen_row_gap = (
        max(int(row_target) - int(densegen_rows), 0) if row_target is not None and densegen_rows is not None else None
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
        evidence=evidence,
    )


def _first_phase_by_status(phases: list[dict[str, object]], *, status: str) -> dict[str, object] | None:
    for phase in phases:
        if str(phase.get("status") or "").strip() == status:
            return dict(phase)
    return None
