"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/status_adapters/promoter_status/infer_runtime.py

Study-owned infer-runtime projection for checked-in
promoter status surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.status import resolve_path_ref

from .record_normalizer import PromoterStudyResolvedContext


@dataclass(frozen=True)
class PromoterStudyInferRuntimeDependencies:
    resolve_named_path_mapping: Callable[..., dict[str, Path]]
    resolve_infer_runtime_lane_contracts: Callable[..., Sequence[object]]
    derive_infer_notify_profile_paths: Callable[[Mapping[str, Path]], tuple[dict[str, Path], dict[str, str]]]
    load_infer_model_summary: Callable[[Path], dict[str, object]]
    string_or_none: Callable[[object], str | None]
    string_list_or_empty: Callable[[object], list[str]]


@dataclass(frozen=True)
class PromoterInferPhaseTarget:
    phase_id: str
    config_label: str
    runtime_label: str
    runbook_surface_label: str


@dataclass(frozen=True)
class PromoterStudyInferRuntimeModelSummary:
    label: str
    model_id: str | None
    device: str

    @property
    def requires_gpu(self) -> bool:
        return self.device.startswith("cuda")

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "model_id": self.model_id,
            "device": self.device,
        }


@dataclass(frozen=True)
class PromoterStudyInferRuntimeResolvedContext:
    preferred_model_family: str | None
    supported_model_families: tuple[str, ...]
    infer_config_paths: dict[str, Path]
    runtime_lane_contracts: tuple[object, ...]
    runtime_config_paths: dict[str, Path]
    phase_targets: tuple[PromoterInferPhaseTarget, ...]
    phase_targets_by_id: dict[str, PromoterInferPhaseTarget]
    config_phase_ids: dict[str, str]
    runtime_phase_ids: dict[str, str]
    infer_notify_profile_paths: dict[str, Path]
    infer_notify_profile_errors: dict[str, str]
    runtime_model_summaries: tuple[PromoterStudyInferRuntimeModelSummary, ...]
    gpu_required_runtime_labels: tuple[str, ...]


def resolve_promoter_study_infer_runtime_context(
    *,
    study_context: PromoterStudyResolvedContext,
    status_kind: str,
    dependencies: PromoterStudyInferRuntimeDependencies,
) -> PromoterStudyInferRuntimeResolvedContext:
    study_repo_root = study_context.study_repo_root
    if study_repo_root is None:
        raise ValueError("promoter-study infer-runtime resolution requires a resolved study_repo_root")

    infer_payload = dict(study_context.study_pipeline.get("infer") or {})
    preferred_model_family = dependencies.string_or_none(infer_payload.get("preferred_model_family"))
    supported_model_families = tuple(dependencies.string_list_or_empty(infer_payload.get("supported_model_families")))
    infer_config_paths = dependencies.resolve_named_path_mapping(
        infer_payload.get("configs"),
        repo_root=study_repo_root,
        label="infer configs",
        status_kind=status_kind,
    )
    runtime_lane_contracts = tuple(
        dependencies.resolve_infer_runtime_lane_contracts(
            infer_config_paths,
            preferred_model_family=preferred_model_family,
        )
    )
    runtime_config_paths = {
        str(getattr(contract, "runtime_label")): Path(getattr(contract, "config_path"))
        for contract in runtime_lane_contracts
    }
    phase_targets = _resolve_study_infer_runtime_phase_targets(
        runtime_lane_contracts,
        study_context=study_context,
    )
    phase_targets_by_id = {target.phase_id: target for target in phase_targets}
    config_phase_ids = {target.config_label: target.phase_id for target in phase_targets if target.config_label}
    runtime_phase_ids = {target.runtime_label: target.phase_id for target in phase_targets if target.runtime_label}
    derived_notify_profile_paths, derived_notify_profile_errors = dependencies.derive_infer_notify_profile_paths(
        runtime_config_paths
    )
    infer_notify_profile_paths = {
        label: derived_notify_profile_paths[label]
        for label in sorted(runtime_config_paths)
        if label in derived_notify_profile_paths
    }
    infer_notify_profile_errors = {
        label: derived_notify_profile_errors[label]
        for label in sorted(runtime_config_paths)
        if label in derived_notify_profile_errors
    }
    runtime_model_summaries = tuple(
        _resolve_runtime_model_summary(runtime_lane, dependencies=dependencies)
        for runtime_lane in runtime_lane_contracts
    )
    gpu_required_runtime_labels = tuple(summary.label for summary in runtime_model_summaries if summary.requires_gpu)
    return PromoterStudyInferRuntimeResolvedContext(
        preferred_model_family=preferred_model_family,
        supported_model_families=supported_model_families,
        infer_config_paths=infer_config_paths,
        runtime_lane_contracts=runtime_lane_contracts,
        runtime_config_paths=runtime_config_paths,
        phase_targets=phase_targets,
        phase_targets_by_id=phase_targets_by_id,
        config_phase_ids=config_phase_ids,
        runtime_phase_ids=runtime_phase_ids,
        infer_notify_profile_paths=infer_notify_profile_paths,
        infer_notify_profile_errors=infer_notify_profile_errors,
        runtime_model_summaries=runtime_model_summaries,
        gpu_required_runtime_labels=gpu_required_runtime_labels,
    )


def _resolve_runtime_model_summary(
    runtime_lane: object,
    *,
    dependencies: PromoterStudyInferRuntimeDependencies,
) -> PromoterStudyInferRuntimeModelSummary:
    runtime_label = str(getattr(runtime_lane, "runtime_label"))
    config_path = Path(getattr(runtime_lane, "config_path"))
    payload = dependencies.load_infer_model_summary(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"infer model summary loader must return a mapping for {config_path}")
    return PromoterStudyInferRuntimeModelSummary(
        label=runtime_label,
        model_id=dependencies.string_or_none(payload.get("model_id")),
        device=dependencies.string_or_none(payload.get("device")) or "unknown",
    )


def _resolve_study_infer_runtime_phase_targets(
    runtime_lane_contracts: Sequence[object],
    *,
    study_context: PromoterStudyResolvedContext,
) -> tuple[PromoterInferPhaseTarget, ...]:
    study_repo_root = study_context.study_repo_root
    if study_repo_root is None:
        raise ValueError("study-owned infer phase targets require a resolved study_repo_root")
    phase_index = {
        _required_runtime_lane_text(phase.get("id"), label="study pipeline phase id"): dict(phase)
        for phase in study_context.phase_states
    }
    execution_surface_labels_by_path = {
        path.expanduser().resolve(): label for label, path in study_context.execution_surface_index.items()
    }
    targets: list[PromoterInferPhaseTarget] = []
    for runtime_lane in runtime_lane_contracts:
        phase_id = _required_runtime_lane_text(getattr(runtime_lane, "phase_id", None), label="infer phase_id")
        runtime_label = _required_runtime_lane_text(
            getattr(runtime_lane, "runtime_label", None),
            label="infer runtime_label",
        )
        config_label = _required_runtime_lane_text(
            getattr(runtime_lane, "config_label", None),
            label="infer config_label",
        )
        phase_state = phase_index.get(phase_id)
        if phase_state is None:
            raise ValueError(f"infer runtime phase is not declared in study pipeline phases: {phase_id}")
        next_surface = _required_runtime_lane_text(
            phase_state.get("next_surface"),
            label=f"study pipeline next_surface for {phase_id}",
        )
        resolved_next_surface = _resolve_study_surface_path(next_surface, repo_root=study_repo_root)
        runbook_surface_label = execution_surface_labels_by_path.get(resolved_next_surface)
        if runbook_surface_label is None:
            raise ValueError(
                "infer runtime phase next_surface is not declared under study execution_surfaces: "
                f"{phase_id} -> {resolved_next_surface}"
            )
        targets.append(
            PromoterInferPhaseTarget(
                phase_id=phase_id,
                config_label=config_label,
                runtime_label=runtime_label,
                runbook_surface_label=runbook_surface_label,
            )
        )
    return tuple(targets)


def _required_runtime_lane_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _resolve_study_surface_path(raw_path: str, *, repo_root: Path) -> Path:
    return resolve_path_ref(
        raw_path,
        repo_root=repo_root,
        default_base="repo",
        label="ops.study.yaml next_surface",
    )


__all__ = [
    "PromoterStudyInferRuntimeDependencies",
    "PromoterInferPhaseTarget",
    "PromoterStudyInferRuntimeModelSummary",
    "PromoterStudyInferRuntimeResolvedContext",
    "resolve_promoter_study_infer_runtime_context",
]
