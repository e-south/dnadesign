"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/loader.py

Public loading orchestration for compositional RT-lnRNA bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    ResolvedSubjectBinding,
    SubjectBinding,
    SubjectBindingByteBlock,
    SubjectBindingContractError,
    SubjectBindingMaterializationResolution,
    SubjectBindingRegistry,
)
from .registry import assemble_registry

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_DEFAULT_REGISTRY_PATH = _STUDY_DIR / "workbench/provenance/subject_bindings/retron_subject_bindings_v1.yaml"


def load_registered_subject_bindings(*, repo_root: Path | None = None) -> SubjectBindingRegistry:
    root = _resolve_repo_root(repo_root)
    return load_subject_bindings(repo_root=root, registry_path=root / _DEFAULT_REGISTRY_PATH)


def load_resolved_registered_subject_bindings(*, repo_root: Path | None = None) -> tuple[ResolvedSubjectBinding, ...]:
    """Load bindings only when every source owner makes exact sequence bytes available."""

    root = _resolve_repo_root(repo_root)
    return load_resolved_subject_bindings(repo_root=root, registry_path=root / _DEFAULT_REGISTRY_PATH)


def load_registered_subject_binding_materialization(
    *,
    repo_root: Path | None = None,
    subject_ids: tuple[str, ...] | None = None,
) -> SubjectBindingMaterializationResolution:
    """Resolve available bytes independently and report opaque provider blocks."""

    root = _resolve_repo_root(repo_root)
    registry_path = root / _DEFAULT_REGISTRY_PATH
    registry = load_subject_bindings(repo_root=root, registry_path=registry_path)
    selected = _selected_subjects(registry=registry, subject_ids=subject_ids)
    blocked = tuple(
        SubjectBindingByteBlock(
            subject_id=subject.subject_id,
            owner_study_id=subject.rt_part.owner_study_id,
            part_id=subject.rt_part.part_id,
            provider_ref=subject.rt_part.provider_ref or "",
            cds_sha256=subject.rt_part.sequence_sha256,
            reason="provider_publication_omits_rt_cds_bytes",
        )
        for subject in selected
        if subject.rt_part.provider_ref is not None
    )
    if subject_ids is not None and blocked:
        details = ", ".join(f"{item.subject_id} ({item.provider_ref})" for item in blocked)
        raise SubjectBindingContractError(f"exact subject projection is byte-blocked: {details}")
    resolvable_ids = frozenset(subject.subject_id for subject in selected if subject.rt_part.provider_ref is None)
    resolved = (
        ()
        if not resolvable_ids
        else assemble_registry(
            repo_root=root,
            registry_path=registry_path,
            require_sequence_bytes=True,
            selected_subject_ids=resolvable_ids,
        )[1]
    )
    return SubjectBindingMaterializationResolution(
        resolved_subjects=resolved,
        blocked_subjects=blocked,
    )


def load_subject_bindings(*, repo_root: Path, registry_path: Path) -> SubjectBindingRegistry:
    registry, _resolved = assemble_registry(
        repo_root=repo_root,
        registry_path=registry_path,
        require_sequence_bytes=False,
    )
    return registry


def load_resolved_subject_bindings(*, repo_root: Path, registry_path: Path) -> tuple[ResolvedSubjectBinding, ...]:
    """Resolve exact bytes or fail when a provider publishes opaque metadata only."""

    _registry, resolved = assemble_registry(
        repo_root=repo_root,
        registry_path=registry_path,
        require_sequence_bytes=True,
    )
    return resolved


def _selected_subjects(
    *, registry: SubjectBindingRegistry, subject_ids: tuple[str, ...] | None
) -> tuple[SubjectBinding, ...]:
    if subject_ids is None:
        return registry.subjects
    if not subject_ids:
        raise SubjectBindingContractError("subject_ids must contain at least one exact subject id")
    if len(set(subject_ids)) != len(subject_ids):
        raise SubjectBindingContractError("subject_ids must not contain duplicates")
    return tuple(registry.resolve_subject_id(subject_id) for subject_id in subject_ids)


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


__all__ = [
    "load_registered_subject_binding_materialization",
    "load_registered_subject_bindings",
    "load_resolved_registered_subject_bindings",
    "load_resolved_subject_bindings",
    "load_subject_bindings",
]
