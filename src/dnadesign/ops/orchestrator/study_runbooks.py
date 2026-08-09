"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/study_runbooks.py

OPS-owned discovery of Infer runbook refs declared by an external study workspace.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from dnadesign.ops.status.path_ref import resolve_path_ref


def discover_infer_runbook_paths_for_study(*, study_dir: Path, repo_root: Path) -> tuple[Path, ...]:
    resolved_study_dir = study_dir.expanduser().resolve()
    operations_dir = resolved_study_dir / "operations"
    contract_path = operations_dir / "ops.study.yaml"
    payload = _read_yaml_mapping(contract_path)
    surfaces = _load_execution_surfaces(payload, contract_path=contract_path)
    paths: list[Path] = []
    for surface_id, surface in surfaces.items():
        surface_type = str(surface.get("surface_type") or "").strip()
        if not surface_type:
            raise ValueError(
                f"ops.study.yaml execution_surfaces.{surface_id} must define surface_type: {contract_path}"
            )
        if surface_type != "runbook":
            continue
        runbook_ref = str(surface.get("runbook_ref") or "").strip()
        if not runbook_ref:
            raise ValueError(f"ops.study.yaml execution_surfaces.{surface_id} must define runbook_ref: {contract_path}")
        paths.append(
            resolve_path_ref(
                runbook_ref,
                repo_root=repo_root,
                manifest_dir=resolved_study_dir,
                default_base="repo",
                label=f"ops.study.yaml execution_surfaces.{surface_id}.runbook_ref",
            )
        )
    return _dedupe_paths(paths)


def _load_execution_surfaces(payload: Mapping[str, object], *, contract_path: Path) -> dict[str, dict[str, object]]:
    inline_surfaces = payload.get("execution_surfaces")
    parts_payload = payload.get("parts")
    part_refs: object = None
    if parts_payload is not None:
        if not isinstance(parts_payload, Mapping):
            raise ValueError(f"ops.study.yaml parts must be a mapping: {contract_path}")
        part_refs = parts_payload.get("execution_surfaces")
    if inline_surfaces is not None and part_refs is not None:
        raise ValueError(
            f"ops.study.yaml parts.execution_surfaces duplicates inline execution_surfaces: {contract_path}"
        )
    if part_refs is not None:
        surfaces: dict[str, dict[str, object]] = {}
        for part_path in _resolve_part_paths(part_refs, contract_path=contract_path):
            _merge_execution_surfaces(
                surfaces,
                _read_yaml_mapping(part_path),
                label="ops.study.yaml parts.execution_surfaces",
                source=part_path,
            )
        return surfaces
    if inline_surfaces is None:
        return {}
    if not isinstance(inline_surfaces, Mapping):
        raise ValueError(f"ops.study.yaml execution_surfaces must be a mapping: {contract_path}")
    surfaces: dict[str, dict[str, object]] = {}
    _merge_execution_surfaces(
        surfaces,
        inline_surfaces,
        label="ops.study.yaml execution_surfaces",
        source=contract_path,
    )
    return surfaces


def _merge_execution_surfaces(
    target: dict[str, dict[str, object]],
    source_payload: Mapping[str, object],
    *,
    label: str,
    source: Path,
) -> None:
    for raw_surface_id, raw_surface in source_payload.items():
        surface_id = str(raw_surface_id or "").strip()
        if not surface_id:
            raise ValueError(f"{label} contains an empty surface id: {source}")
        if surface_id in target:
            raise ValueError(f"{label} duplicates surface id {surface_id!r}: {source}")
        if not isinstance(raw_surface, Mapping):
            raise ValueError(f"{label}.{surface_id} must be a mapping: {source}")
        target[surface_id] = dict(raw_surface)


def _resolve_part_paths(raw_value: object, *, contract_path: Path) -> tuple[Path, ...]:
    if isinstance(raw_value, list):
        if not raw_value:
            raise ValueError(f"ops.study.yaml parts.execution_surfaces must list at least one path: {contract_path}")
        return tuple(
            _resolve_part_path(
                item,
                contract_path=contract_path,
                label=f"ops.study.yaml parts.execution_surfaces[{index}]",
            )
            for index, item in enumerate(raw_value)
        )
    return (
        _resolve_part_path(
            raw_value,
            contract_path=contract_path,
            label="ops.study.yaml parts.execution_surfaces",
        ),
    )


def _resolve_part_path(raw_value: object, *, contract_path: Path, label: str) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty path: {contract_path}")
    if text.startswith(("repo:", "manifest:")):
        raise ValueError(f"{label} must be relative to the operations directory, not a path ref: {contract_path}")
    relative_path = Path(text)
    if relative_path.is_absolute():
        raise ValueError(f"{label} must be relative to the operations directory: {contract_path}")
    if ".." in relative_path.parts:
        raise ValueError(f"{label} must not escape the operations directory: {contract_path}")
    operations_dir = contract_path.parent.resolve()
    part_path = (operations_dir / relative_path).resolve()
    try:
        part_path.relative_to(operations_dir)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the operations directory: {contract_path}") from exc
    if part_path == contract_path.resolve():
        raise ValueError(f"{label} must not point back at ops.study.yaml: {contract_path}")
    if not part_path.exists():
        raise ValueError(f"{label} file does not exist: {part_path}")
    return part_path


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"yaml root must be a mapping: {path}")
    return payload


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


__all__ = ["discover_infer_runbook_paths_for_study"]
