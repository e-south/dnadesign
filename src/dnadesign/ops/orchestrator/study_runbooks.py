"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/study_runbooks.py

OPS-owned discovery of Infer runbook refs declared by an external study workspace.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dnadesign.ops.status.path_ref import resolve_path_ref
from dnadesign.ops.study.record_loader import load_study_ops_contract


def discover_infer_runbook_paths_for_study(*, study_dir: Path, repo_root: Path) -> tuple[Path, ...]:
    resolved_study_dir = study_dir.expanduser().resolve()
    contract_path = resolved_study_dir / "operations" / "ops.study.yaml"
    contract = load_study_ops_contract(resolved_study_dir)
    paths: list[Path] = []
    for surface_id, surface in contract.execution_surfaces.items():
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
                default_base="manifest",
                label=f"ops.study.yaml execution_surfaces.{surface_id}.runbook_ref",
            )
        )
    return _dedupe_paths(paths)


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
