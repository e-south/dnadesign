"""Shared validation helpers for RT-lnRNA Construct materialization."""

from __future__ import annotations

from pathlib import Path

from .contracts import MaterializationContractError


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MaterializationContractError(f"{label} must be a mapping.")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise MaterializationContractError(f"{label} must be a list.")
    return value


def _span_0(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise MaterializationContractError(f"{label} must be [start, end].")
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end <= start:
        raise MaterializationContractError(f"{label} must be a valid zero-based half-open span.")
    return start, end


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")
