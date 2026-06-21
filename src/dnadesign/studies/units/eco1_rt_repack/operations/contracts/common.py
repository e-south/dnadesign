"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/common.py

Shared fail-fast helpers for Eco1 RT repack contract validators.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _ALLOWED_PHASES,
    _PENDING_VALUES,
    _PHASE_RANK,
    _PLANNED_THREAD_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def _iter_forbidden_field_paths(payload: Any, forbidden_fields: set[str], *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in forbidden_fields:
                paths.append(path)
            paths.extend(_iter_forbidden_field_paths(value, forbidden_fields, prefix=path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_iter_forbidden_field_paths(value, forbidden_fields, prefix=path))
    return paths


def _is_pending_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in _PENDING_VALUES or normalized.startswith("pending_")
    if isinstance(value, list):
        return not value or any(_is_pending_value(item) for item in value)
    return False


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and float(value) > 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_sha256_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())


def _append_mismatch_issue(
    issues: list[ContractIssue],
    *,
    check_id: str,
    message: str,
    profile_value: Any,
    authority_value: Any,
    path: str,
) -> None:
    if profile_value != authority_value:
        issues.append(
            ContractIssue(
                check_id=check_id,
                message=message,
                path=path,
            )
        )


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _nested_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    resolved = output_root or repo_root / _PLANNED_THREAD_ROOT
    resolved = resolved.expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")


def _require_known_phase(phase: str) -> None:
    if phase not in _PHASE_RANK:
        raise ValueError(f"Unknown Eco1 RT repack validation phase {phase!r}; expected one of {_ALLOWED_PHASES}")


def _phase_rank(phase: str) -> int:
    _require_known_phase(phase)
    return _PHASE_RANK[phase]


def _merge_reports(phase: str, reports: tuple[ContractReport, ...]) -> ContractReport:
    return ContractReport(phase=phase, issues=tuple(issue for report in reports for issue in report.issues))
