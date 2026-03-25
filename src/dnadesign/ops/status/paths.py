"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/paths.py

Path-resolution helpers for OPS status manifests, inputs, and study records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .parsing import required_text
from .path_ref import resolve_path_ref


def required_path(
    path: Path | None,
    *,
    flag_name: str,
    progress_kind: str,
    base_dir: Path | None = None,
) -> Path:
    if path is None:
        raise ValueError(f"progress kind '{progress_kind}' requires {flag_name}")
    return resolve_input_path(path, base_dir=base_dir)


def resolve_input_path(path: Path, *, base_dir: Path | None = None) -> Path:
    return resolve_path_ref(
        path,
        manifest_dir=base_dir,
        default_base="manifest" if base_dir is not None else "cwd",
        label="<path>",
    )


def resolve_repo_relative_path(
    *,
    repo_root: Path,
    raw_path: str | None,
    progress_kind: str = "promoter-study-record",
) -> Path:
    normalized = required_text(raw_path, flag_name="<repo-relative-path>", progress_kind=progress_kind)
    return resolve_path_ref(
        normalized,
        repo_root=repo_root,
        default_base="repo",
        label="<repo-relative-path>",
    )


def resolve_named_path_mapping(
    payload: object,
    *,
    repo_root: Path,
    label: str,
    progress_kind: str,
) -> dict[str, Path]:
    if payload and not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    resolved: dict[str, Path] = {}
    for name, raw_path in flatten_named_paths(payload or {}):
        resolved[name] = resolve_repo_relative_path(
            repo_root=repo_root,
            raw_path=raw_path,
            progress_kind=progress_kind,
        )
    return resolved


def flatten_named_paths(payload: object, *, prefix: str = "") -> tuple[tuple[str, str], ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        return (((prefix or "path"), payload),)
    if not isinstance(payload, dict):
        raise ValueError("execution_surfaces entries must be strings or nested mappings")
    flattened: list[tuple[str, str]] = []
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("execution_surfaces keys must be non-empty strings")
        next_prefix = f"{prefix}.{key}" if prefix else key
        flattened.extend(flatten_named_paths(value, prefix=next_prefix))
    return tuple(flattened)


def path_or_none(
    value: object,
    *,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
    default_base: str | None = None,
) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_path_ref(
        text,
        repo_root=repo_root,
        manifest_dir=base_dir,
        default_base=default_base or ("manifest" if base_dir is not None else "cwd"),
        label="<path>",
    )


__all__ = [
    "flatten_named_paths",
    "path_or_none",
    "required_path",
    "resolve_input_path",
    "resolve_named_path_mapping",
    "resolve_repo_relative_path",
]
