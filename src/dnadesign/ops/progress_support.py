"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/progress_support.py

Shared path, parsing, and artifact helpers for read-only ops progress surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from .status.path_ref import resolve_path_ref


def namespace_column_counts(columns: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for column in columns:
        if "__" not in column:
            continue
        namespace = column.split("__", 1)[0].strip()
        if namespace:
            counts[namespace] += 1
    return counts


def overlay_namespace_names(dataset_dir: Path) -> list[str]:
    derived_dir = dataset_dir / "_derived"
    if not derived_dir.exists() or not derived_dir.is_dir():
        return []
    namespaces: list[str] = []
    for entry in sorted(derived_dir.iterdir(), key=lambda item: item.name):
        if entry.is_file() and entry.suffix == ".parquet":
            namespaces.append(entry.stem)
            continue
        if entry.is_dir() and any(entry.glob("part-*.parquet")):
            namespaces.append(entry.name)
    return namespaces


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def file_count(path: Path) -> int:
    return sum(1 for candidate in path.rglob("*") if candidate.is_file())


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


def required_text(value: str | None, *, flag_name: str, progress_kind: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"progress kind '{progress_kind}' requires {flag_name}")
    return str(value).strip()


def load_yaml_mapping(path: Path, *, label: str) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping: {path}")
    return payload


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


def parquet_row_count(records_path: Path) -> int:
    return int(pq.ParquetFile(str(records_path)).metadata.num_rows)


def optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"expected integer value, received: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"expected non-negative integer value, received: {value!r}")
    return parsed


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


def required_metadata_text(value: object, *, label: str, source: Path) -> str:
    text = string_or_none(value)
    if text is None:
        raise ValueError(f"{label} is required in {source}")
    return text


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


def string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def string_list_or_empty(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = string_or_none(item)
        if text is not None:
            result.append(text)
    return result


__all__ = [
    "file_count",
    "flatten_named_paths",
    "line_count",
    "load_yaml_mapping",
    "namespace_column_counts",
    "optional_positive_int",
    "overlay_namespace_names",
    "parquet_row_count",
    "path_or_none",
    "required_metadata_text",
    "required_path",
    "required_text",
    "resolve_input_path",
    "resolve_named_path_mapping",
    "resolve_repo_relative_path",
    "string_list_or_empty",
    "string_or_none",
]
