"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/path_ref.py

Shared path-reference contract for OPS-owned manifests and study configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PathBase = Literal["repo", "manifest", "cwd"]
_PATH_BASES = frozenset({"repo", "manifest", "cwd"})
_PLACEHOLDER_PATH_SENTINELS = frozenset({"n/a", "none", "null", "tbd", "todo"})


def resolve_path_ref(
    raw_value: object,
    *,
    repo_root: Path | None = None,
    manifest_dir: Path | None = None,
    cwd: Path | None = None,
    default_base: PathBase | None = None,
    label: str = "path",
) -> Path:
    path_text = _coerce_path_text(raw_value, label=label)
    _reject_placeholder_path_text(path_text, label=label)
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()

    if path_text.startswith("repo:"):
        if repo_root is None:
            raise ValueError(f"{label} uses repo: but repo_root is not available")
        return _resolve_within_repo(repo_root=repo_root, relative_path=path_text.removeprefix("repo:"), label=label)

    if path_text.startswith("manifest:"):
        if manifest_dir is None:
            raise ValueError(f"{label} uses manifest: but manifest_dir is not available")
        return _resolve_from_base(base_dir=manifest_dir, relative_path=path_text.removeprefix("manifest:"), label=label)

    if manifest_dir is not None and (path_text.startswith("./") or path_text.startswith("../")):
        return _resolve_from_base(base_dir=manifest_dir, relative_path=path_text, label=label)

    base = _normalize_path_base(default_base)
    if base == "repo":
        if repo_root is None:
            raise ValueError(f"{label} requires repo_root for repo-relative resolution")
        return _resolve_within_repo(repo_root=repo_root, relative_path=path_text, label=label)
    if base == "manifest":
        if manifest_dir is None:
            raise ValueError(f"{label} requires manifest_dir for manifest-relative resolution")
        return _resolve_from_base(base_dir=manifest_dir, relative_path=path_text, label=label)

    resolved_cwd = Path.cwd().resolve() if cwd is None else cwd.expanduser().resolve()
    return _resolve_from_base(base_dir=resolved_cwd, relative_path=path_text, label=label)


def _resolve_within_repo(*, repo_root: Path, relative_path: str, label: str) -> Path:
    resolved_repo_root = repo_root.expanduser().resolve()
    resolved_path = _resolve_from_base(base_dir=resolved_repo_root, relative_path=relative_path, label=label)
    try:
        resolved_path.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root: {relative_path}") from exc
    return resolved_path


def _resolve_from_base(*, base_dir: Path, relative_path: str, label: str) -> Path:
    text = str(relative_path or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty path reference")
    return (base_dir.expanduser().resolve() / Path(text).expanduser()).resolve()


def _normalize_path_base(value: PathBase | None) -> PathBase:
    if value is None:
        return "cwd"
    if value not in _PATH_BASES:
        raise ValueError(f"unsupported path base: {value!r}")
    return value


def _coerce_path_text(raw_value: object, *, label: str) -> str:
    if isinstance(raw_value, Path):
        text = str(raw_value)
    else:
        text = str(raw_value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty path reference")
    return text


def _reject_placeholder_path_text(path_text: str, *, label: str) -> None:
    normalized = str(path_text or "").strip()
    if "<" in normalized or ">" in normalized:
        raise ValueError(
            f"{label} contains placeholder path text {normalized!r}; "
            "replace scaffold placeholders with a real path before running this command"
        )

    candidate = normalized
    for prefix in ("repo:", "manifest:"):
        if candidate.startswith(prefix):
            candidate = candidate.removeprefix(prefix).strip()
            break
    if candidate.lower() in _PLACEHOLDER_PATH_SENTINELS:
        raise ValueError(
            f"{label} contains placeholder path text {normalized!r}; "
            "replace narrative placeholders with a real path before running this command"
        )


__all__ = [
    "PathBase",
    "resolve_path_ref",
]
