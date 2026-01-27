"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/versioning.py

Version resolution helpers for dense-arrays metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    while True:
        if (current / "uv.lock").exists() or (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _dense_arrays_version_from_uv_lock(root: Path) -> str | None:
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        return None
    try:
        data = tomllib.loads(lock_path.read_text())
    except Exception:
        return None
    packages = data.get("package", [])
    for pkg in packages:
        name = str(pkg.get("name", "")).lower()
        if name not in {"dense-arrays", "dense_arrays"}:
            continue
        version = pkg.get("version")
        if isinstance(version, str) and version:
            source = pkg.get("source") or {}
            if isinstance(source, dict):
                git_url = source.get("git")
                if isinstance(git_url, str) and "#" in git_url:
                    rev = git_url.split("#")[-1]
                    if rev:
                        return f"{version}+git.{rev[:7]}"
            return version
    return None


def _dense_arrays_version_from_pyproject(root: Path) -> str | None:
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        data = tomllib.loads(pyproject_path.read_text())
    except Exception:
        return None
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        dep_str = str(dep).strip()
        if dep_str.startswith("dense-arrays") or dep_str.startswith("dense_arrays"):
            return dep_str
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if isinstance(sources, dict):
        entry = sources.get("dense-arrays") or sources.get("dense_arrays")
        if isinstance(entry, dict):
            git_url = entry.get("git")
            if isinstance(git_url, str) and git_url:
                return f"git:{git_url}"
    return None


def _resolve_dense_arrays_version(cfg_path: Path) -> tuple[str | None, str]:
    try:
        import dense_arrays as da  # type: ignore

        version = getattr(da, "__version__", None)
        if isinstance(version, str) and version:
            return version, "installed"
    except Exception:
        pass
    for pkg_name in ("dense-arrays", "dense_arrays"):
        try:
            version = importlib.metadata.version(pkg_name)
            return version, "installed"
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            break
    root = _find_project_root(cfg_path)
    if root is None:
        root = _find_project_root(Path(__file__).resolve())
    if root is not None:
        version = _dense_arrays_version_from_uv_lock(root)
        if version:
            return version, "lock"
        version = _dense_arrays_version_from_pyproject(root)
        if version:
            return version, "pyproject"
    return None, "unknown"
