"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/core/paths.py

Study-owned DenseGen axis OPAL probe package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .constants import RUN_ROOT_PREFIX


def _repo_root_from(path: Path) -> Path:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not resolve repo root from {path}")


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _default_run_root(repo_root: Path, run_id: str) -> Path:
    return repo_root / RUN_ROOT_PREFIX / run_id


def validate_run_root_policy(*, repo_root: Path, run_root: Path, allow_custom: bool = False) -> None:
    repo_resolved = repo_root.resolve()
    run_resolved = run_root.resolve()
    expected_root = (repo_resolved / RUN_ROOT_PREFIX).resolve()
    try:
        run_resolved.relative_to(expected_root)
        return
    except ValueError:
        pass
    if allow_custom:
        try:
            run_resolved.relative_to(repo_resolved)
        except ValueError:
            return
        raise ValueError(
            f"custom run root inside the repository is not allowed. Use {expected_root} or an external scratch path."
        )
    raise ValueError(
        "run root must be under "
        f"{expected_root}. Use the default run root or pass --allow-custom-run-root for an external scratch path."
    )
