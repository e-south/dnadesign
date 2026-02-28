"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_export_paths.py

Path normalization helpers for DenseGen notebook export controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _relative_base(*, run_root: Path, repo_root: Path | None) -> Path:
    if repo_root is None:
        return run_root
    candidate = Path(repo_root).expanduser()
    if not candidate.is_absolute():
        return run_root
    return candidate


def resolve_records_export_destination(
    *,
    raw_path: str,
    selected_format: str,
    run_root: Path,
    repo_root: Path | None = None,
) -> Path:
    selected = str(selected_format or "").strip().lower()
    if selected not in {"parquet", "csv"}:
        raise ValueError(f"records export format must be parquet or csv, got `{selected}`.")

    path_text = str(raw_path or "").strip()
    used_default = False
    if not path_text:
        path_text = "outputs/notebooks/records_preview"
        used_default = True

    destination = Path(path_text).expanduser()
    if not destination.is_absolute():
        if used_default:
            destination = run_root / destination
        else:
            destination = _relative_base(run_root=run_root, repo_root=repo_root) / destination
    if destination.suffix.lower() != f".{selected}":
        destination = destination.with_suffix(f".{selected}")
    return destination


def resolve_baserender_export_destination(
    *,
    raw_path: str,
    selected_format: str,
    run_root: Path,
    repo_root: Path | None = None,
) -> Path:
    selected = str(selected_format or "").strip().lower()
    if selected not in {"png", "pdf"}:
        raise ValueError(f"BaseRender export format must be png or pdf, got `{selected}`.")

    path_text = str(raw_path or "").strip()
    if not path_text:
        raise ValueError("BaseRender export path cannot be empty.")

    destination = Path(path_text).expanduser()
    if not destination.is_absolute():
        destination = _relative_base(run_root=run_root, repo_root=repo_root) / destination
    if destination.suffix.lower() != f".{selected}":
        destination = destination.with_suffix(f".{selected}")
    return destination
