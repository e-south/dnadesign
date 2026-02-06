"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/layout.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

ANALYSIS_LAYOUT_VERSION = "v8"
ANALYSIS_DIR_NAME = "analysis"
ARCHIVE_DIR_NAME = "_archive"
MANIFEST_FILE_NAME = "manifest.json"


def analysis_root(run_dir: Path) -> Path:
    return run_dir / ANALYSIS_DIR_NAME


def analysis_meta_root(analysis_root: Path) -> Path:
    return analysis_root


def summary_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / "summary.json"


def report_json_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / "report.json"


def report_md_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / "report.md"


def analysis_used_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / "analysis_used.yaml"


def plot_manifest_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / "plot_manifest.json"


def table_manifest_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / "table_manifest.json"


def analysis_manifest_path(analysis_root: Path) -> Path:
    return analysis_meta_root(analysis_root) / MANIFEST_FILE_NAME


def load_summary(path: Path, *, required: bool = False) -> Optional[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing analysis summary: {path}")
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Invalid analysis summary JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {path}")
    return payload


def load_table_manifest(path: Path, *, required: bool = False) -> Optional[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing analysis table manifest: {path}")
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Invalid analysis table manifest JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"analysis table manifest must be a JSON object: {path}")
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError(f"analysis table manifest must include a 'tables' list: {path}")
    return payload


def table_paths_by_key(analysis_root: Path) -> dict[str, Path]:
    manifest = load_table_manifest(table_manifest_path(analysis_root), required=True)
    if manifest is None:
        return {}
    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise ValueError(f"analysis table manifest must include a 'tables' list: {table_manifest_path(analysis_root)}")
    resolved: dict[str, Path] = {}
    for idx, entry in enumerate(tables):
        if not isinstance(entry, dict):
            raise ValueError(
                f"analysis table manifest contains a non-object table entry at index {idx}: "
                f"{table_manifest_path(analysis_root)}"
            )
        key = entry.get("key")
        rel_path = entry.get("path")
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"analysis table manifest table entry at index {idx} has invalid key: "
                f"{table_manifest_path(analysis_root)}"
            )
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError(
                f"analysis table manifest table entry '{key}' has invalid path: {table_manifest_path(analysis_root)}"
            )
        if key in resolved:
            raise ValueError(
                f"analysis table manifest has duplicate table key '{key}': {table_manifest_path(analysis_root)}"
            )
        resolved[key] = analysis_root / rel_path
    return resolved


def resolve_required_table_paths(analysis_root: Path, *, keys: Sequence[str]) -> dict[str, Path]:
    required_keys = [str(key) for key in keys]
    paths_by_key = table_paths_by_key(analysis_root)
    missing_keys = [key for key in required_keys if key not in paths_by_key]
    if missing_keys:
        raise FileNotFoundError(
            "analysis table manifest missing required table keys: "
            + ", ".join(sorted(missing_keys))
            + f" ({table_manifest_path(analysis_root)})"
        )
    resolved = {key: paths_by_key[key] for key in required_keys}
    missing_files = [f"{key}={path}" for key, path in resolved.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError("analysis table manifest references missing table files: " + ", ".join(missing_files))
    return resolved


def current_summary(run_dir: Path) -> Optional[dict]:
    root = analysis_root(run_dir)
    return load_summary(summary_path(root), required=False)


def current_analysis_id(run_dir: Path) -> Optional[str]:
    summary = current_summary(run_dir)
    if not summary:
        return None
    analysis_id = summary.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ValueError("analysis summary missing analysis_id")
    return analysis_id


def list_analysis_entries(run_dir: Path) -> list[dict]:
    root = analysis_root(run_dir)
    entries: list[dict] = []
    if not root.exists():
        return entries

    try:
        summary = load_summary(summary_path(root), required=False)
    except ValueError as exc:
        logger.warning("Skipping invalid analysis summary: %s", exc)
        summary = None
    if summary:
        analysis_id = summary.get("analysis_id")
        if isinstance(analysis_id, str) and analysis_id:
            entries.append({"id": analysis_id, "path": root, "kind": "latest"})
        else:
            logger.warning("analysis summary missing analysis_id: %s", summary_path(root))

    archive_root = root / ARCHIVE_DIR_NAME
    if archive_root.exists():
        for child in sorted(archive_root.iterdir()):
            if not child.is_dir():
                continue
            try:
                summary = load_summary(summary_path(child), required=False)
            except ValueError as exc:
                logger.warning("Skipping invalid archive summary: %s", exc)
                summary = None
            analysis_id = summary.get("analysis_id") if isinstance(summary, dict) else None
            if not isinstance(analysis_id, str) or not analysis_id:
                logger.warning("Skipping archive summary missing analysis_id: %s", child)
                continue
            entries.append({"id": analysis_id, "path": child, "kind": "archive"})

    return entries


def list_analysis_entries_verbose(run_dir: Path) -> list[dict]:
    """List analysis entries with labels and warning messages for missing/invalid summaries."""
    root = analysis_root(run_dir)
    entries: list[dict] = []
    if not root.exists():
        return entries
    if not root.is_dir():
        raise NotADirectoryError(f"analysis root is not a directory: {root}")

    def _safe_summary(path: Path) -> tuple[Optional[dict], Optional[str]]:
        try:
            return load_summary(path, required=True), None
        except Exception as exc:
            return None, str(exc)

    def _append_entry(
        *,
        analysis_id: str,
        path: Path,
        kind: str,
        label: str,
        warnings: list[str],
    ) -> None:
        entries.append(
            {
                "id": analysis_id,
                "path": str(path),
                "kind": kind,
                "label": label,
                "warnings": warnings,
            }
        )

    summary_file = summary_path(root)
    summary, summary_error = _safe_summary(summary_file)
    warnings: list[str] = []
    analysis_id: Optional[str] = None
    if summary:
        analysis_id = summary.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id:
            warnings.append(f"analysis summary missing analysis_id: {summary_file}")
            analysis_id = None
    elif summary_error:
        warnings.append(summary_error)

    if analysis_id:
        _append_entry(
            analysis_id=analysis_id,
            path=root,
            kind="latest",
            label=f"{analysis_id} (latest)",
            warnings=warnings,
        )
    else:
        try:
            has_contents = any(root.iterdir())
        except Exception as exc:
            warnings.append(f"Failed to read analysis directory contents: {exc}")
            has_contents = False
        if not has_contents:
            warnings.append("analysis directory exists but summary.json is missing or empty")
        _append_entry(
            analysis_id="unindexed",
            path=root,
            kind="unindexed",
            label="analysis (unindexed)",
            warnings=warnings,
        )

    archive_root = root / ARCHIVE_DIR_NAME
    if archive_root.exists():
        for child in sorted(archive_root.iterdir()):
            if not child.is_dir():
                continue
            summary_file = summary_path(child)
            summary, summary_error = _safe_summary(summary_file)
            warnings = []
            if summary_error:
                warnings.append(summary_error)
            analysis_id = summary.get("analysis_id") if isinstance(summary, dict) else None
            if not isinstance(analysis_id, str) or not analysis_id:
                warnings.append(f"archive summary missing analysis_id: {child}")
                logger.warning("Skipping archive summary missing analysis_id: %s", child)
                continue
            _append_entry(
                analysis_id=analysis_id,
                path=child,
                kind="archive",
                label=f"{analysis_id} (archive)",
                warnings=warnings,
            )

    return entries


def resolve_analysis_dir(
    run_dir: Path,
    *,
    analysis_id: Optional[str],
    latest: bool,
) -> tuple[Path, Optional[str]]:
    root = analysis_root(run_dir)
    if latest or analysis_id is None:
        summary = load_summary(summary_path(root), required=True)
        resolved_id = summary.get("analysis_id") if isinstance(summary, dict) else None
        return root, resolved_id

    entries = list_analysis_entries(run_dir)
    for entry in entries:
        if entry.get("id") == analysis_id:
            return Path(entry["path"]), analysis_id
    raise FileNotFoundError(f"Analysis id not found: {analysis_id}")
