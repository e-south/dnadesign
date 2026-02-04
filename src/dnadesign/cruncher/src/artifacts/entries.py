"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/artifacts/entries.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_entry(
    path: Path,
    run_dir: Path,
    *,
    kind: str,
    label: str | None = None,
    stage: str | None = None,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a structured artifact entry relative to the run directory."""
    rel = path
    if root_dir is not None:
        try:
            rel = path.relative_to(root_dir)
        except ValueError:
            rel = path
    if rel == path:
        try:
            rel = path.relative_to(run_dir)
        except ValueError:
            rel = path
    size_bytes = None
    if path.exists():
        size_bytes = path.stat().st_size
    return {
        "path": str(rel),
        "type": kind,
        "label": label or path.name,
        "stage": stage,
        "format": path.suffix.lstrip("."),
        "created_at": _utc_now(),
        "size_bytes": size_bytes,
    }


def normalize_artifacts(raw: Iterable[object] | None) -> List[dict[str, Any]]:
    """Coerce artifact payloads into structured entries."""
    if raw is None:
        return []
    artifacts: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            artifacts.append(item)
            continue
        if isinstance(item, str):
            artifacts.append(
                {
                    "path": item,
                    "type": "file",
                    "label": Path(item).name,
                    "stage": None,
                    "format": Path(item).suffix.lstrip("."),
                }
            )
    return artifacts


def append_artifacts(manifest: dict[str, Any], new_items: Iterable[dict[str, Any]]) -> None:
    """Append structured artifacts to a manifest, preserving older string entries."""
    existing = normalize_artifacts(manifest.get("artifacts"))
    seen = {item.get("path") for item in existing}
    for item in new_items:
        path = item.get("path")
        if path and path in seen:
            continue
        existing.append(item)
        if path:
            seen.add(path)
    manifest["artifacts"] = existing
