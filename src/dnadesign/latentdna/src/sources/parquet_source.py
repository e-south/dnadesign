"""
Parquet source helpers.
"""

from __future__ import annotations

from pathlib import Path


def records_path(path: str, *, workspace_dir: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_dir / candidate
    return candidate.resolve()


def source_provenance(path: str, *, workspace_dir: Path) -> list[dict[str, object]]:
    candidate = records_path(path, workspace_dir=workspace_dir)
    return [
        {
            "kind": "file",
            "id": candidate.name,
            "path": candidate.as_posix(),
            "role": "records",
        }
    ]
