"""
Audit recording for latentdna mutating runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..io.json_io import write_json


def record_audit(audit_dir: Path, *, payload: dict[str, Any], command: str, artifact_id: str | None) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = artifact_id or "workspace"
    audit_path = audit_dir / f"{timestamp}__{command.replace(' ', '_')}__{suffix}.json"
    write_json(audit_path, payload)
    return audit_path
