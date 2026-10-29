"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/runs/recorder.py

Audit recording for latentdna mutating runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contracts.ids import validate_identifier


def record_audit(audit_dir: Path, *, payload: dict[str, Any], command: str, artifact_id: str | None) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = validate_identifier(artifact_id or "workspace", label="audit artifact id")
    command = validate_identifier(command.replace(" ", "_"), label="audit command")
    encoded = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    audit_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{timestamp}__{command}__{suffix}__", suffix=".json", dir=audit_dir)
    audit_path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        audit_path.unlink(missing_ok=True)
        raise
    return audit_path
