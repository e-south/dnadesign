"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/config_inputs.py

Config-driven ingest input resolution for infer run command workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..config import JobConfig
from ..errors import ConfigError, ValidationError
from ..input_parsing import load_nonempty_lines


def _resolve_ingest_path(*, ingest_path: str, config_dir: Path) -> Path:
    candidate = Path(ingest_path).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ConfigError(f"ingest.path not found: {resolved}")
    return resolved


def resolve_config_job_inputs(
    *,
    job: JobConfig,
    config_dir: Path,
    i_know_this_is_pickle: bool,
    guard_pickle: Callable[[bool], None],
) -> Any:
    source = job.ingest.source
    ingest_path = str(job.ingest.path or "").strip()

    if source == "usr":
        return None

    if source == "sequences":
        if not ingest_path:
            raise ConfigError("ingest.source='sequences' requires ingest.path for infer run config workflows")
        return load_nonempty_lines(_resolve_ingest_path(ingest_path=ingest_path, config_dir=config_dir))

    if source == "records":
        if not ingest_path:
            raise ConfigError("ingest.source='records' requires ingest.path for infer run config workflows")
        path = _resolve_ingest_path(ingest_path=ingest_path, config_dir=config_dir)
        records: list[dict[str, Any]] = []
        for line in load_nonempty_lines(path):
            try:
                payload = json.loads(line)
            except Exception as error:
                raise ValidationError(f"Invalid JSON in ingest.path records file {path}: {error}") from error
            if not isinstance(payload, dict):
                raise ValidationError(f"records ingest file must contain JSON objects: {path}")
            records.append(payload)
        return records

    if source == "pt_file":
        guard_pickle(i_know_this_is_pickle)
        if ingest_path:
            return _resolve_ingest_path(ingest_path=ingest_path, config_dir=config_dir).as_posix()
        return (config_dir / f"{job.id}.pt").resolve().as_posix()

    raise ConfigError(f"Unsupported ingest.source for infer run config workflow: {source}")
