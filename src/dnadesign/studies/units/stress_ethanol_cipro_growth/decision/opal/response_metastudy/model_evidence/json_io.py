"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/json_io.py

Atomic mutable-index writes and create-only immutable JSON publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from .contracts import ModelEvidenceError


def publish_immutable_json(path: Path, payload: dict[str, object]) -> None:
    final_dir = path.parent
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = final_dir.parent / f".{final_dir.name}.staging-{uuid4().hex}"
    stage.mkdir()
    try:
        write_json(stage / path.name, payload)
        try:
            stage.rename(final_dir)
        except FileExistsError as exc:
            raise ModelEvidenceError(f"immutable record already exists: {final_dir}") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelEvidenceError(f"{label} is missing: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelEvidenceError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelEvidenceError(f"{label} must be a JSON mapping: {path}")
    return payload


__all__ = ["atomic_write_json", "publish_immutable_json", "read_mapping"]
