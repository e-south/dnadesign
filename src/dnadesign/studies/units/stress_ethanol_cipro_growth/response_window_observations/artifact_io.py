"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_io.py

Confined paths, digests, and atomic directory replacement for observation bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def confined_path(path: Path, *, root: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside allowed root {root}: {resolved}") from exc
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Read one JSON object while rejecting duplicate keys at every depth."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}.")
            result[key] = value
        return result

    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def publish_new_directory(*, staged_dir: Path, output_dir: Path) -> None:
    """Atomically publish a new directory without replacing scientific evidence."""

    if output_dir.exists():
        raise FileExistsError(f"immutable observation bundle already exists: {output_dir}")
    os.rename(staged_dir, output_dir)


__all__ = ["confined_path", "file_sha256", "publish_new_directory", "read_json_object"]
