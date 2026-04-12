"""
Manifest IO helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from .json_io import write_json


def write_manifest(path: Path, payload: dict) -> None:
    write_json(path, payload)
