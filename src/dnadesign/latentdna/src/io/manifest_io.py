"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/io/manifest_io.py

Manifest IO helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .json_io import write_json


def write_manifest(path: Path, payload: dict) -> None:
    write_json(path, payload)
