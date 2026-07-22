"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_released_show.py

Path-oriented integrity checks for released-product snapback bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.snapback_released_show_load import load_released_show_artifacts
from dnadesign.cruncher.app.snapback_released_show_present import build_released_show_payload
from dnadesign.cruncher.app.snapback_released_show_validate import validate_released_show_artifacts


def released_show_payload(run_dir: str | Path) -> dict[str, object]:
    artifacts = load_released_show_artifacts(run_dir)
    validate_released_show_artifacts(artifacts)
    return build_released_show_payload(artifacts)


__all__ = ["released_show_payload"]
