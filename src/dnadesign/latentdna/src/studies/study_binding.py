"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/studies/study_binding.py

Study binding contract constants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

REQUIRED_STUDY_RECORD_FILES: tuple[str, ...] = ("campaign.yaml", "datasets.yaml", "ops.study.yaml", "status.md")
REQUIRED_STUDY_DELIVERABLE_DOC_FILES: tuple[str, ...] = ("study.yaml",)


def missing_required_files(root: Path, required_files: Iterable[str]) -> list[str]:
    return [name for name in required_files if not (root / name).is_file()]
