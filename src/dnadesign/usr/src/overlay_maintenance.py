"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlay_maintenance.py

Dataset-targeted overlay maintenance helpers for CLI and sibling tools.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .dataset import Dataset
from .errors import SequencesError


def remove_dataset_overlay(root: Path, dataset: str, namespace: str, *, mode: str = "error") -> dict[str, object]:
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise SequencesError(f"USR root not found: {resolved_root}")

    selected_mode = str(mode or "error").strip().lower()
    ds = Dataset(resolved_root, dataset)
    result = ds.remove_overlay(str(namespace), mode=selected_mode)
    return {
        "dataset": ds.name,
        "root": str(resolved_root),
        "namespace": str(namespace),
        "mode": selected_mode,
        **result,
    }
