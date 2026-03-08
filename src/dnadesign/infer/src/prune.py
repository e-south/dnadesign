"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/prune.py

Infer namespace prune operations for USR-backed datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.usr.src.errors import SequencesError
from dnadesign.usr.src.overlay_maintenance import remove_dataset_overlay

from .errors import ConfigError


def prune_usr_overlay(*, dataset: str, usr_root: Path, mode: str = "archive") -> dict[str, object]:
    selected_mode = str(mode or "").strip().lower()
    if selected_mode not in {"archive", "delete"}:
        raise ConfigError("prune mode must be one of: archive, delete")

    resolved_root = usr_root.expanduser().resolve()
    try:
        result = remove_dataset_overlay(resolved_root, dataset, "infer", mode=selected_mode)
    except SequencesError as error:
        raise ConfigError(str(error)) from error
    if not bool(result.get("removed")):
        raise ConfigError("Overlay 'infer' not found.")
    return result
