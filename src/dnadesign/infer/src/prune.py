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

from dnadesign.usr import Dataset

from .errors import ConfigError


def prune_usr_overlay(*, dataset: str, usr_root: Path, mode: str = "archive") -> dict[str, object]:
    selected_mode = str(mode or "").strip().lower()
    if selected_mode not in {"archive", "delete"}:
        raise ConfigError("prune mode must be one of: archive, delete")

    resolved_root = usr_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ConfigError(f"USR root not found: {resolved_root}")

    ds = Dataset(resolved_root, dataset)
    if not any(overlay.namespace == "infer" for overlay in ds.list_overlays()):
        raise ConfigError("Overlay 'infer' not found.")
    result = ds.remove_overlay("infer", mode=selected_mode)
    return {
        "dataset": ds.name,
        "root": str(resolved_root),
        "namespace": "infer",
        "mode": selected_mode,
        **result,
    }
