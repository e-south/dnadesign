"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/run_paths.py

Run-scoped path helpers for canonical workspace layout.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

RUN_OUTPUTS_DIR = "outputs"
RUN_META_DIR = "meta"
CANDIDATES_DIR = "candidates"
CANDIDATES_CURRENT_DIR = "current"

RUN_MANIFEST_NAME = "run_manifest.json"
INPUTS_MANIFEST_NAME = "inputs_manifest.json"
RUN_STATE_NAME = "run_state.json"


def run_outputs_root(run_root: Path) -> Path:
    return run_root / RUN_OUTPUTS_DIR


def candidates_root(outputs_root: Path) -> Path:
    return outputs_root / CANDIDATES_DIR / CANDIDATES_CURRENT_DIR


def run_meta_root(run_root: Path) -> Path:
    return run_outputs_root(run_root) / RUN_META_DIR


def ensure_run_meta_dir(run_root: Path) -> Path:
    meta = run_meta_root(run_root)
    meta.mkdir(parents=True, exist_ok=True)
    return meta


def run_manifest_path(run_root: Path) -> Path:
    return run_meta_root(run_root) / RUN_MANIFEST_NAME


def inputs_manifest_path(run_root: Path) -> Path:
    return run_meta_root(run_root) / INPUTS_MANIFEST_NAME


def run_state_path(run_root: Path) -> Path:
    return run_meta_root(run_root) / RUN_STATE_NAME
