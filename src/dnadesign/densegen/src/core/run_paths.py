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

import os
from pathlib import Path

RUN_OUTPUTS_DIR = "outputs"
RUN_META_DIR = "meta"
RUN_LOGS_DIR = "logs"
RUN_POOLS_DIR = "pools"
RUN_LIBRARIES_DIR = "libraries"
RUN_TABLES_DIR = "tables"
RUN_PLOTS_DIR = "plots"
RUN_NOTIFY_DIR = "notify"
CANDIDATES_DIR = "candidates"

RUN_MANIFEST_NAME = "run_manifest.json"
INPUTS_MANIFEST_NAME = "inputs_manifest.json"
RUN_STATE_NAME = "run_state.json"
ID_INDEX_NAME = "_densegen_ids.sqlite"

TABLE_FILES = {
    "records.parquet",
    "attempts.parquet",
    "solutions.parquet",
    "composition.parquet",
    "run_metrics.parquet",
}
IGNORED_OUTPUT_ENTRIES = {".DS_Store", ".gitkeep"}
NON_BLOCKING_OUTPUT_DIRS = {
    RUN_LOGS_DIR,
    RUN_POOLS_DIR,
    RUN_LIBRARIES_DIR,
    RUN_PLOTS_DIR,
    RUN_NOTIFY_DIR,
}
META_BLOCKING_FILES = {
    RUN_MANIFEST_NAME,
    INPUTS_MANIFEST_NAME,
    RUN_STATE_NAME,
    "effective_config.json",
    ID_INDEX_NAME,
}
META_NON_BLOCKING_FILES = {"events.jsonl", "run.lock"}
USR_ROOT_NON_BLOCKING_FILES = {"registry.yaml"}


def run_outputs_root(run_root: Path) -> Path:
    return run_root / RUN_OUTPUTS_DIR


def candidates_root(outputs_root: Path, run_id: str) -> Path:
    run_label = str(run_id).strip()
    if not run_label:
        raise ValueError("run_id must be a non-empty string for candidate artifacts.")
    return outputs_root / RUN_POOLS_DIR / CANDIDATES_DIR


def run_meta_root(run_root: Path) -> Path:
    return run_outputs_root(run_root) / RUN_META_DIR


def run_tables_root(run_root: Path) -> Path:
    return run_outputs_root(run_root) / RUN_TABLES_DIR


def run_plots_root(run_root: Path) -> Path:
    return run_outputs_root(run_root) / RUN_PLOTS_DIR


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


def id_index_path(run_root: Path) -> Path:
    return run_meta_root(run_root) / ID_INDEX_NAME


def dense_arrays_path(run_root: Path) -> Path:
    return run_tables_root(run_root) / "records.parquet"


def attempts_path(run_root: Path) -> Path:
    return run_tables_root(run_root) / "attempts.parquet"


def solutions_path(run_root: Path) -> Path:
    return run_tables_root(run_root) / "solutions.parquet"


def composition_path(run_root: Path) -> Path:
    return run_tables_root(run_root) / "composition.parquet"


def display_path(path: Path, run_root: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        try:
            return os.path.relpath(path, run_root)
        except ValueError:
            return str(path)


def has_existing_run_outputs(run_root: Path) -> bool:
    outputs_root = run_outputs_root(run_root)
    if not outputs_root.exists():
        return False
    for entry in outputs_root.iterdir():
        if entry.name in IGNORED_OUTPUT_ENTRIES:
            continue
        if entry.is_dir():
            if entry.name in NON_BLOCKING_OUTPUT_DIRS:
                continue
            if entry.name == RUN_META_DIR:
                if _meta_has_run_artifacts(entry):
                    return True
                continue
            if entry.name == RUN_TABLES_DIR:
                if _tables_has_run_artifacts(entry):
                    return True
                continue
            if _looks_like_usr_root(entry):
                if _usr_root_has_run_artifacts(entry):
                    return True
                continue
            return True
        return True
    return False


def _meta_has_run_artifacts(meta_dir: Path) -> bool:
    if not meta_dir.exists() or not meta_dir.is_dir():
        return False
    for entry in meta_dir.iterdir():
        if entry.name in IGNORED_OUTPUT_ENTRIES:
            continue
        if entry.name in META_NON_BLOCKING_FILES:
            continue
        if entry.name in META_BLOCKING_FILES:
            return True
        return True
    return False


def _tables_has_run_artifacts(tables_dir: Path) -> bool:
    if not tables_dir.exists() or not tables_dir.is_dir():
        return False
    for entry in tables_dir.iterdir():
        if entry.name in IGNORED_OUTPUT_ENTRIES:
            continue
        if entry.name in TABLE_FILES:
            return True
        if entry.is_dir():
            return True
        return True
    return False


def _looks_like_usr_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "registry.yaml").exists()


def _usr_root_has_run_artifacts(usr_root: Path) -> bool:
    for entry in usr_root.iterdir():
        if entry.name in IGNORED_OUTPUT_ENTRIES:
            continue
        if entry.name in USR_ROOT_NON_BLOCKING_FILES:
            continue
        if entry.is_dir():
            # Any dataset/archive directory under the USR root indicates persisted output state.
            return True
        return True
    return False
