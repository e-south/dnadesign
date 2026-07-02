"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/downstream_inputs.py

Downstream-input staging for Eco1 RT design-class expansion artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    CANDIDATE_POOL_FILE_NAME,
    DEFAULT_DESIGN_CLASSES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    MaterializedDesignClassDownstreamInputs,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

MANIFEST_FILE_NAME = "downstream_inputs_manifest.yaml"
STAGED_CANDIDATE_TABLE_FILE_NAME = "candidate_table.parquet"
SHARED_FILES = (
    Path("residue_map.parquet"),
    Path("conservation_profile.parquet"),
)
SHARED_DIRECTORIES = (
    Path("proteinmpnn_request"),
    Path("conservation_alignments"),
    Path("conservation_sources"),
    Path("biohub_esmc/mutation_scoring"),
)


def materialize_design_class_downstream_inputs(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
) -> MaterializedDesignClassDownstreamInputs:
    """Stage root-local inputs needed by expanded fold-review and ESMC lanes."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_DESIGN_CLASSES_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    candidate_pool_path = class_root / CANDIDATE_POOL_FILE_NAME
    if not candidate_pool_path.exists():
        raise FileNotFoundError(candidate_pool_path)

    copied_rows: list[dict[str, Any]] = []
    candidate_table_path = class_root / STAGED_CANDIDATE_TABLE_FILE_NAME
    copied_rows.append(
        _copy_file(
            source_path=candidate_pool_path,
            destination_path=candidate_table_path,
            role="expanded_candidate_table",
            source_root=class_root,
            destination_root=class_root,
        )
    )
    for relative_path in SHARED_FILES:
        copied_rows.append(
            _copy_file(
                source_path=source_root / relative_path,
                destination_path=class_root / relative_path,
                role="shared_thread_input",
                source_root=source_root,
                destination_root=class_root,
            )
        )
    for relative_path in SHARED_DIRECTORIES:
        copied_rows.extend(
            _copy_directory_files(
                source_dir=source_root / relative_path,
                destination_dir=class_root / relative_path,
                role="shared_thread_directory",
                source_root=source_root,
                destination_root=class_root,
            )
        )

    manifest_path = class_root / MANIFEST_FILE_NAME
    payload = {
        "schema_id": "eco1_rt.design_class_downstream_inputs",
        "schema_version": 1,
        "status": "materialized",
        "output_root": str(class_root),
        "source_output_root": str(source_root),
        "candidate_table_path": str(candidate_table_path),
        "candidate_table_hash": sha256_uri(candidate_table_path),
        "copied_file_count": len(copied_rows),
        "copied_files": copied_rows,
        "mask_policy_note": (
            "No root-level mask_set.yaml is staged here. The expanded candidate pool carries multiple "
            "mask_policy_id values, so mask-specific review must read the per-class mask_set.yaml files."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return MaterializedDesignClassDownstreamInputs(
        candidate_table_path=candidate_table_path,
        manifest_path=manifest_path,
        copied_file_count=len(copied_rows),
    )


def _copy_directory_files(
    *,
    source_dir: Path,
    destination_dir: Path,
    role: str,
    source_root: Path,
    destination_root: Path,
) -> list[dict[str, Any]]:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Expected directory: {source_dir}")
    rows: list[dict[str, Any]] = []
    for source_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source_dir)
        rows.append(
            _copy_file(
                source_path=source_path,
                destination_path=destination_dir / relative_path,
                role=role,
                source_root=source_root,
                destination_root=destination_root,
            )
        )
    if not rows:
        raise ValueError(f"Shared downstream directory contains no files: {source_dir}")
    return rows


def _copy_file(
    *,
    source_path: Path,
    destination_path: Path,
    role: str,
    source_root: Path,
    destination_root: Path,
) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not source_path.is_file():
        raise ValueError(f"Expected file: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return {
        "role": role,
        "source_path": _relative_or_absolute(source_path, source_root),
        "destination_path": _relative_or_absolute(destination_path, destination_root),
        "source_hash": sha256_uri(source_path),
        "destination_hash": sha256_uri(destination_path),
    }


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path, start=root)
    except ValueError:
        return str(path)
