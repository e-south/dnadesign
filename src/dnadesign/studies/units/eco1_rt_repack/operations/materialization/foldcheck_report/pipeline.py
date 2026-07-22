"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/pipeline.py

Materialize Eco1 fold-check reports from ColabFold output directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report.constants import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUNTIME_PARAMETERS,
    REFERENCE_BACKBONE_RELATIVE_PATH,
    REPORT_FILE_NAME,
    REQUEST_MANIFEST_RELATIVE_PATH,
    RESIDUE_MAP_FILE_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report.reference import (
    mapped_reference_positions,
)
from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.thread.adapters.colabfold import build_colabfold_foldcheck_rows
from dnadesign.thread.foldcheck import write_foldcheck_report


@dataclass(frozen=True)
class MaterializedFoldCheckReportArtifacts:
    """Paths emitted by one Eco1 fold-check report materialization pass."""

    foldcheck_report_path: Path


def materialize_foldcheck_report(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    colabfold_output_root: Path,
    runtime_version: str,
    runtime_parameters: Mapping[str, Any] | None = None,
) -> MaterializedFoldCheckReportArtifacts:
    """Normalize ColabFold outputs into Eco1's foldcheck_report.parquet artifact."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    source_root = resolve_output_root(root, source_output_root or DEFAULT_THREAD_OUTPUT_ROOT)
    request_manifest_path = out_root / REQUEST_MANIFEST_RELATIVE_PATH
    if not request_manifest_path.exists():
        raise FileNotFoundError(request_manifest_path)
    reference_pdb_path = _first_existing_path(
        out_root / REFERENCE_BACKBONE_RELATIVE_PATH,
        source_root / REFERENCE_BACKBONE_RELATIVE_PATH,
        description="fold-check reference backbone",
    )
    residue_map_path = _first_existing_path(
        out_root / RESIDUE_MAP_FILE_NAME,
        source_root / RESIDUE_MAP_FILE_NAME,
        description="fold-check residue map",
    )
    reference_mobile_positions = mapped_reference_positions(residue_map_path)
    output_dir = colabfold_output_root.expanduser()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    request_manifest = _load_yaml(request_manifest_path)
    parameters = dict(DEFAULT_RUNTIME_PARAMETERS)
    if runtime_parameters:
        parameters.update(dict(runtime_parameters))
    rows = build_colabfold_foldcheck_rows(
        output_root=output_dir,
        request_manifest=request_manifest,
        runtime_version=runtime_version,
        runtime_parameters=parameters,
        reference_pdb_path=reference_pdb_path,
        reference_mobile_positions=reference_mobile_positions,
    )
    report_path = out_root / REPORT_FILE_NAME
    write_foldcheck_report(report_path, rows, request_hash=str(request_manifest["request_hash"]))
    return MaterializedFoldCheckReportArtifacts(foldcheck_report_path=report_path)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _first_existing_path(*paths: Path, description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    formatted = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"{description} not found at any expected path: {formatted}")
