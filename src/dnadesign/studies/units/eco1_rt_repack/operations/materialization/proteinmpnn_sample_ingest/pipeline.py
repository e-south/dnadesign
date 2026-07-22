"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/pipeline.py

Eco1 wrapper around generic ProteinMPNN backend sample ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest.constants import (
    BACKEND_OUTPUT_DIR,
    DEFAULT_OUTPUT_ROOT,
    REQUEST_MANIFEST,
    SAMPLE_TABLE,
    SAMPLE_TABLES_DIR,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest.models import (
    ProteinMpnnSampleIngestResult,
)
from dnadesign.thread.adapters.proteinmpnn import (
    ProteinMpnnExecutionConfig,
    parse_proteinmpnn_outputs,
    run_official_proteinmpnn_request,
    write_sample_table,
)

ProteinMpnnRunner = Callable[..., dict[str, Any]]


def materialize_proteinmpnn_samples(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    proteinmpnn_root: Path | None = None,
    overwrite: bool = False,
    runner: ProteinMpnnRunner = run_official_proteinmpnn_request,
) -> ProteinMpnnSampleIngestResult:
    """Materialize Eco1 ProteinMPNN backend outputs into sample_table.parquet."""

    out_root = _resolve_output_root(repo_root, output_root)
    request_manifest_path = out_root / REQUEST_MANIFEST
    if not request_manifest_path.exists():
        raise FileNotFoundError(f"Missing ProteinMPNN request manifest: {request_manifest_path}")
    manifest = _load_yaml(request_manifest_path)
    backend_output_dir = out_root / BACKEND_OUTPUT_DIR
    execution_config = ProteinMpnnExecutionConfig(
        batch_id=str(manifest["batch_id"]),
        num_seq_per_target=int(manifest["num_seq_per_target"]),
        batch_size=int(manifest["batch_size"]),
        overwrite=overwrite,
    )
    run_result = runner(
        request_manifest_path=request_manifest_path,
        proteinmpnn_root=proteinmpnn_root,
        output_dir=backend_output_dir,
        execution_config=execution_config,
    )
    backend_run_id = str(run_result["backend_run_id"])
    request_hash = str(run_result["request_hash"])
    rows = parse_proteinmpnn_outputs(
        run_outputs=run_result["run_outputs"],
        backend_run_id=backend_run_id,
        request_hash=request_hash,
        target_name=str(manifest["proteinmpnn_name"]),
        sequence_length=int(manifest["canonical_position_count"]),
    )
    batch_sample_table_path = out_root / SAMPLE_TABLES_DIR / f"{execution_config.batch_id}.parquet"
    write_sample_table(batch_sample_table_path, rows, request_hash=request_hash)
    sample_table_path = out_root / SAMPLE_TABLE
    shutil.copyfile(batch_sample_table_path, sample_table_path)
    active_backend_manifest = backend_output_dir / "backend_run_manifest.yaml"
    shutil.copyfile(Path(run_result["backend_run_manifest_path"]), active_backend_manifest)
    return ProteinMpnnSampleIngestResult(
        sample_table_path=sample_table_path,
        backend_run_manifest_path=active_backend_manifest,
    )


def _resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    root = repo_root.expanduser().resolve()
    resolved = output_root or root / DEFAULT_OUTPUT_ROOT
    return resolved if resolved.is_absolute() else root / resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
