"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/candidate_table/pipeline.py

Eco1 wrapper around generic thread candidate-table construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.thread.candidates import build_proteinmpnn_candidate_rows, write_candidate_table

_DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
_REQUEST_MANIFEST = "proteinmpnn_request/request_manifest.yaml"
_SAMPLE_TABLE = "sample_table.parquet"
_CANDIDATE_TABLE = "candidate_table.parquet"
_CANDIDATE_TABLES_DIR = "candidate_tables"


@dataclass(frozen=True)
class CandidateTableResult:
    """Paths emitted by one Eco1 candidate-table materialization pass."""

    candidate_table_path: Path
    batch_candidate_table_path: Path


def materialize_candidate_table(*, repo_root: Path, output_root: Path | None = None) -> CandidateTableResult:
    """Materialize candidate_table.parquet from the accepted ProteinMPNN sample table."""

    root = repo_root.expanduser().resolve()
    out_root = _resolve_output_root(root, output_root)
    request_manifest_path = out_root / _REQUEST_MANIFEST
    sample_table_path = out_root / _SAMPLE_TABLE
    for required_path in (request_manifest_path, sample_table_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    manifest = _load_yaml(request_manifest_path)
    batch_id = str(manifest["batch_id"])
    rows = build_proteinmpnn_candidate_rows(
        sample_table_path=sample_table_path,
        request_manifest_path=request_manifest_path,
    )
    batch_candidate_table_path = out_root / _CANDIDATE_TABLES_DIR / f"{batch_id}.parquet"
    write_candidate_table(batch_candidate_table_path, rows, request_hash=str(manifest["request_hash"]))
    candidate_table_path = out_root / _CANDIDATE_TABLE
    shutil.copyfile(batch_candidate_table_path, candidate_table_path)
    return CandidateTableResult(
        candidate_table_path=candidate_table_path,
        batch_candidate_table_path=batch_candidate_table_path,
    )


def _resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    resolved = output_root or repo_root / _DEFAULT_OUTPUT_ROOT
    return resolved if resolved.is_absolute() else repo_root / resolved


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
