"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/artifact.py

Materialize and verify study-owned promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .bundle_io import confined_path, file_sha256, publish_complete_bundle
from .contracts import (
    BINDINGS_FILENAME,
    SCHEMA_ID,
    SCHEMA_VERSION,
    STUDY_ID,
    BindingSourceArtifact,
    PromoterCandidateBindingsError,
    PromoterCandidateBindingsPreview,
    PromoterCandidateBindingsVerification,
    PromoterCandidateBindingsWriteResult,
)
from .manifest_contract import build_manifest, validate_manifest, validate_source_artifacts
from .parquet_io import read_bindings, write_bindings
from .resolution import resolve_promoter_candidate_bindings
from .row_contract import validate_binding_rows


def preview_promoter_candidate_bindings(
    *,
    alias_rows: pd.DataFrame,
    candidate_records: pd.DataFrame,
    genbank_annotations: pd.DataFrame,
    candidate_table_id: str,
    candidate_selection_sha256: str,
    source_artifacts: tuple[BindingSourceArtifact, ...],
) -> PromoterCandidateBindingsPreview:
    """Resolve and validate the binding set without writing an artifact."""

    validate_source_artifacts(source_artifacts)
    bindings = resolve_promoter_candidate_bindings(
        alias_rows=alias_rows,
        candidate_records=candidate_records,
        genbank_annotations=genbank_annotations,
        candidate_table_id=candidate_table_id,
        candidate_selection_sha256=candidate_selection_sha256,
    )
    validate_binding_rows(bindings)
    return PromoterCandidateBindingsPreview(
        bindings=bindings,
        candidate_table_id=str(candidate_table_id),
        candidate_selection_sha256=str(candidate_selection_sha256),
        source_artifacts=source_artifacts,
    )


def materialize_promoter_candidate_bindings(
    preview: PromoterCandidateBindingsPreview,
    *,
    out_dir: Path,
    allowed_output_root: Path,
    overwrite: bool = False,
) -> PromoterCandidateBindingsWriteResult:
    """Publish a fully verified bundle within an explicit output root."""

    root = Path(allowed_output_root).expanduser().resolve()
    output_dir = confined_path(Path(out_dir), root=root, label="output directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise PromoterCandidateBindingsError(f"Promoter binding output is not a directory: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise PromoterCandidateBindingsError(
            f"Promoter binding target already exists; pass overwrite=True to replace it: {output_dir}"
        )
    if not isinstance(preview.bindings, pd.DataFrame):
        raise PromoterCandidateBindingsError("Promoter binding preview must carry a pandas DataFrame.")
    validate_binding_rows(preview.bindings)
    validate_source_artifacts(preview.source_artifacts)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent) as temporary:
        staged_dir = Path(temporary)
        bindings_path = staged_dir / BINDINGS_FILENAME
        manifest_path = staged_dir / "manifest.json"
        write_bindings(preview.bindings, bindings_path)
        manifest = build_manifest(preview, bindings_sha256=file_sha256(bindings_path))
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_promoter_candidate_bindings(staged_dir, allowed_root=root)
        publish_complete_bundle(staged_dir=staged_dir, output_dir=output_dir)

    return PromoterCandidateBindingsWriteResult(
        manifest_json=output_dir / "manifest.json",
        bindings_parquet=output_dir / BINDINGS_FILENAME,
        binding_count=len(preview.bindings),
        candidate_count=preview.bindings["candidate_id"].nunique(),
    )


def verify_promoter_candidate_bindings(
    bundle_dir: Path,
    *,
    allowed_root: Path | None = None,
) -> PromoterCandidateBindingsVerification:
    """Verify confinement, digests, manifest semantics, rows, and BaseRender compatibility."""

    bundle = Path(bundle_dir).expanduser().resolve()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        bundle = confined_path(bundle, root=root, label="bundle directory")
    manifest_path = bundle / "manifest.json"
    payload = _read_manifest(manifest_path)
    record = payload.get("record")
    if not isinstance(record, dict):
        raise PromoterCandidateBindingsError("Promoter binding manifest record must be a mapping.")
    relative_path = record.get("path")
    if relative_path != BINDINGS_FILENAME:
        raise PromoterCandidateBindingsError(f"Promoter binding record path must be {BINDINGS_FILENAME!r}.")
    bindings_path = confined_path(bundle / BINDINGS_FILENAME, root=bundle, label="binding record")
    if not bindings_path.is_file():
        raise PromoterCandidateBindingsError(f"Promoter binding record not found: {bindings_path}")
    expected_digest = str(record.get("sha256", "")).lower()
    actual_digest = file_sha256(bindings_path)
    if actual_digest != expected_digest:
        raise PromoterCandidateBindingsError(
            f"Promoter binding record digest mismatch: expected={expected_digest} actual={actual_digest}"
        )
    bindings = read_bindings(bindings_path)
    validate_binding_rows(bindings)
    validate_manifest(payload, bindings=bindings)
    return PromoterCandidateBindingsVerification(
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        study_id=STUDY_ID,
        binding_count=len(bindings),
        candidate_count=bindings["candidate_id"].nunique(),
        manifest_json=manifest_path,
        bindings_parquet=bindings_path,
    )


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PromoterCandidateBindingsError(f"Promoter binding manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromoterCandidateBindingsError(f"Could not parse promoter binding manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromoterCandidateBindingsError("Promoter binding manifest must contain a JSON object.")
    return payload


__all__ = [
    "materialize_promoter_candidate_bindings",
    "preview_promoter_candidate_bindings",
    "verify_promoter_candidate_bindings",
]
