"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/manifest.py

Publish a display-only OPAL projection from one canonical Reader record.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import (
    PROMOTER_EVIDENCE_NON_CLAIM,
    PROMOTER_RESPONSE_SEMANTIC_KIND,
    READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
    READER_EVIDENCE_MANIFEST_ADAPTER,
    READER_EVIDENCE_SCHEMA_VERSION,
    READER_PROMOTER_EVIDENCE_FILENAME,
    READER_PROMOTER_EVIDENCE_MEDIA_DIR,
    TARGET_CAMPAIGN_SLUG,
    ReaderPromoterEvidenceError,
    ReaderPromoterEvidenceWriteResult,
    VerifiedReaderPromoterEvidenceSource,
    canonical_json_sha256,
)
from .display_verification import verify_reader_promoter_evidence_manifest
from .verification import verify_reader_promoter_evidence_source


def preview_reader_promoter_evidence_manifest(
    *,
    reader_root: Path,
    experiment_root: Path,
    projection_path: Path,
    bindings_bundle: Path,
    round_label: str = "r0",
    reader_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify one canonical diagnostic and build its OPAL projection in memory."""

    _validate_round(round_label)
    source = verify_reader_promoter_evidence_source(
        reader_root=reader_root,
        experiment_root=experiment_root,
        projection_path=projection_path,
        bindings_bundle=bindings_bundle,
        reader_command=reader_command,
    )
    return _preview(source, round_label=round_label)


def materialize_reader_promoter_evidence_manifest(
    *,
    reader_root: Path,
    experiment_root: Path,
    projection_path: Path,
    bindings_bundle: Path,
    out_dir: Path,
    round_label: str = "r0",
    filename: str = READER_PROMOTER_EVIDENCE_FILENAME,
    overwrite: bool = False,
    reader_command: Sequence[str] | None = None,
) -> ReaderPromoterEvidenceWriteResult:
    """Atomically stage verified diagnostic bytes, then publish the display manifest."""

    _validate_round(round_label)
    output_name = _safe_filename(filename)
    source = verify_reader_promoter_evidence_source(
        reader_root=reader_root,
        experiment_root=experiment_root,
        projection_path=projection_path,
        bindings_bundle=bindings_bundle,
        reader_command=reader_command,
    )
    payload = _preview(source, round_label=round_label)
    output_dir = Path(out_dir).expanduser().resolve()
    target = output_dir / output_name
    if target.exists() and not overwrite:
        raise ReaderPromoterEvidenceError(f"Reader promoter-evidence manifest already exists: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir / f".{output_name}.staging-{uuid.uuid4().hex}"
    staging_root.mkdir()
    staged = staging_root / output_name
    installed_media: Path | None = None
    try:
        media_relative = _media_relative_path(source)
        staged_media = staging_root / media_relative
        staged_media.parent.mkdir(parents=True)
        shutil.copyfile(source.display.selected_file.path, staged_media)
        with staged.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        staged_verification = verify_reader_promoter_evidence_manifest(staged)
        installed_media = _publish_media(
            staging_root=staging_root,
            output_dir=output_dir,
            media_relative=media_relative,
        )
        if overwrite:
            try:
                os.replace(staged, target)
            except OSError as exc:
                raise ReaderPromoterEvidenceError(f"Could not publish Reader display manifest: {exc}") from exc
        else:
            try:
                os.link(staged, target)
            except FileExistsError as exc:
                raise ReaderPromoterEvidenceError(
                    f"Reader promoter-evidence manifest already exists: {target}"
                ) from exc
            staged.unlink()
    except BaseException:
        if installed_media is not None and installed_media.exists():
            installed_media.unlink()
            _remove_empty_parents(installed_media.parent, stop=output_dir)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return ReaderPromoterEvidenceWriteResult(
        manifest_json=target,
        row_count=staged_verification.row_count,
        artifact_count=staged_verification.artifact_count,
    )


def _preview(source: VerifiedReaderPromoterEvidenceSource, *, round_label: str) -> dict[str, Any]:
    row = _display_row(source)
    return {
        "schema_version": READER_EVIDENCE_SCHEMA_VERSION,
        "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
        "created_at": datetime.now(UTC).isoformat(),
        "campaign_slug": TARGET_CAMPAIGN_SLUG,
        "round": round_label,
        "summary": {
            "rows": 1,
            "distinct_ids": 1,
            "reader_experiments": 1,
            "artifact_count": 1,
            "missing_artifact_rows": 0,
        },
        "rows": [row],
    }


def _display_row(source: VerifiedReaderPromoterEvidenceSource) -> dict[str, Any]:
    response_source = _response_source(source)
    source_receipt_sha256 = canonical_json_sha256(response_source)
    selected_file = source.display.selected_file
    media_type = "image/png" if selected_file.path.suffix.lower() == ".png" else "application/pdf"
    return {
        "id": source.candidate_id,
        "candidate_id": source.candidate_id,
        "design_id": source.design_id,
        "reader_experiment_id": source.source_experiment_id,
        "reduction_id": source.reduction_id,
        "evidence_role": "display_only",
        "claim_status": "objective_neutral",
        "non_claim_boundary": PROMOTER_EVIDENCE_NON_CLAIM,
        "selected_binding": dict(source.selected_binding),
        "sources": {
            "response_window": response_source,
            "candidate_bindings": dict(source.binding_source),
        },
        "artifacts": [
            {
                "semantic_kind": PROMOTER_RESPONSE_SEMANTIC_KIND,
                "kind": "reader_record_projection",
                "record_id": READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
                "scope": "design_reduction",
                "path": _media_relative_path(source).as_posix(),
                "path_label": (
                    f"{source.source_experiment_id}/{source.design_id}/{source.reduction_id}/{selected_file.path.name}"
                ),
                "exists": True,
                "media_type": media_type,
                "bytes": selected_file.size_bytes,
                "sha256": selected_file.content_digest,
                "source_record_revision_digest": source.display.record.revision_digest,
                "source_file_path": selected_file.reader_path,
                "source_receipt_sha256": source_receipt_sha256,
            }
        ],
    }


def _response_source(source: VerifiedReaderPromoterEvidenceSource) -> dict[str, object]:
    records = source.records
    diagnostic = source.display.record.to_dict()
    return {
        "schema_version": "stress_ethanol_cipro_growth.reader_response_record_source.v1",
        "output_experiment_id": records.experiment_id,
        "source_experiment_id": source.source_experiment_id,
        "design_id": source.design_id,
        "reduction_id": source.reduction_id,
        "protocol_id": records.protocol_id,
        "projection_sha256": "sha256:" + records.projection_sha256,
        "catalog": {
            "schema_version": records.source.catalog_schema_version,
            "provenance_epoch_id": records.provenance_epoch_id,
            "sha256": "sha256:" + records.catalog_sha256,
        },
        "records": {
            "designs": records.record_refs["designs"].to_dict(),
            "traces": records.record_refs["traces"].to_dict(),
            "diagnostic": diagnostic,
        },
    }


def _media_relative_path(source: VerifiedReaderPromoterEvidenceSource) -> Path:
    revision = source.display.record.revision_digest.removeprefix("sha256:")
    return Path(READER_PROMOTER_EVIDENCE_MEDIA_DIR) / revision / source.display.selected_file.path.name


def _publish_media(*, staging_root: Path, output_dir: Path, media_relative: Path) -> Path | None:
    source = staging_root / media_relative
    destination = output_dir / media_relative
    _ensure_confined_directory(output_dir, media_relative.parent)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            raise ReaderPromoterEvidenceError(
                f"Content-addressed Reader diagnostic differs from its existing target: {destination}"
            )
        return None
    os.replace(source, destination)
    return destination


def _ensure_confined_directory(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ReaderPromoterEvidenceError(f"Reader diagnostic media directory must not be a symlink: {current}")
        if current.exists():
            if not current.is_dir():
                raise ReaderPromoterEvidenceError(f"Reader diagnostic media parent is not a directory: {current}")
        else:
            current.mkdir()
        try:
            current.resolve().relative_to(root)
        except ValueError as exc:  # pragma: no cover - symlinks are rejected before this guard
            raise ReaderPromoterEvidenceError("Reader diagnostic media directory escapes its output root.") from exc


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    current = path
    while current != stop and current.is_dir() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent


def _validate_round(round_label: str) -> None:
    if not isinstance(round_label, str) or not round_label.startswith("r") or not round_label[1:].isdigit():
        raise ReaderPromoterEvidenceError("round_label must use the OPAL form 'r<integer>'.")


def _safe_filename(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or not value.endswith(".json"):
        raise ReaderPromoterEvidenceError("Reader display filename must be a non-empty .json basename.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or len(posix.parts) != 1 or len(windows.parts) != 1:
        raise ReaderPromoterEvidenceError("Reader display filename must not contain a directory path.")
    return value


__all__ = [
    "materialize_reader_promoter_evidence_manifest",
    "preview_reader_promoter_evidence_manifest",
    "verify_reader_promoter_evidence_manifest",
]
