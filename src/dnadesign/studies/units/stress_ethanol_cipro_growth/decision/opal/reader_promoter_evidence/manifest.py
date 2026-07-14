"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/manifest.py

Build display-only OPAL manifests from verified Reader evidence bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_RESPONSE_SEMANTIC_KIND,
    READER_BUNDLE_SCHEMA_VERSION,
    READER_EVIDENCE_SCHEMA_VERSION,
    READER_PROMOTER_EVIDENCE_FILENAME,
    READER_PROMOTER_EVIDENCE_MEDIA_DIR,
    TARGET_CAMPAIGN_SLUG,
    ReaderPromoterEvidenceError,
    ReaderPromoterEvidenceVerification,
    ReaderPromoterEvidenceWriteResult,
    VerifiedReaderPromoterEvidenceBundle,
)
from .display_verification import verify_reader_promoter_evidence_manifest
from .verification import verify_reader_promoter_evidence_bundle


def preview_reader_promoter_evidence_manifest(
    bundle_dirs: Sequence[Path],
    *,
    bindings_bundle: Path,
    round_label: str = "r0",
) -> dict[str, Any]:
    """Validate Reader bundles and build one display-only manifest in memory."""

    _validate_inputs(bundle_dirs, round_label=round_label)
    bundles = [
        verify_reader_promoter_evidence_bundle(Path(path), bindings_bundle=bindings_bundle) for path in bundle_dirs
    ]
    return _preview_from_bundles(bundles, round_label=round_label)


def materialize_reader_promoter_evidence_manifest(
    bundle_dirs: Sequence[Path],
    *,
    bindings_bundle: Path,
    out_dir: Path,
    round_label: str = "r0",
    filename: str = READER_PROMOTER_EVIDENCE_FILENAME,
    overwrite: bool = False,
) -> ReaderPromoterEvidenceWriteResult:
    """Atomically publish one verified display-only manifest."""

    output_dir = Path(out_dir).expanduser().resolve()
    _validate_inputs(bundle_dirs, round_label=round_label)
    output_name = _safe_filename(filename)
    target = output_dir / output_name
    if target.exists() and not overwrite:
        raise ReaderPromoterEvidenceError(f"Reader promoter-evidence manifest already exists: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir / f".{output_name}.staging-{uuid.uuid4().hex}"
    staging_root.mkdir()
    staged = staging_root / output_name
    installed_media: list[Path] = []
    try:
        bundles = [
            verify_reader_promoter_evidence_bundle(Path(path), bindings_bundle=bindings_bundle) for path in bundle_dirs
        ]
        payload = _preview_from_bundles(bundles, round_label=round_label)
        _stage_media(bundles, staging_root=staging_root)
        with staged.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        verification = verify_reader_promoter_evidence_manifest(staged)
        installed_media = _publish_media(staging_root=staging_root, output_dir=output_dir)
        if overwrite:
            os.replace(staged, target)
        else:
            try:
                os.link(staged, target)
            except FileExistsError as exc:
                raise ReaderPromoterEvidenceError(
                    f"Reader promoter-evidence manifest already exists: {target}"
                ) from exc
            staged.unlink()
        published = verify_reader_promoter_evidence_manifest(target)
    except BaseException:
        for media_dir in reversed(installed_media):
            if media_dir.exists():
                shutil.rmtree(media_dir)
        _remove_empty_media_root(output_dir)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    if published != ReaderPromoterEvidenceVerification(
        manifest_json=target,
        row_count=verification.row_count,
        artifact_count=verification.artifact_count,
    ):
        raise ReaderPromoterEvidenceError("Published Reader promoter-evidence verification changed after staging.")
    return ReaderPromoterEvidenceWriteResult(
        manifest_json=target,
        row_count=published.row_count,
        artifact_count=published.artifact_count,
    )


def _display_row(bundle: VerifiedReaderPromoterEvidenceBundle) -> dict[str, Any]:
    selection = bundle.manifest["selection"]
    artifacts = bundle.manifest["artifacts"]
    candidate_id = str(selection["candidate_id"])
    experiment_id = str(selection["experiment_id"])
    design_id = str(selection["design_id"])
    reduction_id = str(selection["reduction_id"])
    return {
        "id": candidate_id,
        "candidate_id": candidate_id,
        "design_id": design_id,
        "reader_experiment_id": experiment_id,
        "reduction_id": reduction_id,
        "evidence_role": "display_only",
        "claim_status": str(bundle.manifest["claim_status"]),
        "selected_binding": dict(bundle.manifest["selected_binding"]),
        "binding_source": dict(bundle.manifest["sources"]["candidate_bindings"]),
        "artifacts": [
            {
                "semantic_kind": PROMOTER_RESPONSE_SEMANTIC_KIND,
                "kind": "reader_publication",
                "record_id": READER_BUNDLE_SCHEMA_VERSION,
                "scope": "design_reduction",
                "path": _media_relative_path(bundle, artifact_id=artifact_id),
                "path_label": f"{experiment_id}/{design_id}/{reduction_id}/{artifact_id}",
                "exists": True,
                "media_type": "image/png" if artifact_id.endswith(".png") else "application/pdf",
                "bytes": int(artifacts[artifact_id]["bytes"]),
                "sha256": str(artifacts[artifact_id]["sha256"]),
                "source_manifest_sha256": bundle.manifest_sha256,
            }
            for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS
        ],
    }


def _preview_from_bundles(
    bundles: Sequence[VerifiedReaderPromoterEvidenceBundle],
    *,
    round_label: str,
) -> dict[str, Any]:
    rows = [_display_row(bundle) for bundle in bundles]
    rows.sort(
        key=lambda row: (
            str(row["reader_experiment_id"]),
            str(row["design_id"]),
            str(row["reduction_id"]),
            str(row["candidate_id"]),
        )
    )
    identities = [
        (
            str(row["candidate_id"]),
            str(row["design_id"]),
            str(row["reader_experiment_id"]),
            str(row["reduction_id"]),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ReaderPromoterEvidenceError("Reader promoter-evidence inputs contain a duplicate selection identity.")
    return {
        "schema_version": READER_EVIDENCE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "campaign_slug": TARGET_CAMPAIGN_SLUG,
        "round": round_label,
        "summary": {
            "rows": len(rows),
            "distinct_ids": len({str(row["candidate_id"]) for row in rows}),
            "reader_experiments": len({str(row["reader_experiment_id"]) for row in rows}),
            "artifact_count": sum(len(row["artifacts"]) for row in rows),
            "missing_artifact_rows": 0,
        },
        "rows": rows,
    }


def _media_relative_path(bundle: VerifiedReaderPromoterEvidenceBundle, *, artifact_id: str) -> str:
    source_digest = bundle.manifest_sha256.removeprefix("sha256:")
    return f"{READER_PROMOTER_EVIDENCE_MEDIA_DIR}/{source_digest}/{artifact_id}"


def _stage_media(
    bundles: Sequence[VerifiedReaderPromoterEvidenceBundle],
    *,
    staging_root: Path,
) -> None:
    for bundle in bundles:
        for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS:
            destination = staging_root / _media_relative_path(bundle, artifact_id=artifact_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle.root / artifact_id, destination)


def _publish_media(*, staging_root: Path, output_dir: Path) -> list[Path]:
    staging_media = staging_root / READER_PROMOTER_EVIDENCE_MEDIA_DIR
    target_media = output_dir / READER_PROMOTER_EVIDENCE_MEDIA_DIR
    target_media.mkdir(exist_ok=True)
    installed: list[Path] = []
    try:
        for source_dir in sorted(staging_media.iterdir()):
            destination = target_media / source_dir.name
            if destination.exists():
                _verify_existing_media(source_dir=source_dir, destination=destination)
                continue
            os.replace(source_dir, destination)
            installed.append(destination)
    except BaseException:
        for destination in reversed(installed):
            if destination.exists():
                shutil.rmtree(destination)
        _remove_empty_media_root(output_dir)
        raise
    return installed


def _verify_existing_media(*, source_dir: Path, destination: Path) -> None:
    if not destination.is_dir():
        raise ReaderPromoterEvidenceError(f"Reader evidence media target is not a directory: {destination}")
    for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS:
        source = source_dir / artifact_id
        target = destination / artifact_id
        if (
            not target.is_file()
            or source.stat().st_size != target.stat().st_size
            or _file_sha256(source) != _file_sha256(target)
        ):
            raise ReaderPromoterEvidenceError(
                f"Content-addressed Reader evidence media does not match its existing target: {destination}"
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _remove_empty_media_root(output_dir: Path) -> None:
    media_root = output_dir / READER_PROMOTER_EVIDENCE_MEDIA_DIR
    if media_root.is_dir() and not any(media_root.iterdir()):
        media_root.rmdir()


def _validate_inputs(bundle_dirs: Sequence[Path], *, round_label: str) -> None:
    if not bundle_dirs:
        raise ReaderPromoterEvidenceError("At least one Reader promoter-evidence bundle is required.")
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
