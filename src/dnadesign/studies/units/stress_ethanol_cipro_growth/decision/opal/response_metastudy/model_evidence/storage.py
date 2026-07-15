"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/storage.py

Public record, verify, and catalog operations for model-evidence trajectories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from ..runtime.publication import sha256_file, verify_bundle_artifacts
from .contracts import (
    CATALOG_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    LATEST_SCHEMA_VERSION,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA_VERSION,
    ModelEvidenceError,
    content_digest,
    validated_evidence_id,
)
from .json_io import atomic_write_json, publish_immutable_json
from .projection import project_verified_manifest
from .verification import (
    catalog,
    checkpoint_index_record,
    empty_scan,
    scan_immutable,
    verify_latest,
    verify_protocol,
)


def record_checkpoint(
    *,
    metastudy_bundle: Path,
    trajectory_root: Path,
    evidence_id: str,
) -> dict[str, object]:
    """Record one verified metastudy snapshot under an immutable protocol series."""

    bundle_root = Path(metastudy_bundle).resolve()
    manifest = verify_bundle_artifacts(bundle_root)
    projection = project_verified_manifest(
        manifest,
        metastudy_manifest_sha256=sha256_file(bundle_root / "manifest.json"),
    )
    root = Path(trajectory_root).resolve()
    validated_id = validated_evidence_id(evidence_id)
    existing = scan_immutable(root) if root.exists() else empty_scan()
    _ensure_protocol(root, protocol=projection.protocol, protocol_digest=projection.protocol_digest)

    checkpoint_base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evidence_id": validated_id,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": projection.protocol_digest,
        "snapshot": projection.snapshot,
    }
    checkpoint_digest = content_digest(checkpoint_base)
    same_id = [
        row
        for row in existing["checkpoints"]
        if row["protocol_digest"] == projection.protocol_digest and row["evidence_id"] == validated_id
    ]
    if same_id:
        record = same_id[0]
        if record["checkpoint_digest"] != checkpoint_digest:
            raise ModelEvidenceError(
                f"evidence_id {validated_id!r} already exists with different content in protocol "
                f"{projection.protocol_digest}."
            )
        _write_indexes(root, checkpoint=record)
        return _public_record(root, record)

    relative = Path("series") / projection.protocol_digest / "checkpoints" / f"{validated_id}__{checkpoint_digest}"
    checkpoint_path = root / relative / "checkpoint.json"
    payload = {**checkpoint_base, "checkpoint_digest": checkpoint_digest}
    publish_immutable_json(checkpoint_path, payload)
    record = checkpoint_index_record(root, checkpoint_path, payload)
    _write_indexes(root, checkpoint=record)
    return _public_record(root, record)


def verify_trajectory(trajectory_root: Path) -> dict[str, object]:
    """Verify immutable protocol/checkpoint records and report index freshness."""

    root = Path(trajectory_root).resolve()
    scan = scan_immutable(root)
    expected_catalog = catalog(scan["checkpoints"], protocol_count=len(scan["protocols"]))
    catalog_matches = False
    try:
        catalog_matches = json.loads((root / "catalog.json").read_text(encoding="utf-8")) == expected_catalog
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    verify_latest(root, checkpoints=scan["checkpoints"])
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "protocol_count": len(scan["protocols"]),
        "checkpoint_count": len(scan["checkpoints"]),
        "catalog_matches": catalog_matches,
    }


def rebuild_catalog(trajectory_root: Path) -> dict[str, object]:
    """Rebuild the replaceable catalog solely from verified immutable records."""

    root = Path(trajectory_root).resolve()
    scan = scan_immutable(root)
    result = catalog(scan["checkpoints"], protocol_count=len(scan["protocols"]))
    atomic_write_json(root / "catalog.json", result)
    return result


def _ensure_protocol(root: Path, *, protocol: dict[str, object], protocol_digest: str) -> None:
    path = root / "protocols" / protocol_digest / "protocol.json"
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_digest": protocol_digest,
        "protocol": protocol,
    }
    if path.exists():
        verify_protocol(path, expected_digest=protocol_digest)
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ModelEvidenceError(f"frozen protocol {protocol_digest} differs from requested content.")
        return
    publish_immutable_json(path, payload)


def _write_indexes(root: Path, *, checkpoint: dict[str, object]) -> None:
    scan = scan_immutable(root)
    atomic_write_json(
        root / "catalog.json",
        catalog(scan["checkpoints"], protocol_count=len(scan["protocols"])),
    )
    latest = {
        "schema_version": LATEST_SCHEMA_VERSION,
        **{key: checkpoint[key] for key in ("protocol_digest", "evidence_id", "checkpoint_digest", "checkpoint_path")},
    }
    atomic_write_json(root / "latest.json", latest)


def _public_record(root: Path, record: dict[str, object]) -> dict[str, object]:
    return {
        "protocol_digest": record["protocol_digest"],
        "evidence_id": record["evidence_id"],
        "checkpoint_digest": record["checkpoint_digest"],
        "checkpoint_path": str((root / str(record["checkpoint_path"])).resolve()),
    }


__all__ = ["rebuild_catalog", "record_checkpoint", "verify_trajectory"]
