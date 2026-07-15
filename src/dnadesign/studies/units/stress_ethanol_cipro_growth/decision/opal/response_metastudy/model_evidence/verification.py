"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/verification.py

Verification and rebuildable index projection for immutable model evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    CATALOG_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    LATEST_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    ModelEvidenceError,
    content_digest,
    validated_evidence_id,
)
from .json_io import read_mapping


def scan_immutable(root: Path) -> dict[str, list[dict[str, object]]]:
    if not root.is_dir():
        raise ModelEvidenceError(f"model-evidence trajectory root is missing: {root}")
    _verify_immutable_layout(root)
    protocols = [
        verify_protocol(path, expected_digest=path.parent.name)
        for path in sorted((root / "protocols").glob("*/protocol.json"))
    ]
    known_protocols = {str(record["protocol_digest"]) for record in protocols}
    checkpoints: list[dict[str, object]] = []
    for path in sorted((root / "series").glob("*/checkpoints/*/checkpoint.json")):
        record = verify_checkpoint(root, path)
        if record["protocol_digest"] not in known_protocols:
            raise ModelEvidenceError(f"checkpoint {path} references unknown protocol {record['protocol_digest']}.")
        checkpoints.append(record)
    identities = [(row["protocol_digest"], row["evidence_id"]) for row in checkpoints]
    if len(identities) != len(set(identities)):
        raise ModelEvidenceError("trajectory contains duplicate evidence_id records within one protocol series.")
    return {"protocols": protocols, "checkpoints": checkpoints}


def verify_protocol(path: Path, *, expected_digest: str) -> dict[str, object]:
    payload = read_mapping(path, label="protocol")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ModelEvidenceError(f"protocol schema is unsupported: {path}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ModelEvidenceError(f"protocol payload is missing: {path}")
    actual = content_digest(protocol)
    if payload.get("protocol_digest") != expected_digest or actual != expected_digest:
        raise ModelEvidenceError(f"protocol content digest mismatch: {path}")
    if path.parent.name != expected_digest:
        raise ModelEvidenceError(f"protocol directory digest mismatch: {path}")
    return {"protocol_digest": expected_digest, "path": path}


def verify_checkpoint(root: Path, path: Path) -> dict[str, object]:
    payload = read_mapping(path, label="checkpoint")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ModelEvidenceError(f"checkpoint schema is unsupported: {path}")
    checkpoint_digest = payload.get("checkpoint_digest")
    if not isinstance(checkpoint_digest, str) or len(checkpoint_digest) != 64:
        raise ModelEvidenceError(f"checkpoint digest is invalid: {path}")
    base = dict(payload)
    del base["checkpoint_digest"]
    if content_digest(base) != checkpoint_digest:
        raise ModelEvidenceError(f"checkpoint content digest mismatch: {path}")
    evidence_id = validated_evidence_id(payload.get("evidence_id"))
    protocol_digest = payload.get("protocol_digest")
    if not isinstance(protocol_digest, str) or len(protocol_digest) != 64:
        raise ModelEvidenceError(f"checkpoint protocol digest is invalid: {path}")
    expected_dir = f"{evidence_id}__{checkpoint_digest}"
    if path.parent.name != expected_dir or path.parents[2].name != protocol_digest:
        raise ModelEvidenceError(f"checkpoint path identity mismatch: {path}")
    return checkpoint_index_record(root, path, payload)


def checkpoint_index_record(root: Path, path: Path, payload: dict[str, object]) -> dict[str, object]:
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ModelEvidenceError(f"checkpoint snapshot is missing: {path}")
    gates = snapshot.get("decision_gates")
    source = snapshot.get("source_metastudy")
    corpus = snapshot.get("corpus")
    campaign_model = snapshot.get("campaign_model")
    challenger = snapshot.get("best_fixed_challenger")
    if not all(isinstance(value, dict) for value in (gates, source, corpus, campaign_model, challenger)):
        raise ModelEvidenceError(f"checkpoint index fields are missing: {path}")
    model_screen_candidate_count = corpus.get("model_screen_candidate_count")
    if isinstance(model_screen_candidate_count, bool) or not isinstance(model_screen_candidate_count, int):
        raise ModelEvidenceError(f"checkpoint model-screen candidate count is invalid: {path}")
    return {
        "protocol_digest": str(payload["protocol_digest"]),
        "evidence_id": str(payload["evidence_id"]),
        "checkpoint_digest": str(payload["checkpoint_digest"]),
        "checkpoint_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "evidence_timing": snapshot.get("evidence_timing"),
        "metastudy_manifest_sha256": source.get("manifest_sha256"),
        "model_screen_candidate_count": model_screen_candidate_count,
        "decision_gates": dict(gates),
        "campaign_model_summary": _model_summary(campaign_model, path=path),
        "best_fixed_challenger_summary": _model_summary(challenger, path=path),
    }


def _model_summary(model: dict[str, object], *, path: Path) -> dict[str, object]:
    fields = (
        "model_id",
        "representation_id",
        "median_channel_spearman",
        "minimum_channel_spearman",
        "response_magnitude_mae",
        "weakest_target_view_response_separation_spearman",
        "weakest_target_view_feasibility_spearman",
        "minimum_defined_group_count",
    )
    missing = [field for field in fields if field not in model]
    if missing:
        raise ModelEvidenceError(f"checkpoint model summary missing {missing[0]}: {path}")
    return {field: model[field] for field in fields}


def verify_latest(root: Path, *, checkpoints: list[dict[str, object]]) -> None:
    if not checkpoints:
        if (root / "latest.json").exists():
            raise ModelEvidenceError("latest model-evidence pointer exists without an immutable checkpoint.")
        return
    latest = read_mapping(root / "latest.json", label="latest pointer")
    if latest.get("schema_version") != LATEST_SCHEMA_VERSION:
        raise ModelEvidenceError("latest model-evidence pointer schema is unsupported.")
    keys = ("protocol_digest", "evidence_id", "checkpoint_digest", "checkpoint_path")
    identity = tuple(latest.get(key) for key in keys)
    known = {tuple(row[key] for key in keys) for row in checkpoints}
    if identity not in known:
        raise ModelEvidenceError("latest model-evidence pointer does not reference a verified checkpoint.")


def _verify_immutable_layout(root: Path) -> None:
    protocol_root = root / "protocols"
    series_root = root / "series"
    actual_protocol_files = {path for path in protocol_root.rglob("*") if path.is_file()}
    expected_protocol_files = set(protocol_root.glob("*/protocol.json"))
    actual_checkpoint_files = {path for path in series_root.rglob("*") if path.is_file()}
    expected_checkpoint_files = set(series_root.glob("*/checkpoints/*/checkpoint.json"))
    unexpected = sorted(
        (actual_protocol_files - expected_protocol_files) | (actual_checkpoint_files - expected_checkpoint_files)
    )
    if unexpected:
        raise ModelEvidenceError(f"unexpected file in immutable model-evidence namespace: {unexpected[0]}")


def catalog(checkpoints: list[dict[str, object]], *, protocol_count: int) -> dict[str, object]:
    ordered = sorted(checkpoints, key=lambda row: (str(row["protocol_digest"]), str(row["evidence_id"])))
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "record_kind": "rebuildable_index_not_scientific_authority",
        "protocol_count": protocol_count,
        "checkpoint_count": len(ordered),
        "checkpoints": ordered,
    }


def empty_scan() -> dict[str, list[dict[str, object]]]:
    return {"protocols": [], "checkpoints": []}


__all__ = [
    "catalog",
    "checkpoint_index_record",
    "empty_scan",
    "scan_immutable",
    "verify_latest",
    "verify_protocol",
]
