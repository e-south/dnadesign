"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/verification.py

Independent verification of Reader promoter-evidence bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from .contracts import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_EVIDENCE_NON_CLAIM,
    READER_BUNDLE_SCHEMA_VERSION,
    ReaderPromoterEvidenceError,
    VerifiedReaderPromoterEvidenceBundle,
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_FIELDS = {
    "schema_version",
    "created_at",
    "claim_status",
    "non_claim_boundary",
    "selection",
    "selected_binding",
    "sources",
    "objective_overlay",
    "artifacts",
}


def verify_reader_promoter_evidence_bundle(bundle_dir: Path) -> VerifiedReaderPromoterEvidenceBundle:
    """Independently verify one Reader bundle without importing Reader."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ReaderPromoterEvidenceError(f"Reader promoter-evidence manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReaderPromoterEvidenceError(f"Could not parse Reader promoter-evidence manifest: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ReaderPromoterEvidenceError(
            f"Reader promoter-evidence manifest fields must be exactly {sorted(_MANIFEST_FIELDS)}."
        )
    if payload["schema_version"] != READER_BUNDLE_SCHEMA_VERSION:
        raise ReaderPromoterEvidenceError(f"Reader promoter evidence must use {READER_BUNDLE_SCHEMA_VERSION!r}.")
    _verify_created_at(payload["created_at"])
    claim_status = payload["claim_status"]
    if claim_status not in {"objective_neutral", "screen_only"}:
        raise ReaderPromoterEvidenceError("Reader promoter evidence has an unsupported claim_status.")
    if payload["non_claim_boundary"] != PROMOTER_EVIDENCE_NON_CLAIM:
        raise ReaderPromoterEvidenceError("Reader promoter evidence changed its non-claim boundary.")
    _verify_selection(payload["selection"])
    _verify_sources(payload["sources"])
    _verify_selected_binding(
        payload["selected_binding"],
        baserender_adapter_kind=payload["sources"]["baserender"]["adapter_kind"],
    )
    _verify_overlay(payload["objective_overlay"], claim_status=str(claim_status))
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(PROMOTER_EVIDENCE_ARTIFACT_IDS):
        raise ReaderPromoterEvidenceError(
            f"Reader promoter-evidence artifacts must be exactly {sorted(PROMOTER_EVIDENCE_ARTIFACT_IDS)}."
        )
    for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS:
        _verify_artifact(root, artifact_id=artifact_id, value=artifacts[artifact_id])
    return VerifiedReaderPromoterEvidenceBundle(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        manifest=payload,
    )


def _verify_artifact(root: Path, *, artifact_id: str, value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} metadata is malformed.")
    if value["path"] != artifact_id:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} path disagrees with its identity.")
    path = (root / artifact_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} escapes its bundle root.") from exc
    size = value["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or not path.is_file():
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} is missing or has an invalid size.")
    if path.stat().st_size != size or _sha256(path) != value["sha256"]:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} digest or size mismatch.")
    signature = path.read_bytes()[:8]
    if artifact_id.endswith(".pdf") and not signature.startswith(b"%PDF"):
        raise ReaderPromoterEvidenceError("Reader promoter-evidence PDF signature is invalid.")
    if artifact_id.endswith(".png") and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderPromoterEvidenceError("Reader promoter-evidence PNG signature is invalid.")


def _verify_selection(value: object) -> None:
    fields = {"experiment_id", "design_id", "candidate_id", "reduction_id"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError(f"Reader selection fields must be exactly {sorted(fields)}.")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ReaderPromoterEvidenceError("Reader selection values must be non-empty strings.")


def _verify_sources(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"response_window", "candidate_bindings", "baserender"}:
        raise ReaderPromoterEvidenceError(
            "Reader sources must name response_window, candidate_bindings, and baserender."
        )
    response = value["response_window"]
    if not isinstance(response, dict) or set(response) != {"schema_version", "request_id", "manifest_sha256"}:
        raise ReaderPromoterEvidenceError("Reader response-window source metadata is malformed.")
    if response["schema_version"] != "reader.response_window.bundle.v3" or not _is_sha256(response["manifest_sha256"]):
        raise ReaderPromoterEvidenceError("Reader response-window source is not a verified bundle v3 record.")
    binding = value["candidate_bindings"]
    binding_fields = {
        "schema_id",
        "schema_version",
        "study_id",
        "manifest_sha256",
        "records_sha256",
        "candidate_table_id",
        "candidate_selection_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != binding_fields:
        raise ReaderPromoterEvidenceError("Reader candidate-binding source metadata is malformed.")
    if (
        binding["schema_id"] != "dnadesign.study.promoter_candidate_bindings.v1"
        or binding["schema_version"] != "1"
        or binding["study_id"] != "stress_ethanol_cipro_growth"
        or any(not _is_sha256(binding[key]) for key in binding if key.endswith("sha256"))
        or not _nonempty(binding["candidate_table_id"])
    ):
        raise ReaderPromoterEvidenceError("Candidate-binding source is not the supported exact study contract.")
    baserender = value["baserender"]
    baserender_fields = {
        "contract_id",
        "contract_version",
        "style_profile",
        "renderer_name",
        "adapter_kind",
        "sequence_length_bp",
        "feature_count",
        "strand_count",
        "legend_entries",
        "image_width_px",
        "image_height_px",
    }
    if not isinstance(baserender, dict) or set(baserender) != baserender_fields:
        raise ReaderPromoterEvidenceError("Reader BaseRender diagnostics are malformed.")
    if (
        baserender["contract_id"] != "dnadesign.baserender.sequence_panel.v1"
        or str(baserender["contract_version"]) != "1"
        or baserender["adapter_kind"] not in {"densegen_tfbs", "usr_genbank_annotations_v1"}
        or not _nonempty(baserender["style_profile"])
        or not _nonempty(baserender["renderer_name"])
        or any(
            not _int_at_least(baserender[key], 1)
            for key in ("sequence_length_bp", "strand_count", "image_width_px", "image_height_px")
        )
        or not _int_at_least(baserender["feature_count"], 0)
        or not isinstance(baserender["legend_entries"], list)
        or any(not _nonempty(item) for item in baserender["legend_entries"])
    ):
        raise ReaderPromoterEvidenceError("Reader BaseRender diagnostics contain invalid values.")


def _verify_selected_binding(
    value: object,
    *,
    baserender_adapter_kind: object,
) -> None:
    fields = {
        "sequence_sha256",
        "sequence_authority_dataset_id",
        "sequence_authority_id",
        "sequence_authority_sha256",
        "source_class",
        "design_family",
        "binding_status",
        "binding_method",
        "densegen_plan",
        "densegen_run_id",
        "densegen_sampling_library_hash",
    }
    densegen_fields = {"densegen_plan", "densegen_run_id", "densegen_sampling_library_hash"}
    digest_fields = {"sequence_sha256", "sequence_authority_sha256"}
    required_text = fields - densegen_fields - digest_fields
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError(f"Reader selected_binding fields must be exactly {sorted(fields)}.")
    if (
        any(not _is_sha256(value[field]) for field in digest_fields)
        or any(not _nonempty(value[field]) for field in required_text)
        or value["binding_status"] != "resolved"
        or value["binding_method"] != "exact_alias"
    ):
        raise ReaderPromoterEvidenceError("Reader selected_binding sequence or exact-binding provenance is malformed.")
    if baserender_adapter_kind == "densegen_tfbs":
        if any(not _nonempty(value[field]) for field in densegen_fields):
            raise ReaderPromoterEvidenceError(
                "DenseGen Reader evidence requires selected_binding plan, run, and library provenance."
            )
    elif baserender_adapter_kind == "usr_genbank_annotations_v1":
        if any(value[field] is not None for field in densegen_fields):
            raise ReaderPromoterEvidenceError(
                "GenBank Reader evidence requires null DenseGen selected_binding provenance."
            )
    else:  # pragma: no cover - adapter rejected by source verification first
        raise ReaderPromoterEvidenceError("Reader selected_binding uses an unsupported BaseRender adapter.")


def _verify_overlay(value: object, *, claim_status: str) -> None:
    if value is None:
        if claim_status != "objective_neutral":
            raise ReaderPromoterEvidenceError("Screen-only Reader evidence requires an objective overlay record.")
        return
    fields = {"schema_version", "objective_id", "claim_status", "manifest_sha256", "components"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError(f"Reader objective overlay fields must be exactly {sorted(fields)}.")
    if (
        value["schema_version"] != "reader.response_window.objective_display_overlay.v1"
        or value["claim_status"] != claim_status
        or claim_status != "screen_only"
        or not _is_sha256(value["manifest_sha256"])
        or not _nonempty(value["objective_id"])
    ):
        raise ReaderPromoterEvidenceError("Reader objective overlay identity or claim status is invalid.")
    components = value["components"]
    component_fields = {"component_id", "label", "value", "unit"}
    if not isinstance(components, list) or not 1 <= len(components) <= 6:
        raise ReaderPromoterEvidenceError(
            "Reader objective overlay components must contain between one and six raw components."
        )
    ids: list[str] = []
    for component in components:
        if (
            not isinstance(component, dict)
            or set(component) != component_fields
            or not _nonempty(component["component_id"])
            or not _nonempty(component["label"])
            or not _nonempty(component["unit"])
            or not _finite(component["value"])
        ):
            raise ReaderPromoterEvidenceError("Reader objective overlay component is malformed.")
        ids.append(str(component["component_id"]))
    if len(ids) != len(set(ids)):
        raise ReaderPromoterEvidenceError("Reader objective overlay component identities must be unique.")


def _verify_created_at(value: object) -> None:
    if not isinstance(value, str):
        raise ReaderPromoterEvidenceError("Reader created_at must be an ISO-8601 timestamp.")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError("Reader created_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ReaderPromoterEvidenceError("Reader created_at must include a timezone.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _int_at_least(value: object, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = ["verify_reader_promoter_evidence_bundle"]
