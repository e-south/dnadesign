"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/binding_verification.py

Verify study-owned candidate identity for one Reader promoter-evidence selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    READER_ALIAS_NAMESPACE,
    PromoterCandidateBindingsError,
    load_promoter_candidate_bindings,
)

from .contracts import ReaderPromoterEvidenceError


def verify_reader_study_binding(payload: dict[str, object], *, bindings_bundle: Path) -> None:
    """Match a Reader selection to one exact row in the explicit study artifact."""

    try:
        bindings = load_promoter_candidate_bindings(bindings_bundle)
    except PromoterCandidateBindingsError as exc:
        raise ReaderPromoterEvidenceError(f"Study promoter-candidate bindings did not verify: {exc}") from exc
    selection = payload["selection"]
    if not isinstance(selection, dict):  # pragma: no cover - verified by the caller
        raise ReaderPromoterEvidenceError("Reader selection is malformed.")
    matches = bindings.loc[
        (bindings["alias_namespace"].astype(str) == READER_ALIAS_NAMESPACE)
        & (bindings["alias"].astype(str) == str(selection["design_id"]))
    ]
    if len(matches) != 1:
        raise ReaderPromoterEvidenceError(
            "Reader design alias must resolve exactly once in the explicit study promoter-candidate bindings."
        )
    row = matches.iloc[0]
    if str(selection["candidate_id"]) != str(row["candidate_id"]):
        raise ReaderPromoterEvidenceError("Reader selection candidate identity disagrees with the study binding.")

    manifest_path = Path(bindings_bundle).expanduser().resolve() / "manifest.json"
    binding_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source = {
        "schema_id": str(binding_manifest["schema_id"]),
        "schema_version": str(binding_manifest["schema_version"]),
        "study_id": str(binding_manifest["study_id"]),
        "manifest_sha256": _sha256(manifest_path),
        "records_sha256": "sha256:" + str(binding_manifest["record"]["sha256"]),
        "candidate_table_id": str(binding_manifest["candidate_table"]["dataset_id"]),
        "candidate_selection_sha256": "sha256:" + str(binding_manifest["candidate_table"]["selection_sha256"]),
    }
    sources = payload["sources"]
    if not isinstance(sources, dict) or sources["candidate_bindings"] != expected_source:
        raise ReaderPromoterEvidenceError(
            "Reader candidate-binding source claim disagrees with the explicit study binding artifact."
        )

    expected_binding = {
        "reader_design_id": str(row["alias"]),
        "candidate_id": str(row["candidate_id"]),
        "sequence_sha256": "sha256:" + str(row["sequence_sha256"]),
        "sequence_authority_dataset_id": str(row["sequence_authority_dataset_id"]),
        "sequence_authority_id": str(row["sequence_authority_id"]),
        "sequence_authority_sha256": "sha256:" + str(row["sequence_authority_sha256"]),
        "source_class": str(row["source_class"]),
        "design_family": str(row["design_family"]),
        "binding_status": str(row["binding_status"]),
        "binding_method": str(row["binding_method"]),
        "densegen_plan": _optional_text(row["densegen__plan"]),
        "densegen_run_id": _optional_text(row["densegen__run_id"]),
        "densegen_sampling_library_hash": _optional_text(row["densegen__sampling_library_hash"]),
    }
    if payload["selected_binding"] != expected_binding:
        raise ReaderPromoterEvidenceError(
            "Reader selected sequence or binding provenance disagrees with the explicit study binding artifact."
        )
    baserender = sources["baserender"]
    if not isinstance(baserender, dict) or baserender["adapter_kind"] != row["baserender_adapter_kind"]:
        raise ReaderPromoterEvidenceError("Reader BaseRender adapter disagrees with the explicit study binding.")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = ["verify_reader_study_binding"]
