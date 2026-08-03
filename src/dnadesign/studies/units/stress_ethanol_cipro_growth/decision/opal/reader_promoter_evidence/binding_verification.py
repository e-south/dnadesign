"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/binding_verification.py

Resolve one Reader design through the study-owned promoter binding artifact.

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


def resolve_reader_study_binding(
    design_id: str,
    *,
    bindings_bundle: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    """Return the exact selected binding and its immutable source receipt."""

    try:
        bindings = load_promoter_candidate_bindings(bindings_bundle)
    except PromoterCandidateBindingsError as exc:
        raise ReaderPromoterEvidenceError(f"Study promoter-candidate bindings did not verify: {exc}") from exc
    matches = bindings.loc[
        (bindings["alias_namespace"].astype(str) == READER_ALIAS_NAMESPACE)
        & (bindings["alias"].astype(str) == design_id)
    ]
    if len(matches) != 1:
        raise ReaderPromoterEvidenceError(
            "Reader design alias must resolve exactly once in the study promoter-candidate bindings."
        )
    row = matches.iloc[0]
    manifest_path = Path(bindings_bundle).expanduser().resolve() / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderPromoterEvidenceError(f"Could not read the promoter-binding manifest: {exc}") from exc
    source = {
        "schema_id": str(manifest["schema_id"]),
        "schema_version": str(manifest["schema_version"]),
        "study_id": str(manifest["study_id"]),
        "manifest_sha256": _sha256(manifest_path),
        "records_sha256": "sha256:" + str(manifest["record"]["sha256"]),
        "candidate_table_id": str(manifest["candidate_table"]["dataset_id"]),
        "candidate_selection_sha256": "sha256:" + str(manifest["candidate_table"]["selection_sha256"]),
    }
    selected = {
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
    if selected["binding_status"] != "resolved" or selected["binding_method"] != "exact_alias":
        raise ReaderPromoterEvidenceError("Promoter display requires one resolved exact Reader alias binding.")
    return selected, source


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
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["resolve_reader_study_binding"]
