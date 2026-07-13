"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/resolution.py

Exact namespace-qualified alias resolution for study promoter candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .contracts import ExactPromoterCandidateIdentity, PromoterCandidateBindingsError
from .render_projection import annotation_index, project_baserender_values
from .values import optional_value, require_columns, required_sha256, required_text

_ALIAS_COLUMNS = (
    "alias_namespace",
    "alias",
    "display_label",
    "candidate_id",
    "authority_sequence",
    "sequence_authority_dataset_id",
    "sequence_authority_id",
    "sequence_authority_sha256",
)


def resolve_exact_promoter_candidate_identity(
    *,
    alias_namespace: str,
    alias: str,
    candidate_id: str,
    authority_sequence: str,
    candidate_records: pd.DataFrame,
) -> ExactPromoterCandidateIdentity:
    """Bind one exact typed alias to one sequence-matched candidate."""

    require_columns(candidate_records, ("id", "sequence"), label="candidate table selection")
    return _resolve_exact_identity(
        alias_namespace=alias_namespace,
        alias=alias,
        candidate_id=candidate_id,
        authority_sequence=authority_sequence,
        by_candidate=_candidate_index(candidate_records),
    )


def resolve_promoter_candidate_bindings(
    *,
    alias_rows: pd.DataFrame,
    candidate_records: pd.DataFrame,
    genbank_annotations: pd.DataFrame,
    candidate_table_id: str,
    candidate_selection_sha256: str,
) -> pd.DataFrame:
    """Resolve exact aliases without embedding Reader, metric, or campaign semantics."""

    require_columns(alias_rows, _ALIAS_COLUMNS, label="promoter alias table")
    require_columns(
        candidate_records,
        (
            "id",
            "sequence",
            "opal_candidate__source_class",
            "opal_candidate__design_family",
        ),
        label="candidate table selection",
    )
    aliases = alias_rows.copy()
    aliases["alias_namespace"] = aliases["alias_namespace"].astype(str)
    aliases["alias"] = aliases["alias"].astype(str)
    duplicates = aliases.duplicated(subset=["alias_namespace", "alias"], keep=False)
    if duplicates.any():
        values = sorted(
            {
                f"{row.alias_namespace}:{row.alias}"
                for row in aliases.loc[duplicates, ["alias_namespace", "alias"]].itertuples(index=False)
            }
        )
        raise PromoterCandidateBindingsError(f"Promoter aliases must be unique within each namespace: {values}")

    by_candidate = _candidate_index(candidate_records)
    annotations = annotation_index(genbank_annotations)
    table_id = required_text(candidate_table_id, field="candidate table ID")
    table_digest = required_sha256(candidate_selection_sha256, field="candidate selection SHA-256")
    rows = [
        _binding_row(
            alias_record,
            by_candidate=by_candidate,
            annotations=annotations,
            candidate_table_id=table_id,
            candidate_selection_sha256=table_digest,
        )
        for alias_record in aliases.to_dict(orient="records")
    ]
    return pd.DataFrame(rows)


def _binding_row(
    alias_record: dict[str, Any],
    *,
    by_candidate: pd.DataFrame,
    annotations: pd.DataFrame,
    candidate_table_id: str,
    candidate_selection_sha256: str,
) -> dict[str, Any]:
    identity = _resolve_exact_identity(
        alias_namespace=alias_record["alias_namespace"],
        alias=alias_record["alias"],
        candidate_id=alias_record["candidate_id"],
        authority_sequence=alias_record["authority_sequence"],
        by_candidate=by_candidate,
    )
    candidate = by_candidate.loc[identity.candidate_id]
    source_class = required_text(
        candidate["opal_candidate__source_class"],
        field="source class",
        row_id=identity.candidate_id,
    )
    adapter_kind, annotation_column, adapter_values = project_baserender_values(
        candidate=candidate,
        candidate_id=identity.candidate_id,
        source_class=source_class,
        annotation_by_id=annotations,
        canonical_sequence=identity.canonical_sequence,
    )
    densegen_values = {
        field: optional_value(candidate.get(field)) if adapter_kind == "densegen_tfbs" else None
        for field in ("densegen__plan", "densegen__run_id", "densegen__sampling_library_hash")
    }
    return {
        "alias_namespace": identity.alias_namespace,
        "alias": identity.alias,
        "display_label": required_text(alias_record["display_label"], field="display label", row_id=identity.alias),
        "candidate_id": identity.candidate_id,
        "canonical_sequence": identity.canonical_sequence,
        "sequence_sha256": identity.sequence_sha256,
        "candidate_table_id": candidate_table_id,
        "candidate_selection_sha256": candidate_selection_sha256,
        "sequence_authority_dataset_id": required_text(
            alias_record["sequence_authority_dataset_id"], field="sequence authority dataset ID"
        ),
        "sequence_authority_id": required_text(alias_record["sequence_authority_id"], field="sequence authority ID"),
        "sequence_authority_sha256": required_sha256(
            alias_record["sequence_authority_sha256"], field="sequence authority SHA-256"
        ),
        "source_class": source_class,
        "design_family": required_text(
            candidate["opal_candidate__design_family"], field="design family", row_id=identity.candidate_id
        ),
        "baserender_adapter_kind": adapter_kind,
        "baserender_annotation_column": annotation_column,
        **densegen_values,
        "densegen__used_tfbs_detail": adapter_values["densegen__used_tfbs_detail"],
        "densegen__required_regulators": adapter_values["densegen__required_regulators"],
        "seq_annot__features": adapter_values["seq_annot__features"],
        "seq_annot__source_file": adapter_values["seq_annot__source_file"],
        "usr_label__primary": adapter_values["usr_label__primary"],
        "derived__product_kind": adapter_values["derived__product_kind"],
        "binding_status": identity.binding_status,
        "binding_method": identity.binding_method,
    }


def _candidate_index(candidate_records: pd.DataFrame) -> pd.DataFrame:
    candidates = candidate_records.copy()
    candidates["id"] = candidates["id"].astype(str)
    duplicates = sorted(candidates.loc[candidates["id"].duplicated(keep=False), "id"].unique().tolist())
    if duplicates:
        raise PromoterCandidateBindingsError(f"Candidate table selection IDs must be unique: {duplicates}")
    return candidates.set_index("id", drop=False)


def _resolve_exact_identity(
    *,
    alias_namespace: Any,
    alias: Any,
    candidate_id: Any,
    authority_sequence: Any,
    by_candidate: pd.DataFrame,
) -> ExactPromoterCandidateIdentity:
    namespace = required_text(alias_namespace, field="alias namespace")
    alias_value = required_text(alias, field="alias")
    candidate_key = required_text(candidate_id, field="candidate ID", row_id=f"{namespace}:{alias_value}")
    if candidate_key not in by_candidate.index:
        raise PromoterCandidateBindingsError(
            f"Alias {namespace}:{alias_value!r} has no exact candidate {candidate_key!r}."
        )
    candidate_sequence = required_text(
        by_candidate.loc[candidate_key, "sequence"], field="candidate sequence", row_id=candidate_key
    )
    expected_sequence = required_text(authority_sequence, field="authority sequence", row_id=alias_value)
    if expected_sequence != candidate_sequence:
        raise PromoterCandidateBindingsError(
            f"Alias {namespace}:{alias_value!r} sequence does not match candidate {candidate_key!r}."
        )
    return ExactPromoterCandidateIdentity(
        alias_namespace=namespace,
        alias=alias_value,
        candidate_id=candidate_key,
        canonical_sequence=candidate_sequence,
        sequence_sha256=hashlib.sha256(candidate_sequence.encode("utf-8")).hexdigest(),
    )
