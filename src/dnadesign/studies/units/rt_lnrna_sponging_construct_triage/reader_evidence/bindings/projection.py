"""Canonical JSON projection for Reader evidence-binding artifacts."""

from __future__ import annotations

from dataclasses import asdict, fields

from .contracts import ReaderEvidenceBinding, ReaderEvidenceBindingSet


def binding_artifact_payload(
    binding_set: ReaderEvidenceBindingSet,
    *,
    include_digest: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": binding_set.schema_id,
        "artifact_id": binding_set.artifact_id,
        "subject_binding_set_id": binding_set.subject_binding_set_id,
        "binding_count": len(binding_set.rows),
        "unbound_count": binding_set.unbound_count,
        "bindings": [_binding_payload(row) for row in binding_set.rows],
    }
    if include_digest:
        payload["artifact_digest"] = binding_set.artifact_digest
    return payload


def _binding_payload(row: ReaderEvidenceBinding) -> dict[str, object]:
    """Project typed Reader lineage without weakening it to caller mappings."""

    payload = {item.name: getattr(row, item.name) for item in fields(row)}
    payload["reader_record_producer"] = row.reader_record_producer.to_dict()
    payload["reader_record_inputs"] = [item.to_dict() for item in row.reader_record_inputs]
    payload["observation_identity_values"] = list(row.observation_identity_values)
    payload["biological_replicate_identity_scopes"] = [
        asdict(item) for item in row.biological_replicate_identity_scopes
    ]
    return payload


__all__: list[str] = []
