"""Canonical JSON projection for Reader evidence-binding artifacts."""

from __future__ import annotations

from dataclasses import asdict

from .contracts import ReaderEvidenceBindingSet


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
        "bindings": [asdict(row) for row in binding_set.rows],
    }
    if include_digest:
        payload["artifact_digest"] = binding_set.artifact_digest
    return payload


__all__: list[str] = []
