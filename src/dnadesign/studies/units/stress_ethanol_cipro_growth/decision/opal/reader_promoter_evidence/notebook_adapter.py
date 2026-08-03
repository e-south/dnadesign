"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/notebook_adapter.py

Register the study-owned Reader evidence validator with OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from dnadesign.opal import (
    ReaderEvidenceArtifactAdapter,
    register_reader_evidence_artifact_adapter,
)

from .contracts import PROMOTER_RESPONSE_SEMANTIC_KIND, ReaderPromoterEvidenceError
from .display_verification import verify_reader_promoter_evidence_manifest

_MAX_NOTEBOOK_MEDIA_BYTES = 32 * 1024 * 1024


def register_notebook_adapter() -> None:
    """Register this study's exact validator and detail renderer."""

    register_reader_evidence_artifact_adapter(
        ReaderEvidenceArtifactAdapter(
            semantic_kind=PROMOTER_RESPONSE_SEMANTIC_KIND,
            verify_artifact=verify_notebook_artifact,
            render_details=render_notebook_details,
            verification_label="Promoter-response evidence",
            display_label="Promoter response evidence",
        )
    )


def verify_notebook_artifact(row: Mapping[str, Any]) -> Path:
    """Run the authoritative study verifier, then bind one flattened notebook row."""

    manifest_path = _manifest_path(row.get("manifest_path"))
    verify_reader_promoter_evidence_manifest(manifest_path)
    publication_row, publication_artifact = _publication_match(row, manifest_path=manifest_path)
    _verify_flattened_binding(row, publication_row=publication_row, publication_artifact=publication_artifact)
    size = publication_artifact["bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size > _MAX_NOTEBOOK_MEDIA_BYTES:
        raise ReaderPromoterEvidenceError(
            f"Reader display artifact exceeds the {_MAX_NOTEBOOK_MEDIA_BYTES}-byte notebook render ceiling."
        )
    return (manifest_path.parent / str(publication_artifact["path"])).resolve()


def render_notebook_details(row: Mapping[str, Any], *, mo: Any) -> Any:
    """Render study-owned explanatory fields after authoritative verification."""

    response = row["sources"]["response_window"]
    records = response["records"]
    catalog = response["catalog"]
    bindings = row["sources"]["candidate_bindings"]
    selected = row["selected_binding"]
    details = [
        ("Claim", "Objective-neutral display evidence"),
        ("Reader source experiment", str(row["reader_experiment_id"])),
        ("Reader output experiment", str(response["output_experiment_id"])),
        ("Design", str(row["design_id"])),
        ("Candidate", str(row["candidate_id"])),
        ("Response summary", _reduction_label(row["reduction_id"])),
        (
            "Diagnostic panels",
            "growth trajectory; response trajectory; reference-relative magnitude; reduced four-state components",
        ),
        (
            "Reader catalog",
            f"v{catalog['schema_version']} · {_short_digest(catalog['sha256'])} · "
            f"epoch {catalog['provenance_epoch_id']}",
        ),
        (
            "Exact records",
            f"designs r{records['designs']['revision']} {_short_digest(records['designs']['revision_digest'])}; "
            f"traces r{records['traces']['revision']} {_short_digest(records['traces']['revision_digest'])}; "
            f"diagnostic r{records['diagnostic']['revision']} "
            f"{_short_digest(records['diagnostic']['revision_digest'])}",
        ),
        (
            "Candidate binding",
            f"{selected['binding_method']} · {bindings['candidate_table_id']} · "
            f"{_short_digest(bindings['manifest_sha256'])}",
        ),
    ]
    markdown = [
        "| Evidence field | Verified value |",
        "|---|---|",
        *[f"| {_escape(label)} | {_escape(value)} |" for label, value in details],
        "",
        f"> {_escape(row['non_claim_boundary'])}",
    ]
    return mo.accordion({"Evidence details": mo.md("\n".join(markdown))}, multiple=False)


def _manifest_path(value: object) -> Path:
    raw = str(value or "")
    path = Path(raw).expanduser()
    if not raw or raw != str(path) or not path.is_absolute() or path.resolve() != path:
        raise ReaderPromoterEvidenceError("Reader display manifest path must be exact and absolute.")
    return path


def _publication_match(
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for publication_row in payload["rows"]:
        for artifact in publication_row["artifacts"]:
            if artifact["path"] == row.get("path"):
                matches.append((publication_row, artifact))
    if len(matches) != 1:
        raise ReaderPromoterEvidenceError(
            "Reader display artifact must resolve exactly once in its verified publication manifest."
        )
    return matches[0]


def _verify_flattened_binding(
    row: Mapping[str, Any],
    *,
    publication_row: Mapping[str, Any],
    publication_artifact: Mapping[str, Any],
) -> None:
    row_fields = (
        "id",
        "candidate_id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        "evidence_role",
        "claim_status",
        "non_claim_boundary",
        "selected_binding",
        "sources",
    )
    artifact_fields = {
        "semantic_kind": "semantic_kind",
        "kind": "kind",
        "record_id": "artifact_record_id",
        "scope": "scope",
        "path": "path",
        "path_label": "path_label",
        "exists": "exists",
        "media_type": "media_type",
        "bytes": "bytes",
        "sha256": "sha256",
        "source_record_revision_digest": "source_record_revision_digest",
        "source_file_path": "source_file_path",
        "source_receipt_sha256": "source_receipt_sha256",
    }
    if any(publication_row[field] != row.get(field) for field in row_fields):
        raise ReaderPromoterEvidenceError("Reader display row disagrees with its verified publication manifest.")
    if any(publication_artifact[field] != row.get(flattened) for field, flattened in artifact_fields.items()):
        raise ReaderPromoterEvidenceError("Reader display artifact disagrees with its verified publication manifest.")


def _short_digest(value: object) -> str:
    text = str(value or "")
    if text.startswith("sha256:") and len(text) == 71:
        return f"sha256:{text[7:17]}…{text[-8:]}"
    return text


def _reduction_label(value: object) -> str:
    token = str(value or "").strip()
    match = re.search(r"_(\d+(?:p\d+)?)_(\d+(?:p\d+)?)h_post$", token)
    if token.startswith("event_") and match is not None:
        start, end = (part.replace("p", ".") for part in match.groups())
        return f"{start}–{end} h post-event"
    return token.replace("_", " ").strip().title()


def _escape(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


__all__ = ["register_notebook_adapter", "render_notebook_details", "verify_notebook_artifact"]
