"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_promoter_evidence_contract.py

Validate OPAL's public projection of Reader promoter-evidence bundle v5 metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

PROMOTER_EVIDENCE_BUNDLE_RECORD_ID = "reader.response_window.promoter_evidence_bundle.v5"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_SOURCE_FIELDS = frozenset({"response_window", "candidate_bindings", "baserender"})


class ReaderPromoterEvidenceContextError(ValueError):
    """Raised when projected Reader v5 evidence metadata is incomplete or inconsistent."""


def verify_reader_promoter_evidence_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structured v5 provenance without recomputing Reader-owned evidence."""

    non_claim_boundary = _text(row.get("non_claim_boundary"), field="non_claim_boundary")
    claim_status = str(row.get("claim_status") or "")
    if claim_status not in {"objective_neutral", "screen_only"}:
        raise ReaderPromoterEvidenceContextError("Promoter evidence claim_status is unsupported.")
    sources = row.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != _SOURCE_FIELDS:
        raise ReaderPromoterEvidenceContextError(
            "Promoter evidence sources must name response_window, candidate_bindings, and baserender."
        )
    response = _verify_response_source(sources["response_window"], row=row)
    bindings = _verify_binding_source(sources["candidate_bindings"])
    baserender = _verify_baserender_source(sources["baserender"])
    if response["study_id"] != bindings["study_id"]:
        raise ReaderPromoterEvidenceContextError("Promoter evidence source study identities disagree.")
    selected_binding = _verify_selected_binding(
        row.get("selected_binding"),
        row=row,
        baserender_adapter_kind=str(baserender["adapter_kind"]),
    )
    overlay = _verify_objective_overlay(row.get("objective_overlay"), row=row, claim_status=claim_status)
    return {
        "non_claim_boundary": non_claim_boundary,
        "claim_status": claim_status,
        "response_window": response,
        "candidate_bindings": bindings,
        "baserender": baserender,
        "selected_binding": selected_binding,
        "objective_overlay": overlay,
    }


def _verify_response_source(value: object, *, row: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "study_id",
        "request_id",
        "experiment_id",
        "reduction_id",
        "manifest_sha256",
    }
    source = _exact_mapping(value, fields=fields, field="sources.response_window")
    if source["schema_version"] != "reader.response_window.bundle.v5":
        raise ReaderPromoterEvidenceContextError("Promoter evidence requires a Reader response-window bundle v5.")
    for field in ("study_id", "request_id", "experiment_id", "reduction_id"):
        _text(source[field], field=f"sources.response_window.{field}")
    _digest(source["manifest_sha256"], field="sources.response_window.manifest_sha256")
    if source["experiment_id"] != row.get("reader_experiment_id"):
        raise ReaderPromoterEvidenceContextError("Response-window experiment disagrees with the displayed evidence.")
    if source["reduction_id"] != row.get("reduction_id"):
        raise ReaderPromoterEvidenceContextError("Response-window reduction disagrees with the displayed evidence.")
    return source


def _verify_binding_source(value: object) -> dict[str, Any]:
    fields = {
        "schema_id",
        "schema_version",
        "study_id",
        "manifest_sha256",
        "records_sha256",
        "candidate_table_id",
        "candidate_selection_sha256",
    }
    source = _exact_mapping(value, fields=fields, field="sources.candidate_bindings")
    if source["schema_id"] != "dnadesign.study.promoter_candidate_bindings.v1":
        raise ReaderPromoterEvidenceContextError("Promoter evidence uses an unsupported candidate-binding contract.")
    if source["schema_version"] != "1":
        raise ReaderPromoterEvidenceContextError("Promoter evidence candidate-binding schema version must be 1.")
    for field in ("schema_version", "study_id", "candidate_table_id"):
        _text(source[field], field=f"sources.candidate_bindings.{field}")
    for field in ("manifest_sha256", "records_sha256", "candidate_selection_sha256"):
        _digest(source[field], field=f"sources.candidate_bindings.{field}")
    return source


def _verify_baserender_source(value: object) -> dict[str, Any]:
    fields = {
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
    source = _exact_mapping(value, fields=fields, field="sources.baserender")
    if source["contract_id"] != "dnadesign.baserender.sequence_panel.v1":
        raise ReaderPromoterEvidenceContextError("Promoter evidence uses an unsupported BaseRender contract.")
    if source["contract_version"] != "1":
        raise ReaderPromoterEvidenceContextError("Promoter evidence BaseRender contract version must be 1.")
    if source["adapter_kind"] not in {"densegen_tfbs", "usr_genbank_annotations_v1"}:
        raise ReaderPromoterEvidenceContextError("Promoter evidence uses an unsupported BaseRender adapter.")
    for field in ("contract_version", "style_profile", "renderer_name"):
        _text(source[field], field=f"sources.baserender.{field}")
    for field, minimum in (
        ("sequence_length_bp", 1),
        ("feature_count", 0),
        ("strand_count", 1),
        ("image_width_px", 1),
        ("image_height_px", 1),
    ):
        _integer(source[field], field=f"sources.baserender.{field}", minimum=minimum)
    legend = source["legend_entries"]
    if not isinstance(legend, list) or any(not isinstance(item, str) or not item.strip() for item in legend):
        raise ReaderPromoterEvidenceContextError("Promoter evidence BaseRender legend entries are malformed.")
    return source


def _verify_selected_binding(
    value: object,
    *,
    row: Mapping[str, Any],
    baserender_adapter_kind: str,
) -> dict[str, Any]:
    fields = {
        "reader_design_id",
        "candidate_id",
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
    selected = _exact_mapping(value, fields=fields, field="selected_binding")
    for field in ("sequence_sha256", "sequence_authority_sha256"):
        _digest(selected[field], field=f"selected_binding.{field}")
    for field in (
        "reader_design_id",
        "candidate_id",
        "sequence_authority_dataset_id",
        "sequence_authority_id",
        "source_class",
        "design_family",
        "binding_status",
        "binding_method",
    ):
        _text(selected[field], field=f"selected_binding.{field}")
    if selected["binding_status"] != "resolved" or selected["binding_method"] != "exact_alias":
        raise ReaderPromoterEvidenceContextError("Promoter evidence selected binding must be exact and resolved.")
    if selected["reader_design_id"] != row.get("design_id") or selected["candidate_id"] != row.get("candidate_id"):
        raise ReaderPromoterEvidenceContextError("Promoter evidence selected-binding identity is inconsistent.")
    densegen_fields = ("densegen_plan", "densegen_run_id", "densegen_sampling_library_hash")
    if baserender_adapter_kind == "densegen_tfbs":
        for field in densegen_fields:
            _text(selected[field], field=f"selected_binding.{field}")
    elif any(selected[field] is not None for field in densegen_fields):
        raise ReaderPromoterEvidenceContextError(
            "GenBank promoter evidence must not claim DenseGen selected-binding provenance."
        )
    return selected


def _verify_objective_overlay(
    value: object,
    *,
    row: Mapping[str, Any],
    claim_status: str,
) -> dict[str, Any] | None:
    if value is None:
        if claim_status != "objective_neutral":
            raise ReaderPromoterEvidenceContextError("Screen-only promoter evidence requires an objective overlay.")
        return None
    fields = {
        "schema_version",
        "objective_id",
        "objective_display_label",
        "claim_status",
        "experiment_id",
        "reader_design_id",
        "reduction_id",
        "manifest_sha256",
        "components",
    }
    overlay = _exact_mapping(value, fields=fields, field="objective_overlay")
    if (
        overlay["schema_version"] != "reader.response_window.objective_display_overlay.v2"
        or overlay["claim_status"] != "screen_only"
        or claim_status != "screen_only"
    ):
        raise ReaderPromoterEvidenceContextError("Promoter evidence objective overlay identity is invalid.")
    for field in ("objective_id", "objective_display_label", "experiment_id", "reader_design_id", "reduction_id"):
        _text(overlay[field], field=f"objective_overlay.{field}")
    _digest(overlay["manifest_sha256"], field="objective_overlay.manifest_sha256")
    expected = {
        "experiment_id": row.get("reader_experiment_id"),
        "reader_design_id": row.get("design_id"),
        "reduction_id": row.get("reduction_id"),
    }
    if any(overlay[field] != expected_value for field, expected_value in expected.items()):
        raise ReaderPromoterEvidenceContextError("Promoter evidence objective overlay selection is inconsistent.")
    components = overlay["components"]
    if not isinstance(components, list) or not 1 <= len(components) <= 6:
        raise ReaderPromoterEvidenceContextError("Promoter evidence objective overlay components are malformed.")
    component_fields = {"component_id", "label", "value", "unit"}
    ids: list[str] = []
    for index, component in enumerate(components):
        item = _exact_mapping(component, fields=component_fields, field=f"objective_overlay.components[{index}]")
        for field in ("component_id", "label", "unit"):
            _text(item[field], field=f"objective_overlay.components[{index}].{field}")
        number = item["value"]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise ReaderPromoterEvidenceContextError("Promoter evidence objective overlay value must be finite.")
        ids.append(str(item["component_id"]))
    if len(ids) != len(set(ids)):
        raise ReaderPromoterEvidenceContextError("Promoter evidence objective overlay component IDs must be unique.")
    return overlay


def _exact_mapping(value: object, *, fields: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReaderPromoterEvidenceContextError(f"Promoter evidence {field} fields are malformed.")
    return dict(value)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReaderPromoterEvidenceContextError(f"Promoter evidence {field} must be trimmed non-empty text.")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReaderPromoterEvidenceContextError(f"Promoter evidence {field} must be a SHA-256 digest.")
    return value


def _integer(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReaderPromoterEvidenceContextError(f"Promoter evidence {field} must be an integer >= {minimum}.")
    return value


__all__ = [
    "PROMOTER_EVIDENCE_BUNDLE_RECORD_ID",
    "ReaderPromoterEvidenceContextError",
    "verify_reader_promoter_evidence_context",
]
