"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/bundle_semantics.py

Verify identity, source, sequence, and overlay semantics in one Reader evidence manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re

from .contracts import ReaderPromoterEvidenceError

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def verify_reader_bundle_semantics(payload: dict[str, object], *, claim_status: str) -> dict[str, str]:
    selection = _verify_selection(payload["selection"])
    sources = payload["sources"]
    _verify_sources(sources, selection=selection)
    _verify_selected_binding(
        payload["selected_binding"],
        baserender_adapter_kind=sources["baserender"]["adapter_kind"],
        reader_design_id=selection["design_id"],
        candidate_id=selection["candidate_id"],
    )
    _verify_overlay(payload["objective_overlay"], claim_status=claim_status, selection=selection)
    return selection


def _verify_selection(value: object) -> dict[str, str]:
    fields = {"experiment_id", "design_id", "candidate_id", "reduction_id"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError(f"Reader selection fields must be exactly {sorted(fields)}.")
    if any(not _nonempty(item) for item in value.values()):
        raise ReaderPromoterEvidenceError("Reader selection values must be non-empty strings.")
    return {field: str(value[field]) for field in fields}


def _verify_sources(value: object, *, selection: dict[str, str]) -> None:
    if not isinstance(value, dict) or set(value) != {"response_window", "candidate_bindings", "baserender"}:
        raise ReaderPromoterEvidenceError(
            "Reader sources must name response_window, candidate_bindings, and baserender."
        )
    response = value["response_window"]
    response_fields = {
        "schema_version",
        "study_id",
        "request_id",
        "experiment_id",
        "reduction_id",
        "manifest_sha256",
    }
    if not isinstance(response, dict) or set(response) != response_fields:
        raise ReaderPromoterEvidenceError("Reader response-window source metadata is malformed.")
    if (
        response["schema_version"] != "reader.response_window.bundle.v5"
        or response["study_id"] != "stress_ethanol_cipro_growth"
        or any(not _nonempty(response[field]) for field in ("request_id", "experiment_id", "reduction_id"))
        or not _is_sha256(response["manifest_sha256"])
    ):
        raise ReaderPromoterEvidenceError("Reader response-window source is not a verified bundle v5 record.")
    for field in ("experiment_id", "reduction_id"):
        if response[field] != selection[field]:
            raise ReaderPromoterEvidenceError(
                f"Reader response-window source {field} disagrees with selection {field}."
            )
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
    if binding["study_id"] != response["study_id"]:
        raise ReaderPromoterEvidenceError("Reader evidence source study identities disagree.")
    _verify_baserender(value["baserender"])


def _verify_baserender(value: object) -> None:
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
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError("Reader BaseRender diagnostics are malformed.")
    if (
        value["contract_id"] != "dnadesign.baserender.sequence_panel.v1"
        or str(value["contract_version"]) != "1"
        or value["adapter_kind"] not in {"densegen_tfbs", "usr_genbank_annotations_v1"}
        or any(not _nonempty(value[field]) for field in ("style_profile", "renderer_name"))
        or any(
            not _int_at_least(value[key], 1)
            for key in ("sequence_length_bp", "strand_count", "image_width_px", "image_height_px")
        )
        or not _int_at_least(value["feature_count"], 0)
        or not isinstance(value["legend_entries"], list)
        or any(not _nonempty(item) for item in value["legend_entries"])
    ):
        raise ReaderPromoterEvidenceError("Reader BaseRender diagnostics contain invalid values.")


def _verify_selected_binding(
    value: object,
    *,
    baserender_adapter_kind: object,
    reader_design_id: str,
    candidate_id: str,
) -> None:
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
    densegen_fields = {"densegen_plan", "densegen_run_id", "densegen_sampling_library_hash"}
    digest_fields = {"sequence_sha256", "sequence_authority_sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError(f"Reader selected_binding fields must be exactly {sorted(fields)}.")
    if (
        any(not _is_sha256(value[field]) for field in digest_fields)
        or any(not _nonempty(value[field]) for field in fields - densegen_fields - digest_fields)
        or value["binding_status"] != "resolved"
        or value["binding_method"] != "exact_alias"
    ):
        raise ReaderPromoterEvidenceError("Reader selected_binding sequence or exact-binding provenance is malformed.")
    if value["reader_design_id"] != reader_design_id:
        raise ReaderPromoterEvidenceError(
            "Reader selected_binding reader_design_id disagrees with selection design_id."
        )
    if value["candidate_id"] != candidate_id:
        raise ReaderPromoterEvidenceError("Reader selected_binding candidate_id disagrees with selection candidate_id.")
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
    else:  # pragma: no cover - source verification rejects this first
        raise ReaderPromoterEvidenceError("Reader selected_binding uses an unsupported BaseRender adapter.")


def _verify_overlay(value: object, *, claim_status: str, selection: dict[str, str]) -> None:
    if value is None:
        if claim_status != "objective_neutral":
            raise ReaderPromoterEvidenceError("Screen-only Reader evidence requires an objective overlay record.")
        return
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
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError(f"Reader objective overlay fields must be exactly {sorted(fields)}.")
    if (
        value["schema_version"] != "reader.response_window.objective_display_overlay.v2"
        or value["claim_status"] != claim_status
        or claim_status != "screen_only"
        or not _is_sha256(value["manifest_sha256"])
        or any(
            not _nonempty(value[field])
            for field in ("objective_id", "experiment_id", "reader_design_id", "reduction_id")
        )
    ):
        raise ReaderPromoterEvidenceError("Reader objective overlay identity or claim status is invalid.")
    objective_display_label = value["objective_display_label"]
    if (
        not isinstance(objective_display_label, str)
        or objective_display_label != objective_display_label.strip()
        or not objective_display_label
        or not objective_display_label.isprintable()
        or len(objective_display_label) > 40
    ):
        raise ReaderPromoterEvidenceError(
            "Reader objective overlay display label must be a trimmed, printable, single-line string of at most "
            "40 characters."
        )
    selection_fields = {
        "experiment_id": "experiment_id",
        "reader_design_id": "design_id",
        "reduction_id": "reduction_id",
    }
    for field, selection_field in selection_fields.items():
        if value[field] != selection[selection_field]:
            raise ReaderPromoterEvidenceError(
                f"Reader objective overlay {field} disagrees with selection {selection_field}."
            )
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
            or any(not _nonempty(component[field]) for field in ("component_id", "label", "unit"))
            or not _finite(component["value"])
        ):
            raise ReaderPromoterEvidenceError("Reader objective overlay component is malformed.")
        ids.append(str(component["component_id"]))
    if len(ids) != len(set(ids)):
        raise ReaderPromoterEvidenceError("Reader objective overlay component identities must be unique.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _int_at_least(value: object, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = ["verify_reader_bundle_semantics"]
