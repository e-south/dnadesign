"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/representation_contract.py

Representation-table and Infer handoff validation for the RT-lnRNA sponging
construct triage study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_REPRESENTATION_TABLE_CONTRACT_PATH = _STUDY_DIR / "operations/contract/schemas/representation-table.schema.yaml"
_INPUT_DATASET = "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1"
_OUTPUT_DATASET = "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1"
_EXPECTED_MODEL = "evo2_7b"
_EXPECTED_INTERMEDIATE_BLOCK = 26
_EXPECTED_INTERMEDIATE_SELECTOR = "block26_mlp_out"
_EXPECTED_INTERMEDIATE_DIMENSION = 4096
_EXPECTED_OUTPUT_LAYER_DIMENSION = 512
REQUIRED_SOURCE_VIEW_NAMES = (
    "dual_cassette_2000bp_seq_mean",
    "dual_cassette_2000bp_fwd_rc_concat",
    "lnrna_span_in_construct_anchor_mean",
    "lnrna_span_in_construct_reverse_complement_anchor_mean",
    "rt_cds_span_in_construct_anchor_mean",
    "rt_cds_span_in_construct_reverse_complement_anchor_mean",
)
_VIEW_POOLING = {
    "dual_cassette_2000bp_seq_mean": ("seq_mean", None),
    "dual_cassette_2000bp_fwd_rc_concat": ("seq_mean", None),
    "lnrna_span_in_construct_anchor_mean": ("anchor_mean", "sequence_view"),
    "lnrna_span_in_construct_reverse_complement_anchor_mean": ("anchor_mean", "sequence_view"),
    "rt_cds_span_in_construct_anchor_mean": ("anchor_mean", "sequence_view"),
    "rt_cds_span_in_construct_reverse_complement_anchor_mean": ("anchor_mean", "sequence_view"),
}
_EXPECTED_OVERLAY_REFERENCE_IDS = (
    "khan_cross_retron_rt_dna_abundance_v1",
    "crawford_eco1_lnrna_msd_abundance_v1",
    "crawford_eco1_lnrna_msd_design_reference_v1",
)
_EXPECTED_FIXED_SIZE_VECTORS = {
    "intermediate_embedding_7b_dual_cassette_2000bp_fwd_rc_concat": ("float32", 8192),
    "intermediate_embedding_7b_lnrna_span_in_construct_anchor_mean_bidir_concat": ("float32", 8192),
    "intermediate_embedding_7b_rt_cds_span_in_construct_anchor_mean_bidir_concat": ("float32", 8192),
    "intermediate_embedding_7b_lnrna_rt_slot_pair_anchor_mean_concat": ("float32", 16384),
}
_EXPECTED_SCALAR_KINDS = ("log_likelihood__total", "log_likelihood__mean_per_token")
_EXPECTED_CONSTRUCT_SUBJECT_SEQUENCE_FIELDS = (
    "construct_subject__lnrna_sequence",
    "construct_subject__rt_cds_sequence",
)


@dataclass(frozen=True)
class RepresentationTableContractAudit:
    errors: tuple[str, ...]
    source_view_names: tuple[str, ...] = ()
    fixed_size_vectors: dict[str, tuple[str, int]] = field(default_factory=dict)
    overlay_reference_ids: tuple[str, ...] = ()
    row_key_source: dict[str, str] = field(default_factory=dict)
    construct_subject_promotion: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class InferFeatureBundleAudit:
    errors: tuple[str, ...]
    selected_view_names: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_registered_representation_table_contract(
    *,
    repo_root: Path | None = None,
) -> RepresentationTableContractAudit:
    root = _resolve_repo_root(repo_root)
    payload = yaml.safe_load((root / _REPRESENTATION_TABLE_CONTRACT_PATH).read_text(encoding="utf-8"))
    return validate_representation_table_contract_payload(payload)


def validate_representation_table_contract_payload(payload: object) -> RepresentationTableContractAudit:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return RepresentationTableContractAudit(errors=("representation table contract must be a mapping",))
    if _string(payload.get("schema_id")) != "rt_lnrna_sponging_construct_triage_representation_table_v1":
        errors.append("schema_id must be rt_lnrna_sponging_construct_triage_representation_table_v1")
    if _string(payload.get("row_key")) != "construct_subject__id":
        errors.append("row_key must be construct_subject__id")

    source_views = _list(payload.get("source_sequence_views"), label="source_sequence_views", errors=errors)
    source_view_names = tuple(
        _string(_mapping(view, label="source_sequence_views[]", errors=errors).get("view_name"))
        for view in source_views
    )
    if source_view_names != REQUIRED_SOURCE_VIEW_NAMES:
        errors.append("source_sequence_views must declare the six required RT-lnRNA view names in order")
    for view in source_views:
        item = _mapping(view, label="source_sequence_views[]", errors=errors)
        if _string(item.get("dataset")) != _OUTPUT_DATASET:
            errors.append(f"{_string(item.get('view_name'))}: dataset must be {_OUTPUT_DATASET}")
        if _string(item.get("context_kind")) != "template_custom":
            errors.append(f"{_string(item.get('view_name'))}: context_kind must be template_custom")

    _validate_feature_outputs(payload.get("infer_feature_outputs"), errors=errors)
    fixed_size_vectors = _parse_fixed_size_vectors(payload.get("fixed_size_vector_exports"), errors=errors)
    for vector_id, expected in _EXPECTED_FIXED_SIZE_VECTORS.items():
        if fixed_size_vectors.get(vector_id) != expected:
            errors.append(
                f"fixed_size_vector_exports must declare {vector_id} as dtype={expected[0]} dimension={expected[1]}"
            )

    overlay_reference_ids = tuple(
        _string(_mapping(item, label="source_overlay_inputs[]", errors=errors).get("reference_overlay_id"))
        for item in _list(payload.get("source_overlay_inputs"), label="source_overlay_inputs", errors=errors)
    )
    if overlay_reference_ids != _EXPECTED_OVERLAY_REFERENCE_IDS:
        errors.append(
            "source_overlay_inputs must declare Khan abundance, Crawford abundance, and Crawford design references"
        )
    construct_subject_promotion = _parse_construct_subject_promotion(
        payload.get("construct_subject_promotion"),
        errors=errors,
    )
    row_key_source = _parse_row_key_source(payload.get("row_key_source"), errors=errors)
    return RepresentationTableContractAudit(
        errors=tuple(errors),
        source_view_names=source_view_names,
        fixed_size_vectors=fixed_size_vectors,
        overlay_reference_ids=overlay_reference_ids,
        row_key_source=row_key_source,
        construct_subject_promotion=construct_subject_promotion,
    )


def validate_infer_feature_bundle_payload(payload: object) -> InferFeatureBundleAudit:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return InferFeatureBundleAudit(errors=("Infer feature bundle payload must be a mapping",))
    if _string(payload.get("kind")) != "evo2_sequence_feature_v1":
        errors.append("feature_bundle.kind must be evo2_sequence_feature_v1")
    if int(payload.get("intermediate_block") or _EXPECTED_INTERMEDIATE_BLOCK) != _EXPECTED_INTERMEDIATE_BLOCK:
        errors.append(f"feature_bundle.intermediate_block must be {_EXPECTED_INTERMEDIATE_BLOCK} for evo2_7b")
    if payload.get("collect_log_likelihood") is not True:
        errors.append("feature_bundle.collect_log_likelihood must be true")
    if payload.get("collect_output_layer_mean") is not True:
        errors.append("feature_bundle.collect_output_layer_mean must be true")
    if payload.get("collect_intermediate_embedding") is not True:
        errors.append("feature_bundle.collect_intermediate_embedding must be true")

    selected_view_names: list[str] = []
    sequence_view_inputs = _list(
        payload.get("sequence_view_inputs"),
        label="feature_bundle.sequence_view_inputs",
        errors=errors,
    )
    for index, input_payload in enumerate(sequence_view_inputs, start=1):
        input_cfg = _mapping(
            input_payload,
            label=f"feature_bundle.sequence_view_inputs[{index}]",
            errors=errors,
        )
        if _string(input_cfg.get("dataset")) != _OUTPUT_DATASET:
            continue
        selector = _mapping(
            input_cfg.get("view_selector"),
            label=f"feature_bundle.sequence_view_inputs[{index}].view_selector",
            errors=errors,
        )
        pooling = _mapping(
            input_cfg.get("pooling"),
            label=f"feature_bundle.sequence_view_inputs[{index}].pooling",
            errors=errors,
        )
        view_name = _string(selector.get("view_name"))
        if not view_name:
            errors.append(
                f"feature_bundle.sequence_view_inputs[{index}] for {_OUTPUT_DATASET} must select by explicit "
                "view_name; product_kind plus orientation is ambiguous for RT-lnRNA construct views"
            )
            continue
        if view_name not in REQUIRED_SOURCE_VIEW_NAMES:
            errors.append(
                f"feature_bundle.sequence_view_inputs[{index}].view_selector.view_name is not a v1 RT-lnRNA view"
            )
            continue
        selected_view_names.append(view_name)
        expected_operation, expected_bounds_from = _VIEW_POOLING[view_name]
        if _string(pooling.get("operation")) != expected_operation:
            errors.append(f"{view_name}: pooling.operation must be {expected_operation}")
        if _string(pooling.get("bounds_from")) != _string(expected_bounds_from):
            expected = "null" if expected_bounds_from is None else expected_bounds_from
            errors.append(f"{view_name}: pooling.bounds_from must be {expected}")
    selected = tuple(selected_view_names)
    if selected != REQUIRED_SOURCE_VIEW_NAMES:
        errors.append("feature_bundle.sequence_view_inputs must select the six RT-lnRNA source views in contract order")
    return InferFeatureBundleAudit(errors=tuple(errors), selected_view_names=selected)


def _validate_feature_outputs(payload: object, *, errors: list[str]) -> None:
    outputs = _list(payload, label="infer_feature_outputs", errors=errors)
    by_view: dict[str, list[dict[str, object]]] = {}
    for item in outputs:
        output = _mapping(item, label="infer_feature_outputs[]", errors=errors)
        by_view.setdefault(_string(output.get("view_name")), []).append(output)
    for view_name in REQUIRED_SOURCE_VIEW_NAMES:
        outputs_for_view = by_view.get(view_name, [])
        vector_kinds = {
            _string(item.get("representation_kind")): item
            for item in outputs_for_view
            if _string(item.get("output_kind")) == "vector"
        }
        intermediate = vector_kinds.get("intermediate_embedding")
        if intermediate is None:
            errors.append(f"{view_name}: missing intermediate_embedding vector output")
        else:
            _validate_vector_output(
                intermediate,
                view_name=view_name,
                representation_kind="intermediate_embedding",
                expected_dimension=_EXPECTED_INTERMEDIATE_DIMENSION,
                errors=errors,
            )
        output_layer = vector_kinds.get("output_layer_mean")
        if output_layer is None:
            errors.append(f"{view_name}: missing output_layer_mean vector output")
        else:
            _validate_vector_output(
                output_layer,
                view_name=view_name,
                representation_kind="output_layer_mean",
                expected_dimension=_EXPECTED_OUTPUT_LAYER_DIMENSION,
                errors=errors,
            )
        scalar_kinds = {
            _string(item.get("scalar_kind"))
            for item in outputs_for_view
            if _string(item.get("output_kind")) == "scalar"
        }
        if scalar_kinds != set(_EXPECTED_SCALAR_KINDS):
            errors.append(
                f"{view_name}: scalar outputs must be log_likelihood__total and log_likelihood__mean_per_token"
            )


def _validate_vector_output(
    output: dict[str, object],
    *,
    view_name: str,
    representation_kind: str,
    expected_dimension: int,
    errors: list[str],
) -> None:
    if _string(output.get("model_name")) != _EXPECTED_MODEL:
        errors.append(f"{view_name}/{representation_kind}: model_name must be {_EXPECTED_MODEL}")
    if (
        representation_kind == "intermediate_embedding"
        and _string(output.get("layer_name")) != _EXPECTED_INTERMEDIATE_SELECTOR
    ):
        errors.append(f"{view_name}/{representation_kind}: layer_name must be {_EXPECTED_INTERMEDIATE_SELECTOR}")
    if int(output.get("dimension") or 0) != expected_dimension:
        errors.append(f"{view_name}/{representation_kind}: dimension must be {expected_dimension}")
    if _string(output.get("sidecar_value_dtype")) != "float64_list":
        errors.append(f"{view_name}/{representation_kind}: sidecar_value_dtype must be float64_list")
    if _string(output.get("fixed_size_export_dtype")) != "float32":
        errors.append(f"{view_name}/{representation_kind}: fixed_size_export_dtype must be float32")


def _parse_fixed_size_vectors(payload: object, *, errors: list[str]) -> dict[str, tuple[str, int]]:
    parsed: dict[str, tuple[str, int]] = {}
    for item in _list(payload, label="fixed_size_vector_exports", errors=errors):
        output = _mapping(item, label="fixed_size_vector_exports[]", errors=errors)
        vector_id = _string(output.get("view_id"))
        dtype = _string(output.get("dtype"))
        dimension = int(output.get("dimension") or 0)
        if not vector_id:
            errors.append("fixed_size_vector_exports[].view_id is required")
            continue
        parsed[vector_id] = (dtype, dimension)
    return parsed


def _parse_construct_subject_promotion(value: object, *, errors: list[str]) -> dict[str, object]:
    payload = _mapping(value, label="construct_subject_promotion", errors=errors)
    source_dataset = _string(payload.get("source_dataset"))
    consolidated_construct_dataset = _string(payload.get("consolidated_construct_dataset"))
    sequence_fields = tuple(
        _string(field)
        for field in _list(
            payload.get("required_sequence_fields"),
            label="construct_subject_promotion.required_sequence_fields",
            errors=errors,
        )
    )
    construct_views = tuple(
        _string(view)
        for view in _list(
            payload.get("required_construct_views"),
            label="construct_subject_promotion.required_construct_views",
            errors=errors,
        )
    )
    if source_dataset != _INPUT_DATASET:
        errors.append(f"construct_subject_promotion.source_dataset must be {_INPUT_DATASET}")
    if consolidated_construct_dataset != _OUTPUT_DATASET:
        errors.append(f"construct_subject_promotion.consolidated_construct_dataset must be {_OUTPUT_DATASET}")
    if sequence_fields != _EXPECTED_CONSTRUCT_SUBJECT_SEQUENCE_FIELDS:
        errors.append(
            "construct_subject_promotion.required_sequence_fields must be construct_subject__lnrna_sequence, "
            "construct_subject__rt_cds_sequence"
        )
    if construct_views != REQUIRED_SOURCE_VIEW_NAMES:
        errors.append("construct_subject_promotion.required_construct_views must match the six source views")
    if payload.get("requires_explicit_rt_plus_lnrna_authority") is not True:
        errors.append("construct_subject_promotion.requires_explicit_rt_plus_lnrna_authority must be true")
    if payload.get("forbid_overlay_only_infer_rows") is not True:
        errors.append("construct_subject_promotion.forbid_overlay_only_infer_rows must be true")
    return {
        "source_dataset": source_dataset,
        "consolidated_construct_dataset": consolidated_construct_dataset,
        "required_sequence_fields": sequence_fields,
        "required_construct_views": construct_views,
    }


def _parse_row_key_source(value: object, *, errors: list[str]) -> dict[str, str]:
    payload = _mapping(value, label="row_key_source", errors=errors)
    dataset = _string(payload.get("dataset"))
    namespace = _string(payload.get("namespace"))
    column = _string(payload.get("column"))
    materialized_by = _string(payload.get("materialized_by"))
    output_join_field = _string(payload.get("output_join_field"))
    input_dataset = _string(payload.get("input_dataset"))
    input_construct_subject_field = _string(payload.get("input_construct_subject_field"))

    if dataset != _OUTPUT_DATASET:
        errors.append(f"row_key_source.dataset must be {_OUTPUT_DATASET}")
    if namespace != "construct_subject":
        errors.append("row_key_source.namespace must be construct_subject")
    if column != "construct_subject__id":
        errors.append("row_key_source.column must be construct_subject__id")
    if materialized_by != "construct_output_subject_bridge":
        errors.append("row_key_source.materialized_by must be construct_output_subject_bridge")
    if output_join_field != "construct__input_id":
        errors.append("row_key_source.output_join_field must be construct__input_id")
    if input_dataset != _INPUT_DATASET:
        errors.append(f"row_key_source.input_dataset must be {_INPUT_DATASET}")
    if input_construct_subject_field != "construct_subject__id":
        errors.append("row_key_source.input_construct_subject_field must be construct_subject__id")
    return {
        "dataset": dataset,
        "namespace": namespace,
        "column": column,
        "materialized_by": materialized_by,
        "output_join_field": output_join_field,
        "input_dataset": input_dataset,
        "input_construct_subject_field": input_construct_subject_field,
    }


def _mapping(value: object, *, label: str, errors: list[str]) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    errors.append(f"{label} must be a mapping")
    return {}


def _list(value: object, *, label: str, errors: list[str]) -> list[object]:
    if isinstance(value, list):
        return value
    errors.append(f"{label} must be a list")
    return []


def _string(value: object) -> str:
    return str(value or "").strip()


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")
