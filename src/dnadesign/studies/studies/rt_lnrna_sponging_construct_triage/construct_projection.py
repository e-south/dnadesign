"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/construct_projection.py

Construct projection manifest validation for the RT-lnRNA sponging construct
triage study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_DEFAULT_MANIFEST_PATH = _STUDY_DIR / "operations/contract/fixtures/construct/construct-projection-manifest.yaml"
_EXPECTED_STRATEGY_ID = "construct_multi_slot_assembly_v1"
_EXPECTED_CONSTRUCT_CONTRACT = "dual_cassette_rt_lnrna_expression_v1"
_EXPECTED_REPRESENTATION_CONTRACT = "dual_cassette_construct_context_embedding_v1"
_EXPECTED_TARGET_LENGTH_NT = 1600
_EXPECTED_TARGET_START_0 = 56
_EXPECTED_TARGET_END_0 = _EXPECTED_TARGET_START_0 + _EXPECTED_TARGET_LENGTH_NT
_EXPECTED_TARGET_CONTEXT_SOURCE_ID = "genbank:1600bp-region.gb#record"
_EXPECTED_TARGET_CONTEXT_AUTHORITY_ID = "dual_cassette_1600bp_region"
_EXPECTED_REQUIRED_SLOTS = ("lnrna", "rt_cds")
_EXPECTED_SLOT_CONTRACTS = {
    "lnrna": {
        "role": "lnrna_cassette",
        "sequence_field": "candidate__lnrna_sequence",
    },
    "rt_cds": {
        "role": "rt_cds",
        "sequence_field": "candidate__rt_cds_sequence",
    },
}
_REQUIRED_VIEW_NAMES = (
    "dual_cassette_1600bp_seq_mean",
    "dual_cassette_1600bp_fwd_rc_concat",
    "lnrna_span_in_construct_anchor_mean",
    "lnrna_span_in_construct_reverse_complement_anchor_mean",
    "rt_cds_span_in_construct_anchor_mean",
    "rt_cds_span_in_construct_reverse_complement_anchor_mean",
)
_ANCHOR_VIEW_CONTRACTS = {
    "lnrna_span_in_construct_anchor_mean": {
        "orientation": "forward",
        "pooling_slot": "lnrna",
    },
    "lnrna_span_in_construct_reverse_complement_anchor_mean": {
        "orientation": "reverse_complement",
        "pooling_slot": "lnrna",
    },
    "rt_cds_span_in_construct_anchor_mean": {
        "orientation": "forward",
        "pooling_slot": "rt_cds",
    },
    "rt_cds_span_in_construct_reverse_complement_anchor_mean": {
        "orientation": "reverse_complement",
        "pooling_slot": "rt_cds",
    },
}


@dataclass(frozen=True)
class ProjectionManifestAudit:
    errors: tuple[str, ...]
    strategy_id: str = ""
    construct_template_id: str = ""
    required_view_names: tuple[str, ...] = ()
    candidate_count: int = 0
    candidate_spans: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class _ProjectionSlot:
    slot_id: str
    role: str
    sequence_field: str
    template_start_0: int
    template_end_0: int
    required: bool

    @property
    def template_length(self) -> int:
        return self.template_end_0 - self.template_start_0


def validate_registered_projection_manifest(*, repo_root: Path | None = None) -> ProjectionManifestAudit:
    root = _resolve_repo_root(repo_root)
    path = root / _DEFAULT_MANIFEST_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_projection_manifest_payload(payload)


def validate_projection_manifest_payload(payload: object) -> ProjectionManifestAudit:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ProjectionManifestAudit(errors=("projection manifest must be a mapping",))

    strategy = _mapping(payload.get("strategy"), label="strategy", errors=errors)
    strategy_id = _string(strategy.get("strategy_id")) if strategy is not None else ""
    if strategy_id != _EXPECTED_STRATEGY_ID:
        errors.append(f"strategy.strategy_id must be {_EXPECTED_STRATEGY_ID}")
    runtime_contract = _mapping(
        strategy.get("construct_runtime_contract") if strategy is not None else None,
        label="strategy.construct_runtime_contract",
        errors=errors,
    )
    if runtime_contract is not None:
        _validate_runtime_contract(runtime_contract, errors=errors)

    if _string(payload.get("construct_contract")) != _EXPECTED_CONSTRUCT_CONTRACT:
        errors.append(f"construct_contract must be {_EXPECTED_CONSTRUCT_CONTRACT}")
    if _string(payload.get("representation_contract")) != _EXPECTED_REPRESENTATION_CONTRACT:
        errors.append(f"representation_contract must be {_EXPECTED_REPRESENTATION_CONTRACT}")

    template = _mapping(payload.get("construct_template"), label="construct_template", errors=errors)
    template_id = _string(template.get("construct_template_id")) if template is not None else ""
    target_context = _mapping(
        template.get("target_context") if template is not None else None,
        label="construct_template.target_context",
        errors=errors,
    )
    if template is not None:
        _validate_template(template, errors=errors)
    target_length = (
        _positive_int(target_context.get("length_nt"), label="target_context.length_nt", errors=errors)
        if target_context is not None
        else None
    )
    target_start = _nonnegative_int(target_context.get("window_start_0")) if target_context is not None else None
    target_end = _nonnegative_int(target_context.get("window_end_0")) if target_context is not None else None
    if target_length is not None and target_length != _EXPECTED_TARGET_LENGTH_NT:
        errors.append(f"target_context.length_nt must be {_EXPECTED_TARGET_LENGTH_NT}")

    slots = _parse_slots(payload.get("slots"), errors=errors)
    required_slots = tuple(slot.slot_id for slot in slots if slot.required)
    if required_slots != _EXPECTED_REQUIRED_SLOTS:
        errors.append("slots must declare required lnrna and rt_cds slots in construct order")

    view_names = _parse_view_names(payload.get("representation_views"), errors=errors)
    if view_names != _REQUIRED_VIEW_NAMES:
        errors.append("representation_views must declare the six v1 RT-lnRNA construct views in order")

    candidate_spans: dict[str, dict[str, tuple[int, int]]] = {}
    candidates = _list(payload.get("candidates"), label="candidates", errors=errors)
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            errors.append(f"candidates[{index}] must be a mapping")
            continue
        candidate_id = _string(candidate.get("candidate_id"))
        if not candidate_id:
            errors.append(f"candidates[{index}].candidate_id is required")
            continue
        if _string(candidate.get("construct_projection_status")) != "representable":
            errors.append(f"{candidate_id}: construct_projection_status must be representable for this fixture")
        slot_bindings = _mapping(candidate.get("slot_bindings"), label=f"{candidate_id}.slot_bindings", errors=errors)
        declared_spans = _mapping(
            candidate.get("emitted_slot_spans_0"),
            label=f"{candidate_id}.emitted_slot_spans_0",
            errors=errors,
        )
        view_declarations = tuple(
            str(value)
            for value in _list(
                candidate.get("construct_context_view_declarations"),
                label=f"{candidate_id}.construct_context_view_declarations",
                errors=errors,
            )
        )
        if view_declarations != _REQUIRED_VIEW_NAMES:
            errors.append(f"{candidate_id}: construct_context_view_declarations must match representation_views")
        if slot_bindings is None or declared_spans is None:
            continue
        _validate_candidate_slot_bindings(
            candidate_id=candidate_id,
            slots=slots,
            slot_bindings=slot_bindings,
            errors=errors,
        )
        computed = _computed_candidate_spans(
            candidate_id=candidate_id,
            slots=slots,
            slot_bindings=slot_bindings,
            target_length=target_length,
            target_start=target_start,
            target_end=target_end,
            errors=errors,
        )
        if not computed:
            continue
        candidate_spans[candidate_id] = computed
        for slot_id, span in computed.items():
            declared = _span_0(declared_spans.get(slot_id), label=f"{candidate_id}.emitted_slot_spans_0.{slot_id}")
            if declared is None:
                errors.append(f"{candidate_id}: emitted_slot_spans_0.{slot_id} must be [start, end]")
                continue
            if declared != span:
                errors.append(f"{candidate_id}: emitted_slot_spans_0.{slot_id} is {declared}, expected {span}")

    return ProjectionManifestAudit(
        errors=tuple(errors),
        strategy_id=strategy_id,
        construct_template_id=template_id,
        required_view_names=view_names,
        candidate_count=len(candidates),
        candidate_spans=candidate_spans,
    )


def _validate_runtime_contract(runtime_contract: dict[str, object], *, errors: list[str]) -> None:
    if _string(runtime_contract.get("mode")) != "realize_template":
        errors.append("strategy.construct_runtime_contract.mode must be realize_template")
    if runtime_contract.get("input_primary_field") is not None:
        errors.append("strategy.construct_runtime_contract.input_primary_field must be null for multi-slot rows")
    required_slots = tuple(
        str(value)
        for value in _list(
            runtime_contract.get("required_slots"),
            label="strategy.construct_runtime_contract.required_slots",
            errors=errors,
        )
    )
    if required_slots != _EXPECTED_REQUIRED_SLOTS:
        errors.append("strategy.construct_runtime_contract.required_slots must be lnrna, rt_cds")
    if runtime_contract.get("parts_are_named_slots") is not True:
        errors.append("strategy.construct_runtime_contract.parts_are_named_slots must be true")


def _validate_template(template: dict[str, object], *, errors: list[str]) -> None:
    for field_name in (
        "construct_template_id",
        "plasmid_context_source_id",
        "target_context_source_id",
        "source_authority_id",
        "target_context_source_authority_id",
    ):
        if not _string(template.get(field_name)):
            errors.append(f"construct_template.{field_name} is required")
    if _string(template.get("target_context_source_id")) != _EXPECTED_TARGET_CONTEXT_SOURCE_ID:
        errors.append(f"construct_template.target_context_source_id must be {_EXPECTED_TARGET_CONTEXT_SOURCE_ID}")
    if _string(template.get("target_context_source_authority_id")) != _EXPECTED_TARGET_CONTEXT_AUTHORITY_ID:
        errors.append(
            f"construct_template.target_context_source_authority_id must be {_EXPECTED_TARGET_CONTEXT_AUTHORITY_ID}"
        )
    target_context = _mapping(
        template.get("target_context"),
        label="construct_template.target_context",
        errors=errors,
    )
    if target_context is None:
        return
    if _string(target_context.get("context_id")) != "dual_cassette_1600bp_context_v1":
        errors.append("construct_template.target_context.context_id must be dual_cassette_1600bp_context_v1")
    if _string(target_context.get("coordinate_basis")) != "zero_based_half_open":
        errors.append("construct_template.target_context.coordinate_basis must be zero_based_half_open")
    if _nonnegative_int(target_context.get("window_start_0")) != _EXPECTED_TARGET_START_0:
        errors.append(f"construct_template.target_context.window_start_0 must be {_EXPECTED_TARGET_START_0}")
    if _nonnegative_int(target_context.get("window_end_0")) != _EXPECTED_TARGET_END_0:
        errors.append(f"construct_template.target_context.window_end_0 must be {_EXPECTED_TARGET_END_0}")
    if _string(target_context.get("padding_policy")) != "real_plasmid_sequence_only":
        errors.append("construct_template.target_context.padding_policy must be real_plasmid_sequence_only")
    if _string(target_context.get("truncation_policy")) != "fail":
        errors.append("construct_template.target_context.truncation_policy must be fail")


def _validate_candidate_slot_bindings(
    *,
    candidate_id: str,
    slots: tuple[_ProjectionSlot, ...],
    slot_bindings: dict[str, object],
    errors: list[str],
) -> None:
    for slot in slots:
        binding = _mapping(
            slot_bindings.get(slot.slot_id),
            label=f"{candidate_id}.slot_bindings.{slot.slot_id}",
            errors=errors,
        )
        if binding is None:
            continue
        if not _string(binding.get("sequence_id")):
            errors.append(f"{candidate_id}.slot_bindings.{slot.slot_id}.sequence_id is required")
        sequence_length = _positive_int(
            binding.get("sequence_length_nt"),
            label=f"{candidate_id}.slot_bindings.{slot.slot_id}.sequence_length_nt",
            errors=errors,
        )
        source_span = _span_0(
            binding.get("source_sequence_span_0"),
            label=f"{candidate_id}.slot_bindings.{slot.slot_id}.source_sequence_span_0",
        )
        if source_span is None:
            errors.append(f"{candidate_id}.slot_bindings.{slot.slot_id}.source_sequence_span_0 must be [start, end]")
        elif sequence_length is not None and source_span[1] - source_span[0] != sequence_length:
            errors.append(
                f"{candidate_id}.slot_bindings.{slot.slot_id}.source_sequence_span_0 length "
                f"must equal sequence_length_nt"
            )


def _computed_candidate_spans(
    *,
    candidate_id: str,
    slots: tuple[_ProjectionSlot, ...],
    slot_bindings: dict[str, object],
    target_length: int | None,
    target_start: int | None,
    target_end: int | None,
    errors: list[str],
) -> dict[str, tuple[int, int]]:
    cursor = 0
    out_len = 0
    full_spans: dict[str, tuple[int, int]] = {}
    for slot in slots:
        binding = _mapping(
            slot_bindings.get(slot.slot_id),
            label=f"{candidate_id}.slot_bindings.{slot.slot_id}",
            errors=errors,
        )
        if binding is None:
            continue
        sequence_length = _positive_int(
            binding.get("sequence_length_nt"),
            label=f"{candidate_id}.slot_bindings.{slot.slot_id}.sequence_length_nt",
            errors=errors,
        )
        if sequence_length is None:
            continue
        if slot.template_start_0 < cursor:
            errors.append(f"{candidate_id}: slot {slot.slot_id} overlaps a prior template slot")
            continue
        out_len += slot.template_start_0 - cursor
        start = out_len
        out_len += sequence_length
        end = out_len
        cursor = slot.template_end_0
        full_spans[slot.slot_id] = (start, end)
    if target_length is None or target_start is None or target_end is None:
        return {}
    if target_end - target_start != target_length:
        errors.append(f"{candidate_id}: target context span must equal target_context.length_nt")
        return {}
    window_start = _candidate_window_start(slots=slots, full_spans=full_spans, target_start=target_start)
    spans: dict[str, tuple[int, int]] = {}
    for slot_id, (start, end) in full_spans.items():
        relative = (start - window_start, end - window_start)
        spans[slot_id] = relative
        if relative[0] < 0 or relative[1] > target_length:
            errors.append(
                f"{candidate_id}: required slot {slot_id} resolves to {relative}, "
                f"outside target context length {target_length}"
            )
    return spans


def _candidate_window_start(
    *,
    slots: tuple[_ProjectionSlot, ...],
    full_spans: dict[str, tuple[int, int]],
    target_start: int,
) -> int:
    lnrna_slot = next(slot for slot in slots if slot.slot_id == "lnrna")
    full_lnrna_start, full_lnrna_end = full_spans["lnrna"]
    base_center = lnrna_slot.template_start_0 + (lnrna_slot.template_length // 2)
    full_center = full_lnrna_start + ((full_lnrna_end - full_lnrna_start) // 2)
    return target_start + (full_center - base_center)


def _parse_slots(payload: object, *, errors: list[str]) -> tuple[_ProjectionSlot, ...]:
    slots: list[_ProjectionSlot] = []
    seen: set[str] = set()
    for index, item in enumerate(_list(payload, label="slots", errors=errors), start=1):
        if not isinstance(item, dict):
            errors.append(f"slots[{index}] must be a mapping")
            continue
        slot_id = _string(item.get("slot_id"))
        role = _string(item.get("role"))
        sequence_field = _string(item.get("sequence_field"))
        span = _span_0(item.get("template_span_0"), label=f"slots[{index}].template_span_0")
        if not slot_id or not role or not sequence_field or span is None:
            errors.append(f"slots[{index}] must define slot_id, role, sequence_field, and template_span_0")
            continue
        expected = _EXPECTED_SLOT_CONTRACTS.get(slot_id)
        if expected is not None and (role != expected["role"] or sequence_field != expected["sequence_field"]):
            errors.append(
                f"slot {slot_id} must use role={expected['role']} and sequence_field={expected['sequence_field']}"
            )
        if _string(item.get("placement_kind")) != "replace":
            errors.append(f"slots[{index}].placement_kind must be replace")
        if _string(item.get("orientation")) != "forward":
            errors.append(f"slots[{index}].orientation must be forward")
        guard = _mapping(item.get("guard"), label=f"slots[{index}].guard", errors=errors)
        if guard is not None:
            if not _string(guard.get("source_feature")):
                errors.append(f"slots[{index}].guard.source_feature is required")
            guard_span = _positive_int(
                guard.get("replaced_span_bp"),
                label=f"slots[{index}].guard.replaced_span_bp",
                errors=errors,
            )
            if guard_span is not None and guard_span != span[1] - span[0]:
                errors.append(f"slots[{index}].guard.replaced_span_bp must equal template_span_0 length")
        if slot_id in seen:
            errors.append(f"duplicate slot_id {slot_id!r}")
            continue
        seen.add(slot_id)
        slots.append(
            _ProjectionSlot(
                slot_id=slot_id,
                role=role,
                sequence_field=sequence_field,
                template_start_0=span[0],
                template_end_0=span[1],
                required=bool(item.get("required")),
            )
        )
    slots.sort(key=lambda slot: slot.template_start_0)
    return tuple(slots)


def _parse_view_names(payload: object, *, errors: list[str]) -> tuple[str, ...]:
    names: list[str] = []
    for index, item in enumerate(_list(payload, label="representation_views", errors=errors), start=1):
        if not isinstance(item, dict):
            errors.append(f"representation_views[{index}] must be a mapping")
            continue
        view_name = _string(item.get("view_name"))
        if not view_name:
            errors.append(f"representation_views[{index}].view_name is required")
            continue
        if _string(item.get("product_kind")) != "realized_context":
            errors.append(f"{view_name}: product_kind must be realized_context")
        _validate_view_shape(view_name=view_name, view=item, errors=errors)
        names.append(view_name)
    return tuple(names)


def _validate_view_shape(*, view_name: str, view: dict[str, object], errors: list[str]) -> None:
    if view_name == "dual_cassette_1600bp_seq_mean":
        if _string(view.get("orientation")) != "forward":
            errors.append(f"{view_name}: orientation must be forward")
        if _string(view.get("pooling_operation")) != "seq_mean":
            errors.append(f"{view_name}: pooling_operation must be seq_mean")
        if _string(view.get("construct_output_anchor_part")):
            errors.append(f"{view_name}: construct_output_anchor_part must be empty")
    if view_name == "dual_cassette_1600bp_fwd_rc_concat":
        required_orientations = tuple(
            str(value)
            for value in _list(
                view.get("required_orientations"),
                label=f"{view_name}.required_orientations",
                errors=errors,
            )
        )
        if required_orientations != ("forward", "reverse_complement"):
            errors.append(f"{view_name}: required_orientations must be forward, reverse_complement")
        if _string(view.get("downstream_transform")) != "block_normalized_concatenate":
            errors.append(f"{view_name}: downstream_transform must be block_normalized_concatenate")
        if _string(view.get("construct_output_anchor_part")):
            errors.append(f"{view_name}: construct_output_anchor_part must be empty")
    if view_name in _ANCHOR_VIEW_CONTRACTS:
        _validate_anchor_view(view_name=view_name, view=view, errors=errors)
    required_slots = tuple(
        str(value)
        for value in _list(
            view.get("required_slots"),
            label=f"{view_name}.required_slots",
            errors=errors,
        )
    )
    if required_slots != _EXPECTED_REQUIRED_SLOTS:
        errors.append(f"{view_name}: required_slots must be lnrna, rt_cds")


def _validate_anchor_view(*, view_name: str, view: dict[str, object], errors: list[str]) -> None:
    contract = _ANCHOR_VIEW_CONTRACTS[view_name]
    expected_orientation = contract["orientation"]
    expected_slot = contract["pooling_slot"]
    if _string(view.get("orientation")) != expected_orientation:
        errors.append(f"{view_name}: orientation must be {expected_orientation}")
    if _string(view.get("pooling_operation")) != "anchor_mean":
        errors.append(f"{view_name}: pooling_operation must be anchor_mean")
    if _string(view.get("pooling_slot")) != expected_slot:
        errors.append(f"{view_name}: pooling_slot must be {expected_slot}")
    if _string(view.get("construct_output_anchor_part")) != expected_slot:
        errors.append(f"{view_name}: construct_output_anchor_part must be {expected_slot}")


def _mapping(value: object, *, label: str, errors: list[str]) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    errors.append(f"{label} must be a mapping")
    return None


def _list(value: object, *, label: str, errors: list[str]) -> list[object]:
    if isinstance(value, list):
        return value
    errors.append(f"{label} must be a list")
    return []


def _span_0(value: object, *, label: str) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return start, end


def _positive_int(value: object, *, label: str, errors: list[str]) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a positive integer")
        return None
    if integer <= 0:
        errors.append(f"{label} must be a positive integer")
        return None
    return integer


def _nonnegative_int(value: object) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer < 0:
        return None
    return integer


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
