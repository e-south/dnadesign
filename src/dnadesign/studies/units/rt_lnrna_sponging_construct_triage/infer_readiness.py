"""
Infer-readiness checks for RT-lnRNA Construct materializations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from dnadesign.usr import Dataset, load_sequence_views

from .representation_contract import REQUIRED_SOURCE_VIEW_NAMES

REQUIRED_INFER_READY_VIEW_NAMES = REQUIRED_SOURCE_VIEW_NAMES
_EXPECTED_OUTPUT_ORIENTATIONS = ("forward", "reverse_complement")
_EXPECTED_CONTEXT_LENGTH_NT = 2000
_VIEW_EXPECTATIONS = {
    "dual_cassette_2000bp_seq_mean": ("forward", "seq_mean", False),
    "dual_cassette_2000bp_reverse_complement_seq_mean": ("reverse_complement", "seq_mean", False),
    "lnrna_span_in_construct_anchor_mean": ("forward", "anchor_mean", True),
    "lnrna_span_in_construct_reverse_complement_anchor_mean": ("reverse_complement", "anchor_mean", True),
    "rt_cds_span_in_construct_anchor_mean": ("forward", "anchor_mean", True),
    "rt_cds_span_in_construct_reverse_complement_anchor_mean": ("reverse_complement", "anchor_mean", True),
}


class ConstructInferReadinessError(ValueError):
    """Raised when materialized RT-lnRNA Construct outputs are not Infer-ready."""


@dataclass(frozen=True)
class ConstructInferReadinessAudit:
    errors: tuple[str, ...]
    input_count: int
    output_count: int
    sequence_view_count: int
    construct_subject_count: int
    view_names: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_construct_infer_readiness(
    *,
    usr_root: Path,
    input_dataset: str,
    output_dataset: str,
    expected_construct_subject_ids: Iterable[str] | None = None,
) -> ConstructInferReadinessAudit:
    """Audit the study-owned Construct output shape required by Infer.

    This validates the concrete USR datasets, not a config fixture. Each
    construct subject must have two realized output rows and the six explicit
    sequence-view names consumed by the RT-lnRNA Infer feature bundle.
    """

    input_ds = Dataset(Path(usr_root), input_dataset)
    output_ds = Dataset(Path(usr_root), output_dataset)
    input_rows = _dataset_rows(input_ds)
    output_rows = _dataset_rows(output_ds)
    views = load_sequence_views(output_ds)
    errors: list[str] = []

    input_by_id, subject_by_input_id, input_subject_ids = _audit_inputs(input_rows, errors=errors)
    expected_subject_ids = {str(value) for value in expected_construct_subject_ids or ()}
    if expected_subject_ids:
        missing = sorted(expected_subject_ids - input_subject_ids)
        unexpected = sorted(input_subject_ids - expected_subject_ids)
        if missing:
            errors.append("Construct input dataset is missing expected construct subject(s): " + ", ".join(missing[:5]))
        if unexpected:
            errors.append("Construct input dataset has unexpected construct subject(s): " + ", ".join(unexpected[:5]))

    output_by_id, output_ids_by_subject = _audit_outputs(
        output_rows,
        input_by_id=input_by_id,
        subject_by_input_id=subject_by_input_id,
        expected_subject_ids=input_subject_ids,
        errors=errors,
    )
    view_names = _audit_sequence_views(
        views,
        output_dataset=output_dataset,
        output_by_id=output_by_id,
        output_ids_by_subject=output_ids_by_subject,
        subject_by_input_id=subject_by_input_id,
        expected_subject_ids=input_subject_ids,
        errors=errors,
    )

    expected_output_count = len(input_subject_ids) * 2
    if len(output_rows) != expected_output_count:
        errors.append(f"Construct output row count {len(output_rows)} must equal 2 per construct subject.")
    expected_view_count = len(input_subject_ids) * len(REQUIRED_INFER_READY_VIEW_NAMES)
    if len(views) != expected_view_count:
        errors.append(f"Sequence view row count {len(views)} must equal 6 per construct subject.")

    return ConstructInferReadinessAudit(
        errors=tuple(errors),
        input_count=len(input_rows),
        output_count=len(output_rows),
        sequence_view_count=len(views),
        construct_subject_count=len(input_subject_ids),
        view_names=view_names,
    )


def require_construct_infer_readiness(
    *,
    usr_root: Path,
    input_dataset: str,
    output_dataset: str,
    expected_construct_subject_ids: Iterable[str] | None = None,
) -> ConstructInferReadinessAudit:
    audit = validate_construct_infer_readiness(
        usr_root=usr_root,
        input_dataset=input_dataset,
        output_dataset=output_dataset,
        expected_construct_subject_ids=expected_construct_subject_ids,
    )
    if audit.ok:
        return audit
    preview = "; ".join(audit.errors[:8])
    raise ConstructInferReadinessError(f"RT-lnRNA Construct output is not Infer-ready: {preview}")


def _dataset_rows(dataset: Dataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in dataset.scan(include_overlays=True):
        rows.extend(dict(row) for row in batch.to_pylist())
    return rows


def _audit_inputs(
    rows: list[dict[str, object]],
    *,
    errors: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, str], set[str]]:
    input_by_id: dict[str, dict[str, object]] = {}
    subject_by_input_id: dict[str, str] = {}
    subject_ids: list[str] = []
    if not rows:
        errors.append("Construct input dataset has no rows.")
        return input_by_id, subject_by_input_id, set()
    for row in rows:
        input_id = _required_text(row, "id", errors=errors)
        subject_id = _required_text(row, "construct_subject__id", errors=errors)
        if not input_id or not subject_id:
            continue
        if input_id in input_by_id:
            errors.append(f"Construct input dataset has duplicate id: {input_id}")
        input_by_id[input_id] = row
        subject_by_input_id[input_id] = subject_id
        subject_ids.append(subject_id)
        if _text(row.get("construct_subject__record_kind")) != "construct_subject_envelope":
            errors.append(f"{subject_id}: input construct_subject__record_kind must be construct_subject_envelope.")
        if _text(row.get("construct_subject__sequence_authority")) != "overlay_only":
            errors.append(f"{subject_id}: input construct_subject__sequence_authority must be overlay_only.")
        _require_dna_field(row, "construct_subject__lnrna_sequence", subject_id=subject_id, errors=errors)
        _require_dna_field(row, "construct_subject__rt_cds_sequence", subject_id=subject_id, errors=errors)
    duplicates = sorted(value for value, count in Counter(subject_ids).items() if count > 1)
    if duplicates:
        errors.append(
            "Construct input dataset has duplicate construct_subject__id value(s): " + ", ".join(duplicates[:5])
        )
    return input_by_id, subject_by_input_id, set(subject_ids)


def _audit_outputs(
    rows: list[dict[str, object]],
    *,
    input_by_id: Mapping[str, Mapping[str, object]],
    subject_by_input_id: Mapping[str, str],
    expected_subject_ids: set[str],
    errors: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    output_by_id: dict[str, dict[str, object]] = {}
    output_ids_by_subject: dict[str, list[str]] = {subject_id: [] for subject_id in expected_subject_ids}
    if not rows:
        errors.append("Construct output dataset has no rows.")
        return output_by_id, output_ids_by_subject
    for row in rows:
        output_id = _required_text(row, "id", errors=errors)
        input_id = _required_text(row, "construct__input_id", errors=errors)
        if output_id:
            if output_id in output_by_id:
                errors.append(f"Construct output dataset has duplicate id: {output_id}")
            output_by_id[output_id] = row
        if not output_id or not input_id:
            continue
        expected_subject_id = subject_by_input_id.get(input_id)
        if expected_subject_id is None:
            errors.append(f"{output_id}: construct__input_id does not resolve to a selected construct subject.")
            continue
        output_subject_id = _text(row.get("construct_subject__id"))
        if output_subject_id != expected_subject_id:
            errors.append(
                f"{output_id}: construct_subject__id bridge mismatch; "
                f"expected {expected_subject_id}, found {output_subject_id or '<missing>'}."
            )
        if _text(row.get("construct_subject__record_kind")) != "construct_output":
            errors.append(f"{output_id}: output construct_subject__record_kind must be construct_output.")
        if _text(row.get("construct_subject__sequence_authority")) != "realized_construct_sequence":
            errors.append(
                f"{output_id}: output construct_subject__sequence_authority must be realized_construct_sequence."
            )
        if input_id not in input_by_id:
            errors.append(f"{output_id}: output construct__input_id is absent from input dataset.")
        sequence = _text(row.get("sequence"))
        if len(sequence) != _EXPECTED_CONTEXT_LENGTH_NT:
            errors.append(f"{output_id}: realized context length must be {_EXPECTED_CONTEXT_LENGTH_NT} nt.")
        output_ids_by_subject.setdefault(expected_subject_id, []).append(output_id)

    for subject_id in sorted(expected_subject_ids):
        subject_outputs = output_ids_by_subject.get(subject_id, [])
        if len(subject_outputs) != 2:
            errors.append(f"{subject_id}: must have exactly two realized output rows.")
            continue
        orientations = {_text(output_by_id[output_id].get("construct__orientation")) for output_id in subject_outputs}
        if orientations != set(_EXPECTED_OUTPUT_ORIENTATIONS):
            errors.append(f"{subject_id}: realized output rows must include forward and reverse_complement.")
    return output_by_id, output_ids_by_subject


def _audit_sequence_views(
    views: list[object],
    *,
    output_dataset: str,
    output_by_id: Mapping[str, Mapping[str, object]],
    output_ids_by_subject: Mapping[str, list[str]],
    subject_by_input_id: Mapping[str, str],
    expected_subject_ids: set[str],
    errors: list[str],
) -> tuple[str, ...]:
    view_names_seen: set[str] = set()
    views_by_subject: dict[str, dict[str, list[object]]] = {subject_id: {} for subject_id in expected_subject_ids}
    view_ids = [str(getattr(view, "view_id", "") or "") for view in views]
    duplicate_view_ids = sorted(value for value, count in Counter(view_ids).items() if value and count > 1)
    if duplicate_view_ids:
        errors.append("Sequence views have duplicate view_id value(s): " + ", ".join(duplicate_view_ids[:5]))

    for view in views:
        view_name = _text(getattr(view, "view_name", None))
        sequence_id = _text(getattr(view, "sequence_id", None))
        parent_sequence_id = _text(getattr(view, "parent_sequence_id", None))
        view_names_seen.add(view_name)
        output_row = output_by_id.get(sequence_id)
        if output_row is None:
            errors.append(f"{view_name or '<missing-view-name>'}: sequence_id does not resolve to an output row.")
            continue
        expected_input_id = _text(output_row.get("construct__input_id"))
        if parent_sequence_id != expected_input_id:
            errors.append(f"{view_name}: parent_sequence_id must match the output row construct__input_id.")
        subject_id = subject_by_input_id.get(expected_input_id)
        if subject_id is None:
            errors.append(f"{view_name}: parent_sequence_id does not resolve to a construct subject.")
            continue
        views_by_subject.setdefault(subject_id, {}).setdefault(view_name, []).append(view)
        expected = _VIEW_EXPECTATIONS.get(view_name)
        if expected is None:
            errors.append(f"{view_name or '<missing-view-name>'}: unsupported RT-lnRNA Infer source view.")
            continue
        expected_orientation, expected_pooling, requires_anchor = expected
        if _text(getattr(view, "source_dataset_id", None)) != output_dataset:
            errors.append(f"{view_name}: source_dataset_id must be {output_dataset}.")
        if _text(getattr(view, "product_kind", None)) != "realized_context":
            errors.append(f"{view_name}: product_kind must be realized_context.")
        if _text(getattr(view, "context_kind", None)) != "template_custom":
            errors.append(f"{view_name}: context_kind must be template_custom.")
        if _text(getattr(view, "orientation", None)) != expected_orientation:
            errors.append(f"{view_name}: orientation must be {expected_orientation}.")
        if _text(output_row.get("construct__orientation")) != expected_orientation:
            errors.append(f"{view_name}: view orientation must match the output row orientation.")
        if _text(getattr(view, "recommended_pooling", None)) != expected_pooling:
            errors.append(f"{view_name}: recommended_pooling must be {expected_pooling}.")
        has_anchor = (
            getattr(view, "anchor_start_0", None) is not None and getattr(view, "anchor_end_0", None) is not None
        )
        if requires_anchor and not has_anchor:
            errors.append(f"{view_name}: anchor_mean view must carry anchor_start_0 and anchor_end_0.")

    for subject_id in sorted(expected_subject_ids):
        names_for_subject = views_by_subject.get(subject_id, {})
        missing = [name for name in REQUIRED_INFER_READY_VIEW_NAMES if name not in names_for_subject]
        extra = sorted(set(names_for_subject) - set(REQUIRED_INFER_READY_VIEW_NAMES))
        duplicate_names = sorted(name for name, items in names_for_subject.items() if len(items) > 1)
        if missing:
            errors.append(f"{subject_id}: missing Infer source view(s): " + ", ".join(missing))
        if extra:
            errors.append(f"{subject_id}: unexpected Infer source view(s): " + ", ".join(extra))
        if duplicate_names:
            errors.append(f"{subject_id}: duplicate Infer source view(s): " + ", ".join(duplicate_names))
        output_ids = set(output_ids_by_subject.get(subject_id, []))
        view_output_ids = {
            _text(getattr(view, "sequence_id", None)) for items in names_for_subject.values() for view in items
        }
        if output_ids and not view_output_ids <= output_ids:
            errors.append(f"{subject_id}: sequence views reference output rows outside the construct subject.")

    ordered_seen = [name for name in REQUIRED_INFER_READY_VIEW_NAMES if name in view_names_seen]
    ordered_seen.extend(sorted(name for name in view_names_seen if name not in set(REQUIRED_INFER_READY_VIEW_NAMES)))
    return tuple(ordered_seen)


def _required_text(row: Mapping[str, object], field_name: str, *, errors: list[str]) -> str:
    value = _text(row.get(field_name))
    if not value:
        errors.append(f"Required field is missing or blank: {field_name}")
    return value


def _require_dna_field(
    row: Mapping[str, object],
    field_name: str,
    *,
    subject_id: str,
    errors: list[str],
) -> None:
    value = _text(row.get(field_name))
    if not value:
        errors.append(f"{subject_id}: {field_name} must be non-empty.")
        return
    invalid = sorted(set(value.upper()) - {"A", "C", "G", "T"})
    if invalid:
        errors.append(f"{subject_id}: {field_name} must be DNA4.")


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
