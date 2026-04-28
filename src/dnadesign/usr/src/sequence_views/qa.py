"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sequence_views/qa.py

Dataset-local QA helpers for sequence-view sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..contracts import SchemaError
from .store import load_sequence_views

if TYPE_CHECKING:
    from ..dataset import Dataset


@dataclass(frozen=True)
class SequenceViewContractExpectation:
    total_records: int | None = None
    total_views: int | None = None
    counts_by_product_kind: dict[str, int] = field(default_factory=dict)
    counts_by_orientation: dict[str, int] = field(default_factory=dict)
    counts_by_context_kind: dict[str, int] = field(default_factory=dict)
    counts_by_recommended_pooling: dict[str, int] = field(default_factory=dict)
    exact_lengths_by_product_kind: dict[str, int] = field(default_factory=dict)
    require_bounds_for_pooling: tuple[str, ...] = ("anchor_mean",)


@dataclass(frozen=True)
class SequenceViewContractReport:
    dataset: str
    total_records: int
    total_views: int
    counts_by_product_kind: dict[str, int]
    counts_by_orientation: dict[str, int]
    counts_by_context_kind: dict[str, int]
    counts_by_recommended_pooling: dict[str, int]
    invalid_bounds: int
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def _record_lengths(dataset: Dataset) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for batch in dataset.scan(columns=["id", "length"], include_overlays=False):
        ids = batch.column(batch.schema.get_field_index("id")).to_pylist()
        row_lengths = batch.column(batch.schema.get_field_index("length")).to_pylist()
        for row_id, length in zip(ids, row_lengths, strict=True):
            if row_id is None:
                continue
            lengths[str(row_id)] = int(length)
    return lengths


def validate_sequence_view_contract(
    dataset: Dataset,
    *,
    expectation: SequenceViewContractExpectation | None = None,
    raise_on_error: bool = True,
) -> SequenceViewContractReport:
    expectation = expectation or SequenceViewContractExpectation()
    lengths_by_id = _record_lengths(dataset)
    views = load_sequence_views(dataset)

    product_counts = Counter(str(view.product_kind) for view in views)
    orientation_counts = Counter(str(view.orientation) for view in views)
    context_counts = Counter(str(view.context_kind) for view in views if view.context_kind is not None)
    pooling_counts = Counter(str(view.recommended_pooling) for view in views if view.recommended_pooling is not None)

    errors: list[str] = []
    invalid_bounds = 0

    if expectation.total_records is not None and len(lengths_by_id) != expectation.total_records:
        errors.append(f"expected {expectation.total_records} records, observed {len(lengths_by_id)}")
    if expectation.total_views is not None and len(views) != expectation.total_views:
        errors.append(f"expected {expectation.total_views} sequence views, observed {len(views)}")

    for label, expected_counts, observed_counts in (
        ("product_kind", expectation.counts_by_product_kind, product_counts),
        ("orientation", expectation.counts_by_orientation, orientation_counts),
        ("context_kind", expectation.counts_by_context_kind, context_counts),
        ("recommended_pooling", expectation.counts_by_recommended_pooling, pooling_counts),
    ):
        for key, expected in expected_counts.items():
            observed = int(observed_counts.get(key, 0))
            if observed != int(expected):
                errors.append(f"expected {label}={key} count {expected}, observed {observed}")

    required_bound_pooling = set(expectation.require_bounds_for_pooling)
    for view in views:
        length = lengths_by_id.get(view.sequence_id)
        if length is None:
            errors.append(f"view_id={view.view_id} references missing sequence_id={view.sequence_id}")
            continue
        if view.recommended_pooling in required_bound_pooling and (
            view.anchor_start_0 is None or view.anchor_end_0 is None
        ):
            invalid_bounds += 1
            errors.append(f"view_id={view.view_id} pooling={view.recommended_pooling} is missing anchor bounds")
        for label, start, end in (
            ("anchor", view.anchor_start_0, view.anchor_end_0),
            ("forward_anchor", view.forward_anchor_start_0, view.forward_anchor_end_0),
        ):
            if start is None and end is None:
                continue
            if start is None or end is None or start < 0 or end <= start or end > length:
                invalid_bounds += 1
                errors.append(
                    f"view_id={view.view_id} has invalid {label} bounds {start}:{end} for sequence length {length}"
                )
        expected_length = expectation.exact_lengths_by_product_kind.get(str(view.product_kind))
        if expected_length is not None and length != int(expected_length):
            errors.append(
                f"view_id={view.view_id} product_kind={view.product_kind} expected length "
                f"{expected_length}, observed {length}"
            )

    report = SequenceViewContractReport(
        dataset=dataset.name,
        total_records=len(lengths_by_id),
        total_views=len(views),
        counts_by_product_kind=dict(sorted(product_counts.items())),
        counts_by_orientation=dict(sorted(orientation_counts.items())),
        counts_by_context_kind=dict(sorted(context_counts.items())),
        counts_by_recommended_pooling=dict(sorted(pooling_counts.items())),
        invalid_bounds=invalid_bounds,
        errors=tuple(errors),
    )
    if raise_on_error and errors:
        preview = "; ".join(errors[:5])
        suffix = "" if len(errors) <= 5 else f"; ... {len(errors) - 5} more"
        raise SchemaError(f"Sequence-view contract validation failed for dataset '{dataset.name}': {preview}{suffix}")
    return report
