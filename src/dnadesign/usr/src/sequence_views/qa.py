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

import pyarrow.parquet as pq

from ..contracts import SchemaError
from .store import sequence_views_path

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

    product_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    pooling_counts: Counter[str] = Counter()

    errors: list[str] = []
    invalid_bounds = 0
    total_views = 0
    required_bound_pooling = set(expectation.require_bounds_for_pooling)

    path = sequence_views_path(dataset)
    if path.exists():
        for batch in pq.ParquetFile(path).iter_batches(
            columns=[
                "view_id",
                "sequence_id",
                "product_kind",
                "context_kind",
                "orientation",
                "anchor_start_0",
                "anchor_end_0",
                "forward_anchor_start_0",
                "forward_anchor_end_0",
                "recommended_pooling",
            ],
            batch_size=65_536,
        ):
            payload = batch.to_pydict()
            total_views += batch.num_rows
            for row_index in range(batch.num_rows):
                view_id = str(payload["view_id"][row_index])
                sequence_id = str(payload["sequence_id"][row_index])
                product_kind = str(payload["product_kind"][row_index])
                context_kind = payload["context_kind"][row_index]
                orientation = str(payload["orientation"][row_index])
                recommended_pooling = payload["recommended_pooling"][row_index]
                anchor_start_0 = payload["anchor_start_0"][row_index]
                anchor_end_0 = payload["anchor_end_0"][row_index]
                forward_anchor_start_0 = payload["forward_anchor_start_0"][row_index]
                forward_anchor_end_0 = payload["forward_anchor_end_0"][row_index]

                product_counts[product_kind] += 1
                orientation_counts[orientation] += 1
                if context_kind is not None:
                    context_counts[str(context_kind)] += 1
                if recommended_pooling is not None:
                    pooling_counts[str(recommended_pooling)] += 1

                length = lengths_by_id.get(sequence_id)
                if length is None:
                    errors.append(f"view_id={view_id} references missing sequence_id={sequence_id}")
                    continue
                if recommended_pooling in required_bound_pooling and (anchor_start_0 is None or anchor_end_0 is None):
                    invalid_bounds += 1
                    errors.append(f"view_id={view_id} pooling={recommended_pooling} is missing anchor bounds")
                for label, start, end in (
                    ("anchor", anchor_start_0, anchor_end_0),
                    ("forward_anchor", forward_anchor_start_0, forward_anchor_end_0),
                ):
                    if start is None and end is None:
                        continue
                    if start is None or end is None or int(start) < 0 or int(end) <= int(start) or int(end) > length:
                        invalid_bounds += 1
                        errors.append(
                            f"view_id={view_id} has invalid {label} bounds {start}:{end} for sequence length {length}"
                        )
                expected_length = expectation.exact_lengths_by_product_kind.get(product_kind)
                if expected_length is not None and length != int(expected_length):
                    errors.append(
                        f"view_id={view_id} product_kind={product_kind} expected length "
                        f"{expected_length}, observed {length}"
                    )

    if expectation.total_records is not None and len(lengths_by_id) != expectation.total_records:
        errors.append(f"expected {expectation.total_records} records, observed {len(lengths_by_id)}")
    if expectation.total_views is not None and total_views != expectation.total_views:
        errors.append(f"expected {expectation.total_views} sequence views, observed {total_views}")

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

    report = SequenceViewContractReport(
        dataset=dataset.name,
        total_records=len(lengths_by_id),
        total_views=total_views,
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
