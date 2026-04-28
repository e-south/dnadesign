"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/completion_planner.py

Dry-run completion planning for sequence-view feature bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dnadesign.usr import Dataset

from ..contracts import infer_usr_column_name
from ..runtime.resume_planner import read_usr_columns
from .aliases import load_feature_alias_ids, load_feature_vector_keys
from .contracts import PromoterFeatureBundleConfig, SequenceContextRecord
from .execution import (
    _sequence_view_feature_vector_specs,
    build_feature_metadata_rows,
)
from .selectors import resolve_intermediate_selector
from .sequence_views import load_sequence_view_input_records_with_status, resolve_sequence_view_contexts


@dataclass(frozen=True)
class FeatureCompletionCommands:
    construct_completion: list[str] = field(default_factory=list)
    infer_backfill: list[str] = field(default_factory=list)
    alias_backfill: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureCompletionPlan:
    dataset: str
    bundle_id: str
    model_family: str
    required_views: int
    required_vectors: int
    existing_vectors: int
    reusable_vectors: int
    stale_vectors: int
    missing_vectors: int
    missing_products: int
    persisted_vector_reusable: int
    legacy_digest_reusable: int
    legacy_unclassified_vectors: int
    existing_aliases: int
    by_product_kind: dict[str, int]
    by_orientation: dict[str, int]
    by_pooling_operation: dict[str, int]
    missing_product_selectors: list[dict[str, object]] = field(default_factory=list)
    commands: FeatureCompletionCommands = field(default_factory=FeatureCompletionCommands)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["commands"] = asdict(self.commands)
        return payload


def _dataset_label(
    records: list[dict[str, object]],
    *,
    missing_product_selectors: list[dict[str, object]] | None = None,
) -> str:
    labels = {
        str(record["_infer_source_dataset_id"])
        for record in records
        if str(record.get("_infer_source_dataset_id") or "").strip()
    }
    labels.update(
        str(selector["dataset"])
        for selector in (missing_product_selectors or [])
        if str(selector.get("dataset") or "").strip()
    )
    labels = sorted(labels)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ",".join(labels)


def _group_required_keys_by_dataset(specs: list[dict[str, object]]) -> dict[tuple[str, str], set[str]]:
    keys_by_dataset: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        keys_by_dataset.setdefault(dataset_key, set()).add(str(spec["feature_vector_key"]))
    return keys_by_dataset


def _persisted_feature_vector_keys(specs: list[dict[str, object]]) -> set[str]:
    existing: set[str] = set()
    for (dataset_root, dataset_id), keys in _group_required_keys_by_dataset(specs).items():
        loaded = load_feature_vector_keys(dataset_root=dataset_root, dataset_id=dataset_id, keys=keys)
        existing.update(loaded)
    return existing


def _existing_alias_ids(specs: list[dict[str, object]]) -> set[str]:
    alias_ids: set[str] = set()
    for dataset_root, dataset_id in _group_required_keys_by_dataset(specs):
        alias_ids.update(load_feature_alias_ids(dataset_root=dataset_root, dataset_id=dataset_id))
    return alias_ids


def _legacy_overlay_status(
    *,
    contexts: list[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    specs: list[dict[str, object]],
    model_id: str,
    job_id: str,
) -> dict[tuple[int, str], str]:
    row_indexes_by_dataset: dict[tuple[str, str], set[int]] = {}
    for spec in specs:
        row_index = int(spec["row_index"])
        context = contexts[row_index]
        if context.source_dataset_root is None or context.source_dataset_id is None:
            continue
        key = (context.source_dataset_root, context.source_dataset_id)
        row_indexes_by_dataset.setdefault(key, set()).add(row_index)

    status_by_spec: dict[tuple[int, str], str] = {}
    specs_by_dataset: dict[tuple[str, str], list[dict[str, object]]] = {}
    for spec in specs:
        key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        specs_by_dataset.setdefault(key, []).append(spec)

    for (dataset_root, dataset_id), row_index_set in row_indexes_by_dataset.items():
        row_indexes = sorted(row_index_set)
        dataset = Dataset(Path(dataset_root), dataset_id)
        ids = [contexts[row_index].sequence_id for row_index in row_indexes]
        output_columns = {
            str(spec["out_id"]): infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=str(spec["out_id"]))
            for spec in specs_by_dataset.get((dataset_root, dataset_id), [])
        }
        digest_column = infer_usr_column_name(
            model_id=model_id,
            job_id=job_id,
            out_id="metadata__feature_request_digest",
        )
        column_values = read_usr_columns(
            ds=dataset,
            ids=ids,
            column_names=[*output_columns.values(), digest_column],
        )
        row_position = {row_index: position for position, row_index in enumerate(row_indexes)}
        for spec in specs_by_dataset.get((dataset_root, dataset_id), []):
            row_index = int(spec["row_index"])
            position = row_position[row_index]
            column_name = output_columns[str(spec["out_id"])]
            value = column_values.get(column_name, [None] * len(ids))[position]
            if value is None:
                continue
            observed_digest = column_values.get(digest_column, [None] * len(ids))[position]
            expected_digest = metadata_rows[row_index].get("feature_request_digest")
            if observed_digest == expected_digest:
                status_by_spec[(row_index, str(spec["feature_vector_key"]))] = "reusable"
            elif observed_digest is None:
                status_by_spec[(row_index, str(spec["feature_vector_key"]))] = "legacy_unclassified"
            else:
                status_by_spec[(row_index, str(spec["feature_vector_key"]))] = "stale"
    return status_by_spec


def plan_sequence_view_feature_completion(
    *,
    bundle: PromoterFeatureBundleConfig,
    model_id: str,
    job_id: str,
    bundle_id: str | None = None,
    infer_command: str | None = None,
) -> FeatureCompletionPlan:
    load_result = load_sequence_view_input_records_with_status(bundle=bundle)
    records = load_result.records
    missing_product_selectors = [item.as_dict() for item in load_result.missing_products]
    contexts = resolve_sequence_view_contexts(records=records)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id=model_id)
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )

    persisted_keys = _persisted_feature_vector_keys(specs)
    remaining_specs = [spec for spec in specs if str(spec["feature_vector_key"]) not in persisted_keys]
    legacy_status = (
        _legacy_overlay_status(
            contexts=contexts,
            metadata_rows=metadata_rows,
            specs=remaining_specs,
            model_id=model_id,
            job_id=job_id,
        )
        if remaining_specs
        else {}
    )

    persisted_reusable = 0
    legacy_reusable = 0
    legacy_unclassified = 0
    stale = 0
    missing = 0
    for spec in specs:
        key = str(spec["feature_vector_key"])
        status_key = (int(spec["row_index"]), key)
        if key in persisted_keys:
            persisted_reusable += 1
            continue
        status = legacy_status.get(status_key)
        if status == "reusable":
            legacy_reusable += 1
        elif status == "legacy_unclassified":
            legacy_unclassified += 1
            stale += 1
        elif status == "stale":
            stale += 1
        else:
            missing += 1

    product_counts = Counter(str(context.product_kind) for context in contexts if context.product_kind is not None)
    orientation_counts = Counter(
        str(context.orientation or context.anchor_orientation or "forward") for context in contexts
    )
    pooling_counts = Counter(str(context.pooling_operation or "seq_mean") for context in contexts)
    reusable = persisted_reusable + legacy_reusable
    construct_completion = [
        _missing_product_command(selector_payload) for selector_payload in missing_product_selectors
    ]
    commands = FeatureCompletionCommands(
        construct_completion=construct_completion,
        infer_backfill=[infer_command] if infer_command and missing else [],
        alias_backfill=[infer_command] if infer_command and (legacy_reusable or legacy_unclassified) else [],
    )
    return FeatureCompletionPlan(
        dataset=_dataset_label(records, missing_product_selectors=missing_product_selectors),
        bundle_id=bundle_id or job_id,
        model_family=model_id,
        required_views=len(contexts),
        required_vectors=len(specs),
        existing_vectors=persisted_reusable + legacy_reusable + legacy_unclassified + stale,
        reusable_vectors=reusable,
        stale_vectors=stale,
        missing_vectors=missing,
        missing_products=len(missing_product_selectors),
        persisted_vector_reusable=persisted_reusable,
        legacy_digest_reusable=legacy_reusable,
        legacy_unclassified_vectors=legacy_unclassified,
        existing_aliases=len(_existing_alias_ids(specs)),
        by_product_kind=dict(sorted(product_counts.items())),
        by_orientation=dict(sorted(orientation_counts.items())),
        by_pooling_operation=dict(sorted(pooling_counts.items())),
        missing_product_selectors=missing_product_selectors,
        commands=commands,
    )


def _missing_product_command(selector_payload: dict[str, object]) -> str:
    dataset = str(selector_payload.get("dataset") or "").strip()
    product_kind = str(selector_payload.get("product_kind") or "").strip()
    orientation = str(selector_payload.get("orientation") or "").strip()
    pooling = str(selector_payload.get("pooling_operation") or "").strip()
    parts = [f"complete sequence products dataset={dataset}"]
    if product_kind:
        parts.append(f"product_kind={product_kind}")
    if orientation:
        parts.append(f"orientation={orientation}")
    if pooling:
        parts.append(f"pooling={pooling}")
    return " ".join(parts)
