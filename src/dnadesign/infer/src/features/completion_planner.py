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

from .aliases import (
    load_feature_alias_ids,
    load_feature_scalar_alias_ids,
    load_feature_scalar_keys,
    load_feature_vector_keys,
)
from .contracts import PromoterFeatureBundleConfig
from .execution import (
    _sequence_view_feature_scalar_specs,
    _sequence_view_feature_vector_specs,
    build_feature_metadata_rows,
)
from .selectors import resolve_intermediate_selector
from .sequence_views import load_sequence_view_input_records_with_status, resolve_sequence_view_contexts


@dataclass(frozen=True)
class FeatureCompletionCommands:
    construct_completion: list[str] = field(default_factory=list)
    infer_backfill: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureCompletionPlan:
    dataset: str
    bundle_id: str
    model_family: str
    required_views: int
    required_vectors: int
    required_scalars: int
    existing_vectors: int
    existing_scalars: int
    reusable_vectors: int
    reusable_scalars: int
    stale_vectors: int
    stale_scalars: int
    missing_vectors: int
    missing_scalars: int
    missing_products: int
    persisted_vector_reusable: int
    persisted_scalar_reusable: int
    existing_aliases: int
    existing_scalar_aliases: int
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


def _group_required_scalar_keys_by_dataset(specs: list[dict[str, object]]) -> dict[tuple[str, str], set[str]]:
    keys_by_dataset: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        keys_by_dataset.setdefault(dataset_key, set()).add(str(spec["feature_scalar_key"]))
    return keys_by_dataset


def _persisted_feature_vector_keys(specs: list[dict[str, object]]) -> set[str]:
    existing: set[str] = set()
    for (dataset_root, dataset_id), keys in _group_required_keys_by_dataset(specs).items():
        loaded = load_feature_vector_keys(dataset_root=dataset_root, dataset_id=dataset_id, keys=keys)
        existing.update(loaded)
    return existing


def _persisted_feature_scalar_keys(specs: list[dict[str, object]]) -> set[str]:
    existing: set[str] = set()
    for (dataset_root, dataset_id), keys in _group_required_scalar_keys_by_dataset(specs).items():
        loaded = load_feature_scalar_keys(dataset_root=dataset_root, dataset_id=dataset_id, keys=keys)
        existing.update(loaded)
    return existing


def _existing_alias_ids(specs: list[dict[str, object]]) -> set[str]:
    alias_ids: set[str] = set()
    for dataset_root, dataset_id in _group_required_keys_by_dataset(specs):
        alias_ids.update(load_feature_alias_ids(dataset_root=dataset_root, dataset_id=dataset_id))
    return alias_ids


def _existing_scalar_alias_ids(specs: list[dict[str, object]]) -> set[str]:
    alias_ids: set[str] = set()
    for dataset_root, dataset_id in _group_required_scalar_keys_by_dataset(specs):
        alias_ids.update(load_feature_scalar_alias_ids(dataset_root=dataset_root, dataset_id=dataset_id))
    return alias_ids


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
    scalar_specs = _sequence_view_feature_scalar_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
    )

    persisted_keys = _persisted_feature_vector_keys(specs)
    persisted_reusable = 0
    missing = 0
    for spec in specs:
        key = str(spec["feature_vector_key"])
        if key in persisted_keys:
            persisted_reusable += 1
            continue
        missing += 1

    persisted_scalar_keys = _persisted_feature_scalar_keys(scalar_specs)
    persisted_scalar_reusable = 0
    missing_scalars = 0
    for spec in scalar_specs:
        key = str(spec["feature_scalar_key"])
        if key in persisted_scalar_keys:
            persisted_scalar_reusable += 1
            continue
        missing_scalars += 1

    product_counts = Counter(str(context.product_kind) for context in contexts if context.product_kind is not None)
    orientation_counts = Counter(
        str(context.orientation or context.anchor_orientation or "forward") for context in contexts
    )
    pooling_counts = Counter(str(context.pooling_operation or "seq_mean") for context in contexts)
    construct_completion = [
        _missing_product_command(selector_payload) for selector_payload in missing_product_selectors
    ]
    commands = FeatureCompletionCommands(
        construct_completion=construct_completion,
        infer_backfill=[infer_command] if infer_command and (missing or missing_scalars) else [],
    )
    return FeatureCompletionPlan(
        dataset=_dataset_label(records, missing_product_selectors=missing_product_selectors),
        bundle_id=bundle_id or job_id,
        model_family=model_id,
        required_views=len(contexts),
        required_vectors=len(specs),
        required_scalars=len(scalar_specs),
        existing_vectors=persisted_reusable,
        existing_scalars=persisted_scalar_reusable,
        reusable_vectors=persisted_reusable,
        reusable_scalars=persisted_scalar_reusable,
        stale_vectors=0,
        stale_scalars=0,
        missing_vectors=missing,
        missing_scalars=missing_scalars,
        missing_products=len(missing_product_selectors),
        persisted_vector_reusable=persisted_reusable,
        persisted_scalar_reusable=persisted_scalar_reusable,
        existing_aliases=len(_existing_alias_ids(specs)),
        existing_scalar_aliases=len(_existing_scalar_alias_ids(scalar_specs)),
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
