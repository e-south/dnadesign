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

import pyarrow.parquet as pq

from dnadesign.usr import Dataset, sequence_views_path

from .aliases import (
    load_feature_alias_ids,
    load_feature_alias_rows,
    load_feature_scalar_alias_ids,
    load_feature_scalar_alias_rows,
    load_feature_scalar_keys,
    load_feature_vector_keys,
)
from .contracts import SequenceFeatureBundleConfig
from .execution import (
    _sequence_view_feature_scalar_specs,
    _sequence_view_feature_vector_specs,
    build_feature_metadata_rows,
)
from .selectors import resolve_intermediate_selector
from .sequence_views import (
    _resolve_usr_root,
    load_sequence_view_input_records_with_status,
    resolve_sequence_view_contexts,
)


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


def _vector_representation_specs(
    *,
    bundle: SequenceFeatureBundleConfig,
    selector: str,
) -> tuple[tuple[str, str | None], ...]:
    specs: list[tuple[str, str | None]] = []
    if bundle.collect_output_layer_mean:
        specs.append(("output_layer_mean", None))
    if bundle.collect_intermediate_embedding:
        specs.append(("intermediate_embedding", selector))
    return tuple(specs)


def _scalar_kind_specs(*, bundle: SequenceFeatureBundleConfig) -> tuple[str, ...]:
    if not bundle.collect_log_likelihood:
        return ()
    return ("log_likelihood__total", "log_likelihood__mean_per_token")


def _inventory_orientation(value: object) -> str:
    text = str(value or "").strip()
    return text or "forward"


def _sequence_view_aliases_contain(aliases: object, value: str) -> bool:
    if not isinstance(aliases, list):
        return False
    return value.casefold() in {str(alias).casefold() for alias in aliases}


def _selected_sequence_view_rows(
    ds: Dataset,
    *,
    product_kind: str | None,
    view_name: str | None,
    alias: str | None,
    orientation: str | None,
) -> list[dict[str, object]]:
    path = sequence_views_path(ds)
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    table = pq.read_table(
        path,
        columns=[
            "view_id",
            "product_kind",
            "view_name",
            "aliases",
            "orientation",
        ],
    )
    for raw in table.to_pylist():
        if product_kind is not None and raw.get("product_kind") != product_kind:
            continue
        if view_name is not None and raw.get("view_name") != view_name:
            continue
        if alias is not None and not _sequence_view_aliases_contain(raw.get("aliases"), alias):
            continue
        if orientation is not None and raw.get("orientation") != orientation:
            continue
        rows.append(raw)
    return rows


def _load_feature_vector_key_inventory(
    *,
    dataset_root: str,
    dataset_id: str,
    keys: set[str],
) -> set[str]:
    return load_feature_vector_keys(dataset_root=dataset_root, dataset_id=dataset_id, keys=keys)


def _load_feature_scalar_key_inventory(
    *,
    dataset_root: str,
    dataset_id: str,
    keys: set[str],
) -> set[str]:
    return load_feature_scalar_keys(dataset_root=dataset_root, dataset_id=dataset_id, keys=keys)


def _alias_row_vector_identity(row: dict[str, object]) -> tuple[str, str, str | None, str, str]:
    return (
        str(row.get("view_id") or ""),
        str(row.get("representation_kind") or ""),
        str(row.get("layer_name") or "") or None,
        str(row.get("pooling_operation") or ""),
        _inventory_orientation(row.get("orientation")),
    )


def _alias_row_scalar_identity(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("view_id") or ""),
        str(row.get("scalar_kind") or ""),
        _inventory_orientation(row.get("orientation")),
    )


def _apply_payload_state(
    state_by_slot: dict[tuple[object, ...], str],
    *,
    slot: tuple[object, ...],
    has_payload: bool,
) -> None:
    current = state_by_slot.get(slot)
    if has_payload:
        state_by_slot[slot] = "reusable"
    elif current is None:
        state_by_slot[slot] = "stale"


def plan_sequence_view_feature_inventory_completion(
    *,
    bundle: SequenceFeatureBundleConfig,
    model_id: str,
    job_id: str,
    bundle_id: str | None = None,
    infer_command: str | None = None,
) -> FeatureCompletionPlan:
    """Plan completion from sequence-view and alias inventories only.

    This path is intentionally cheaper than the exact planner. It does not
    derive feature keys for unfilled records; instead it treats canonical alias
    rows plus payload-key presence as the reusable-work contract. That keeps
    status snapshots bounded while still exposing corrupt alias-to-payload
    references as stale work.
    """

    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    vector_representations = _vector_representation_specs(bundle=bundle, selector=selector.intermediate_selector)
    scalar_kinds = _scalar_kind_specs(bundle=bundle)

    product_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    pooling_counts: Counter[str] = Counter()
    missing_product_selectors: list[dict[str, object]] = []
    dataset_labels: set[str] = set()
    expected_vectors_by_dataset: dict[tuple[str, str], set[tuple[str, str, str | None, str, str]]] = {}
    expected_scalars_by_dataset: dict[tuple[str, str], set[tuple[str, str, str]]] = {}

    required_views = 0
    selected_cache: dict[tuple[str, str, str | None, str | None, str | None, str | None], list[dict[str, object]]] = {}
    for input_cfg in bundle.sequence_view_inputs:
        root = _resolve_usr_root(input_cfg.root)
        dataset_labels.add(input_cfg.dataset)
        ds = Dataset(root, input_cfg.dataset)
        cache_key = (
            str(root),
            input_cfg.dataset,
            input_cfg.view_selector.product_kind,
            input_cfg.view_selector.view_name,
            input_cfg.view_selector.alias,
            input_cfg.view_selector.orientation,
        )
        selected = selected_cache.get(cache_key)
        if selected is None:
            selected = _selected_sequence_view_rows(
                ds,
                product_kind=input_cfg.view_selector.product_kind,
                view_name=input_cfg.view_selector.view_name,
                alias=input_cfg.view_selector.alias,
                orientation=input_cfg.view_selector.orientation,
            )
            selected_cache[cache_key] = selected
        if not selected:
            missing_product_selectors.append(
                {
                    "dataset": input_cfg.dataset,
                    "root": str(root),
                    "product_kind": input_cfg.view_selector.product_kind,
                    "view_name": input_cfg.view_selector.view_name,
                    "alias": input_cfg.view_selector.alias,
                    "orientation": input_cfg.view_selector.orientation,
                    "pooling_operation": input_cfg.pooling.operation,
                }
            )
            continue

        dataset_key = (str(root), input_cfg.dataset)
        expected_vectors = expected_vectors_by_dataset.setdefault(dataset_key, set())
        expected_scalars = expected_scalars_by_dataset.setdefault(dataset_key, set())
        pooling_operation = str(input_cfg.pooling.operation)
        for view in selected:
            required_views += 1
            product_counts[str(view.get("product_kind"))] += 1
            orientation = _inventory_orientation(view.get("orientation"))
            orientation_counts[orientation] += 1
            pooling_counts[pooling_operation] += 1
            for representation_kind, layer_name in vector_representations:
                expected_vectors.add(
                    (
                        str(view.get("view_id")),
                        representation_kind,
                        layer_name,
                        pooling_operation,
                        orientation,
                    )
                )
            for scalar_kind in scalar_kinds:
                expected_scalars.add((str(view.get("view_id")), scalar_kind, orientation))

    required_vectors = required_views * len(vector_representations)
    required_scalars = required_views * len(scalar_kinds)
    existing_aliases = 0
    existing_scalar_aliases = 0
    reusable_vectors = 0
    stale_vectors = 0
    reusable_scalars = 0
    stale_scalars = 0

    for dataset_key, expected_vectors in expected_vectors_by_dataset.items():
        dataset_root, dataset_id = dataset_key
        alias_rows = load_feature_alias_rows(dataset_root=dataset_root, dataset_id=dataset_id)
        existing_aliases += len(alias_rows)
        alias_keys = {
            str(row.get("feature_vector_key") or "")
            for row in alias_rows
            if str(row.get("feature_vector_key") or "").strip()
        }
        payload_keys = _load_feature_vector_key_inventory(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            keys=alias_keys,
        )
        state_by_slot: dict[tuple[object, ...], str] = {}
        for row in alias_rows:
            if str(row.get("model_name") or "") != model_id:
                continue
            identity = _alias_row_vector_identity(row)
            if identity not in expected_vectors:
                continue
            key = str(row.get("feature_vector_key") or "")
            _apply_payload_state(state_by_slot, slot=identity, has_payload=key in payload_keys)
        reusable_vectors += sum(1 for state in state_by_slot.values() if state == "reusable")
        stale_vectors += sum(1 for state in state_by_slot.values() if state == "stale")

    for dataset_key, expected_scalars in expected_scalars_by_dataset.items():
        dataset_root, dataset_id = dataset_key
        alias_rows = load_feature_scalar_alias_rows(dataset_root=dataset_root, dataset_id=dataset_id)
        existing_scalar_aliases += len(alias_rows)
        alias_keys = {
            str(row.get("feature_scalar_key") or "")
            for row in alias_rows
            if str(row.get("feature_scalar_key") or "").strip()
        }
        payload_keys = _load_feature_scalar_key_inventory(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            keys=alias_keys,
        )
        state_by_slot: dict[tuple[object, ...], str] = {}
        for row in alias_rows:
            if str(row.get("model_name") or "") != model_id:
                continue
            identity = _alias_row_scalar_identity(row)
            if identity not in expected_scalars:
                continue
            key = str(row.get("feature_scalar_key") or "")
            _apply_payload_state(state_by_slot, slot=identity, has_payload=key in payload_keys)
        reusable_scalars += sum(1 for state in state_by_slot.values() if state == "reusable")
        stale_scalars += sum(1 for state in state_by_slot.values() if state == "stale")

    missing_vectors = max(required_vectors - reusable_vectors - stale_vectors, 0)
    missing_scalars = max(required_scalars - reusable_scalars - stale_scalars, 0)
    needs_infer_backfill = bool(missing_vectors or missing_scalars or stale_vectors or stale_scalars)
    commands = FeatureCompletionCommands(
        construct_completion=[
            _missing_product_command(selector_payload) for selector_payload in missing_product_selectors
        ],
        infer_backfill=[infer_command] if infer_command and needs_infer_backfill else [],
    )
    return FeatureCompletionPlan(
        dataset=",".join(sorted(dataset_labels)),
        bundle_id=bundle_id or job_id,
        model_family=model_id,
        required_views=required_views,
        required_vectors=required_vectors,
        required_scalars=required_scalars,
        existing_vectors=reusable_vectors + stale_vectors,
        existing_scalars=reusable_scalars + stale_scalars,
        reusable_vectors=reusable_vectors,
        reusable_scalars=reusable_scalars,
        stale_vectors=stale_vectors,
        stale_scalars=stale_scalars,
        missing_vectors=missing_vectors,
        missing_scalars=missing_scalars,
        missing_products=len(missing_product_selectors),
        persisted_vector_reusable=reusable_vectors,
        persisted_scalar_reusable=reusable_scalars,
        existing_aliases=existing_aliases,
        existing_scalar_aliases=existing_scalar_aliases,
        by_product_kind=dict(sorted(product_counts.items())),
        by_orientation=dict(sorted(orientation_counts.items())),
        by_pooling_operation=dict(sorted(pooling_counts.items())),
        missing_product_selectors=missing_product_selectors,
        commands=commands,
    )


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
    bundle: SequenceFeatureBundleConfig,
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
