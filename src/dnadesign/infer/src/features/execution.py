"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/execution.py

Execution helpers for Evo2 feature bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional

import torch

from ..contracts import infer_usr_column_name
from ..errors import CapabilityError, RuntimeOOMError
from ..runtime.resume_planner import read_usr_columns
from .aliases import (
    compute_feature_alias_id,
    compute_feature_scalar_alias_id,
    compute_feature_scalar_key,
    load_feature_scalar_rows,
    load_feature_vector_rows,
    persist_feature_alias_rows,
    persist_feature_scalar_alias_rows,
    persist_feature_scalar_rows,
    persist_feature_vector_rows,
    record_feature_bundle_complete,
    record_feature_bundle_progress,
)
from .cache_keys import compute_feature_vector_key, compute_forward_pass_key
from .context import resolve_sequence_contexts
from .contracts import SequenceContextRecord, SequenceFeatureBundleConfig
from .selectors import canonical_selector_for_block, resolve_intermediate_selector
from .sequence_views import bundle_uses_sequence_views, resolve_sequence_view_contexts

_LOG_LIKELIHOOD_TOTAL = "log_likelihood__total"
_LOG_LIKELIHOOD_MEAN = "log_likelihood__mean_per_token"
_OUTPUT_LAYER_SEQ_MEAN = "output_layer_mean__seq_mean"
_OUTPUT_LAYER_ANCHOR_MEAN = "output_layer_mean__anchor_mean"
_OUTPUT_LAYER_CORE60_MEAN = "output_layer_mean__core60_mean"
_FEATURE_BUNDLE_PROGRESS_STEP_PCT_SMALL_TARGET = 25
_FEATURE_BUNDLE_PROGRESS_STEP_PCT_LARGE_TARGET = 10
_FEATURE_BUNDLE_SMALL_TARGET_CONTEXTS_THRESHOLD = 200
_FEATURE_BUNDLE_PROGRESS_HEARTBEAT_SECONDS = 1800.0
_METADATA_OUTPUT_FIELDS = (
    ("metadata__sequence_id", "sequence_id"),
    ("metadata__anchor_id", "anchor_id"),
    ("metadata__is_wildtype", "is_wildtype"),
    ("metadata__context_id", "context_id"),
    ("metadata__context_kind", "context_kind"),
    ("metadata__view_id", "view_id"),
    ("metadata__view_name", "view_name"),
    ("metadata__product_kind", "product_kind"),
    ("metadata__orientation", "orientation"),
    ("metadata__template_id", "template_id"),
    ("metadata__resolved_length", "resolved_length"),
    ("metadata__anchor_start", "anchor_start"),
    ("metadata__anchor_end", "anchor_end"),
    ("metadata__pooling_operation", "pooling_operation"),
    ("metadata__pooling_start_0", "pooling_start_0"),
    ("metadata__pooling_end_0", "pooling_end_0"),
    ("metadata__model_name", "model_name"),
    ("metadata__provider_name", "provider_name"),
    ("metadata__provider_version", "provider_version"),
    ("metadata__intermediate_block", "intermediate_block"),
    ("metadata__intermediate_selector", "intermediate_selector"),
    ("metadata__pooling_modes", "pooling_modes"),
    ("metadata__forward_pass_key", "forward_pass_key"),
    ("metadata__feature_vector_key", "feature_vector_key"),
    ("metadata__parent_sequence_id", "parent_sequence_id"),
    ("metadata__derivation_id", "derivation_id"),
    ("metadata__feature_schema_version", "feature_schema_version"),
    ("metadata__construct_version", "construct_version"),
    ("metadata__timestamp", "timestamp"),
    ("metadata__feature_request_digest", "feature_request_digest"),
)


def _templated_anchor_mean_enabled(bundle: SequenceFeatureBundleConfig) -> bool:
    return bundle.context.kind != "anchor_only" and bool(bundle.pooling.anchor_mean_for_templated)


def _sequence_view_pooling_modes(bundle: SequenceFeatureBundleConfig) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for input_cfg in bundle.sequence_view_inputs:
        operation = input_cfg.pooling.operation
        if operation in seen:
            continue
        seen.add(operation)
        ordered.append(operation)
    return ordered


def _pooling_modes(bundle: SequenceFeatureBundleConfig) -> list[str]:
    if bundle_uses_sequence_views(bundle):
        return _sequence_view_pooling_modes(bundle)
    modes: list[str] = []
    if bundle.pooling.seq_mean:
        modes.append("seq_mean")
    if _templated_anchor_mean_enabled(bundle):
        modes.append("anchor_mean")
    return modes


def _output_layer_out_id(pool_scope: str) -> str:
    if pool_scope == "seq_mean":
        return _OUTPUT_LAYER_SEQ_MEAN
    if pool_scope == "anchor_mean":
        return _OUTPUT_LAYER_ANCHOR_MEAN
    if pool_scope == "core60_mean":
        return _OUTPUT_LAYER_CORE60_MEAN
    raise CapabilityError(f"Unsupported pooling scope '{pool_scope}'.")


def _intermediate_out_id(selector: str, pool_scope: str) -> str:
    return f"intermediate_embedding__{selector}__{pool_scope}"


def infer_output_family(out_id: str) -> str:
    output_id = str(out_id).strip()
    if output_id.startswith("log_likelihood__"):
        return "log_likelihood"
    if output_id.startswith("output_layer_mean__"):
        return "output_layer_mean"
    if output_id.startswith("intermediate_embedding__"):
        return "intermediate_embedding"
    if output_id.startswith("metadata__"):
        return "metadata"
    family, _, _ = output_id.partition("__")
    return family or output_id


def infer_output_kind(out_id: str) -> str:
    return "metadata" if infer_output_family(out_id) == "metadata" else "feature"


def _progress_pct(*, completed: int, target: int) -> float:
    if target <= 0:
        return 100.0
    ratio = float(completed) * 100.0 / float(target)
    return max(0.0, min(100.0, ratio))


def build_feature_bundle_outputs(
    *,
    bundle: SequenceFeatureBundleConfig,
    model_id: str | None = None,
) -> list[dict[str, object]]:
    selector = (
        resolve_intermediate_selector(
            model_id=model_id,
            intermediate_block=bundle.intermediate_block,
        ).intermediate_selector
        if model_id is not None
        else canonical_selector_for_block(bundle.intermediate_block)
    )
    outputs: list[dict[str, object]] = []
    pooling_modes = _pooling_modes(bundle)

    if bundle.collect_log_likelihood:
        outputs.append(
            {
                "id": _LOG_LIKELIHOOD_TOTAL,
                "fn": "evo2.log_likelihood",
                "params": {"method": "native", "reduction": "sum"},
                "format": "float",
            }
        )
        outputs.append(
            {
                "id": _LOG_LIKELIHOOD_MEAN,
                "fn": "evo2.log_likelihood",
                "params": {"method": "native", "reduction": "mean"},
                "format": "float",
            }
        )

    if bundle.collect_output_layer_mean:
        for pool_scope in pooling_modes:
            outputs.append(
                {
                    "id": _output_layer_out_id(pool_scope),
                    "fn": "evo2.logits",
                    "params": {
                        "pool": {"method": "mean", "dim": 1},
                        "feature_group": "output_layer_mean",
                        "pool_scope": pool_scope,
                    },
                    "format": "list",
                }
            )

    if bundle.collect_intermediate_embedding:
        for pool_scope in pooling_modes:
            outputs.append(
                {
                    "id": _intermediate_out_id(selector, pool_scope),
                    "fn": "evo2.embedding",
                    "params": {
                        "layer": selector,
                        "pool": {"method": "mean", "dim": 1},
                        "feature_group": "intermediate_embedding",
                        "intermediate_block": bundle.intermediate_block,
                        "intermediate_selector": selector,
                        "pool_scope": pool_scope,
                    },
                    "format": "list",
                }
            )

    return outputs


def _pool_tensor_scopes(
    tensor: torch.Tensor,
    *,
    context: SequenceContextRecord,
) -> tuple[list[float], list[float]]:
    token_count = int(tensor.shape[0])
    if token_count != int(context.resolved_length):
        raise CapabilityError(
            "Anchor-aware Evo2 pooling requires one token per base. "
            f"id={context.sequence_id} token_count={token_count} resolved_length={context.resolved_length}"
        )
    if context.anchor_start < 0 or context.anchor_end <= context.anchor_start or context.anchor_end > token_count:
        raise CapabilityError(
            "Anchor-aware Evo2 pooling received an invalid anchor span. "
            f"id={context.sequence_id} start={context.anchor_start} end={context.anchor_end} token_count={token_count}"
        )
    seq_mean = tensor.mean(dim=0).detach().cpu().tolist()
    anchor_mean = tensor[context.anchor_start : context.anchor_end].mean(dim=0).detach().cpu().tolist()
    return seq_mean, anchor_mean


def _pool_tensor_for_context(
    tensor: torch.Tensor,
    *,
    context: SequenceContextRecord,
) -> list[float]:
    token_count = int(tensor.shape[0])
    if token_count != int(context.resolved_length):
        raise CapabilityError(
            "Sequence-view Evo2 pooling requires one token per base. "
            f"id={context.context_id} token_count={token_count} resolved_length={context.resolved_length}"
        )
    operation = str(context.pooling_operation or "seq_mean")
    if operation == "seq_mean":
        return tensor.mean(dim=0).detach().cpu().tolist()
    start_0 = context.pooling_start_0
    end_0 = context.pooling_end_0
    if start_0 is None or end_0 is None:
        raise CapabilityError(f"{operation} requires explicit pooling bounds for context '{context.context_id}'.")
    if start_0 < 0 or end_0 <= start_0 or end_0 > token_count:
        raise CapabilityError(
            f"{operation} received invalid pooling bounds for context '{context.context_id}': "
            f"{start_0}:{end_0} length={token_count}"
        )
    if operation == "core60_mean" and (end_0 - start_0) != 60:
        raise CapabilityError(
            f"core60_mean requires an exact 60 bp pooling span for context '{context.context_id}'. "
            f"Observed span {end_0 - start_0}."
        )
    return tensor[start_0:end_0].mean(dim=0).detach().cpu().tolist()


def _feature_pooling_identity(context: SequenceContextRecord) -> tuple[str, int | None, int | None]:
    operation = str(context.pooling_operation or "seq_mean")
    return operation, context.pooling_start_0, context.pooling_end_0


def _feature_bundle_log_likelihoods(adapter, seq_chunk: list[str]) -> tuple[list[float], list[float]]:
    fused = getattr(adapter, "log_likelihood_total_and_mean", None)
    if callable(fused):
        totals, means = fused(seq_chunk, method="native")
        return list(totals), list(means)
    totals = adapter.log_likelihood(
        seq_chunk,
        method="native",
        reduction="sum",
    )
    means = adapter.log_likelihood(
        seq_chunk,
        method="native",
        reduction="mean",
    )
    return list(totals), list(means)


def _feature_bundle_logits_and_embedding(
    adapter,
    *,
    seq_chunk: list[str],
    selector: str,
) -> tuple[list[object] | None, list[object] | None]:
    fused = getattr(adapter, "logits_and_embedding", None)
    if callable(fused):
        logits_tensors, embedding_tensors = fused(seq_chunk, layer=selector, fmt="tensor")
        return list(logits_tensors), list(embedding_tensors)
    return None, None


def _resolve_feature_bundle_adapter(adapter, adapter_factory: Callable[[], object] | None):
    if adapter is not None:
        return adapter
    if adapter_factory is None:
        raise CapabilityError("Feature bundle execution requires an adapter or adapter factory.")
    return adapter_factory()


def _stable_feature_bundle_eval_batch_size(
    *,
    model_id: str,
    bundle: SequenceFeatureBundleConfig,
    micro_batch_size: int,
) -> int | None:
    if micro_batch_size <= 0:
        return None
    if model_id == "evo2_20b" and bundle.context.kind == "anchor_only":
        return int(micro_batch_size)
    return None


def _pad_feature_bundle_eval_sequences(
    *,
    seq_chunk: list[str],
    eval_batch_size: int | None,
) -> list[str]:
    if eval_batch_size is None or eval_batch_size <= len(seq_chunk) or not seq_chunk:
        return list(seq_chunk)
    padded = list(seq_chunk)
    source_count = len(seq_chunk)
    for offset in range(eval_batch_size - source_count):
        padded.append(seq_chunk[offset % source_count])
    return padded


def _sequence_view_group_output_needs(
    *,
    all_vals: Dict[str, List[object]],
    contexts: List[SequenceContextRecord],
    row_indexes: list[int],
    bundle: SequenceFeatureBundleConfig,
    selector: str,
) -> tuple[bool, bool, bool]:
    needs_scalars = False
    needs_logits = False
    needs_embedding = False
    for row_index in row_indexes:
        context = contexts[row_index]
        pool_scope = str(context.pooling_operation or "seq_mean")
        if bundle.collect_log_likelihood and (
            all_vals[_LOG_LIKELIHOOD_TOTAL][row_index] is None or all_vals[_LOG_LIKELIHOOD_MEAN][row_index] is None
        ):
            needs_scalars = True
        if bundle.collect_output_layer_mean and all_vals[_output_layer_out_id(pool_scope)][row_index] is None:
            needs_logits = True
        if (
            bundle.collect_intermediate_embedding
            and all_vals[_intermediate_out_id(selector, pool_scope)][row_index] is None
        ):
            needs_embedding = True
    return needs_scalars, needs_logits, needs_embedding


def _feature_request_digest(
    *,
    bundle: SequenceFeatureBundleConfig,
    context: SequenceContextRecord,
    model_id: str,
    selector: str,
) -> str:
    payload = {
        "feature_schema_version": bundle.feature_schema_version,
        "model_id": model_id,
        "context_id": context.context_id,
        "context_kind": context.context_kind,
        "view_id": context.view_id,
        "product_kind": context.product_kind,
        "orientation": context.orientation,
        "template_id": context.template_id,
        "resolved_sequence": context.resolved_sequence,
        "intermediate_selector": selector,
        "pooling_operation": context.pooling_operation,
        "pooling_start_0": context.pooling_start_0,
        "pooling_end_0": context.pooling_end_0,
        "outputs": {
            "log_likelihood": bundle.collect_log_likelihood,
            "output_layer_mean": bundle.collect_output_layer_mean,
            "intermediate_embedding": bundle.collect_intermediate_embedding,
        },
        "pooling": {
            "seq_mean": bundle.pooling.seq_mean,
            "anchor_mean_for_templated": bundle.pooling.anchor_mean_for_templated,
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _forward_pass_key_for_context(
    *,
    context: SequenceContextRecord,
    model_id: str,
    selector: str,
    bundle: SequenceFeatureBundleConfig,
) -> str:
    requested_layers = [selector] if bundle.collect_intermediate_embedding else []
    return compute_forward_pass_key(
        provider="evo2",
        model_name=model_id,
        model_revision=None,
        tokenizer_revision=None,
        requested_layers=requested_layers,
        normalized_input_sequence=context.resolved_sequence,
        provider_params={},
        orientation=str(context.orientation or context.anchor_orientation or "forward"),
    )


def _primary_feature_vector_key_for_context(
    *,
    context: SequenceContextRecord,
    model_id: str,
    selector: str,
    bundle: SequenceFeatureBundleConfig,
    forward_pass_key: str,
) -> str | None:
    pooling_operation, pooling_start_0, pooling_end_0 = _feature_pooling_identity(context)
    if pooling_operation is None:
        return None
    if bundle.collect_intermediate_embedding:
        return compute_feature_vector_key(
            forward_pass_key=forward_pass_key,
            representation_kind="intermediate_embedding",
            layer_name=selector,
            pooling_operation=pooling_operation,
            pooling_start_0=pooling_start_0,
            pooling_end_0=pooling_end_0,
            dtype_or_storage_format="list<float64>",
        )
    if bundle.collect_output_layer_mean:
        return compute_feature_vector_key(
            forward_pass_key=forward_pass_key,
            representation_kind="output_layer_mean",
            layer_name=None,
            pooling_operation=pooling_operation,
            pooling_start_0=pooling_start_0,
            pooling_end_0=pooling_end_0,
            dtype_or_storage_format="list<float64>",
        )
    return None


def _feature_vector_key_for_representation(
    *,
    context: SequenceContextRecord,
    forward_pass_key: str,
    representation_kind: str,
    selector: str,
) -> str:
    pooling_operation, pooling_start_0, pooling_end_0 = _feature_pooling_identity(context)
    return compute_feature_vector_key(
        forward_pass_key=forward_pass_key,
        representation_kind=representation_kind,
        layer_name=selector if representation_kind == "intermediate_embedding" else None,
        pooling_operation=pooling_operation,
        pooling_start_0=pooling_start_0,
        pooling_end_0=pooling_end_0,
        dtype_or_storage_format="list<float64>",
    )


def build_feature_metadata_rows(
    *,
    contexts: List[SequenceContextRecord],
    bundle: SequenceFeatureBundleConfig,
    model_id: str,
    include_feature_request_digest: bool = True,
) -> list[dict[str, object]]:
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    requested_layers = (selector.intermediate_selector,) if bundle.collect_intermediate_embedding else ()
    forward_pass_key_cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for context in contexts:
        orientation = str(context.orientation or context.anchor_orientation or "forward")
        forward_pass_cache_key = (
            context.resolved_sequence,
            orientation,
            requested_layers,
        )
        forward_pass_key = forward_pass_key_cache.get(forward_pass_cache_key)
        if forward_pass_key is None:
            forward_pass_key = _forward_pass_key_for_context(
                context=context,
                model_id=model_id,
                selector=selector.intermediate_selector,
                bundle=bundle,
            )
            forward_pass_key_cache[forward_pass_cache_key] = forward_pass_key
        rows.append(
            {
                "sequence_id": context.sequence_id,
                "anchor_id": context.anchor_id,
                "is_wildtype": context.is_wildtype,
                "context_id": context.context_id,
                "context_kind": context.context_kind,
                "view_id": context.view_id,
                "view_name": context.view_name,
                "product_kind": context.product_kind,
                "orientation": context.orientation or context.anchor_orientation,
                "template_id": context.template_id,
                "resolved_length": context.resolved_length,
                "anchor_start": context.anchor_start,
                "anchor_end": context.anchor_end,
                "pooling_operation": context.pooling_operation,
                "pooling_start_0": context.pooling_start_0,
                "pooling_end_0": context.pooling_end_0,
                "model_name": model_id,
                "provider_name": "evo2",
                "provider_version": None,
                "intermediate_block": selector.intermediate_block,
                "intermediate_selector": selector.intermediate_selector,
                "pooling_modes": _pooling_modes(bundle),
                "forward_pass_key": forward_pass_key,
                "feature_vector_key": _primary_feature_vector_key_for_context(
                    context=context,
                    model_id=model_id,
                    selector=selector.intermediate_selector,
                    bundle=bundle,
                    forward_pass_key=forward_pass_key,
                ),
                "parent_sequence_id": context.parent_sequence_id,
                "derivation_id": context.derivation_id,
                "feature_schema_version": bundle.feature_schema_version,
                "construct_version": context.construct_version,
                "timestamp": timestamp,
                "feature_request_digest": (
                    _feature_request_digest(
                        bundle=bundle,
                        context=context,
                        model_id=model_id,
                        selector=selector.intermediate_selector,
                    )
                    if include_feature_request_digest
                    else None
                ),
            }
        )
    return rows


def feature_metadata_output_ids() -> list[str]:
    return [out_id for out_id, _field_name in _METADATA_OUTPUT_FIELDS]


def build_feature_metadata_columnar(metadata_rows: list[dict[str, object]]) -> Dict[str, List[object]]:
    columnar: Dict[str, List[object]] = {}
    for out_id, field_name in _METADATA_OUTPUT_FIELDS:
        columnar[out_id] = [row.get(field_name) for row in metadata_rows]
    return columnar


def _apply_digest_resume_guard(
    *,
    ds,
    ids: Optional[List[str]],
    model_id: str,
    job_id: str,
    feature_values: Dict[str, List[object]],
    metadata_columnar: Mapping[str, List[object]],
    existing_digests: Optional[List[object]] = None,
) -> List[int]:
    if ds is None or ids is None:
        return []
    digest_out_id = "metadata__feature_request_digest"
    if existing_digests is None:
        digest_column = infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=digest_out_id)
        existing_digests = read_usr_columns(ds=ds, ids=ids, column_names=[digest_column]).get(
            digest_column,
            [None] * len(ids),
        )
    expected_digests = metadata_columnar[digest_out_id]

    stale_idx: list[int] = []
    for row_index, expected in enumerate(expected_digests):
        if existing_digests[row_index] == expected:
            continue
        if any(values[row_index] is not None for values in feature_values.values()):
            stale_idx.append(row_index)
            for values in feature_values.values():
                values[row_index] = None
    return stale_idx


def _existing_feature_metadata_values(
    *,
    ds,
    ids: Optional[List[str]],
    model_id: str,
    job_id: str,
) -> Dict[str, List[object]]:
    if ds is None or ids is None:
        return {out_id: [] for out_id in feature_metadata_output_ids()}
    metadata_columns = {
        out_id: infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=out_id)
        for out_id in feature_metadata_output_ids()
    }
    existing_by_column = read_usr_columns(ds=ds, ids=ids, column_names=list(metadata_columns.values()))
    return {
        out_id: existing_by_column.get(column_name, [None] * len(ids))
        for out_id, column_name in metadata_columns.items()
    }


def _missing_rows_by_output(columnar: Mapping[str, List[object]]) -> Dict[str, set[int]]:
    return {
        out_id: {row_index for row_index, value in enumerate(values) if value is None}
        for out_id, values in columnar.items()
    }


def _write_chunk_subset(
    *,
    writer: Optional[Callable[..., None]],
    idx_chunk: List[int],
    values: List[object],
    row_indexes: set[int],
    overwrite_override: bool | None = None,
    progress: Mapping[str, object] | None = None,
) -> None:
    if writer is None or not row_indexes:
        return
    subset_pairs = [
        (row_index, values[position]) for position, row_index in enumerate(idx_chunk) if row_index in row_indexes
    ]
    if not subset_pairs:
        return
    writer(
        [row_index for row_index, _value in subset_pairs],
        [value for _row_index, value in subset_pairs],
        overwrite_override=overwrite_override,
        progress=progress,
    )


def _group_columnar_by_row_indexes(
    *,
    columnar: Mapping[str, List[object]],
    row_indexes_by_output: Mapping[str, set[int]],
    idx_chunk: List[int] | None = None,
) -> list[tuple[list[int], Dict[str, List[object]]]]:
    groups: dict[tuple[int, ...], Dict[str, List[object]]] = {}
    position_by_row_index = {row_index: position for position, row_index in enumerate(idx_chunk)} if idx_chunk else None
    for out_id, values in columnar.items():
        row_indexes = tuple(sorted(row_indexes_by_output.get(out_id, set())))
        if not row_indexes:
            continue
        group_columnar = groups.setdefault(row_indexes, {})
        if position_by_row_index is None:
            group_columnar[out_id] = [values[row_index] for row_index in row_indexes]
            continue
        try:
            group_columnar[out_id] = [values[position_by_row_index[row_index]] for row_index in row_indexes]
        except KeyError as exc:
            raise CapabilityError(
                f"Grouped chunk write received row index {exc.args[0]} outside the current chunk."
            ) from exc
    return [(list(row_indexes), payload) for row_indexes, payload in groups.items()]


def _group_event_args_for_columnar(
    *,
    grouped_columnar: Mapping[str, List[object]],
    feature_event_args: Mapping[str, object],
) -> dict[str, object]:
    if any(infer_output_kind(out_id) == "feature" for out_id in grouped_columnar):
        return dict(feature_event_args)
    return {"infer_notify_suppress": True}


def _sequence_view_feature_alias_rows(
    *,
    contexts: List[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    bundle: SequenceFeatureBundleConfig,
    selector: str,
    model_id: str,
) -> list[dict[str, object]]:
    if not bundle.deduplicate.write_alias_map:
        return []
    alias_rows: list[dict[str, object]] = []
    for context, metadata in zip(contexts, metadata_rows, strict=True):
        if context.view_id is None or context.source_dataset_id is None or context.source_dataset_root is None:
            continue
        forward_pass_key = str(metadata["forward_pass_key"])
        created_at = str(metadata["timestamp"])
        for representation_kind, layer_name in (
            ("output_layer_mean", None),
            ("intermediate_embedding", selector if bundle.collect_intermediate_embedding else None),
        ):
            if representation_kind == "output_layer_mean" and not bundle.collect_output_layer_mean:
                continue
            if representation_kind == "intermediate_embedding" and not bundle.collect_intermediate_embedding:
                continue
            if context.pooling_operation is None:
                continue
            pooling_operation, pooling_start_0, pooling_end_0 = _feature_pooling_identity(context)
            feature_vector_key = (
                str(metadata["feature_vector_key"])
                if (
                    metadata.get("feature_vector_key") is not None
                    and (
                        representation_kind == "intermediate_embedding"
                        or (representation_kind == "output_layer_mean" and not bundle.collect_intermediate_embedding)
                    )
                )
                else compute_feature_vector_key(
                    forward_pass_key=forward_pass_key,
                    representation_kind=representation_kind,
                    layer_name=layer_name,
                    pooling_operation=pooling_operation,
                    pooling_start_0=pooling_start_0,
                    pooling_end_0=pooling_end_0,
                    dtype_or_storage_format="list<float64>",
                )
            )
            alias_rows.append(
                {
                    "_dataset_root": context.source_dataset_root,
                    "_dataset_id": context.source_dataset_id,
                    "alias_id": compute_feature_alias_id(
                        view_id=context.view_id,
                        sequence_id=context.sequence_id,
                        feature_vector_key=feature_vector_key,
                        representation_kind=representation_kind,
                    ),
                    "view_id": context.view_id,
                    "view_name": context.view_name,
                    "sequence_id": context.sequence_id,
                    "feature_vector_key": feature_vector_key,
                    "forward_pass_key": forward_pass_key,
                    "provider": "evo2",
                    "model_name": model_id,
                    "model_revision": None,
                    "layer_name": layer_name,
                    "representation_kind": representation_kind,
                    "pooling_operation": pooling_operation,
                    "pooling_start_0": pooling_start_0,
                    "pooling_end_0": pooling_end_0,
                    "orientation": context.orientation or context.anchor_orientation or "forward",
                    "source_dataset_id": context.source_dataset_id,
                    "feature_request_digest": str(metadata["feature_request_digest"]),
                    "created_at": created_at,
                }
            )
    return alias_rows


def _feature_scalar_key_for_context(
    *,
    metadata: Mapping[str, object],
    scalar_kind: str,
) -> str:
    return compute_feature_scalar_key(
        forward_pass_key=str(metadata["forward_pass_key"]),
        scalar_kind=scalar_kind,
        dtype_or_storage_format="float64",
    )


def _sequence_view_feature_scalar_alias_rows(
    *,
    contexts: List[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    bundle: SequenceFeatureBundleConfig,
    model_id: str,
) -> list[dict[str, object]]:
    if not bundle.deduplicate.write_alias_map or not bundle.collect_log_likelihood:
        return []
    alias_rows: list[dict[str, object]] = []
    scalar_key_cache: dict[tuple[str, str], str] = {}
    for context, metadata in zip(contexts, metadata_rows, strict=True):
        if context.view_id is None or context.source_dataset_id is None or context.source_dataset_root is None:
            continue
        created_at = str(metadata["timestamp"])
        for scalar_kind in (_LOG_LIKELIHOOD_TOTAL, _LOG_LIKELIHOOD_MEAN):
            forward_pass_key = str(metadata["forward_pass_key"])
            scalar_cache_key = (forward_pass_key, scalar_kind)
            feature_scalar_key = scalar_key_cache.get(scalar_cache_key)
            if feature_scalar_key is None:
                feature_scalar_key = _feature_scalar_key_for_context(metadata=metadata, scalar_kind=scalar_kind)
                scalar_key_cache[scalar_cache_key] = feature_scalar_key
            alias_rows.append(
                {
                    "_dataset_root": context.source_dataset_root,
                    "_dataset_id": context.source_dataset_id,
                    "alias_id": compute_feature_scalar_alias_id(
                        view_id=context.view_id,
                        sequence_id=context.sequence_id,
                        feature_scalar_key=feature_scalar_key,
                        scalar_kind=scalar_kind,
                    ),
                    "view_id": context.view_id,
                    "view_name": context.view_name,
                    "sequence_id": context.sequence_id,
                    "feature_scalar_key": feature_scalar_key,
                    "forward_pass_key": forward_pass_key,
                    "provider": "evo2",
                    "model_name": model_id,
                    "model_revision": None,
                    "scalar_kind": scalar_kind,
                    "orientation": context.orientation or context.anchor_orientation or "forward",
                    "source_dataset_id": context.source_dataset_id,
                    "feature_request_digest": str(metadata["feature_request_digest"]),
                    "created_at": created_at,
                }
            )
    return alias_rows


def _sequence_view_feature_vector_specs(
    *,
    contexts: List[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    bundle: SequenceFeatureBundleConfig,
    selector: str,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for row_index, (context, metadata) in enumerate(zip(contexts, metadata_rows, strict=True)):
        if context.source_dataset_id is None or context.source_dataset_root is None:
            continue
        forward_pass_key = str(metadata["forward_pass_key"])
        pool_scope = str(context.pooling_operation or "seq_mean")
        if bundle.collect_output_layer_mean:
            feature_vector_key = (
                str(metadata["feature_vector_key"])
                if not bundle.collect_intermediate_embedding and metadata.get("feature_vector_key") is not None
                else _feature_vector_key_for_representation(
                    context=context,
                    forward_pass_key=forward_pass_key,
                    representation_kind="output_layer_mean",
                    selector=selector,
                )
            )
            specs.append(
                {
                    "row_index": row_index,
                    "out_id": _output_layer_out_id(pool_scope),
                    "feature_vector_key": feature_vector_key,
                    "dataset_root": context.source_dataset_root,
                    "dataset_id": context.source_dataset_id,
                }
            )
        if bundle.collect_intermediate_embedding:
            feature_vector_key = (
                str(metadata["feature_vector_key"])
                if metadata.get("feature_vector_key") is not None
                else _feature_vector_key_for_representation(
                    context=context,
                    forward_pass_key=forward_pass_key,
                    representation_kind="intermediate_embedding",
                    selector=selector,
                )
            )
            specs.append(
                {
                    "row_index": row_index,
                    "out_id": _intermediate_out_id(selector, pool_scope),
                    "feature_vector_key": feature_vector_key,
                    "dataset_root": context.source_dataset_root,
                    "dataset_id": context.source_dataset_id,
                }
            )
    return specs


def _sequence_view_feature_scalar_specs(
    *,
    contexts: List[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    bundle: SequenceFeatureBundleConfig,
) -> list[dict[str, object]]:
    if not bundle.collect_log_likelihood:
        return []
    specs: list[dict[str, object]] = []
    scalar_key_cache: dict[tuple[str, str], str] = {}
    for row_index, (context, metadata) in enumerate(zip(contexts, metadata_rows, strict=True)):
        if context.source_dataset_id is None or context.source_dataset_root is None:
            continue
        for out_id in (_LOG_LIKELIHOOD_TOTAL, _LOG_LIKELIHOOD_MEAN):
            forward_pass_key = str(metadata["forward_pass_key"])
            scalar_cache_key = (forward_pass_key, out_id)
            feature_scalar_key = scalar_key_cache.get(scalar_cache_key)
            if feature_scalar_key is None:
                feature_scalar_key = _feature_scalar_key_for_context(metadata=metadata, scalar_kind=out_id)
                scalar_key_cache[scalar_cache_key] = feature_scalar_key
            specs.append(
                {
                    "row_index": row_index,
                    "out_id": out_id,
                    "feature_scalar_key": feature_scalar_key,
                    "dataset_root": context.source_dataset_root,
                    "dataset_id": context.source_dataset_id,
                }
            )
    return specs


def _apply_persisted_sequence_view_vectors(
    *,
    all_vals: Dict[str, List[object]],
    specs: list[dict[str, object]],
) -> None:
    keys_by_dataset: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        keys_by_dataset.setdefault(dataset_key, set()).add(str(spec["feature_vector_key"]))

    loaded_by_dataset: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (dataset_root, dataset_id), keys in keys_by_dataset.items():
        loaded_by_dataset[(dataset_root, dataset_id)] = load_feature_vector_rows(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            keys=keys,
        )

    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        value = loaded_by_dataset.get(dataset_key, {}).get(str(spec["feature_vector_key"]))
        if value is None:
            continue
        all_vals[str(spec["out_id"])][int(spec["row_index"])] = value


def _apply_persisted_sequence_view_scalars(
    *,
    all_vals: Dict[str, List[object]],
    specs: list[dict[str, object]],
) -> None:
    keys_by_dataset: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        keys_by_dataset.setdefault(dataset_key, set()).add(str(spec["feature_scalar_key"]))

    loaded_by_dataset: dict[tuple[str, str], dict[str, float]] = {}
    for (dataset_root, dataset_id), keys in keys_by_dataset.items():
        loaded_by_dataset[(dataset_root, dataset_id)] = load_feature_scalar_rows(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            keys=keys,
        )

    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        value = loaded_by_dataset.get(dataset_key, {}).get(str(spec["feature_scalar_key"]))
        if value is None:
            continue
        all_vals[str(spec["out_id"])][int(spec["row_index"])] = float(value)


def _persist_sequence_view_feature_vectors(
    *,
    all_vals: Dict[str, List[object]],
    specs: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
) -> None:
    rows: list[dict[str, object]] = []
    for spec in specs:
        row_index = int(spec["row_index"])
        value = all_vals[str(spec["out_id"])][row_index]
        if value is None:
            continue
        rows.append(
            {
                "_dataset_root": spec["dataset_root"],
                "_dataset_id": spec["dataset_id"],
                "feature_vector_key": spec["feature_vector_key"],
                "value": [float(item) for item in value],
                "created_at": str(metadata_rows[row_index]["timestamp"]),
            }
        )
    if rows:
        persist_feature_vector_rows(rows)


def _persist_sequence_view_feature_scalars(
    *,
    all_vals: Dict[str, List[object]],
    specs: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
) -> None:
    rows: list[dict[str, object]] = []
    for spec in specs:
        row_index = int(spec["row_index"])
        value = all_vals[str(spec["out_id"])][row_index]
        if value is None:
            continue
        rows.append(
            {
                "_dataset_root": spec["dataset_root"],
                "_dataset_id": spec["dataset_id"],
                "feature_scalar_key": spec["feature_scalar_key"],
                "value": float(value),
                "created_at": str(metadata_rows[row_index]["timestamp"]),
            }
        )
    if rows:
        persist_feature_scalar_rows(rows)


def _record_sequence_view_feature_bundle_complete(
    *,
    contexts: list[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    vector_specs: list[dict[str, object]],
    scalar_specs: list[dict[str, object]],
    job_id: str,
    model_id: str,
    run_elapsed_seconds: float | None = None,
) -> None:
    dataset_keys = {
        (str(context.source_dataset_root), str(context.source_dataset_id))
        for context in contexts
        if context.source_dataset_root is not None and context.source_dataset_id is not None
    }
    for dataset_root, dataset_id in sorted(dataset_keys):
        dataset_context_indexes = [
            row_index
            for row_index, context in enumerate(contexts)
            if str(context.source_dataset_root) == dataset_root and str(context.source_dataset_id) == dataset_id
        ]
        dataset_row_indexes = set(dataset_context_indexes)
        vector_keys = {
            str(spec["feature_vector_key"])
            for spec in vector_specs
            if str(spec["dataset_root"]) == dataset_root and str(spec["dataset_id"]) == dataset_id
        }
        scalar_keys = {
            str(spec["feature_scalar_key"])
            for spec in scalar_specs
            if str(spec["dataset_root"]) == dataset_root and str(spec["dataset_id"]) == dataset_id
        }
        forward_passes = {str(metadata_rows[row_index]["forward_pass_key"]) for row_index in dataset_row_indexes}
        record_feature_bundle_complete(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            job_id=job_id,
            model_id=model_id,
            contexts_completed=len(dataset_context_indexes),
            unique_forward_passes=len(forward_passes),
            required_vector_keys=len(vector_keys),
            required_scalar_keys=len(scalar_keys),
            run_elapsed_seconds=run_elapsed_seconds,
        )


def _sequence_view_dataset_key(context: SequenceContextRecord) -> tuple[str, str] | None:
    if context.source_dataset_root is None or context.source_dataset_id is None:
        return None
    return (str(context.source_dataset_root), str(context.source_dataset_id))


def _build_sequence_view_feature_progress_state(
    *,
    contexts: list[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    vector_specs: list[dict[str, object]],
    scalar_specs: list[dict[str, object]],
    need_idx: list[int],
) -> dict[tuple[str, str], dict[str, object]]:
    state: dict[tuple[str, str], dict[str, object]] = {}
    need_set = {int(row_index) for row_index in need_idx}
    for row_index in sorted(need_set):
        key = _sequence_view_dataset_key(contexts[row_index])
        if key is None:
            continue
        entry = state.setdefault(
            key,
            {
                "contexts_total": 0,
                "contexts_completed": 0,
                "forward_total": set(),
                "forward_completed": set(),
                "required_vector_keys": set(),
                "required_scalar_keys": set(),
                "last_step": -1,
                "last_emit_monotonic": None,
            },
        )
        entry["contexts_total"] = int(entry["contexts_total"]) + 1
        forward_total = entry["forward_total"]
        if isinstance(forward_total, set):
            forward_total.add(str(metadata_rows[row_index]["forward_pass_key"]))
    for spec in vector_specs:
        key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        entry = state.setdefault(
            key,
            {
                "contexts_total": 0,
                "contexts_completed": 0,
                "forward_total": set(),
                "forward_completed": set(),
                "required_vector_keys": set(),
                "required_scalar_keys": set(),
                "last_step": -1,
                "last_emit_monotonic": None,
            },
        )
        required_vector_keys = entry.get("required_vector_keys")
        if isinstance(required_vector_keys, set):
            required_vector_keys.add(str(spec["feature_vector_key"]))
    for spec in scalar_specs:
        key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        entry = state.setdefault(
            key,
            {
                "contexts_total": 0,
                "contexts_completed": 0,
                "forward_total": set(),
                "forward_completed": set(),
                "required_vector_keys": set(),
                "required_scalar_keys": set(),
                "last_step": -1,
                "last_emit_monotonic": None,
            },
        )
        required_scalar_keys = entry.get("required_scalar_keys")
        if isinstance(required_scalar_keys, set):
            required_scalar_keys.add(str(spec["feature_scalar_key"]))
    return state


def _maybe_record_sequence_view_feature_progress(
    *,
    progress_state: dict[tuple[str, str], dict[str, object]],
    contexts: list[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    completed_row_indexes: list[int],
    job_id: str,
    model_id: str,
    run_elapsed_seconds: float | None,
) -> None:
    touched: set[tuple[str, str]] = set()
    for row_index in completed_row_indexes:
        key = _sequence_view_dataset_key(contexts[row_index])
        if key is None:
            continue
        entry = progress_state.get(key)
        if entry is None:
            continue
        entry["contexts_completed"] = int(entry.get("contexts_completed", 0)) + 1
        forward_completed = entry.get("forward_completed")
        if isinstance(forward_completed, set):
            forward_completed.add(str(metadata_rows[row_index]["forward_pass_key"]))
        touched.add(key)

    now = time.monotonic()
    for dataset_root, dataset_id in sorted(touched):
        entry = progress_state[(dataset_root, dataset_id)]
        contexts_total = int(entry.get("contexts_total", 0))
        if contexts_total <= 0:
            continue
        contexts_completed = int(entry.get("contexts_completed", 0))
        progress_pct = float(contexts_completed) * 100.0 / float(contexts_total)
        if progress_pct >= 100.0:
            continue
        step_pct = (
            _FEATURE_BUNDLE_PROGRESS_STEP_PCT_SMALL_TARGET
            if contexts_total <= _FEATURE_BUNDLE_SMALL_TARGET_CONTEXTS_THRESHOLD
            else _FEATURE_BUNDLE_PROGRESS_STEP_PCT_LARGE_TARGET
        )
        progress_step = int(progress_pct // float(step_pct))
        last_step = int(entry.get("last_step", -1))
        last_emit = entry.get("last_emit_monotonic")
        elapsed_since_emit = None if not isinstance(last_emit, float) else now - last_emit
        first_emit = last_emit is None
        step_emit = progress_step > last_step
        heartbeat_emit = (
            elapsed_since_emit is not None and elapsed_since_emit >= _FEATURE_BUNDLE_PROGRESS_HEARTBEAT_SECONDS
        )
        if not first_emit and not step_emit and not heartbeat_emit:
            continue
        entry["last_step"] = max(last_step, progress_step)
        entry["last_emit_monotonic"] = now
        forward_completed = entry.get("forward_completed")
        forward_total = entry.get("forward_total")
        required_vector_keys = entry.get("required_vector_keys")
        required_scalar_keys = entry.get("required_scalar_keys")
        record_feature_bundle_progress(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            job_id=job_id,
            model_id=model_id,
            contexts_completed=contexts_completed,
            contexts_total=contexts_total,
            unique_forward_passes_completed=len(forward_completed) if isinstance(forward_completed, set) else 0,
            unique_forward_passes_total=len(forward_total) if isinstance(forward_total, set) else 0,
            required_vector_keys=len(required_vector_keys) if isinstance(required_vector_keys, set) else 0,
            required_scalar_keys=len(required_scalar_keys) if isinstance(required_scalar_keys, set) else 0,
            run_elapsed_seconds=run_elapsed_seconds,
        )


def _persist_sequence_view_feature_sidecars(
    *,
    contexts: list[SequenceContextRecord],
    metadata_rows: list[dict[str, object]],
    bundle: SequenceFeatureBundleConfig,
    selector: str,
    model_id: str,
    job_id: str,
    all_vals: Dict[str, List[object]],
    vector_specs: list[dict[str, object]],
    scalar_specs: list[dict[str, object]],
    run_elapsed_seconds: float | None = None,
) -> None:
    alias_rows = _sequence_view_feature_alias_rows(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector,
        model_id=model_id,
    )
    if alias_rows:
        persist_feature_alias_rows(alias_rows)
    scalar_alias_rows = _sequence_view_feature_scalar_alias_rows(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        model_id=model_id,
    )
    if scalar_alias_rows:
        persist_feature_scalar_alias_rows(scalar_alias_rows)
    if bundle.deduplicate.by_feature_vector_key:
        _persist_sequence_view_feature_vectors(all_vals=all_vals, specs=vector_specs, metadata_rows=metadata_rows)
    _persist_sequence_view_feature_scalars(all_vals=all_vals, specs=scalar_specs, metadata_rows=metadata_rows)
    _record_sequence_view_feature_bundle_complete(
        contexts=contexts,
        metadata_rows=metadata_rows,
        vector_specs=vector_specs,
        scalar_specs=scalar_specs,
        job_id=job_id,
        model_id=model_id,
        run_elapsed_seconds=run_elapsed_seconds,
    )


def _execute_sequence_view_feature_bundle(
    *,
    seqs: List[str],
    records,
    model_id: str,
    job_id: str,
    bundle: SequenceFeatureBundleConfig,
    existing: Mapping[str, List[object]],
    need_idx: List[int],
    adapter,
    micro_batch_size: int,
    default_batch_size: int,
    auto_derate: bool,
    is_oom: Callable[[BaseException], bool],
    on_progress: Callable[[int], None],
    adapter_factory: Callable[[], object] | None = None,
) -> tuple[Dict[str, List[object]], list[dict[str, object]]]:
    if records is None:
        raise CapabilityError("Sequence-view feature bundles require materialized record payloads.")
    contexts = resolve_sequence_view_contexts(records=list(records))
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id=model_id)
    metadata_columnar = build_feature_metadata_columnar(metadata_rows)
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    all_vals: Dict[str, List[object]] = {key: list(value) for key, value in existing.items()}
    for out_id in all_vals:
        if len(all_vals[out_id]) < len(seqs):
            all_vals[out_id] = list(all_vals[out_id]) + [None] * (len(seqs) - len(all_vals[out_id]))
    vector_specs = _sequence_view_feature_vector_specs(
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
    for spec in vector_specs:
        all_vals.setdefault(str(spec["out_id"]), [None] * len(seqs))
    for spec in scalar_specs:
        all_vals.setdefault(str(spec["out_id"]), [None] * len(seqs))
    if bundle.deduplicate.by_feature_vector_key:
        _apply_persisted_sequence_view_vectors(all_vals=all_vals, specs=vector_specs)
    _apply_persisted_sequence_view_scalars(all_vals=all_vals, specs=scalar_specs)
    missing_feature_rows = {
        int(spec["row_index"]) for spec in vector_specs if all_vals[str(spec["out_id"])][int(spec["row_index"])] is None
    }
    missing_scalar_rows = {
        int(spec["row_index"]) for spec in scalar_specs if all_vals[str(spec["out_id"])][int(spec["row_index"])] is None
    }
    missing_feature_rows.update(missing_scalar_rows)
    if need_idx:
        need_idx = sorted(set(need_idx).intersection(missing_feature_rows))
    else:
        need_idx = sorted(missing_feature_rows)

    run_started_monotonic = time.monotonic()
    if not need_idx:
        _persist_sequence_view_feature_sidecars(
            contexts=contexts,
            metadata_rows=metadata_rows,
            bundle=bundle,
            selector=selector.intermediate_selector,
            model_id=model_id,
            job_id=job_id,
            all_vals=all_vals,
            vector_specs=vector_specs,
            scalar_specs=scalar_specs,
            run_elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        return {**all_vals, **metadata_columnar}, metadata_rows

    adapter = _resolve_feature_bundle_adapter(adapter, adapter_factory)
    stable_eval_batch_size = None
    start_bs = micro_batch_size if micro_batch_size > 0 else min(len(need_idx), default_batch_size)
    bs = start_bs
    unique_need = sorted(set(need_idx))
    forward_groups: dict[str, list[int]] = {}
    representative_contexts: list[SequenceContextRecord] = []
    representative_group_keys: list[str] = []
    representative_forward_keys: list[str] = []
    for row_index in unique_need:
        forward_pass_key = str(metadata_rows[row_index]["forward_pass_key"])
        group_key = (
            forward_pass_key if bundle.deduplicate.by_forward_pass_key else f"{forward_pass_key}:row:{row_index}"
        )
        members = forward_groups.setdefault(group_key, [])
        if not members:
            representative_contexts.append(contexts[row_index])
            representative_group_keys.append(group_key)
            representative_forward_keys.append(forward_pass_key)
        members.append(row_index)
    progress_state = _build_sequence_view_feature_progress_state(
        contexts=contexts,
        metadata_rows=metadata_rows,
        vector_specs=vector_specs,
        scalar_specs=scalar_specs,
        need_idx=unique_need,
    )

    start = 0
    while start < len(representative_contexts):
        take = min(bs, len(representative_contexts) - start)
        context_chunk = representative_contexts[start : start + take]
        group_key_chunk = representative_group_keys[start : start + take]
        forward_key_chunk = representative_forward_keys[start : start + take]
        eval_seq_chunk = [context.resolved_sequence for context in context_chunk]
        try:
            scalar_by_position: dict[int, tuple[float, float]] = {}
            logits_by_position: dict[int, object] = {}
            embedding_by_position: dict[int, object] = {}
            scalar_positions: list[int] = []
            logits_positions: list[int] = []
            embedding_positions: list[int] = []
            for position, group_key in enumerate(group_key_chunk):
                needs_scalars, needs_logits, needs_embedding = _sequence_view_group_output_needs(
                    all_vals=all_vals,
                    contexts=contexts,
                    row_indexes=forward_groups[group_key],
                    bundle=bundle,
                    selector=selector.intermediate_selector,
                )
                if needs_scalars:
                    scalar_positions.append(position)
                if needs_logits:
                    logits_positions.append(position)
                if needs_embedding:
                    embedding_positions.append(position)

            if scalar_positions:
                totals, means = _feature_bundle_log_likelihoods(
                    adapter,
                    seq_chunk=[eval_seq_chunk[position] for position in scalar_positions],
                )
                for local_index, position in enumerate(scalar_positions):
                    scalar_by_position[position] = (totals[local_index], means[local_index])

            logits_position_set = set(logits_positions)
            embedding_position_set = set(embedding_positions)
            fused_positions = sorted(logits_position_set.intersection(embedding_position_set))
            if fused_positions:
                logits_tensors, embedding_tensors = _feature_bundle_logits_and_embedding(
                    adapter,
                    seq_chunk=[eval_seq_chunk[position] for position in fused_positions],
                    selector=selector.intermediate_selector,
                )
                if logits_tensors is not None and embedding_tensors is not None:
                    for local_index, position in enumerate(fused_positions):
                        logits_by_position[position] = logits_tensors[local_index]
                        embedding_by_position[position] = embedding_tensors[local_index]
                else:
                    logits_positions = sorted(logits_position_set)
                    embedding_positions = sorted(embedding_position_set)

            missing_logits_positions = [position for position in logits_positions if position not in logits_by_position]
            if missing_logits_positions:
                logits_tensors = adapter.logits(
                    [eval_seq_chunk[position] for position in missing_logits_positions],
                    fmt="tensor",
                )
                for local_index, position in enumerate(missing_logits_positions):
                    logits_by_position[position] = logits_tensors[local_index]

            missing_embedding_positions = [
                position for position in embedding_positions if position not in embedding_by_position
            ]
            if missing_embedding_positions:
                embedding_tensors = adapter.embedding(
                    [eval_seq_chunk[position] for position in missing_embedding_positions],
                    layer=selector.intermediate_selector,
                    fmt="tensor",
                )
                for local_index, position in enumerate(missing_embedding_positions):
                    embedding_by_position[position] = embedding_tensors[local_index]
        except RuntimeError as exc:
            if is_oom(exc) and stable_eval_batch_size is not None:
                raise RuntimeOOMError(
                    "Feature bundle requires fixed evaluation batch size "
                    f"{stable_eval_batch_size} for stable resume values; refusing to auto-derate."
                ) from exc
            if is_oom(exc) and auto_derate and bs > 1:
                bs = max(1, bs // 2)
                continue
            raise RuntimeOOMError(str(exc)) from exc

        completed_rows = 0
        completed_row_indexes: list[int] = []
        for position, group_key in enumerate(group_key_chunk):
            forward_pass_key = forward_key_chunk[position]
            row_indexes = forward_groups[group_key]
            completed_rows += len(row_indexes)
            completed_row_indexes.extend(row_indexes)
            scalar_pair = scalar_by_position.get(position)
            if scalar_pair is not None:
                total, mean = scalar_pair
                for row_index in row_indexes:
                    if all_vals[_LOG_LIKELIHOOD_TOTAL][row_index] is None:
                        all_vals[_LOG_LIKELIHOOD_TOTAL][row_index] = total
                    if all_vals[_LOG_LIKELIHOOD_MEAN][row_index] is None:
                        all_vals[_LOG_LIKELIHOOD_MEAN][row_index] = mean
            logits_tensor = logits_by_position.get(position)
            embedding_tensor = embedding_by_position.get(position)
            pooled_cache: dict[tuple[object, ...], list[float]] = {}
            for row_index in row_indexes:
                context = contexts[row_index]
                pool_scope = str(context.pooling_operation or "seq_mean")
                if bundle.collect_output_layer_mean:
                    output_out_id = _output_layer_out_id(pool_scope)
                    if all_vals[output_out_id][row_index] is not None:
                        continue
                    assert logits_tensor is not None
                    pooling_key = (
                        "output_layer_mean",
                        _feature_vector_key_for_representation(
                            context=context,
                            forward_pass_key=forward_pass_key,
                            representation_kind="output_layer_mean",
                            selector=selector.intermediate_selector,
                        ),
                    )
                    cache_key = pooling_key if bundle.deduplicate.by_feature_vector_key else (*pooling_key, row_index)
                    if cache_key not in pooled_cache:
                        pooled_cache[cache_key] = _pool_tensor_for_context(logits_tensor, context=context)
                    all_vals[output_out_id][row_index] = pooled_cache[cache_key]
                if bundle.collect_intermediate_embedding:
                    intermediate_out_id = _intermediate_out_id(selector.intermediate_selector, pool_scope)
                    if all_vals[intermediate_out_id][row_index] is not None:
                        continue
                    assert embedding_tensor is not None
                    pooling_key = (
                        "intermediate_embedding",
                        _feature_vector_key_for_representation(
                            context=context,
                            forward_pass_key=forward_pass_key,
                            representation_kind="intermediate_embedding",
                            selector=selector.intermediate_selector,
                        ),
                    )
                    cache_key = pooling_key if bundle.deduplicate.by_feature_vector_key else (*pooling_key, row_index)
                    if cache_key not in pooled_cache:
                        pooled_cache[cache_key] = _pool_tensor_for_context(embedding_tensor, context=context)
                    all_vals[intermediate_out_id][row_index] = pooled_cache[cache_key]
        on_progress(completed_rows)
        _maybe_record_sequence_view_feature_progress(
            progress_state=progress_state,
            contexts=contexts,
            metadata_rows=metadata_rows,
            completed_row_indexes=completed_row_indexes,
            job_id=job_id,
            model_id=model_id,
            run_elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        start += take

    _persist_sequence_view_feature_sidecars(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
        model_id=model_id,
        job_id=job_id,
        all_vals=all_vals,
        vector_specs=vector_specs,
        scalar_specs=scalar_specs,
        run_elapsed_seconds=time.monotonic() - run_started_monotonic,
    )
    return {**all_vals, **metadata_columnar}, metadata_rows


def execute_feature_bundle(
    *,
    seqs: List[str],
    source: str,
    ids: Optional[List[str]],
    records,
    ds,
    model_id: str,
    job_id: str,
    bundle: SequenceFeatureBundleConfig,
    existing: Mapping[str, List[object]],
    need_idx: List[int],
    adapter,
    micro_batch_size: int,
    default_batch_size: int,
    auto_derate: bool,
    is_oom: Callable[[BaseException], bool],
    on_progress: Callable[[int], None],
    on_chunk_by_output: Mapping[str, Optional[Callable[..., None]]],
    on_chunk_by_metadata: Mapping[str, Optional[Callable[..., None]]],
    on_chunk_output_group: Optional[Callable[..., None]] = None,
    on_chunk_metadata_group: Optional[Callable[..., None]] = None,
    adapter_factory: Callable[[], object] | None = None,
) -> tuple[Dict[str, List[object]], list[dict[str, object]]]:
    if bundle_uses_sequence_views(bundle):
        return _execute_sequence_view_feature_bundle(
            seqs=seqs,
            records=records,
            model_id=model_id,
            job_id=job_id,
            bundle=bundle,
            existing=existing,
            need_idx=need_idx,
            adapter=adapter,
            adapter_factory=adapter_factory,
            micro_batch_size=micro_batch_size,
            default_batch_size=default_batch_size,
            auto_derate=auto_derate,
            is_oom=is_oom,
            on_progress=on_progress,
        )
    contexts = resolve_sequence_contexts(
        seqs=seqs,
        source=source,
        ids=ids,
        records=records,
        ds=ds,
        bundle=bundle,
    )
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id=model_id)
    metadata_columnar = build_feature_metadata_columnar(metadata_rows)
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)

    all_vals: Dict[str, List[object]] = {key: list(value) for key, value in existing.items()}
    metadata_existing = _existing_feature_metadata_values(
        ds=ds if source == "usr" else None,
        ids=ids,
        model_id=model_id,
        job_id=job_id,
    )
    stale_idx = _apply_digest_resume_guard(
        ds=ds if source == "usr" else None,
        ids=ids,
        model_id=model_id,
        job_id=job_id,
        feature_values=all_vals,
        metadata_columnar=metadata_columnar,
        existing_digests=metadata_existing.get("metadata__feature_request_digest"),
    )
    stale_idx_set = set(stale_idx)
    feature_resume_idx = set(need_idx) | stale_idx_set

    metadata_missing_by_output = _missing_rows_by_output(metadata_existing)
    metadata_missing_idx = set().union(*metadata_missing_by_output.values()) if metadata_missing_by_output else set()
    metadata_only_idx = metadata_missing_idx - feature_resume_idx
    stable_eval_batch_size = _stable_feature_bundle_eval_batch_size(
        model_id=model_id,
        bundle=bundle,
        micro_batch_size=micro_batch_size,
    )
    feature_target_rows_by_output = {
        out_id: len({row_index for row_index, value in enumerate(values) if value is None} | stale_idx_set)
        for out_id, values in existing.items()
    }
    feature_target_units_by_family: Dict[str, int] = {}
    for out_id, target_rows in feature_target_rows_by_output.items():
        family = infer_output_family(out_id)
        if infer_output_kind(out_id) != "feature":
            continue
        feature_target_units_by_family[family] = feature_target_units_by_family.get(family, 0) + int(target_rows)
    total_feature_units = sum(feature_target_units_by_family.values())
    feature_written_rows_by_output = {out_id: 0 for out_id in feature_target_rows_by_output}
    feature_written_units_by_family = {family: 0 for family in feature_target_units_by_family}
    metadata_target_rows_by_output = {
        out_id: len(set(row_indexes) | stale_idx_set) for out_id, row_indexes in metadata_missing_by_output.items()
    }
    metadata_written_rows_by_output = {out_id: 0 for out_id in metadata_target_rows_by_output}

    def _feature_progress(out_id: str, *, rows_written_now: int) -> dict[str, object]:
        if rows_written_now <= 0:
            return {}
        family = infer_output_family(out_id)
        target_rows = feature_target_rows_by_output.get(out_id, 0)
        feature_written_rows_by_output[out_id] = min(
            target_rows,
            feature_written_rows_by_output.get(out_id, 0) + int(rows_written_now),
        )
        if infer_output_kind(out_id) != "feature":
            return {"infer_notify_suppress": True}
        family_target_units = feature_target_units_by_family.get(family, 0)
        feature_written_units_by_family[family] = min(
            family_target_units,
            feature_written_units_by_family.get(family, 0) + int(rows_written_now),
        )
        overall_completed_units = sum(feature_written_units_by_family.values())
        family_progress_pct_map = {
            family_name: round(
                _progress_pct(completed=completed_units, target=feature_target_units_by_family[family_name]),
                1,
            )
            for family_name, completed_units in feature_written_units_by_family.items()
        }
        return {
            "infer_progress": {
                "target_rows": target_rows,
                "completed_rows": feature_written_rows_by_output[out_id],
                "output_progress_pct": _progress_pct(
                    completed=feature_written_rows_by_output[out_id],
                    target=target_rows,
                ),
                "family_target_units": family_target_units,
                "family_completed_units": feature_written_units_by_family[family],
                "family_progress_pct": _progress_pct(
                    completed=feature_written_units_by_family[family],
                    target=family_target_units,
                ),
                "family_progress_pct_map": family_progress_pct_map,
                "overall_target_units": total_feature_units,
                "overall_completed_units": overall_completed_units,
                "overall_progress_pct": _progress_pct(
                    completed=overall_completed_units,
                    target=total_feature_units,
                ),
            }
        }

    def _feature_group_progress() -> dict[str, object]:
        overall_completed_units = sum(feature_written_units_by_family.values())
        family_progress_pct_map = {
            family_name: round(
                _progress_pct(completed=completed_units, target=feature_target_units_by_family[family_name]),
                1,
            )
            for family_name, completed_units in feature_written_units_by_family.items()
        }
        return {
            "infer_progress": {
                "family_progress_pct_map": family_progress_pct_map,
                "overall_target_units": total_feature_units,
                "overall_completed_units": overall_completed_units,
                "overall_progress_pct": _progress_pct(
                    completed=overall_completed_units,
                    target=total_feature_units,
                ),
            }
        }

    def _metadata_progress(out_id: str, *, rows_written_now: int) -> dict[str, object]:
        if rows_written_now > 0:
            target_rows = metadata_target_rows_by_output.get(out_id, 0)
            metadata_written_rows_by_output[out_id] = min(
                target_rows,
                metadata_written_rows_by_output.get(out_id, 0) + int(rows_written_now),
            )
        else:
            target_rows = metadata_target_rows_by_output.get(out_id, 0)
        return {
            "infer_notify_suppress": True,
            "infer_progress": {
                "target_rows": target_rows,
                "completed_rows": metadata_written_rows_by_output.get(out_id, 0),
                "output_progress_pct": _progress_pct(
                    completed=metadata_written_rows_by_output.get(out_id, 0),
                    target=target_rows,
                ),
            },
        }

    if on_chunk_metadata_group is not None:
        metadata_only_groups = _group_columnar_by_row_indexes(
            columnar=metadata_columnar,
            row_indexes_by_output={
                out_id: metadata_missing_by_output[out_id].intersection(metadata_only_idx)
                for out_id in metadata_columnar
            },
        )
        for row_indexes, grouped_columnar in metadata_only_groups:
            on_chunk_metadata_group(
                row_indexes,
                grouped_columnar,
                event_args={"infer_notify_suppress": True},
            )
    else:
        for out_id, values in metadata_columnar.items():
            metadata_row_indexes = sorted(metadata_missing_by_output[out_id].intersection(metadata_only_idx))
            _write_chunk_subset(
                writer=on_chunk_by_metadata.get(out_id),
                idx_chunk=metadata_row_indexes,
                values=[values[row_index] for row_index in metadata_row_indexes],
                row_indexes=metadata_missing_by_output[out_id],
                progress=_metadata_progress(out_id, rows_written_now=len(metadata_row_indexes)),
            )

    need_idx = sorted(feature_resume_idx)
    if len(need_idx) == 0:
        return {**all_vals, **metadata_columnar}, metadata_rows

    adapter = _resolve_feature_bundle_adapter(adapter, adapter_factory)
    start_bs = (
        stable_eval_batch_size
        if stable_eval_batch_size is not None
        else (micro_batch_size if micro_batch_size > 0 else min(len(need_idx), default_batch_size))
    )
    bs = start_bs
    start = 0
    while start < len(need_idx):
        take = min(bs, len(need_idx) - start)
        idx_chunk = need_idx[start : start + take]
        seq_chunk = [seqs[row_index] for row_index in idx_chunk]
        context_chunk = [contexts[row_index] for row_index in idx_chunk]
        eval_seq_chunk = _pad_feature_bundle_eval_sequences(
            seq_chunk=seq_chunk,
            eval_batch_size=stable_eval_batch_size,
        )

        try:
            chunk_outputs: dict[str, list[object]] = {}
            logits_tensors = None
            embedding_tensors = None

            if bundle.collect_log_likelihood:
                totals, means = _feature_bundle_log_likelihoods(adapter, eval_seq_chunk)
                chunk_outputs[_LOG_LIKELIHOOD_TOTAL] = totals[: len(idx_chunk)]
                chunk_outputs[_LOG_LIKELIHOOD_MEAN] = means[: len(idx_chunk)]

            if bundle.collect_output_layer_mean and bundle.collect_intermediate_embedding:
                logits_tensors, embedding_tensors = _feature_bundle_logits_and_embedding(
                    adapter,
                    seq_chunk=eval_seq_chunk,
                    selector=selector.intermediate_selector,
                )

            if bundle.collect_output_layer_mean:
                if logits_tensors is None:
                    logits_tensors = adapter.logits(eval_seq_chunk, fmt="tensor")
                seq_means: list[list[float]] = []
                anchor_means: list[list[float]] = []
                for tensor, context in zip(logits_tensors[: len(idx_chunk)], context_chunk, strict=True):
                    seq_mean, anchor_mean = _pool_tensor_scopes(tensor, context=context)
                    if bundle.pooling.seq_mean:
                        seq_means.append(seq_mean)
                    anchor_means.append(anchor_mean)
                if bundle.pooling.seq_mean:
                    chunk_outputs[_OUTPUT_LAYER_SEQ_MEAN] = seq_means
                if _templated_anchor_mean_enabled(bundle):
                    chunk_outputs[_OUTPUT_LAYER_ANCHOR_MEAN] = anchor_means

            if bundle.collect_intermediate_embedding:
                if embedding_tensors is None:
                    embedding_tensors = adapter.embedding(
                        eval_seq_chunk,
                        layer=selector.intermediate_selector,
                        fmt="tensor",
                    )
                seq_means = []
                anchor_means = []
                for tensor, context in zip(embedding_tensors[: len(idx_chunk)], context_chunk, strict=True):
                    seq_mean, anchor_mean = _pool_tensor_scopes(tensor, context=context)
                    if bundle.pooling.seq_mean:
                        seq_means.append(seq_mean)
                    anchor_means.append(anchor_mean)
                intermediate_seq_id = f"intermediate_embedding__{selector.intermediate_selector}__seq_mean"
                if bundle.pooling.seq_mean:
                    chunk_outputs[intermediate_seq_id] = seq_means
                if _templated_anchor_mean_enabled(bundle):
                    intermediate_anchor_id = f"intermediate_embedding__{selector.intermediate_selector}__anchor_mean"
                    chunk_outputs[intermediate_anchor_id] = anchor_means

        except RuntimeError as exc:
            if is_oom(exc) and stable_eval_batch_size is not None:
                raise RuntimeOOMError(
                    "Feature bundle requires fixed evaluation batch size "
                    f"{stable_eval_batch_size} for stable resume values; refusing to auto-derate."
                ) from exc
            if is_oom(exc) and auto_derate and bs > 1:
                bs = max(1, bs // 2)
                continue
            raise RuntimeOOMError(str(exc)) from exc

        for out_id, values in chunk_outputs.items():
            if len(values) != len(idx_chunk):
                raise CapabilityError(f"Feature bundle returned wrong number of outputs for '{out_id}'.")
            target = all_vals[out_id]
            for value_index, row_index in enumerate(idx_chunk):
                target[row_index] = values[value_index]
        if on_chunk_output_group is not None:
            idx_chunk_set = set(idx_chunk)
            metadata_chunk = {
                out_id: [values[row_index] for row_index in idx_chunk] for out_id, values in metadata_columnar.items()
            }
            feature_missing_by_output = {}
            for out_id in chunk_outputs:
                missing_rows = {
                    row_index for row_index in idx_chunk if existing[out_id][row_index] is None
                } - stale_idx_set
                feature_missing_by_output[out_id] = missing_rows
                _feature_progress(out_id, rows_written_now=len(missing_rows))
            combined_missing_groups = _group_columnar_by_row_indexes(
                columnar={**chunk_outputs, **metadata_chunk},
                row_indexes_by_output={
                    **feature_missing_by_output,
                    **{
                        out_id: (metadata_missing_by_output[out_id] - stale_idx_set).intersection(idx_chunk_set)
                        for out_id in metadata_columnar
                    },
                },
                idx_chunk=idx_chunk,
            )
            feature_group_event_args = _feature_group_progress()
            for row_indexes, grouped_columnar in combined_missing_groups:
                on_chunk_output_group(
                    row_indexes,
                    grouped_columnar,
                    event_args=_group_event_args_for_columnar(
                        grouped_columnar=grouped_columnar,
                        feature_event_args=feature_group_event_args,
                    ),
                )
            combined_stale_groups = _group_columnar_by_row_indexes(
                columnar={**chunk_outputs, **metadata_chunk},
                row_indexes_by_output={
                    **{out_id: stale_idx_set.intersection(idx_chunk_set) for out_id in chunk_outputs},
                    **{out_id: stale_idx_set.intersection(idx_chunk_set) for out_id in metadata_columnar},
                },
                idx_chunk=idx_chunk,
            )
            if combined_stale_groups:
                for out_id in chunk_outputs:
                    _feature_progress(out_id, rows_written_now=len(stale_idx_set.intersection(idx_chunk_set)))
                feature_group_event_args = _feature_group_progress()
                for row_indexes, grouped_columnar in combined_stale_groups:
                    on_chunk_output_group(
                        row_indexes,
                        grouped_columnar,
                        overwrite_override=True,
                        event_args=_group_event_args_for_columnar(
                            grouped_columnar=grouped_columnar,
                            feature_event_args=feature_group_event_args,
                        ),
                    )
        else:
            for out_id, values in chunk_outputs.items():
                missing_rows = {
                    row_index for row_index in idx_chunk if existing[out_id][row_index] is None
                } - stale_idx_set
                missing_rows_in_chunk = len(missing_rows)
                _write_chunk_subset(
                    writer=on_chunk_by_output.get(out_id),
                    idx_chunk=idx_chunk,
                    values=values,
                    row_indexes=missing_rows,
                    progress=_feature_progress(out_id, rows_written_now=missing_rows_in_chunk),
                )
                stale_rows_in_chunk = len(stale_idx_set.intersection(idx_chunk))
                _write_chunk_subset(
                    writer=on_chunk_by_output.get(out_id),
                    idx_chunk=idx_chunk,
                    values=values,
                    row_indexes=stale_idx_set,
                    overwrite_override=True,
                    progress=_feature_progress(out_id, rows_written_now=stale_rows_in_chunk),
                )
        if on_chunk_output_group is None:
            if on_chunk_metadata_group is not None:
                idx_chunk_set = set(idx_chunk)
                metadata_chunk = {
                    out_id: [values[row_index] for row_index in idx_chunk]
                    for out_id, values in metadata_columnar.items()
                }
                metadata_missing_groups = _group_columnar_by_row_indexes(
                    columnar=metadata_chunk,
                    row_indexes_by_output={
                        out_id: (metadata_missing_by_output[out_id] - stale_idx_set).intersection(idx_chunk_set)
                        for out_id in metadata_columnar
                    },
                    idx_chunk=idx_chunk,
                )
                for row_indexes, grouped_columnar in metadata_missing_groups:
                    on_chunk_metadata_group(
                        row_indexes,
                        grouped_columnar,
                        event_args={"infer_notify_suppress": True},
                    )
                metadata_stale_groups = _group_columnar_by_row_indexes(
                    columnar=metadata_chunk,
                    row_indexes_by_output={
                        out_id: stale_idx_set.intersection(idx_chunk_set) for out_id in metadata_columnar
                    },
                    idx_chunk=idx_chunk,
                )
                for row_indexes, grouped_columnar in metadata_stale_groups:
                    on_chunk_metadata_group(
                        row_indexes,
                        grouped_columnar,
                        overwrite_override=True,
                        event_args={"infer_notify_suppress": True},
                    )
            else:
                for out_id, values in metadata_columnar.items():
                    chunk_metadata = [values[row_index] for row_index in idx_chunk]
                    missing_rows = metadata_missing_by_output[out_id] - stale_idx_set
                    metadata_missing_rows_in_chunk = len(missing_rows.intersection(idx_chunk))
                    _write_chunk_subset(
                        writer=on_chunk_by_metadata.get(out_id),
                        idx_chunk=idx_chunk,
                        values=chunk_metadata,
                        row_indexes=missing_rows,
                        progress=_metadata_progress(out_id, rows_written_now=metadata_missing_rows_in_chunk),
                    )
                    metadata_stale_rows_in_chunk = len(stale_idx_set.intersection(idx_chunk))
                    _write_chunk_subset(
                        writer=on_chunk_by_metadata.get(out_id),
                        idx_chunk=idx_chunk,
                        values=chunk_metadata,
                        row_indexes=stale_idx_set,
                        overwrite_override=True,
                        progress=_metadata_progress(out_id, rows_written_now=metadata_stale_rows_in_chunk),
                    )
        on_progress(len(idx_chunk))
        start += take

    return {**all_vals, **metadata_columnar}, metadata_rows
