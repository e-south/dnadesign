"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/execution.py

Execution helpers for Evo2 promoter-feature bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional

import torch

from ..contracts import infer_usr_column_name
from ..errors import CapabilityError, RuntimeOOMError
from ..runtime.resume_planner import read_usr_column_values
from .context import resolve_sequence_contexts
from .contracts import PromoterFeatureBundleConfig, SequenceContextRecord
from .selectors import canonical_selector_for_block, resolve_intermediate_selector

_LOG_LIKELIHOOD_TOTAL = "log_likelihood__total"
_LOG_LIKELIHOOD_MEAN = "log_likelihood__mean_per_token"
_OUTPUT_LAYER_SEQ_MEAN = "output_layer_mean__seq_mean"
_OUTPUT_LAYER_ANCHOR_MEAN = "output_layer_mean__anchor_mean"
_METADATA_OUTPUT_FIELDS = (
    ("metadata__sequence_id", "sequence_id"),
    ("metadata__anchor_id", "anchor_id"),
    ("metadata__is_wildtype", "is_wildtype"),
    ("metadata__context_id", "context_id"),
    ("metadata__context_kind", "context_kind"),
    ("metadata__template_id", "template_id"),
    ("metadata__resolved_length", "resolved_length"),
    ("metadata__anchor_start", "anchor_start"),
    ("metadata__anchor_end", "anchor_end"),
    ("metadata__model_name", "model_name"),
    ("metadata__provider_name", "provider_name"),
    ("metadata__provider_version", "provider_version"),
    ("metadata__intermediate_block", "intermediate_block"),
    ("metadata__intermediate_selector", "intermediate_selector"),
    ("metadata__pooling_modes", "pooling_modes"),
    ("metadata__feature_schema_version", "feature_schema_version"),
    ("metadata__construct_version", "construct_version"),
    ("metadata__timestamp", "timestamp"),
    ("metadata__feature_request_digest", "feature_request_digest"),
)


def _templated_anchor_mean_enabled(bundle: PromoterFeatureBundleConfig) -> bool:
    return bundle.context.kind != "anchor_only" and bool(bundle.pooling.anchor_mean_for_templated)


def _pooling_modes(bundle: PromoterFeatureBundleConfig) -> list[str]:
    modes: list[str] = []
    if bundle.pooling.seq_mean:
        modes.append("seq_mean")
    if _templated_anchor_mean_enabled(bundle):
        modes.append("anchor_mean")
    return modes


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


def build_feature_bundle_outputs(*, bundle: PromoterFeatureBundleConfig) -> list[dict[str, object]]:
    selector = canonical_selector_for_block(bundle.intermediate_block)
    outputs: list[dict[str, object]] = []

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
        if bundle.pooling.seq_mean:
            outputs.append(
                {
                    "id": _OUTPUT_LAYER_SEQ_MEAN,
                    "fn": "evo2.logits",
                    "params": {
                        "pool": {"method": "mean", "dim": 1},
                        "feature_group": "output_layer_mean",
                        "pool_scope": "seq_mean",
                    },
                    "format": "list",
                }
            )
        if _templated_anchor_mean_enabled(bundle):
            outputs.append(
                {
                    "id": _OUTPUT_LAYER_ANCHOR_MEAN,
                    "fn": "evo2.logits",
                    "params": {
                        "pool": {"method": "mean", "dim": 1},
                        "feature_group": "output_layer_mean",
                        "pool_scope": "anchor_mean",
                    },
                    "format": "list",
                }
            )

    if bundle.collect_intermediate_embedding:
        if bundle.pooling.seq_mean:
            outputs.append(
                {
                    "id": f"intermediate_embedding__{selector}__seq_mean",
                    "fn": "evo2.embedding",
                    "params": {
                        "layer": selector,
                        "pool": {"method": "mean", "dim": 1},
                        "feature_group": "intermediate_embedding",
                        "intermediate_block": bundle.intermediate_block,
                        "intermediate_selector": selector,
                        "pool_scope": "seq_mean",
                    },
                    "format": "list",
                }
            )
        if _templated_anchor_mean_enabled(bundle):
            outputs.append(
                {
                    "id": f"intermediate_embedding__{selector}__anchor_mean",
                    "fn": "evo2.embedding",
                    "params": {
                        "layer": selector,
                        "pool": {"method": "mean", "dim": 1},
                        "feature_group": "intermediate_embedding",
                        "intermediate_block": bundle.intermediate_block,
                        "intermediate_selector": selector,
                        "pool_scope": "anchor_mean",
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


def _feature_request_digest(
    *,
    bundle: PromoterFeatureBundleConfig,
    context: SequenceContextRecord,
    model_id: str,
    selector: str,
) -> str:
    payload = {
        "feature_schema_version": bundle.feature_schema_version,
        "model_id": model_id,
        "context_id": context.context_id,
        "context_kind": context.context_kind,
        "template_id": context.template_id,
        "resolved_sequence": context.resolved_sequence,
        "intermediate_selector": selector,
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


def build_feature_metadata_rows(
    *,
    contexts: List[SequenceContextRecord],
    bundle: PromoterFeatureBundleConfig,
    model_id: str,
) -> list[dict[str, object]]:
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    timestamp = datetime.now(timezone.utc).isoformat()
    return [
        {
            "sequence_id": context.sequence_id,
            "anchor_id": context.anchor_id,
            "is_wildtype": context.is_wildtype,
            "context_id": context.context_id,
            "context_kind": context.context_kind,
            "template_id": context.template_id,
            "resolved_length": context.resolved_length,
            "anchor_start": context.anchor_start,
            "anchor_end": context.anchor_end,
            "model_name": model_id,
            "provider_name": "evo2",
            "provider_version": None,
            "intermediate_block": selector.intermediate_block,
            "intermediate_selector": selector.intermediate_selector,
            "pooling_modes": _pooling_modes(bundle),
            "feature_schema_version": bundle.feature_schema_version,
            "construct_version": context.construct_version,
            "timestamp": timestamp,
            "feature_request_digest": _feature_request_digest(
                bundle=bundle,
                context=context,
                model_id=model_id,
                selector=selector.intermediate_selector,
            ),
        }
        for context in contexts
    ]


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
) -> List[int]:
    if ds is None or ids is None:
        return []
    digest_out_id = "metadata__feature_request_digest"
    digest_column = infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=digest_out_id)
    existing_digests = read_usr_column_values(ds=ds, ids=ids, column_name=digest_column)
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
    return {
        out_id: read_usr_column_values(
            ds=ds,
            ids=ids,
            column_name=infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=out_id),
        )
        for out_id in feature_metadata_output_ids()
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


def execute_feature_bundle(
    *,
    seqs: List[str],
    source: str,
    ids: Optional[List[str]],
    records,
    ds,
    model_id: str,
    job_id: str,
    bundle: PromoterFeatureBundleConfig,
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
) -> tuple[Dict[str, List[object]], list[dict[str, object]]]:
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
    stale_idx = _apply_digest_resume_guard(
        ds=ds if source == "usr" else None,
        ids=ids,
        model_id=model_id,
        job_id=job_id,
        feature_values=all_vals,
        metadata_columnar=metadata_columnar,
    )
    stale_idx_set = set(stale_idx)
    feature_resume_idx = set(need_idx) | stale_idx_set

    metadata_existing = _existing_feature_metadata_values(
        ds=ds if source == "usr" else None,
        ids=ids,
        model_id=model_id,
        job_id=job_id,
    )
    metadata_missing_by_output = _missing_rows_by_output(metadata_existing)
    metadata_missing_idx = set().union(*metadata_missing_by_output.values()) if metadata_missing_by_output else set()
    metadata_only_idx = metadata_missing_idx - feature_resume_idx
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

    start_bs = micro_batch_size if micro_batch_size > 0 else min(len(need_idx), default_batch_size)
    bs = start_bs
    start = 0
    while start < len(need_idx):
        take = min(bs, len(need_idx) - start)
        idx_chunk = need_idx[start : start + take]
        seq_chunk = [seqs[row_index] for row_index in idx_chunk]
        context_chunk = [contexts[row_index] for row_index in idx_chunk]

        try:
            chunk_outputs: dict[str, list[object]] = {}

            if bundle.collect_log_likelihood:
                chunk_outputs[_LOG_LIKELIHOOD_TOTAL] = adapter.log_likelihood(
                    seq_chunk,
                    method="native",
                    reduction="sum",
                )
                chunk_outputs[_LOG_LIKELIHOOD_MEAN] = adapter.log_likelihood(
                    seq_chunk,
                    method="native",
                    reduction="mean",
                )

            if bundle.collect_output_layer_mean:
                logits_tensors = adapter.logits(seq_chunk, fmt="tensor")
                seq_means: list[list[float]] = []
                anchor_means: list[list[float]] = []
                for tensor, context in zip(logits_tensors, context_chunk, strict=True):
                    seq_mean, anchor_mean = _pool_tensor_scopes(tensor, context=context)
                    if bundle.pooling.seq_mean:
                        seq_means.append(seq_mean)
                    anchor_means.append(anchor_mean)
                if bundle.pooling.seq_mean:
                    chunk_outputs[_OUTPUT_LAYER_SEQ_MEAN] = seq_means
                if _templated_anchor_mean_enabled(bundle):
                    chunk_outputs[_OUTPUT_LAYER_ANCHOR_MEAN] = anchor_means

            if bundle.collect_intermediate_embedding:
                embedding_tensors = adapter.embedding(
                    seq_chunk,
                    layer=selector.intermediate_selector,
                    fmt="tensor",
                )
                seq_means = []
                anchor_means = []
                for tensor, context in zip(embedding_tensors, context_chunk, strict=True):
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
            missing_rows = {row_index for row_index in idx_chunk if existing[out_id][row_index] is None} - stale_idx_set
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
