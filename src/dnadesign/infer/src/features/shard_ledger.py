"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/shard_ledger.py

Shard checkpoint ledger helpers for resumable sequence-view Infer backfill.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_FEATURE_SHARD_SIZE_VIEWS = 50_000
SHARD_LEDGER_SCHEMA_VERSION = "infer_feature_shard_ledger_v1"
SHARD_COMMIT_POLICY = "temp_validate_promote"
SHARD_RESUME_POLICY = "skip_committed_retry_failed"

SHARD_STATUS_PENDING = "pending"
SHARD_STATUS_RUNNING = "running"
SHARD_STATUS_COMMITTED = "committed"
SHARD_STATUS_FAILED = "failed"
VALID_SHARD_STATUSES = frozenset(
    {
        SHARD_STATUS_PENDING,
        SHARD_STATUS_RUNNING,
        SHARD_STATUS_COMMITTED,
        SHARD_STATUS_FAILED,
    }
)


@dataclass(frozen=True)
class FeatureShardLedgerEntry:
    shard_index: int
    status: str = SHARD_STATUS_PENDING
    input_selector: dict[str, object] | None = None
    expected_views: int = 0
    expected_vector_keys: int = 0
    expected_scalar_keys: int = 0
    committed_views: int = 0
    committed_vector_keys: int = 0
    committed_scalar_keys: int = 0
    checksum: str | None = None
    last_heartbeat_at: str | None = None
    updated_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class FeatureShardLedger:
    bundle_id: str
    runtime_fingerprint_key: str
    shard_size_views: int
    shard_count: int
    pending_view_estimate: int
    pending_vector_keys: int
    pending_scalar_keys: int
    shards: tuple[FeatureShardLedgerEntry, ...]
    schema_version: str = SHARD_LEDGER_SCHEMA_VERSION
    commit_policy: str = SHARD_COMMIT_POLICY
    resume_policy: str = SHARD_RESUME_POLICY

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _partition_count(total: int, parts: int) -> tuple[int, ...]:
    if parts <= 0:
        return ()
    base, remainder = divmod(max(0, int(total)), parts)
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


def build_initial_shard_ledger(
    *,
    bundle_id: str,
    runtime_fingerprint_key: str,
    shard_size_views: int = DEFAULT_FEATURE_SHARD_SIZE_VIEWS,
    pending_view_estimate: int,
    pending_vector_keys: int,
    pending_scalar_keys: int,
) -> FeatureShardLedger:
    shard_size = max(1, int(shard_size_views))
    views = max(0, int(pending_view_estimate))
    shard_count = (views + shard_size - 1) // shard_size if views else 0
    vector_partitions = _partition_count(pending_vector_keys, shard_count)
    scalar_partitions = _partition_count(pending_scalar_keys, shard_count)
    shards: list[FeatureShardLedgerEntry] = []
    for index in range(shard_count):
        view_offset = index * shard_size
        expected_views = min(shard_size, max(views - view_offset, 0))
        shards.append(
            FeatureShardLedgerEntry(
                shard_index=index,
                input_selector={
                    "view_offset": view_offset,
                    "view_limit": expected_views,
                },
                expected_views=expected_views,
                expected_vector_keys=vector_partitions[index],
                expected_scalar_keys=scalar_partitions[index],
            )
        )
    return FeatureShardLedger(
        bundle_id=str(bundle_id),
        runtime_fingerprint_key=str(runtime_fingerprint_key),
        shard_size_views=shard_size,
        shard_count=shard_count,
        pending_view_estimate=views,
        pending_vector_keys=max(0, int(pending_vector_keys)),
        pending_scalar_keys=max(0, int(pending_scalar_keys)),
        shards=tuple(shards),
    )


def feature_shard_ledger_from_dict(payload: dict[str, Any]) -> FeatureShardLedger:
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != SHARD_LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Infer shard ledger schema_version={schema_version!r}; "
            f"expected {SHARD_LEDGER_SCHEMA_VERSION!r}."
        )
    shards = tuple(_entry_from_dict(raw) for raw in payload.get("shards", ()) if isinstance(raw, dict))
    ledger = FeatureShardLedger(
        bundle_id=str(payload.get("bundle_id") or ""),
        runtime_fingerprint_key=str(payload.get("runtime_fingerprint_key") or ""),
        shard_size_views=int(payload.get("shard_size_views") or 0),
        shard_count=int(payload.get("shard_count") or 0),
        pending_view_estimate=int(payload.get("pending_view_estimate") or 0),
        pending_vector_keys=int(payload.get("pending_vector_keys") or 0),
        pending_scalar_keys=int(payload.get("pending_scalar_keys") or 0),
        shards=shards,
        schema_version=schema_version,
        commit_policy=str(payload.get("commit_policy") or SHARD_COMMIT_POLICY),
        resume_policy=str(payload.get("resume_policy") or SHARD_RESUME_POLICY),
    )
    _validate_ledger(ledger)
    return ledger


def _entry_from_dict(payload: dict[str, Any]) -> FeatureShardLedgerEntry:
    return FeatureShardLedgerEntry(
        shard_index=int(payload.get("shard_index") or 0),
        status=str(payload.get("status") or SHARD_STATUS_PENDING),
        input_selector=dict(payload.get("input_selector") or {}),
        expected_views=int(payload.get("expected_views") or 0),
        expected_vector_keys=int(payload.get("expected_vector_keys") or 0),
        expected_scalar_keys=int(payload.get("expected_scalar_keys") or 0),
        committed_views=int(payload.get("committed_views") or 0),
        committed_vector_keys=int(payload.get("committed_vector_keys") or 0),
        committed_scalar_keys=int(payload.get("committed_scalar_keys") or 0),
        checksum=payload.get("checksum"),
        last_heartbeat_at=payload.get("last_heartbeat_at"),
        updated_at=payload.get("updated_at"),
        error=payload.get("error"),
    )


def _validate_ledger(ledger: FeatureShardLedger) -> None:
    if ledger.shard_count != len(ledger.shards):
        raise ValueError(f"Infer shard ledger shard_count={ledger.shard_count} but has {len(ledger.shards)} entries.")
    seen: set[int] = set()
    for entry in ledger.shards:
        if entry.shard_index in seen:
            raise ValueError(f"Infer shard ledger has duplicate shard_index={entry.shard_index}.")
        seen.add(entry.shard_index)
        if entry.status not in VALID_SHARD_STATUSES:
            raise ValueError(f"Infer shard ledger has unsupported shard status={entry.status!r}.")


def write_shard_ledger(path: str | Path, ledger: FeatureShardLedger) -> Path:
    _validate_ledger(ledger)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp")
    temp.write_text(json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


def load_shard_ledger(path: str | Path) -> FeatureShardLedger:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Infer shard ledger must be a JSON object.")
    return feature_shard_ledger_from_dict(payload)


def mark_shard_status(
    ledger: FeatureShardLedger,
    *,
    shard_index: int,
    status: str,
    **updates: object,
) -> FeatureShardLedger:
    if status not in VALID_SHARD_STATUSES:
        raise ValueError(f"Unsupported Infer shard status={status!r}.")
    replaced = False
    shards: list[FeatureShardLedgerEntry] = []
    for entry in ledger.shards:
        if entry.shard_index != shard_index:
            shards.append(entry)
            continue
        allowed_updates = {
            key: value for key, value in updates.items() if key in FeatureShardLedgerEntry.__dataclass_fields__
        }
        shards.append(replace(entry, status=status, **allowed_updates))
        replaced = True
    if not replaced:
        raise ValueError(f"Infer shard ledger has no shard_index={shard_index}.")
    return replace(ledger, shards=tuple(shards))


def resume_shard_indices(ledger: FeatureShardLedger) -> tuple[int, ...]:
    _validate_ledger(ledger)
    return tuple(entry.shard_index for entry in ledger.shards if entry.status != SHARD_STATUS_COMMITTED)


def committed_shard_indices(ledger: FeatureShardLedger) -> tuple[int, ...]:
    _validate_ledger(ledger)
    return tuple(entry.shard_index for entry in ledger.shards if entry.status == SHARD_STATUS_COMMITTED)
