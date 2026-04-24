"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/lifecycle/write_session.py

Explicit single-lock write session for producer-style dataset mutations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import pyarrow as pa

from ...contracts import ARROW_SCHEMA, AddSequencesResult, SequencesError, with_base_metadata
from ...storage.locking import dataset_write_lock
from ...storage.parquet import now_utc, write_parquet_atomic
from ..core.ingest import add_sequences_dataset, import_rows_dataset
from ..overlay import write_overlay_dataset

if TYPE_CHECKING:
    import pandas as pd

    from ...dataset import Dataset


@contextmanager
def _held_write_lock(_dataset_dir):
    yield


def init_dataset(
    dataset: Dataset,
    *,
    source: str = "",
    notes: str = "",
    actor: Optional[dict] = None,
    write_lock=dataset_write_lock,
    if_missing: bool = False,
) -> bool:
    """Initialize a dataset, optionally treating an existing dataset as a no-op."""
    with write_lock(dataset.dir):
        dataset._require_registry_for_mutation("init")  # noqa: SLF001
        dataset.dir.mkdir(parents=True, exist_ok=True)
        if dataset.records_path.exists():
            if if_missing:
                return False
            raise SequencesError(f"Dataset already initialized: {dataset.records_path}")
        ts = now_utc()
        empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in ARROW_SCHEMA], schema=ARROW_SCHEMA)
        reg_hash = dataset._registry_hash(required=True)  # noqa: SLF001
        empty = with_base_metadata(empty, created_at=ts, registry_hash=reg_hash)
        write_parquet_atomic(empty, dataset.records_path, dataset.snapshot_dir)
        dataset._auto_freeze_registry()  # noqa: SLF001
        date = ts.split("T")[0]
        meta_md = (
            f"name: {dataset.name}\n"
            f"created_at: {ts}\n"
            f"source: {source}\n"
            f"notes: {notes}\n"
            f"schema: USR v1\n\n"
            f"### Updates ({date})\n"
            f"- {ts}: initialized dataset.\n"
        )
        dataset.meta_path.write_text(meta_md, encoding="utf-8")
        dataset._record_event(  # noqa: SLF001
            "init",
            args={"source": source},
            actor=actor,
        )
        return True


def _namespace_policy() -> tuple[Any, set[str]]:
    # Imported lazily to avoid a dataset <-> write-session import cycle.
    from ...dataset import _NS_RE, MUTATION_RESERVED_NAMESPACES

    return _NS_RE, MUTATION_RESERVED_NAMESPACES


class DatasetWriteSession(AbstractContextManager["DatasetWriteSession"]):
    """Hold the dataset write lock once across a producer mutation sequence."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._lock_cm = None

    def __enter__(self) -> DatasetWriteSession:
        if self._lock_cm is not None:
            raise RuntimeError("DatasetWriteSession cannot be re-entered.")
        self._lock_cm = dataset_write_lock(self._dataset.dir)
        self._lock_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._lock_cm is None:
            return None
        lock_cm = self._lock_cm
        self._lock_cm = None
        return lock_cm.__exit__(exc_type, exc, tb)

    def _require_active(self) -> None:
        if self._lock_cm is None:
            raise RuntimeError("DatasetWriteSession must be used inside a 'with' block.")

    def init(self, source: str = "", notes: str = "", actor: Optional[dict] = None) -> None:
        self._require_active()
        init_dataset(self._dataset, source=source, notes=notes, actor=actor, write_lock=_held_write_lock)

    def init_if_missing(self, source: str = "", notes: str = "", actor: Optional[dict] = None) -> bool:
        self._require_active()
        return init_dataset(
            self._dataset,
            source=source,
            notes=notes,
            actor=actor,
            write_lock=_held_write_lock,
            if_missing=True,
        )

    def import_rows(
        self,
        rows: Union[pd.DataFrame, Sequence[Dict[str, object]]],
        *,
        default_bio_type: str = "dna",
        default_alphabet: str = "dna_4",
        source: Optional[str] = None,
        strict_id_check: bool = True,
        actor: Optional[dict] = None,
        prevalidated_new_ids: bool = False,
    ) -> int:
        self._require_active()
        return import_rows_dataset(
            self._dataset,
            rows,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source,
            strict_id_check=strict_id_check,
            actor=actor,
            prevalidated_new_ids=prevalidated_new_ids,
            write_lock=_held_write_lock,
        )

    def add_sequences(
        self,
        rows_or_sequences: Union[pd.DataFrame, Sequence[Dict[str, object]], Sequence[str]],
        *,
        bio_type: str,
        alphabet: str,
        source: str = "",
        created_at: Optional[str] = None,
        on_conflict: str = "error",
        actor: Optional[dict] = None,
    ) -> AddSequencesResult:
        self._require_active()
        return add_sequences_dataset(
            self._dataset,
            rows_or_sequences,
            bio_type=bio_type,
            alphabet=alphabet,
            source=source,
            created_at=created_at,
            on_conflict=on_conflict,
            actor=actor,
            write_lock=_held_write_lock,
        )

    def write_overlay(
        self,
        namespace: str,
        table_or_batches,
        *,
        key: str = "id",
        overwrite: bool = False,
        allow_missing: bool = False,
        note: str = "",
        actor: Optional[dict] = None,
    ) -> int:
        self._require_active()
        namespace_pattern, reserved_namespaces = _namespace_policy()
        return write_overlay_dataset(
            dataset=self._dataset,
            namespace=namespace,
            table_or_batches=table_or_batches,
            key=key,
            overwrite=overwrite,
            allow_missing=allow_missing,
            note=note,
            actor=actor,
            namespace_pattern=namespace_pattern,
            reserved_namespaces=reserved_namespaces,
            write_lock=_held_write_lock,
        )
