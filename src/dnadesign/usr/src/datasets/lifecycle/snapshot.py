"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/lifecycle/snapshot.py

Dataset snapshot helper extracted from the Dataset coordinator.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...storage.locking import dataset_write_lock
from ...storage.parquet import snapshot_parquet_file


class DatasetSnapshotHost(Protocol):
    dir: Path
    records_path: Path
    snapshot_dir: Path

    def _require_exists(self) -> None: ...

    def _require_registry_for_mutation(self, action: str) -> dict: ...

    def _record_event(
        self,
        action: str,
        *,
        args: dict | None = None,
        metrics: dict | None = None,
        artifacts: dict | None = None,
        maintenance: dict | None = None,
        target_path: Path | None = None,
        registry_hash: str | None = None,
        actor: dict | None = None,
    ) -> None: ...


def snapshot_dataset(dataset: DatasetSnapshotHost) -> None:
    """Write a timestamped snapshot and persist the current table atomically."""
    with dataset_write_lock(dataset.dir):
        dataset._require_exists()
        dataset._require_registry_for_mutation("snapshot")
        snapshot_parquet_file(dataset.records_path, dataset.snapshot_dir)
        dataset._record_event(
            "snapshot",
            args={},
        )
