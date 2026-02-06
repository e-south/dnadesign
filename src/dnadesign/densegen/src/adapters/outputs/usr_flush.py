"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/outputs/usr_flush.py

Transactional flush helpers for DenseGen USR writes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import pyarrow as pa

from dnadesign.usr import Dataset


@dataclass(frozen=True)
class OrphanArtifact:
    npz_ref: str
    sha256: str | None = None
    size_bytes: int | None = None


class DensegenUsrFlushError(RuntimeError):
    """Raised when artifact staging fails after writing one or more artifacts."""

    def __init__(self, message: str, *, orphan_artifacts: Sequence[OrphanArtifact] | None = None) -> None:
        super().__init__(message)
        self.orphan_artifacts = list(orphan_artifacts or [])


class DensegenUsrFlushTransaction:
    def __init__(
        self,
        dataset: Dataset,
        namespace: str,
        *,
        run_id: str | None = None,
        delete_orphans_on_failure: bool = False,
    ) -> None:
        self.dataset = dataset
        self.namespace = str(namespace)
        self.run_id = str(run_id) if run_id is not None else None
        self.delete_orphans_on_failure = bool(delete_orphans_on_failure)
        self.orphan_manifest_path = self.dataset.dir / "_artifacts" / "orphans.jsonl"
        self._rows_incoming = 0
        self._rows_new = 0

    def begin(self, *, rows_incoming: int, rows_new: int) -> None:
        self._rows_incoming = int(rows_incoming)
        self._rows_new = int(rows_new)

    def stage_artifacts(
        self,
        staged: tuple[pa.Table, Sequence[OrphanArtifact]],
    ) -> tuple[pa.Table, list[OrphanArtifact]]:
        table, artifacts = staged
        return table, list(artifacts)

    def commit_overlay_part(self, table: pa.Table, *, key: str) -> int:
        return int(self.dataset.write_overlay_part(self.namespace, table, key=key))

    def on_failure(self, exc: Exception, *, orphan_artifacts: Sequence[OrphanArtifact]) -> None:
        artifacts = list(orphan_artifacts)
        reason = f"{type(exc).__name__}: {exc}"
        if artifacts:
            self._write_orphan_manifest(artifacts, reason=reason)
            if self.delete_orphans_on_failure:
                self._delete_orphans(artifacts)
        self.dataset.log_event(
            "densegen_flush_failed",
            args={
                "namespace": self.namespace,
                "run_id": self.run_id,
                "rows_incoming": self._rows_incoming,
                "rows_new": self._rows_new,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            metrics={
                "rows_incoming": self._rows_incoming,
                "rows_new": self._rows_new,
                "orphan_artifacts": len(artifacts),
            },
            artifacts={
                "overlay": {"namespace": self.namespace, "status": "failed"},
                "orphan_artifacts": [artifact.npz_ref for artifact in artifacts],
            },
            target_path=self.dataset.records_path,
        )

    def _write_orphan_manifest(self, artifacts: Sequence[OrphanArtifact], *, reason: str) -> None:
        self.orphan_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.orphan_manifest_path.open("a", encoding="utf-8") as handle:
            for artifact in artifacts:
                payload = {
                    "timestamp_utc": timestamp,
                    "npz_ref": artifact.npz_ref,
                    "sha256": artifact.sha256,
                    "bytes": artifact.size_bytes,
                    "reason": reason,
                    "run_id": self.run_id,
                }
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _delete_orphans(self, artifacts: Sequence[OrphanArtifact]) -> None:
        dataset_root = self.dataset.dir.resolve()
        for artifact in artifacts:
            target = (self.dataset.dir / artifact.npz_ref).resolve()
            if dataset_root != target and dataset_root not in target.parents:
                continue
            target.unlink(missing_ok=True)


def to_orphan_artifact(payload: Mapping[str, object]) -> OrphanArtifact:
    ref = payload.get("npz_ref")
    if ref is None:
        ref = payload.get("ref")
    if ref is None:
        raise ValueError("Missing artifact ref/npz_ref.")
    sha = payload.get("sha256")
    size = payload.get("bytes")
    return OrphanArtifact(
        npz_ref=str(ref),
        sha256=None if sha is None else str(sha),
        size_bytes=None if size is None else int(size),
    )
