"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/usr_writer.py

USR adapter: buffered import + overlay-part writes for DenseGen.

- Ensures a USR dataset exists (creates if missing).
    - Buffers sequences (essential columns) and derived metadata (namespaced).
    - De-duplicates against existing ids in records.parquet before import.
    - Imports via Dataset.import_rows (no JSONL intermediates).
    - Writes namespaced columns keyed by 'id' as overlay parts.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pyarrow as pa

from dnadesign.usr import Dataset

from ...adapters.outputs.parquet import _meta_arrow_type
from ...artifacts.npz_store import describe_npz, write_npz_atomic
from ...core.metadata_schema import META_FIELDS, validate_metadata
from .base import AlignmentDigest
from .id_index import INDEX_FILENAME, IdIndex
from .record import OutputRecord
from .usr_flush import (
    DensegenUsrFlushError,
    DensegenUsrFlushTransaction,
    OrphanArtifact,
    to_orphan_artifact,
)


@dataclass
class USRWriter:
    dataset: str
    root: Optional[Path] = None
    namespace: str = "densegen"
    chunk_size: int = 128
    health_event_interval_seconds: float = 60.0
    default_bio_type: str = "dna"
    default_alphabet: str = "dna_4"
    deduplicate: bool = True
    npz_fields: List[str] = field(default_factory=list)
    npz_root: Optional[Path] = None
    run_quota: Optional[int] = None
    run_id: Optional[str] = None

    # internal buffers
    _records: List[OutputRecord] = field(default_factory=list)
    _seen_ids: set = field(default_factory=set)
    _id_index: Optional[IdIndex] = field(default=None, init=False, repr=False)
    _npz_root: Optional[Path] = field(default=None, init=False, repr=False)
    _pending_overlay_records: List[OutputRecord] = field(default_factory=list, init=False, repr=False)
    _rows_incoming_session: int = field(default=0, init=False, repr=False)
    _rows_written_session: int = field(default=0, init=False, repr=False)
    _flush_count: int = field(default=0, init=False, repr=False)
    _last_health_event_ts: Optional[float] = field(default=None, init=False, repr=False)
    _last_health_meta: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _run_started_monotonic: float = field(default_factory=time.monotonic, init=False, repr=False)
    _resume_detected: bool = field(default=False, init=False, repr=False)
    _lifecycle_emitted: bool = field(default=False, init=False, repr=False)
    _default_run_id: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        if self.root is None:
            raise ValueError("USRWriter requires an explicit root path.")
        if float(self.health_event_interval_seconds) <= 0:
            raise ValueError("USRWriter.health_event_interval_seconds must be > 0.")
        if self.run_quota is not None and int(self.run_quota) <= 0:
            raise ValueError("USRWriter.run_quota must be > 0 when provided.")
        configured_run_id = str(self.run_id or "").strip()
        if self.run_id is not None and not configured_run_id:
            raise ValueError("USRWriter.run_id must be a non-empty string when provided.")
        self._default_run_id = configured_run_id or f"densegen-pid-{os.getpid()}"
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_registry()
        self.ds = Dataset(self.root, self.dataset)
        if not self.ds.records_path.exists():
            self.ds.init(source="densegen init")
        self._configure_npz()
        self._configure_id_index()
        digest = self.alignment_digest()
        self._resume_detected = bool(digest is not None and int(digest.id_count) > 0)

    def _configure_npz(self) -> None:
        fields = [str(f).strip() for f in self.npz_fields if str(f).strip()]
        if len(fields) != len(set(fields)):
            raise ValueError("USRWriter.npz_fields must not contain duplicates.")
        meta_fields = {field.name for field in META_FIELDS}
        unknown = sorted(set(fields) - meta_fields)
        if unknown:
            raise ValueError(f"USRWriter.npz_fields contains unknown metadata fields: {', '.join(unknown)}")
        self.npz_fields = fields
        if not self.npz_fields:
            self._npz_root = None
            return
        dataset_dir = self.root / self.dataset
        npz_root = self.npz_root
        if npz_root is None:
            npz_root = dataset_dir / "_artifacts" / "densegen_npz"
        else:
            npz_root = Path(npz_root)
            if not npz_root.is_absolute():
                npz_root = (self.root / npz_root).resolve()
        if dataset_dir != npz_root and dataset_dir not in npz_root.parents:
            raise ValueError("USRWriter.npz_root must live under the dataset directory.")
        self._npz_root = npz_root

    def _configure_id_index(self) -> None:
        index_path = self.ds.dir / "_artifacts" / INDEX_FILENAME
        self._id_index = IdIndex(index_path)
        if self.ds.records_path.exists():
            self._id_index.bootstrap_from_parquet(self.ds.records_path)

    def add(self, record: OutputRecord) -> bool:
        if record.bio_type != self.default_bio_type or record.alphabet != self.default_alphabet:
            raise ValueError(
                "OutputRecord bio_type/alphabet mismatch for USR sink. "
                f"record=({record.bio_type}, {record.alphabet}) "
                f"sink=({self.default_bio_type}, {self.default_alphabet})"
            )
        validate_metadata(record.meta)
        if not self._lifecycle_emitted:
            lifecycle_status = "resumed" if self._resume_detected else "started"
            self._emit_densegen_health_event(
                status=lifecycle_status,
                record=record,
                rows_incoming=0,
                rows_written=0,
                force=True,
            )
            self._lifecycle_emitted = True
        seq_id = record.id
        if self.deduplicate:
            if seq_id in self._seen_ids:
                return False
            if self._id_exists(seq_id):
                return False  # skip existing ids or local duplicates
        elif seq_id in self._seen_ids:
            return False

        self._seen_ids.add(seq_id)
        self._records.append(record)

        if len(self._records) >= self.chunk_size:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._records and not self._pending_overlay_records:
            return

        incoming_records = list(self._records)
        rows_incoming_count = len(incoming_records)
        rows = [
            {
                "id": r.id,
                "sequence": r.sequence,
                "bio_type": r.bio_type,
                "alphabet": r.alphabet,
                "source": r.source,
            }
            for r in self._records
        ]

        incoming_ids = [r.id for r in self._records]
        if self.deduplicate:
            existing_ids = self._id_exists_many(incoming_ids)
            mask = [rid not in existing_ids for rid in incoming_ids]
            rows_new = [rows[i] for i, keep in enumerate(mask) if keep]
            records_new = [self._records[i] for i, keep in enumerate(mask) if keep]
        else:
            rows_new = rows
            records_new = self._records
        rows_written_count = len(rows_new)

        records_for_overlay = self._records_for_overlay(records_new)
        run_id = records_new[0].meta.get("run_id") if records_new else None
        if run_id is None and self._pending_overlay_records:
            run_id = self._pending_overlay_records[0].meta.get("run_id")
        actor_run_id = self._resolve_run_id(run_id)
        actor = self._densegen_actor(actor_run_id)
        tx = DensegenUsrFlushTransaction(
            self.ds,
            self.namespace,
            run_id=actor_run_id,
            actor=actor,
        )
        tx.begin(rows_incoming=len(incoming_records), rows_new=len(rows_new))
        orphan_artifacts: list[OrphanArtifact] = []
        import_done = False

        try:
            if rows_new:
                self.ds.import_rows(
                    rows_new,
                    default_bio_type=self.default_bio_type,
                    default_alphabet=self.default_alphabet,
                    source=None,
                    strict_id_check=True,
                    actor=actor,
                )
                import_done = True
                self._add_ids_to_index(rec.id for rec in records_new)

            if records_for_overlay:
                table, orphan_artifacts = tx.stage_artifacts(self._build_overlay_table(records_for_overlay))
                tx.commit_overlay_part(table, key="id")
        except Exception as exc:
            if isinstance(exc, DensegenUsrFlushError):
                orphan_artifacts = list(exc.orphan_artifacts)
            if records_for_overlay and (import_done or not rows_new):
                self._pending_overlay_records = list(records_for_overlay)
            if import_done:
                self._records.clear()
                self._seen_ids.clear()
            tx.on_failure(exc, orphan_artifacts=orphan_artifacts)
            raise
        else:
            self._rows_incoming_session += int(rows_incoming_count)
            self._rows_written_session += int(rows_written_count)
            self._flush_count += 1
            health_record = (
                records_for_overlay[-1] if records_for_overlay else (incoming_records[-1] if incoming_records else None)
            )
            self._emit_densegen_health_event(
                status="running",
                record=health_record,
                rows_incoming=rows_incoming_count,
                rows_written=rows_written_count,
                force=False,
            )
            self._pending_overlay_records.clear()
            self._records.clear()
            self._seen_ids.clear()

    def finalize(self) -> None:
        self.flush()
        self._emit_densegen_health_event(
            status="completed",
            record=None,
            rows_incoming=0,
            rows_written=0,
            force=True,
        )

    def on_run_failure(self, exc: Exception) -> None:
        self._emit_densegen_health_event(
            status="failed",
            record=None,
            rows_incoming=0,
            rows_written=0,
            force=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def existing_ids(self) -> set:
        if self._id_index is None:
            return set(self._seen_ids)
        return set(self._id_index.existing_ids()).union(self._seen_ids)

    def alignment_digest(self):
        if self._id_index is None:
            return AlignmentDigest(0, "0" * 32)
        return self._id_index.alignment_digest()

    def _ensure_registry(self) -> None:
        registry_path = self.root / "registry.yaml"
        if registry_path.exists():
            return
        raise RuntimeError(
            f"USR registry not found at {registry_path}. Create registry.yaml before writing USR outputs."
        )

    def _build_overlay_table(self, records: List[OutputRecord]) -> tuple[pa.Table, list[OrphanArtifact]]:
        if not records:
            return pa.table({"id": []}), []
        ids = [rec.id for rec in records]
        offloaded = [set() for _ in records]
        npz_refs: List[Optional[str]] = [None] * len(records)
        npz_sha256: List[Optional[str]] = [None] * len(records)
        npz_bytes: List[Optional[int]] = [None] * len(records)
        npz_fields: List[Optional[List[str]]] = [None] * len(records)
        written_artifacts: list[OrphanArtifact] = []
        if self.npz_fields and self._npz_root is not None:
            self._npz_root.mkdir(parents=True, exist_ok=True)
            for idx, rec in enumerate(records):
                payload = {}
                for field in self.npz_fields:
                    if field not in rec.meta:
                        raise ValueError(f"USRWriter missing npz field in metadata: {field}")
                    value = rec.meta.get(field)
                    if value is None:
                        continue
                    payload[field] = self._npz_pack_value(value)
                if not payload:
                    continue
                npz_path = self._npz_root / f"{rec.id}.npz"
                if npz_path.exists():
                    try:
                        artifact = describe_npz(npz_path)
                    except Exception as exc:
                        raise DensegenUsrFlushError(
                            f"Failed to validate existing NPZ artifact: {npz_path}",
                            orphan_artifacts=written_artifacts,
                        ) from exc
                else:
                    try:
                        artifact = write_npz_atomic(payload, npz_path)
                    except Exception as exc:
                        raise DensegenUsrFlushError(
                            f"Failed to write NPZ artifact: {npz_path}",
                            orphan_artifacts=written_artifacts,
                        ) from exc
                npz_refs[idx] = str(npz_path.relative_to(self.ds.dir))
                npz_sha256[idx] = artifact.sha256
                npz_bytes[idx] = artifact.bytes
                npz_fields[idx] = sorted(payload.keys())
                offloaded[idx] = set(payload.keys())
                written_artifacts.append(
                    to_orphan_artifact(
                        {
                            "npz_ref": npz_refs[idx],
                            "sha256": artifact.sha256,
                            "bytes": artifact.bytes,
                        }
                    )
                )
        arrays = [pa.array(ids, type=pa.string())]
        fields = [pa.field("id", pa.string())]

        for meta_field in META_FIELDS:
            name = meta_field.name
            dtype = _meta_arrow_type(name, pa)
            values = [(None if name in offloaded[idx] else rec.meta.get(name)) for idx, rec in enumerate(records)]
            arrays.append(pa.array(values, type=dtype))
            fields.append(pa.field(f"{self.namespace}__{name}", dtype, nullable=True))

        if self.npz_fields:
            arrays.append(pa.array(npz_refs, type=pa.string()))
            fields.append(pa.field(f"{self.namespace}__npz_ref", pa.string(), nullable=True))
            arrays.append(pa.array(npz_sha256, type=pa.string()))
            fields.append(pa.field(f"{self.namespace}__npz_sha256", pa.string(), nullable=True))
            arrays.append(pa.array(npz_bytes, type=pa.int64()))
            fields.append(pa.field(f"{self.namespace}__npz_bytes", pa.int64(), nullable=True))
            arrays.append(pa.array(npz_fields, type=pa.list_(pa.string())))
            fields.append(pa.field(f"{self.namespace}__npz_fields", pa.list_(pa.string()), nullable=True))

        return pa.Table.from_arrays(arrays, schema=pa.schema(fields)), written_artifacts

    def _records_for_overlay(self, records_new: List[OutputRecord]) -> List[OutputRecord]:
        merged = []
        by_id: dict[str, OutputRecord] = {}
        for rec in [*self._pending_overlay_records, *records_new]:
            if rec.id not in by_id:
                merged.append(rec.id)
            by_id[rec.id] = rec
        return [by_id[record_id] for record_id in merged]

    def _emit_densegen_health_event(
        self,
        *,
        status: str,
        record: Optional[OutputRecord],
        rows_incoming: int,
        rows_written: int,
        force: bool,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = time.monotonic()
        if (
            not force
            and self._last_health_event_ts is not None
            and (now - self._last_health_event_ts) < float(self.health_event_interval_seconds)
        ):
            return

        meta = dict(record.meta) if record is not None else dict(self._last_health_meta)
        if meta:
            self._last_health_meta = dict(meta)

        run_id = self._resolve_run_id(meta.get("run_id"))
        quota = int(self.run_quota) if self.run_quota is not None else 0
        quota_progress_pct = 0.0
        if quota > 0:
            quota_progress_pct = float(self._rows_written_session) / float(quota) * 100.0
        run_elapsed_seconds = max(0.0, float(now - self._run_started_monotonic))
        used_tfbs_raw = meta.get("used_tfbs")
        used_tfbs_unique_count = 0
        if isinstance(used_tfbs_raw, list):
            used_tfbs_unique_count = len({str(item).strip() for item in used_tfbs_raw if str(item).strip()})
        tfbs_total_library = self._to_int(meta.get("library_unique_tfbs_count")) or 0
        tfbs_coverage_pct = 0.0
        if tfbs_total_library > 0:
            tfbs_coverage_pct = float(used_tfbs_unique_count) / float(tfbs_total_library) * 100.0
        plans_attempted = int(self._rows_incoming_session)
        plans_solved = int(self._rows_written_session)

        metrics: dict[str, Any] = {
            "rows_incoming_flush": int(rows_incoming),
            "rows_written_flush": int(rows_written),
            "rows_incoming_session": int(self._rows_incoming_session),
            "rows_written_session": int(self._rows_written_session),
            "flush_count": int(self._flush_count),
            "run_quota": quota,
            "quota_progress_pct": quota_progress_pct,
            "compression_ratio": self._to_float(meta.get("compression_ratio")),
            "gc_total": self._to_float(meta.get("gc_total")),
            "gc_core": self._to_float(meta.get("gc_core")),
            "library_unique_tf_count": self._to_int(meta.get("library_unique_tf_count")),
            "library_unique_tfbs_count": self._to_int(meta.get("library_unique_tfbs_count")),
            "solver_solve_time_s": self._to_float(meta.get("solver_solve_time_s")),
            "run_elapsed_seconds": run_elapsed_seconds,
            "densegen": {
                "tfbs_total_library": tfbs_total_library,
                "tfbs_unique_used": used_tfbs_unique_count,
                "tfbs_coverage_pct": tfbs_coverage_pct,
                "plans_attempted": plans_attempted,
                "plans_solved": plans_solved,
                "rows_written_session": int(self._rows_written_session),
                "run_quota": quota,
                "quota_progress_pct": quota_progress_pct,
                "run_elapsed_seconds": run_elapsed_seconds,
            },
        }
        args: dict[str, Any] = {
            "status": str(status),
            "namespace": self.namespace,
            "run_id": run_id,
            "plan": meta.get("plan"),
            "input_name": meta.get("input_name"),
            "sampling_library_index": self._to_int(meta.get("sampling_library_index")),
            "sampling_library_hash": meta.get("sampling_library_hash"),
            "solver_status": meta.get("solver_status"),
        }
        if error_type is not None:
            args["error_type"] = str(error_type)
        if error_message is not None:
            args["error"] = str(error_message)
        self.ds.log_event(
            "densegen_health",
            args=args,
            metrics=metrics,
            artifacts={"overlay": {"namespace": self.namespace, "status": str(status)}},
            target_path=self.ds.records_path,
            actor=self._densegen_actor(run_id),
        )
        if str(status).strip().lower() == "running":
            self._last_health_event_ts = now

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _npz_pack_value(value):
        if isinstance(value, np.ndarray):
            return value
        try:
            arr = np.array(value)
            if arr.dtype == object:
                return np.array(value, dtype=object)
            return arr
        except Exception:
            return np.array(value, dtype=object)

    def _id_exists(self, seq_id: str) -> bool:
        if self._id_index is None:
            return False
        return bool(self._id_index.contains(str(seq_id)))

    def _id_exists_many(self, seq_ids: list[str]) -> set[str]:
        if self._id_index is None:
            return set()
        return set(self._id_index.contains_many([str(seq_id) for seq_id in seq_ids if seq_id]))

    def _add_ids_to_index(self, ids) -> None:
        if self._id_index is None:
            return
        self._id_index.add([str(seq_id) for seq_id in ids if seq_id])

    def _resolve_run_id(self, run_id: Any) -> str:
        run_id_text = str(run_id or "").strip()
        if run_id_text:
            return run_id_text
        return self._default_run_id

    @staticmethod
    def _densegen_actor(run_id: Any) -> dict[str, Any]:
        run_id_text = str(run_id or "").strip()
        if not run_id_text:
            raise ValueError("USRWriter actor run_id must be a non-empty string.")
        return {
            "tool": "densegen",
            "run_id": run_id_text,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
