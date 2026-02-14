"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/attempts.py

Attempts logging and failure aggregation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts.records import AttemptRecord, SolutionRecord
from ..run_paths import run_tables_root

ATTEMPTS_CHUNK_SIZE = 256
SOLUTIONS_CHUNK_SIZE = 256


def _flush_attempts(tables_root: Path, buffer: list[dict]) -> None:
    if not buffer:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to write attempts logs.") from exc

    schema = pa.schema(
        [
            pa.field("attempt_id", pa.string()),
            pa.field("attempt_index", pa.int64()),
            pa.field("run_id", pa.string()),
            pa.field("input_name", pa.string()),
            pa.field("plan_name", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("detail_json", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("sequence_hash", pa.string()),
            pa.field("solution_id", pa.string()),
            pa.field("used_tf_counts_json", pa.string()),
            pa.field("used_tf_list", pa.list_(pa.string())),
            pa.field("sampling_library_index", pa.int64()),
            pa.field("sampling_library_hash", pa.string()),
            pa.field("solver_status", pa.string()),
            pa.field("solver_objective", pa.float64()),
            pa.field("solver_solve_time_s", pa.float64()),
            pa.field("dense_arrays_version", pa.string()),
            pa.field("dense_arrays_version_source", pa.string()),
            pa.field("library_tfbs", pa.list_(pa.string())),
            pa.field("library_tfs", pa.list_(pa.string())),
            pa.field("library_site_ids", pa.list_(pa.string())),
            pa.field("library_sources", pa.list_(pa.string())),
        ]
    )
    table = pa.Table.from_pylist(buffer, schema=schema)
    tables_root.mkdir(parents=True, exist_ok=True)
    filename = f"attempts_part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, tables_root / filename)
    buffer.clear()


def _flush_solutions(tables_root: Path, buffer: list[dict]) -> None:
    if not buffer:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to write solutions logs.") from exc

    schema = pa.schema(
        [
            pa.field("solution_id", pa.string()),
            pa.field("attempt_id", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("input_name", pa.string()),
            pa.field("plan_name", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("sequence_hash", pa.string()),
            pa.field("sampling_library_index", pa.int64()),
            pa.field("sampling_library_hash", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(buffer, schema=schema)
    tables_root.mkdir(parents=True, exist_ok=True)
    filename = f"solutions_part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, tables_root / filename)
    buffer.clear()


def _coerce_attempt_list(value: object, *, field: str, input_name: str, plan_name: str) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception as exc:
            raise RuntimeError(
                f"attempts.parquet {field} JSON parse failed for {input_name}/{plan_name}: {exc}"
            ) from exc
        if isinstance(parsed, list):
            return parsed
        raise RuntimeError(f"attempts.parquet {field} JSON did not yield a list for {input_name}/{plan_name}.")
    if isinstance(value, float) and math.isnan(value):
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    raise RuntimeError(
        f"attempts.parquet {field} expected list-like values for {input_name}/{plan_name} "
        f"but found {type(value).__name__}."
    )


def _load_failure_counts_from_attempts(
    tables_root: Path,
) -> dict[tuple[str, str, str, str, str | None], dict[str, int]]:
    attempts_path = tables_root / "attempts.parquet"
    if not attempts_path.exists():
        alt_path = tables_root / "tables" / "attempts.parquet"
        if alt_path.exists():
            attempts_path = alt_path
    if not attempts_path.exists():
        return {}
    try:
        df = pd.read_parquet(attempts_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] = {}
    for row in df.itertuples(index=False):
        status = str(getattr(row, "status", "") or "")
        if status == "success":
            continue
        reason = str(getattr(row, "reason", None) or "unknown")
        input_name = str(getattr(row, "input_name", None) or "")
        plan_name = str(getattr(row, "plan_name", None) or "")
        library_tfbs = _coerce_attempt_list(
            getattr(row, "library_tfbs", None),
            field="library_tfbs",
            input_name=input_name,
            plan_name=plan_name,
        )
        library_tfs = _coerce_attempt_list(
            getattr(row, "library_tfs", None),
            field="library_tfs",
            input_name=input_name,
            plan_name=plan_name,
        )
        library_site_ids = _coerce_attempt_list(
            getattr(row, "library_site_ids", None),
            field="library_site_ids",
            input_name=input_name,
            plan_name=plan_name,
        )
        for idx, tfbs in enumerate(library_tfbs):
            tf = str(library_tfs[idx]) if idx < len(library_tfs) else ""
            site_id_raw = library_site_ids[idx] if idx < len(library_site_ids) else None
            site_id = None
            if site_id_raw not in (None, "", "None"):
                site_id = str(site_id_raw)
            key = (input_name, plan_name, tf, str(tfbs), site_id)
            reasons = counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1
    return counts


def _load_existing_library_index(tables_root: Path) -> int:
    attempts_path = tables_root / "attempts.parquet"
    paths: list[Path] = []
    if attempts_path.exists():
        paths.append(attempts_path)
    paths.extend(sorted(tables_root.glob("attempts_part-*.parquet")))
    if not paths:
        return 0
    max_idx = 0
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["sampling_library_index"])
        except Exception:
            continue
        if df.empty or "sampling_library_index" not in df.columns:
            continue
        try:
            current = int(pd.to_numeric(df["sampling_library_index"], errors="coerce").dropna().max() or 0)
        except Exception:
            continue
        max_idx = max(max_idx, current)
    return max_idx


def _load_existing_library_index_by_plan(
    tables_root: Path,
) -> dict[tuple[str, str], int]:
    attempts_path = tables_root / "attempts.parquet"
    paths: list[Path] = []
    if attempts_path.exists():
        paths.append(attempts_path)
    paths.extend(sorted(tables_root.glob("attempts_part-*.parquet")))
    if not paths:
        return {}
    max_by_plan: dict[tuple[str, str], int] = {}
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["input_name", "plan_name", "sampling_library_index"])
        except Exception:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            input_name = str(row.get("input_name") or "")
            plan_name = str(row.get("plan_name") or "")
            idx = row.get("sampling_library_index")
            try:
                idx_val = int(idx) if idx is not None else 0
            except Exception:
                idx_val = 0
            key = (input_name, plan_name)
            max_by_plan[key] = max(max_by_plan.get(key, 0), idx_val)
    return max_by_plan


def _load_existing_library_build_count_by_plan(
    tables_root: Path,
) -> dict[tuple[str, str], int]:
    attempts_path = tables_root / "attempts.parquet"
    paths: list[Path] = []
    if attempts_path.exists():
        paths.append(attempts_path)
    paths.extend(sorted(tables_root.glob("attempts_part-*.parquet")))
    if not paths:
        return {}

    counts_by_plan: dict[tuple[str, str], set[int]] = {}
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["input_name", "plan_name", "sampling_library_index"])
        except Exception:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            input_name = str(row.get("input_name") or "")
            plan_name = str(row.get("plan_name") or "")
            if not input_name or not plan_name:
                continue
            idx = row.get("sampling_library_index")
            try:
                idx_val = int(idx) if idx is not None else 0
            except Exception:
                idx_val = 0
            if idx_val <= 0:
                continue
            key = (input_name, plan_name)
            values = counts_by_plan.setdefault(key, set())
            values.add(idx_val)

    return {key: int(len(values)) for key, values in counts_by_plan.items()}


def _load_existing_attempt_index_by_plan(tables_root: Path) -> dict[tuple[str, str], int]:
    attempts_path = tables_root / "attempts.parquet"
    paths: list[Path] = []
    if attempts_path.exists():
        paths.append(attempts_path)
    paths.extend(sorted(tables_root.glob("attempts_part-*.parquet")))
    if not paths:
        return {}
    max_by_plan: dict[tuple[str, str], int] = {}
    for path in paths:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "attempt_index" not in df.columns:
            raise RuntimeError(
                f"attempts file missing attempt_index column: {path}. "
                "Regenerate outputs with the current DenseGen version."
            )
        for _, row in df.iterrows():
            input_name = str(row.get("input_name") or "")
            plan_name = str(row.get("plan_name") or "")
            key = (input_name, plan_name)
            try:
                idx_val = int(row.get("attempt_index") or 0)
            except Exception:
                idx_val = 0
            max_by_plan[key] = max(max_by_plan.get(key, 0), idx_val)
    return max_by_plan


def _append_attempt(
    tables_root: Path,
    *,
    run_id: str,
    input_name: str,
    plan_name: str,
    attempt_index: int,
    status: str,
    reason: str,
    detail: dict | None,
    sequence: str | None,
    used_tf_counts: dict[str, int] | None,
    used_tf_list: list[str] | None,
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    solution_id: str | None = None,
    library_tfbs: list[str] | None = None,
    library_tfs: list[str] | None = None,
    library_site_ids: list[str | None] | None = None,
    library_sources: list[str | None] | None = None,
    attempts_buffer: list[dict] | None = None,
) -> str:
    sequence_val = sequence or ""
    lib_tfbs = [str(x) for x in (library_tfbs or [])]
    lib_tfs = [str(x) for x in (library_tfs or [])]
    lib_site_ids = [str(x) if x is not None else "" for x in (library_site_ids or [])]
    lib_sources = [str(x) if x is not None else "" for x in (library_sources or [])]
    created_at = datetime.now(timezone.utc).isoformat()
    seq_hash = hashlib.sha256(sequence_val.encode("utf-8")).hexdigest() if sequence_val else ""
    record = AttemptRecord.build(
        attempt_index=int(attempt_index),
        run_id=run_id,
        input_name=input_name,
        plan_name=plan_name,
        created_at=created_at,
        status=status,
        reason=reason,
        detail_json=json.dumps(detail or {}),
        sequence=sequence_val,
        sequence_hash=seq_hash,
        solution_id=solution_id,
        used_tf_counts_json=json.dumps(used_tf_counts or {}),
        used_tf_list=used_tf_list or [],
        sampling_library_index=int(sampling_library_index),
        sampling_library_hash=str(sampling_library_hash),
        solver_status=solver_status,
        solver_objective=solver_objective,
        solver_solve_time_s=solver_solve_time_s,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        library_tfbs=lib_tfbs,
        library_tfs=lib_tfs,
        library_site_ids=lib_site_ids,
        library_sources=lib_sources,
    )
    payload = record.to_dict()
    if attempts_buffer is not None:
        attempts_buffer.append(payload)
        if len(attempts_buffer) >= ATTEMPTS_CHUNK_SIZE:
            _flush_attempts(tables_root, attempts_buffer)
        return record.attempt_id
    _flush_attempts(tables_root, [payload])
    return record.attempt_id


def _log_rejection(
    tables_root: Path,
    *,
    run_id: str,
    input_name: str,
    plan_name: str,
    attempt_index: int,
    reason: str,
    detail: dict | None,
    sequence: str,
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    library_tfbs: list[str] | None = None,
    library_tfs: list[str] | None = None,
    library_site_ids: list[str | None] | None = None,
    library_sources: list[str | None] | None = None,
    attempts_buffer: list[dict] | None = None,
) -> None:
    status = "duplicate" if reason == "output_duplicate" else "rejected"
    _append_attempt(
        tables_root,
        run_id=run_id,
        input_name=input_name,
        plan_name=plan_name,
        attempt_index=attempt_index,
        status=status,
        reason=reason,
        detail=detail,
        sequence=sequence,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        sampling_library_index=sampling_library_index,
        sampling_library_hash=sampling_library_hash,
        solver_status=solver_status,
        solver_objective=solver_objective,
        solver_solve_time_s=solver_solve_time_s,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        solution_id=None,
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        library_site_ids=library_site_ids,
        library_sources=library_sources,
        attempts_buffer=attempts_buffer,
    )


def _record_solution(
    solutions_buffer: list[dict],
    *,
    attempt_id: str,
    run_id: str,
    input_name: str,
    plan_name: str,
    sequence: str,
    sampling_library_index: int,
    sampling_library_hash: str,
) -> SolutionRecord:
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    record = SolutionRecord.build(
        attempt_id=attempt_id,
        run_id=run_id,
        input_name=input_name,
        plan_name=plan_name,
        sequence=sequence,
        sequence_hash=seq_hash,
        sampling_library_index=sampling_library_index,
        sampling_library_hash=sampling_library_hash,
    )
    solutions_buffer.append(record.to_dict())
    if len(solutions_buffer) >= SOLUTIONS_CHUNK_SIZE:
        _flush_solutions(run_tables_root(Path(record.run_root)), solutions_buffer)
    return record
