"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/outputs.py

Output helpers for pipeline artifacts and manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ...adapters.outputs import OutputRecord, SinkBase
from ...config import resolve_relative_path


def _consolidate_parts(tables_root: Path, *, part_glob: str, final_name: str) -> bool:
    parts = sorted(tables_root.glob(part_glob))
    if not parts:
        return False
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required to consolidate parquet parts.") from exc
    final_path = tables_root / final_name
    sources = [str(p) for p in parts]
    if final_path.exists():
        sources.insert(0, str(final_path))
    dataset = ds.dataset(sources, format="parquet")
    tmp_path = tables_root / f".{final_name}.tmp"
    writer = pq.ParquetWriter(tmp_path, schema=dataset.schema)
    scanner = ds.Scanner.from_dataset(dataset, batch_size=4096)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        writer.write_table(pa.Table.from_batches([batch], schema=dataset.schema))
    writer.close()
    tmp_path.replace(final_path)
    for part in parts:
        part.unlink()
    return True


def _emit_event(events_path: Path, *, event: str, payload: dict) -> None:
    record = {"event": event, "created_at": datetime.now(timezone.utc).isoformat()}
    record.update(payload)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _dump_model(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=False)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return dict(value)


def _effective_sampling_caps(input_cfg, cfg_path: Path) -> dict | None:
    sampling = getattr(input_cfg, "sampling", None)
    if sampling is None:
        return None
    mining = getattr(sampling, "mining", None)
    budget = getattr(mining, "budget", None) if mining is not None else None
    return {
        "budget_mode": getattr(budget, "mode", None) if budget is not None else None,
        "budget_candidates": getattr(budget, "candidates", None) if budget is not None else None,
        "budget_target_tier_fraction": getattr(budget, "target_tier_fraction", None) if budget is not None else None,
        "budget_max_candidates": getattr(budget, "max_candidates", None) if budget is not None else None,
        "budget_max_seconds": getattr(budget, "max_seconds", None) if budget is not None else None,
        "budget_min_candidates": getattr(budget, "min_candidates", None) if budget is not None else None,
        "budget_growth_factor": getattr(budget, "growth_factor", None) if budget is not None else None,
    }


def _write_effective_config(
    *,
    cfg,
    cfg_path: Path,
    run_root: Path,
    seeds: dict[str, int],
    outputs_root: Path,
) -> Path:
    resolved_inputs = []
    for inp in cfg.inputs:
        entry = {"name": inp.name, "type": getattr(inp, "type", None)}
        if hasattr(inp, "path"):
            entry["path"] = str(resolve_relative_path(cfg_path, getattr(inp, "path")))
        if hasattr(inp, "paths"):
            paths = getattr(inp, "paths", None)
            if isinstance(paths, list):
                entry["paths"] = [str(resolve_relative_path(cfg_path, p)) for p in paths]
        caps = _effective_sampling_caps(inp, cfg_path)
        if caps is not None:
            entry["sampling_caps"] = caps
        resolved_inputs.append(entry)

    payload = {
        "schema_version": cfg.schema_version,
        "run_id": cfg.run.id,
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "seeds": {k: int(v) for k, v in seeds.items()},
        "inputs": resolved_inputs,
        "config": _dump_model(cfg),
    }
    out_path = outputs_root / "meta" / "effective_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _assert_sink_alignment(sinks: list[SinkBase]) -> None:
    if len(sinks) <= 1:
        return
    sink_types = [type(s).__name__ for s in sinks]
    if len(set(sink_types)) != len(sink_types):
        raise RuntimeError("Duplicate sink types detected; output.targets must be unique.")
    digests = []
    for sink in sinks:
        digest = sink.alignment_digest()
        if digest is None:
            raise RuntimeError(
                f"Sink {type(sink).__name__} does not provide alignment digest; alignment requires digest support."
            )
        digests.append((type(sink).__name__, digest))

    baseline_name, baseline = digests[0]
    mismatches = []
    for name, digest in digests[1:]:
        if digest != baseline:
            mismatches.append((name, digest.id_count, digest.xor_hash))
    if mismatches:
        details = "; ".join(f"{name}: count={cnt} hash={h}" for name, cnt, h in mismatches)
        raise RuntimeError(
            "Output sinks are out of sync before run. "
            f"Baseline={baseline_name} count={baseline.id_count} hash={baseline.xor_hash}. "
            f"Differences: {details}. Remove stale outputs or run with a single target to rebuild."
        )


def _write_to_sinks(sinks: list[SinkBase], record: OutputRecord) -> bool:
    results = [sink.add(record) for sink in sinks]
    if not results:
        raise RuntimeError("No output sinks configured.")
    if all(results):
        return True
    if not any(results):
        return False
    raise RuntimeError("Output sinks are inconsistent (some accepted, some rejected).")
