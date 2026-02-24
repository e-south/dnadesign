"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/optimizer_stats.py

Resolve and validate optimizer stats payloads for analysis workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def _resolve_optimizer_stats(manifest: dict[str, object], run_dir: Path) -> dict[str, object] | None:
    stats_payload = manifest.get("optimizer_stats")
    if stats_payload is None:
        return None
    if not isinstance(stats_payload, dict):
        raise ValueError("Run manifest field 'optimizer_stats' must be a dictionary.")
    optimizer_stats = dict(stats_payload)
    if "swap_events" in optimizer_stats:
        raise ValueError("Run manifest field 'optimizer_stats.swap_events' is unsupported for gibbs_anneal.")
    inline_move_stats = optimizer_stats.get("move_stats")
    if inline_move_stats is not None and not isinstance(inline_move_stats, list):
        raise ValueError("Run manifest field 'optimizer_stats.move_stats' must be a list when provided.")
    move_stats_path = optimizer_stats.get("move_stats_path") or manifest.get("optimizer_stats_path")
    if move_stats_path is None:
        return optimizer_stats
    if not isinstance(move_stats_path, str) or not move_stats_path:
        raise ValueError("Run manifest field 'optimizer_stats_path' must be a non-empty string when provided.")
    relative = Path(move_stats_path)
    if relative.is_absolute():
        raise ValueError("Run manifest field 'optimizer_stats_path' must be relative to the run directory.")
    sidecar = (run_dir / relative).resolve()
    try:
        sidecar.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError("Run manifest field 'optimizer_stats_path' must resolve within the run directory.") from exc
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing optimizer stats sidecar: {sidecar}")
    if sidecar.suffix == ".gz":
        with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(sidecar.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer stats sidecar must contain an object: {sidecar}")
    if "swap_events" in payload:
        raise ValueError(f"Optimizer stats sidecar field 'swap_events' is unsupported: {sidecar}")
    move_stats = payload.get("move_stats")
    if move_stats is not None and not isinstance(move_stats, list):
        raise ValueError(f"Optimizer stats sidecar field 'move_stats' must be a list when present: {sidecar}")
    if move_stats is None:
        raise ValueError(f"Optimizer stats sidecar missing required 'move_stats' list: {sidecar}")
    if move_stats is not None:
        optimizer_stats["move_stats"] = move_stats
    return optimizer_stats
