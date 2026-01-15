"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/run_status.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_METRICS_FIELDS = (
    "run_dir",
    "stage",
    "status",
    "status_message",
    "phase",
    "chain",
    "step",
    "total",
    "progress_pct",
    "beta",
    "beta_softmin",
    "beta_min",
    "beta_max",
    "current_score",
    "score_mean",
    "score_std",
    "best_score",
    "best_chain",
    "best_draw",
    "acceptance_rate",
    "swap_rate",
)


@dataclass
class RunStatusWriter:
    path: Path
    stage: str
    run_dir: Path
    status: str = "running"
    metrics_path: Path | None = None
    metrics_fields: tuple[str, ...] = DEFAULT_METRICS_FIELDS
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = {
            "stage": self.stage,
            "status": self.status,
            "run_dir": str(self.run_dir.resolve()),
            "started_at": _utc_now(),
        }
        base.update(self.payload)
        self.payload = base
        self.write()
        self._append_metrics(event="start")

    def update(self, **fields: Any) -> None:
        self.payload.update(fields)
        self.payload["updated_at"] = _utc_now()
        self.write()
        if _should_emit_metrics(fields):
            self._append_metrics(event="update")

    def finish(self, status: str = "completed", **fields: Any) -> None:
        self.payload.update(fields)
        self.payload["status"] = status
        self.payload["finished_at"] = _utc_now()
        self.write()
        self._append_metrics(event="finish")

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2))

    def _append_metrics(self, *, event: str) -> None:
        if self.metrics_path is None:
            return
        record = {"event": event, "timestamp": _utc_now()}
        for key in self.metrics_fields:
            record[key] = self.payload.get(key)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


def _should_emit_metrics(fields: Dict[str, Any]) -> bool:
    if not fields:
        return False
    keys = set(fields.keys())
    tracked = {
        "progress_pct",
        "current_score",
        "score_mean",
        "score_std",
        "best_score",
        "acceptance_rate",
        "swap_rate",
        "status_message",
        "phase",
        "beta",
        "beta_softmin",
    }
    return bool(keys & tracked)
