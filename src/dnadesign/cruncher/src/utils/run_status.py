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


@dataclass
class RunStatusWriter:
    path: Path
    stage: str
    run_dir: Path
    status: str = "running"
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

    def update(self, **fields: Any) -> None:
        self.payload.update(fields)
        self.payload["updated_at"] = _utc_now()
        self.write()

    def finish(self, status: str = "completed", **fields: Any) -> None:
        self.payload.update(fields)
        self.payload["status"] = status
        self.payload["finished_at"] = _utc_now()
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2))
