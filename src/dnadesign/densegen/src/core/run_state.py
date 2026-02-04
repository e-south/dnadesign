"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/run_state.py

Run-state checkpoints for resumable DenseGen runs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PlanState:
    input_name: str
    plan_name: str
    generated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "plan_name": self.plan_name,
            "generated": int(self.generated),
        }


@dataclass
class RunState:
    run_id: str
    created_at: str
    updated_at: str
    schema_version: str
    config_sha256: str
    run_root: str
    items: list[PlanState]

    def to_dict(self) -> dict[str, Any]:
        total_generated = sum(int(item.generated) for item in self.items)
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "run_root": self.run_root,
            "total_generated": int(total_generated),
            "items": [item.to_dict() for item in self.items],
        }

    def write_json(self, path: Path) -> None:
        payload = json.dumps(self.to_dict(), indent=2)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(path)

    @staticmethod
    def from_counts(
        *,
        run_id: str,
        schema_version: str,
        config_sha256: str,
        run_root: str,
        counts: dict[tuple[str, str], int],
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> "RunState":
        now = datetime.now(timezone.utc).isoformat()
        items = [PlanState(input_name=k[0], plan_name=k[1], generated=int(v)) for k, v in sorted(counts.items())]
        return RunState(
            run_id=run_id,
            created_at=created_at or now,
            updated_at=updated_at or now,
            schema_version=str(schema_version),
            config_sha256=str(config_sha256),
            run_root=str(run_root),
            items=items,
        )


def load_run_state(path: Path) -> RunState:
    data = json.loads(path.read_text())
    items = [
        PlanState(
            input_name=str(item.get("input_name", "")),
            plan_name=str(item.get("plan_name", "")),
            generated=int(item.get("generated", 0)),
        )
        for item in data.get("items", [])
    ]
    return RunState(
        run_id=str(data.get("run_id", "")),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        schema_version=str(data.get("schema_version", "")),
        config_sha256=str(data.get("config_sha256", "")),
        run_root=str(data.get("run_root", "")),
        items=items,
    )
