"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/run_manifest.py

Run-level manifest summaries for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlanManifest:
    input_name: str
    plan_name: str
    generated: int
    duplicates_skipped: int
    failed_solutions: int
    total_resamples: int
    libraries_built: int
    stall_events: int
    failed_min_count_per_tf: int = 0
    failed_required_regulators: int = 0
    failed_min_count_by_regulator: int = 0
    failed_min_required_regulators: int = 0
    duplicate_solutions: int = 0
    leaderboard_latest: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "input_name": self.input_name,
            "plan_name": self.plan_name,
            "generated": int(self.generated),
            "duplicates_skipped": int(self.duplicates_skipped),
            "failed_solutions": int(self.failed_solutions),
            "total_resamples": int(self.total_resamples),
            "libraries_built": int(self.libraries_built),
            "stall_events": int(self.stall_events),
            "failed_min_count_per_tf": int(self.failed_min_count_per_tf),
            "failed_required_regulators": int(self.failed_required_regulators),
            "failed_min_count_by_regulator": int(self.failed_min_count_by_regulator),
            "failed_min_required_regulators": int(self.failed_min_required_regulators),
            "duplicate_solutions": int(self.duplicate_solutions),
        }
        if self.leaderboard_latest is not None:
            payload["leaderboard_latest"] = self.leaderboard_latest
        return payload


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    schema_version: str
    config_sha256: str
    run_root: str
    solver_backend: str | None
    solver_strategy: str
    solver_options: list[str]
    solver_strands: str
    dense_arrays_version: str | None
    dense_arrays_version_source: str
    items: list[PlanManifest]

    def to_dict(self) -> dict[str, Any]:
        total_generated = sum(item.generated for item in self.items)
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "run_root": self.run_root,
            "solver_backend": self.solver_backend,
            "solver_strategy": self.solver_strategy,
            "solver_options": list(self.solver_options),
            "solver_strands": self.solver_strands,
            "dense_arrays_version": self.dense_arrays_version,
            "dense_arrays_version_source": self.dense_arrays_version_source,
            "total_generated": int(total_generated),
            "items": [item.to_dict() for item in self.items],
        }

    def write_json(self, path: Path) -> None:
        payload = json.dumps(self.to_dict(), indent=2)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(path)


def load_run_manifest(path: Path) -> RunManifest:
    data = json.loads(path.read_text())
    items = [
        PlanManifest(
            input_name=str(item.get("input_name", "")),
            plan_name=str(item.get("plan_name", "")),
            generated=int(item.get("generated", 0)),
            duplicates_skipped=int(item.get("duplicates_skipped", 0)),
            failed_solutions=int(item.get("failed_solutions", 0)),
            total_resamples=int(item.get("total_resamples", 0)),
            libraries_built=int(item.get("libraries_built", 0)),
            stall_events=int(item.get("stall_events", 0)),
            failed_min_count_per_tf=int(item.get("failed_min_count_per_tf", 0)),
            failed_required_regulators=int(item.get("failed_required_regulators", 0)),
            failed_min_count_by_regulator=int(item.get("failed_min_count_by_regulator", 0)),
            failed_min_required_regulators=int(item.get("failed_min_required_regulators", 0)),
            duplicate_solutions=int(item.get("duplicate_solutions", 0)),
            leaderboard_latest=item.get("leaderboard_latest"),
        )
        for item in data.get("items", [])
    ]
    return RunManifest(
        run_id=str(data.get("run_id", "")),
        created_at=str(data.get("created_at", "")),
        schema_version=str(data.get("schema_version", "")),
        config_sha256=str(data.get("config_sha256", "")),
        run_root=str(data.get("run_root", "")),
        solver_backend=data.get("solver_backend"),
        solver_strategy=str(data.get("solver_strategy", "")),
        solver_options=list(data.get("solver_options", [])),
        solver_strands=str(data.get("solver_strands", "")),
        dense_arrays_version=data.get("dense_arrays_version"),
        dense_arrays_version_source=str(data.get("dense_arrays_version_source", "")),
        items=items,
    )
