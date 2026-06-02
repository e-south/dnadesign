"""Stage B execution public contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXECUTION_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_execution.v1"


@dataclass(frozen=True)
class TfbsStageBExecutionConfig:
    """Runtime inputs for executing generated Stage B sentinel campaigns."""

    config_manifest_path: Path
    repo_root: Path
    rounds: int | None = None
    campaign_keys: tuple[str, ...] = ()
    resume_existing: bool = False
    machine_readable: bool = True


@dataclass(frozen=True)
class TfbsStageBExecutionResult:
    """Result summary for a Stage B sentinel execution run."""

    status: str
    execution_manifest_path: Path
    campaign_count: int
    round_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "execution_manifest_path": str(self.execution_manifest_path),
            "campaign_count": int(self.campaign_count),
            "round_count": int(self.round_count),
        }
