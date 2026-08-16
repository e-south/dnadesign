"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/contracts.py

Defines immutable records used to inspect and relocate OPAL campaign histories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..state import CampaignState


@dataclass(frozen=True)
class RunHistory:
    round_index: int
    run_id: str
    round_dir: Path
    run_part: Path
    prediction_parts: tuple[Path, ...]
    run_row: dict[str, Any]
    round_context: dict[str, Any]
    invariant_sha256: str
    prediction_row_count: int
    prediction_retention: str


@dataclass(frozen=True)
class RoundColumnEvidence:
    round_index: int
    run_id: str
    round_context_sha256: str


@dataclass(frozen=True)
class HistoryColumnContract:
    campaign_slug: str
    x_column_name: str
    y_column_name: str
    rounds: tuple[RoundColumnEvidence, ...]
    sha256: str


@dataclass(frozen=True)
class CampaignHistory:
    workdir: Path
    campaign_slug: str
    runs: tuple[RunHistory, ...]
    state: CampaignState | None
    retention_manifest: Path | None

    @property
    def rounds(self) -> tuple[int, ...]:
        return tuple(run.round_index for run in self.runs)


@dataclass(frozen=True)
class HistoryRelocationPlan:
    source: CampaignHistory
    target: CampaignHistory
    campaign_slug: str
    canonical_rounds: tuple[int, ...]
    invariant_sha256: str
    column_contract: HistoryColumnContract | None

    @property
    def imported_rounds(self) -> tuple[int, ...]:
        return self.source.rounds

    @property
    def existing_rounds(self) -> tuple[int, ...]:
        return self.target.rounds
