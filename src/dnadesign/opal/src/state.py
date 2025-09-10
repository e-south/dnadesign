"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/state.py

Provides CampaignState and RoundEntry dataclasses to read/write state.json.
This file is the authoritative record of:

- campaign identity and data location,
- representation/label/transform info,
- per-round metrics, artifacts, durations, seeds,
- counts like number_of_candidates_scored_in_round.

Designed to be append-only per round.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .utils import now_iso, read_json, write_json


@dataclass
class RoundEntry:
    round_index: int
    round_name: str
    round_dir: str
    labels_used_rounds: list[int]
    number_of_training_examples_used_in_round: int
    number_of_candidates_scored_in_round: int
    selection_top_k_requested: int
    selection_top_k_effective_after_ties: int
    model: dict
    metrics: dict
    durations_sec: dict
    seeds: dict
    artifacts: dict
    writebacks: dict
    warnings: list[str]
    status: str = "completed"


@dataclass
class CampaignState:
    version: int = 1
    campaign_slug: str = ""
    campaign_name: str = ""
    workdir: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    data_location: dict = field(default_factory=dict)
    representation_column_name: str = ""
    representation_vector_dimension: int = 0
    label_source_column_name: str = ""
    representation_transform: dict = field(default_factory=dict)
    training_policy: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    rounds: List[RoundEntry] = field(default_factory=list)
    backlog: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        raw = read_json(path)
        st = cls(**{k: v for k, v in raw.items() if k != "rounds"})
        st.rounds = [RoundEntry(**r) for r in raw.get("rounds", [])]
        return st

    def save(self, path: Path) -> None:
        self.updated_at = now_iso()
        data = self.__dict__.copy()
        data["rounds"] = [r.__dict__ for r in self.rounds]
        write_json(path, data)

    def add_round(self, entry: RoundEntry) -> None:
        self.rounds.append(entry)
