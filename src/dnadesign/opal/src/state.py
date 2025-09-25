"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/state.py

Campaign state and per-round registry.

This module defines:
  • RoundEntry — append-only snapshot for a completed round
  • CampaignState — campaign identity + config surfaces + round history

Required keys:
  - campaign_slug, campaign_name, workdir, data_location
  - x_column_name, y_column_name

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .utils import now_iso, read_json, write_json


# ----------------------------
# Per-round entry
# ----------------------------
@dataclass
class RoundEntry:
    # ------- non-defaults first -------
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
    # ------- defaults after -------
    status: str = "completed"


# ----------------------------
# Campaign state
# ----------------------------
@dataclass
class CampaignState:
    # ------- required (non-default) -------
    campaign_slug: str
    campaign_name: str
    workdir: str
    data_location: Dict[str, Any]
    x_column_name: str
    y_column_name: str

    # ------- defaults after -------
    version: int = 1
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    representation_vector_dimension: int = 0
    representation_transform: Dict[str, Any] = field(default_factory=dict)
    training_policy: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    rounds: List[RoundEntry] = field(default_factory=list)
    backlog: Dict[str, Any] = field(
        default_factory=lambda: {
            "number_of_selected_but_not_yet_labeled_candidates_total": 0
        }
    )

    # ---- Back-compat aliases (old field names) ----
    @property
    def representation_column_name(self) -> str:
        return self.x_column_name

    @property
    def label_column_name(self) -> str:
        return self.y_column_name

    # ---- persistence ----
    def to_dict(self) -> Dict[str, Any]:
        """Serialize only declared dataclass fields (ignore ad-hoc attrs)."""
        return asdict(self)

    def save(self, path: Path) -> None:
        self.updated_at = now_iso()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        raw = read_json(Path(path))
        # Opinionated: required keys must be present and non-empty.
        required = [
            "campaign_slug",
            "campaign_name",
            "workdir",
            "data_location",
            "x_column_name",
            "y_column_name",
        ]
        missing = [k for k in required if k not in raw or raw[k] in (None, "")]
        if missing:
            raise ValueError(
                f"state.json is missing required keys: {missing}. "
                "Run `opal init` or regenerate the state."
            )

        # Fill sensible defaults if absent (non-breaking for forward fields).
        raw.setdefault("version", 1)
        raw.setdefault("created_at", now_iso())
        raw.setdefault("updated_at", now_iso())
        raw.setdefault("representation_vector_dimension", 0)
        raw.setdefault("representation_transform", {})
        raw.setdefault("training_policy", {})
        raw.setdefault("performance", {})
        raw.setdefault("rounds", [])
        raw.setdefault(
            "backlog",
            {"number_of_selected_but_not_yet_labeled_candidates_total": 0},
        )

        st = cls(
            campaign_slug=raw["campaign_slug"],
            campaign_name=raw["campaign_name"],
            workdir=raw["workdir"],
            data_location=raw["data_location"],
            x_column_name=raw["x_column_name"],
            y_column_name=raw["y_column_name"],
            version=raw["version"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            representation_vector_dimension=raw["representation_vector_dimension"],
            representation_transform=raw["representation_transform"],
            training_policy=raw["training_policy"],
            performance=raw["performance"],
            rounds=[RoundEntry(**r) for r in raw.get("rounds", [])],
            backlog=raw["backlog"],
        )
        return st

    # ---- mutate ----
    def add_round(self, entry: RoundEntry) -> None:
        self.rounds.append(entry)
