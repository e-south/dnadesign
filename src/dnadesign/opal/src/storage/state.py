"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/state.py

Storage helpers for state OPAL storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..core.utils import now_iso, read_json, write_json

STATE_SCHEMA_VERSION = 3
_STATE_REQUIRED_KEYS = (
    "version",
    "campaign_slug",
    "campaign_name",
    "workdir",
    "data_location",
    "x_column_name",
    "y_column_name",
    "created_at",
    "updated_at",
    "representation_vector_dimension",
    "representation_transform",
    "training_policy",
    "performance",
    "rounds",
    "backlog",
)
_ROUND_REQUIRED_KEYS = (
    "round_index",
    "run_id",
    "round_name",
    "round_dir",
    "labels_used_rounds",
    "number_of_training_examples_used_in_round",
    "number_of_candidates_scored_in_round",
    "selection_views",
    "selection_batch",
    "model",
    "metrics",
    "durations_sec",
    "seeds",
    "artifacts",
    "writebacks",
    "warnings",
    "status",
)


def _missing_required_keys(payload: Mapping[str, Any], required: tuple[str, ...]) -> list[str]:
    return [key for key in required if key not in payload or payload[key] in (None, "")]


# ----------------------------
# Per-round entry
# ----------------------------
@dataclass
class RoundEntry:
    # ------- non-defaults first -------
    round_index: int
    run_id: str
    round_name: str
    round_dir: str
    labels_used_rounds: list[int]
    number_of_training_examples_used_in_round: int
    number_of_candidates_scored_in_round: int
    selection_views: dict
    selection_batch: dict
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
    version: int = 3
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    representation_vector_dimension: int = 0
    representation_transform: Dict[str, Any] = field(default_factory=dict)
    training_policy: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    rounds: List[RoundEntry] = field(default_factory=list)
    backlog: Dict[str, Any] = field(
        default_factory=lambda: {"number_of_selected_but_not_yet_labeled_candidates_total": 0}
    )

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
        version = raw.get("version")
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"state.json version must be {STATE_SCHEMA_VERSION}; observed {version!r}. "
                "Run `opal init` or rerun the campaign to regenerate state."
            )

        missing = _missing_required_keys(raw, _STATE_REQUIRED_KEYS)
        if missing:
            raise ValueError(
                f"state.json is missing required keys: {missing}. Run `opal init` or regenerate the state."
            )

        rounds_raw = raw["rounds"]
        if not isinstance(rounds_raw, list):
            raise ValueError("state.json key 'rounds' must be a list.")
        rounds: list[RoundEntry] = []
        for index, round_payload in enumerate(rounds_raw):
            if not isinstance(round_payload, dict):
                raise ValueError(f"state.json round {index} must be an object.")
            round_missing = _missing_required_keys(round_payload, _ROUND_REQUIRED_KEYS)
            if round_missing:
                raise ValueError(
                    f"state.json round {index} is missing required keys: {round_missing}. "
                    "Rerun the round to regenerate state."
                )
            rounds.append(
                RoundEntry(
                    run_id=str(round_payload["run_id"]),
                    round_index=int(round_payload["round_index"]),
                    round_name=str(round_payload["round_name"]),
                    round_dir=str(round_payload["round_dir"]),
                    labels_used_rounds=list(round_payload["labels_used_rounds"]),
                    number_of_training_examples_used_in_round=int(
                        round_payload["number_of_training_examples_used_in_round"]
                    ),
                    number_of_candidates_scored_in_round=int(round_payload["number_of_candidates_scored_in_round"]),
                    selection_views=dict(round_payload["selection_views"]),
                    selection_batch=dict(round_payload["selection_batch"]),
                    model=dict(round_payload["model"]),
                    metrics=dict(round_payload["metrics"]),
                    durations_sec=dict(round_payload["durations_sec"]),
                    seeds=dict(round_payload["seeds"]),
                    artifacts=dict(round_payload["artifacts"]),
                    writebacks=dict(round_payload["writebacks"]),
                    warnings=list(round_payload["warnings"]),
                    status=str(round_payload["status"]),
                )
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
            rounds=rounds,
            backlog=raw["backlog"],
        )
        return st

    # ---- mutate ----
    def add_round(self, entry: RoundEntry) -> None:
        self.rounds.append(entry)
