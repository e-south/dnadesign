"""Contracts for DenseGen TFBS Stage B sentinel config generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....core.constants import DEFAULT_SEED
from ...retention import (
    DEFAULT_TFBS_STAGE_INITIAL_LABELS,
    DEFAULT_TFBS_STAGE_ROUNDS,
    DEFAULT_TFBS_STAGE_SELECTION_K,
)
from ..seed import TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
from ..semantics import TFBS_STAGE_B_DEFAULT_TIE_HANDLING, TFBS_STAGE_B_SPLIT_ID


@dataclass(frozen=True)
class TfbsStageBConfig:
    """Inputs and gates for Stage B sentinel campaign config generation."""

    stage_a_run_root: Path
    out_dir: Path | None = None
    repo_root: Path | None = None
    label_names: tuple[str, ...] = ()
    target_profile_id: str | None = None
    split_id: str = TFBS_STAGE_B_SPLIT_ID
    seed: int = DEFAULT_SEED
    rounds: int = DEFAULT_TFBS_STAGE_ROUNDS
    selection_k: int = DEFAULT_TFBS_STAGE_SELECTION_K
    initial_label_count: int = DEFAULT_TFBS_STAGE_INITIAL_LABELS
    initial_seed_policy: str = TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
    selection_tie_handling: str = TFBS_STAGE_B_DEFAULT_TIE_HANDLING
    validate_configs: bool = True
    replace_out_dir: bool = False
    refresh_existing_execution_state: bool = False
    score_batch_size: int = 1000
    max_x_matrix_gib: float = 8.0


@dataclass(frozen=True)
class TfbsStageBResult:
    """Materialized Stage B config paths and validation status."""

    status: str
    out_dir: Path
    config_manifest_path: Path
    collection_manifest_path: Path
    campaign_count: int
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "out_dir": str(self.out_dir),
            "config_manifest_path": str(self.config_manifest_path),
            "collection_manifest_path": str(self.collection_manifest_path),
            "campaign_count": int(self.campaign_count),
            "validation_status": self.validation_status,
        }
