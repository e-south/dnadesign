"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/round/contracts.py

Defines round-level data contracts for OPAL execution stages. Provides
dataclasses for round inputs, outputs, and stage bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...config.types import RootConfig
from ...core.progress import ProgressFactory
from ...storage.data_access import RecordsStore
from ...storage.workspace import CampaignWorkspace


@dataclass(frozen=True)
class RunRoundRequest:
    cfg: RootConfig
    as_of_round: int
    config_path: Optional[Path] = None
    k_override: Optional[int] = None
    score_batch_size_override: Optional[int] = None
    verbose: bool = True
    allow_resume: bool = False
    progress_factory: Optional[ProgressFactory] = None


@dataclass(frozen=True)
class RunRoundResult:
    ok: bool
    run_id: str
    as_of_round: int
    trained_on: int
    scored: int
    top_k_requested: int
    top_k_effective: int
    ledger_path: str


@dataclass(frozen=True)
class RoundInputs:
    cfg: RootConfig
    req: RunRoundRequest
    ws: CampaignWorkspace
    store: RecordsStore
    df: pd.DataFrame
    rdir: Path


@dataclass(frozen=True)
class TrainingBundle:
    rep: Any
    plan: Any
    train_df: pd.DataFrame
    train_ids: List[str]
    Y_train: np.ndarray
    R_train: np.ndarray
    y_dim: int


@dataclass(frozen=True)
class XBundle:
    X_train: np.ndarray
    id_order_train: List[str]
    X_pool: np.ndarray
    id_order_pool: List[str]
    cand_df: pd.DataFrame


@dataclass(frozen=True)
class ScoreBundle:
    model: Any
    fit_metrics: Any
    fit_duration: float
    Y_hat: np.ndarray
    y_obj_scalar: np.ndarray
    diag: Dict[str, Any]
    obj_summary_stats: Optional[Dict[str, Any]]
    obj_name: str
    obj_params: Dict[str, Any]
    obj_mode: str
    sel_name: str
    sel_params: Dict[str, Any]
    tie_handling: str
    mode: str
    ranks_competition: np.ndarray
    selected_bool: np.ndarray
    selected_effective: int
    top_k: int
    obj_sha: str
    scores: np.ndarray
    uq_scalar: np.ndarray


@dataclass(frozen=True)
class ArtifactBundle:
    apaths: Any
    selected_df: pd.DataFrame
    labels_used_df: Optional[pd.DataFrame]
    artifacts_paths_and_hashes: Dict[str, tuple[str, str]]
