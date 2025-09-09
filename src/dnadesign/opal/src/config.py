"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config.py

Defines pydantic models for campaign configuration (campaign/data/training/
selection/scoring/safety/metadata). Normalizes paths, slugifies names, and
converts YAML into strongly-typed objects consumed by the rest of OPAL.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator

from .utils import ExitCodes, OpalError, slugify


class LocationUSR(BaseModel):
    kind: Literal["usr"]
    dataset: str
    usr_root: str


class LocationLocal(BaseModel):
    kind: Literal["local"]
    path: str  # path to records.parquet in campaign root


class TransformConfig(BaseModel):
    name: Literal["identity", "concat"] = "identity"
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelParamsRF(BaseModel):
    n_estimators: int = 100
    criterion: Literal["friedman_mse", "squared_error"] = "friedman_mse"
    bootstrap: bool = True
    oob_score: bool = True
    random_state: int = 7
    n_jobs: int = -1


class ModelConfig(BaseModel):
    name: Literal["random_forest"] = "random_forest"
    params: ModelParamsRF = Field(default_factory=ModelParamsRF)


class TrainingPolicy(BaseModel):
    cumulative_training: bool = True
    label_cross_round_deduplication_policy: Literal["latest_only"] = "latest_only"
    allow_resuggesting_candidates_until_labeled: bool = True


class SelectionConfig(BaseModel):
    strategy: Literal["top_n"] = "top_n"
    top_k_default: int = 12
    tie_handling: Literal["competition_rank"] = "competition_rank"


class ScoringConfig(BaseModel):
    score_batch_size: int = 10_000
    sort_stability: Literal["(-y_pred, id)"] = "(-y_pred, id)"


class SafetyConfig(BaseModel):
    fail_on_mixed_biotype_or_alphabet: bool = True
    require_biotype_and_alphabet_on_init: bool = True
    conflict_policy_on_duplicate_ids: Literal["error"] = "error"
    write_back_requires_columns_present: bool = True
    accept_x_mismatch: bool = False


class MetaConfig(BaseModel):
    objective: Literal["maximize"] = "maximize"
    notes: str = ""


class DataConfig(BaseModel):
    location: LocationUSR | LocationLocal
    representation_column_name: str
    label_source_column_name: str
    transform: TransformConfig = Field(default_factory=TransformConfig)


class CampaignConfig(BaseModel):
    name: str
    slug: Optional[str] = None
    workdir: str

    @validator("slug", always=True)
    def _auto_slug(cls, v, values):
        if v:
            return slugify(v)
        return slugify(values.get("name", "campaign"))


class RootConfig(BaseModel):
    campaign: CampaignConfig
    data: DataConfig
    training: Dict[str, Any]
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    metadata: MetaConfig = Field(default_factory=MetaConfig)

    @validator("training")
    def _validate_training(cls, v):
        # Accept dict; normalize into {"model": ModelConfig, "policy": TrainingPolicy}
        model_cfg = v.get("model", {})
        policy_cfg = v.get("policy", {})
        model = ModelConfig(**model_cfg)
        policy = TrainingPolicy(**policy_cfg)
        return {"model": model, "policy": policy}


def load_config(path: Path) -> RootConfig:
    if not path.exists():
        raise OpalError(f"Config not found: {path}", ExitCodes.NOT_FOUND)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RootConfig(**raw)
    # normalize workdir path to absolute
    wd = Path(cfg.campaign.workdir).resolve()
    cfg.campaign.workdir = str(wd)
    return cfg
