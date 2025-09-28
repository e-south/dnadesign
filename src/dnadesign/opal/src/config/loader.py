"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/loader.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from typing_extensions import Literal

from .plugin_schemas import validate_params
from .types import (
    CampaignBlock,
    DataBlock,
    LocationLocal,
    LocationUSR,
    MetadataBlock,
    ObjectiveBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBlock,
    TargetNormalizerCfg,
    TrainingBlock,
)


# ---- Strict YAML loader (duplicate keys fail fast) ----
class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep: bool = False):
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise KeyError(f"Duplicate key in YAML: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


# ---- path helpers ----
def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p))))


def _resolve_relative_to(cfg_path: Path, p: Path) -> Path:
    return (cfg_path.parent / p).resolve() if not p.is_absolute() else p


def resolve_path_like(cfg_path: Path, value: str | os.PathLike) -> Path:
    return _resolve_relative_to(cfg_path, _expand(value))


# ---- Pydantic model shells for top-level YAML (strict) ----
class PLocationUSR(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["usr"]
    dataset: str
    path: str  # unified key (replaces former usr_root)


class PLocationLocal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["local"]
    path: str


PLocation = Union[PLocationUSR, PLocationLocal]


class PPluginRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: PLocation
    x_column_name: str
    y_column_name: str
    y_expected_length: Optional[int] = None


class PCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    slug: str
    workdir: str

    @field_validator("slug")
    @classmethod
    def _slug_ok(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9_-]+", v):
            raise ValueError("campaign.slug must match ^[a-z0-9_-]+$")
        return v


class PTargetNormalizer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = True
    method: str = "robust_iqr_per_target"
    minimum_labels_required: int = 5
    center_statistic: str = "median"
    scale_statistic: str = "iqr"


class PTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Dict[str, Any] = Field(default_factory=dict)
    target_normalizer: PTargetNormalizer = Field(default_factory=PTargetNormalizer)


class PScoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_batch_size: int = 10_000  # single source of truth


class PSafety(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fail_on_mixed_biotype_or_alphabet: bool = True
    require_biotype_and_alphabet_on_init: bool = True
    conflict_policy_on_duplicate_ids: str = "error"
    write_back_requires_columns_present: bool = True
    accept_x_mismatch: bool = False


class PMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notes: str = ""


class PRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign: PCampaign
    data: PData
    transforms_x: Optional[PPluginRef] = None
    transforms_y: Optional[PPluginRef] = None
    models: Optional[PPluginRef] = None
    objectives: Optional[PPluginRef] = None
    selection: Optional[PPluginRef] = None
    training: PTraining = Field(default_factory=PTraining)
    scoring: PScoring = Field(default_factory=PScoring)
    safety: PSafety = Field(default_factory=PSafety)
    metadata: PMetadata = Field(default_factory=PMetadata)


# ---- Main loader (intentionally no backcompat for normalizer key) ----
def load_config(path: Path | str) -> RootConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.load(cfg_path.read_text(), Loader=_StrictLoader)

    # --- pragmatic, unrelated migrations kept ---
    # A) data.location.usr_root -> data.location.path
    try:
        loc = raw.get("data", {}).get("location", {})
        if isinstance(loc, dict) and loc.get("kind") == "usr" and "usr_root" in loc:
            loc["path"] = loc.pop("usr_root")
    except Exception:
        pass

    # B) scoring.sort_stability -> removed
    try:
        if "scoring" in raw and isinstance(raw["scoring"], dict):
            raw["scoring"].pop("sort_stability", None)
    except Exception:
        pass

    # C) selection.params.score_batch_size -> scoring.score_batch_size
    try:
        sel = raw.get("selection")
        if isinstance(sel, dict):
            params = sel.get("params", {})
            if isinstance(params, dict) and "score_batch_size" in params:
                raw.setdefault("scoring", {})
                sc = raw["scoring"]
                if "score_batch_size" not in sc:
                    sc["score_batch_size"] = params["score_batch_size"]
                params.pop("score_batch_size", None)
    except Exception:
        pass
    # --- end migrations ---

    # Validate via strict Pydantic shells
    try:
        pyd = PRoot.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid campaign.yaml: {e}") from e

    # Validate plugin params via schema registry (pass-through if unknown)
    tx = pyd.transforms_x or PPluginRef(name="identity", params={})
    ty = pyd.transforms_y or PPluginRef(name="logic5_from_tidy_v1", params={})
    mdl = pyd.models or PPluginRef(name="random_forest", params={})
    obj = pyd.objectives or PPluginRef(name="sfxi_v1", params={})
    sel = pyd.selection or PPluginRef(name="top_n", params={})

    tx_params = validate_params("transform_x", tx.name, tx.params)
    ty_params = validate_params("transform_y", ty.name, ty.params)
    mdl_params = validate_params("model", mdl.name, mdl.params)
    obj_params = validate_params("objective", obj.name, obj.params)
    sel_params = validate_params("selection", sel.name, sel.params)

    # Build dataclasses (public API) with ABS paths
    def _abs(v: str) -> str:
        return str(resolve_path_like(cfg_path, v))

    loc_model = pyd.data.location
    if isinstance(loc_model, PLocationUSR):
        loc_dc = LocationUSR(
            kind="usr", dataset=loc_model.dataset, path=_abs(loc_model.path)
        )
    else:
        loc_dc = LocationLocal(kind="local", path=_abs(loc_model.path))

    data_dc = DataBlock(
        location=loc_dc,
        x_column_name=pyd.data.x_column_name,
        y_column_name=pyd.data.y_column_name,
        y_expected_length=pyd.data.y_expected_length,
        transforms_x=PluginRef(tx.name, tx_params),
        transforms_y=PluginRef(ty.name, ty_params),
    )

    selection_dc = SelectionBlock(
        selection=PluginRef(sel.name, sel_params),
        top_k_default=sel_params.get("top_k_default"),
        tie_handling=sel_params.get("tie_handling"),
        objective=sel_params.get("objective"),
    )
    objective_dc = ObjectiveBlock(objective=PluginRef(obj.name, obj_params))

    training_dc = TrainingBlock(
        models=PluginRef(mdl.name, mdl_params),
        policy=pyd.training.policy
        or {
            "cumulative_training": True,
            "label_cross_round_deduplication_policy": "latest_only",
            "allow_resuggesting_candidates_until_labeled": True,
        },
        target_normalizer=TargetNormalizerCfg(
            enable=pyd.training.target_normalizer.enable,
            method=pyd.training.target_normalizer.method,
            minimum_labels_required=pyd.training.target_normalizer.minimum_labels_required,
            center_statistic=pyd.training.target_normalizer.center_statistic,
            scale_statistic=pyd.training.target_normalizer.scale_statistic,
        ),
    )

    scoring_dc = ScoringBlock(score_batch_size=int(pyd.scoring.score_batch_size))

    safety_dc = SafetyBlock(
        fail_on_mixed_biotype_or_alphabet=pyd.safety.fail_on_mixed_biotype_or_alphabet,
        require_biotype_and_alphabet_on_init=pyd.safety.require_biotype_and_alphabet_on_init,
        conflict_policy_on_duplicate_ids=pyd.safety.conflict_policy_on_duplicate_ids,
        write_back_requires_columns_present=pyd.safety.write_back_requires_columns_present,
        accept_x_mismatch=pyd.safety.accept_x_mismatch,
    )

    root = RootConfig(
        campaign=CampaignBlock(
            name=pyd.campaign.name,
            slug=pyd.campaign.slug,
            workdir=str(resolve_path_like(cfg_path, pyd.campaign.workdir)),
        ),
        data=data_dc,
        selection=selection_dc,
        objective=objective_dc,
        training=training_dc,
        scoring=scoring_dc,
        safety=safety_dc,
        metadata=MetadataBlock(notes=pyd.metadata.notes),
    )
    return root
