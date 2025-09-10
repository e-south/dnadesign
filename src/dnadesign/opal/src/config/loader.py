"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/loader.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
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
    TargetScalerCfg,
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


# ---- Pydantic cores for YAML ----
class PLocationUSR(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["usr"]
    dataset: str
    usr_root: str


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
    # Support both the old and the clearer new names via alias choices.
    representation_column_name: str = Field(
        validation_alias=AliasChoices("x_column_name", "representation_column_name")
    )
    label_source_column_name: str = Field(
        validation_alias=AliasChoices("y_column_name", "label_source_column_name")
    )
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


class PTargetScaler(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = True
    method: str = "robust_iqr_per_target"
    minimum_labels_required: int = 5
    center_statistic: str = "median"
    scale_statistic: str = "iqr"


class PTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Dict[str, Any] = Field(default_factory=dict)
    target_scaler: PTargetScaler = Field(default_factory=PTargetScaler)


class PScoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_batch_size: int = 10_000
    sort_stability: str = (
        "(-opal__{slug}__r{round}__selection_score__logic_plus_effect_v1, id)"
    )


class PSafety(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fail_on_mixed_biotype_or_alphabet: bool = True
    require_biotype_and_alphabet_on_init: bool = True
    conflict_policy_on_duplicate_ids: str = "error"
    write_back_requires_columns_present: bool = True
    accept_x_mismatch: bool = False


class PMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = "maximize"
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


# ---- Helpers ----
def _coerce_location(loc: PLocation):
    if isinstance(loc, PLocationUSR):
        return LocationUSR(kind="usr", dataset=loc.dataset, usr_root=loc.usr_root)
    return LocationLocal(kind="local", path=loc.path)


def _auto_sort_stability(tpl: str, objective_name: str) -> str:
    # If template references __selection_score__ but omits the chosen objective,
    # inject the objective name.
    if "__selection_score__" in tpl and "{objective}" in tpl:
        return tpl.format(objective=objective_name)
    if "__selection_score__" in tpl and objective_name not in tpl:
        return f"(-opal__{{slug}}__r{{round}}__selection_score__{objective_name}, id)"
    return tpl


# ---- Main loader ----
def load_config(path: Path | str) -> RootConfig:
    p = Path(path)
    raw = yaml.load(p.read_text(), Loader=_StrictLoader)
    try:
        cfg = PRoot.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid campaign.yaml: {e}") from e

    # Validate plugin params via schema registry (pass-through if unknown)
    tx = cfg.transforms_x or PPluginRef(name="identity", params={})
    ty = cfg.transforms_y or PPluginRef(name="logic5_from_tidy_v1", params={})
    mdl = cfg.models or PPluginRef(name="random_forest", params={})
    obj = cfg.objectives or PPluginRef(name="logic_plus_effect_v1", params={})
    sel = cfg.selection or PPluginRef(name="top_n", params={})

    tx_params = validate_params("transform_x", tx.name, tx.params)
    ty_params = validate_params("transform_y", ty.name, ty.params)
    mdl_params = validate_params("model", mdl.name, mdl.params)
    obj_params = validate_params("objective", obj.name, obj.params)
    sel_params = validate_params("selection", sel.name, sel.params)

    # Build dataclasses (public API)
    data_dc = DataBlock(
        location=_coerce_location(cfg.data.location),
        representation_column_name=cfg.data.representation_column_name,
        label_source_column_name=cfg.data.label_source_column_name,
        y_expected_length=cfg.data.y_expected_length,
        transforms_x=PluginRef(tx.name, tx_params),
        transforms_y=PluginRef(ty.name, ty_params),
    )

    selection_dc = SelectionBlock(
        selection=PluginRef(sel.name, sel_params),
        top_k_default=sel_params.get("top_k_default"),
        tie_handling=sel_params.get("tie_handling"),
    )
    objective_dc = ObjectiveBlock(objective=PluginRef(obj.name, obj_params))

    training_dc = TrainingBlock(
        models=PluginRef(mdl.name, mdl_params),
        policy=cfg.training.policy
        or {
            "cumulative_training": True,
            "label_cross_round_deduplication_policy": "latest_only",
            "allow_resuggesting_candidates_until_labeled": True,
        },
        target_scaler=TargetScalerCfg(
            enable=cfg.training.target_scaler.enable,
            method=cfg.training.target_scaler.method,
            minimum_labels_required=cfg.training.target_scaler.minimum_labels_required,
            center_statistic=cfg.training.target_scaler.center_statistic,
            scale_statistic=cfg.training.target_scaler.scale_statistic,
        ),
    )

    scoring_dc = ScoringBlock(
        score_batch_size=cfg.scoring.score_batch_size,
        sort_stability=_auto_sort_stability(cfg.scoring.sort_stability, obj.name),
    )

    safety_dc = SafetyBlock(
        fail_on_mixed_biotype_or_alphabet=cfg.safety.fail_on_mixed_biotype_or_alphabet,
        require_biotype_and_alphabet_on_init=cfg.safety.require_biotype_and_alphabet_on_init,
        conflict_policy_on_duplicate_ids=cfg.safety.conflict_policy_on_duplicate_ids,
        write_back_requires_columns_present=cfg.safety.write_back_requires_columns_present,
        accept_x_mismatch=cfg.safety.accept_x_mismatch,
    )

    root = RootConfig(
        campaign=CampaignBlock(
            name=cfg.campaign.name, slug=cfg.campaign.slug, workdir=cfg.campaign.workdir
        ),
        data=data_dc,
        selection=selection_dc,
        objective=objective_dc,
        training=training_dc,
        scoring=scoring_dc,
        safety=safety_dc,
        metadata=MetadataBlock(
            objective=cfg.metadata.objective, notes=cfg.metadata.notes
        ),
    )
    return root
