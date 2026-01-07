"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/loader.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from typing_extensions import Literal

from ..core.utils import ConfigError
from .plugin_schemas import validate_params
from .types import (
    CampaignBlock,
    DataBlock,
    IngestBlock,
    LocationLocal,
    LocationUSR,
    MetadataBlock,
    ObjectiveBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBlock,
    TrainingBlock,
)


# ---- Strict YAML loader ----
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


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p))))


def _resolve_relative_to(cfg_path: Path, p: Path) -> Path:
    return (cfg_path.parent / p).resolve() if not p.is_absolute() else p


def resolve_path_like(cfg_path: Path, value: str | os.PathLike) -> Path:
    return _resolve_relative_to(cfg_path, _expand(value))


# ---- Pydantic shells for strict validation of top-level YAML ----
class PLocationUSR(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["usr"]
    dataset: str
    path: str


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
        import re as _re

        if not _re.fullmatch(r"[a-z0-9_-]+", v):
            raise ValueError("campaign.slug must match ^[a-z0-9_-]+$")
        return v


class PTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Dict[str, Any] = Field(default_factory=dict)
    y_ops: List[PPluginRef] = Field(default_factory=list)


class PScoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_batch_size: int = 10_000


class PIngest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duplicate_policy: Literal["error", "keep_first", "keep_last"] = "error"


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
    transforms_x: PPluginRef
    transforms_y: PPluginRef
    model: PPluginRef
    objective: PPluginRef
    selection: PPluginRef
    training: PTraining = Field(default_factory=PTraining)
    ingest: PIngest = Field(default_factory=PIngest)
    scoring: PScoring = Field(default_factory=PScoring)
    safety: PSafety = Field(default_factory=PSafety)
    metadata: PMetadata = Field(default_factory=PMetadata)
    plot_config: Optional[str] = None
    plot_defaults: Dict[str, Any] = Field(default_factory=dict)
    plot_presets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Non-core, optional block used by the `opal plot` CLI.
    plots: List[Dict[str, Any]] = Field(default_factory=list)


def load_config(path: Path | str) -> RootConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.load(cfg_path.read_text(), Loader=_StrictLoader)

    try:
        pyd = PRoot.model_validate(raw)

    except ValidationError as e:
        raise ConfigError(f"Invalid campaign.yaml: {e}")

    # Validate params with schemas
    tx = pyd.transforms_x
    ty = pyd.transforms_y
    mdl = pyd.model
    obj = pyd.objective
    sel = pyd.selection

    tx_params = validate_params("transform_x", tx.name, tx.params)
    ty_params = validate_params("transform_y", ty.name, ty.params)
    mdl_params = validate_params("model", mdl.name, mdl.params)
    obj_params = validate_params("objective", obj.name, obj.params)
    sel_params = validate_params("selection", sel.name, sel.params)

    # Build dataclasses
    def _abs(v: str) -> str:
        return str(resolve_path_like(cfg_path, v))

    if isinstance(pyd.data.location, PLocationUSR):
        loc_dc = LocationUSR(
            kind="usr",
            dataset=pyd.data.location.dataset,
            path=_abs(pyd.data.location.path),
        )
    else:
        loc_dc = LocationLocal(kind="local", path=_abs(pyd.data.location.path))

    data_dc = DataBlock(
        location=loc_dc,
        x_column_name=pyd.data.x_column_name,
        y_column_name=pyd.data.y_column_name,
        y_expected_length=pyd.data.y_expected_length,
        transforms_x=PluginRef(tx.name, tx_params),
        transforms_y=PluginRef(ty.name, ty_params),
    )

    selection_dc = SelectionBlock(selection=PluginRef(sel.name, sel_params))
    objective_dc = ObjectiveBlock(objective=PluginRef(obj.name, obj_params))
    training_dc = TrainingBlock(
        policy=pyd.training.policy or {},
        y_ops=[PluginRef(t.name, t.params) for t in pyd.training.y_ops],
    )
    ingest_dc = IngestBlock(duplicate_policy=pyd.ingest.duplicate_policy)
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
        model=PluginRef(mdl.name, mdl_params),
        selection=selection_dc,
        objective=objective_dc,
        training=training_dc,
        ingest=ingest_dc,
        scoring=scoring_dc,
        safety=safety_dc,
        metadata=MetadataBlock(notes=pyd.metadata.notes),
        plot_config=(_abs(pyd.plot_config) if pyd.plot_config else None),
    )
    return root
