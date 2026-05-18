"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/loader.py

Loads and validates OPAL campaign YAML into typed config objects. Resolves
paths relative to campaign root and config location.

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

from ..core.config_resolve import resolve_campaign_root
from ..core.utils import ConfigError
from .plugin_schemas import validate_params
from .types import (
    CampaignBlock,
    DataBlock,
    IngestBlock,
    LabelsBlock,
    LabelSourceCampaignHistory,
    LabelSourceUSRSidecar,
    LocationLocal,
    LocationUSR,
    ObjectivesBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBlock,
    TrainingBlock,
    WritebackBlock,
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


def _resolve_relative_to(base_dir: Path, p: Path) -> Path:
    return (base_dir / p).resolve() if not p.is_absolute() else p


def resolve_path_like(cfg_path: Path, value: str | os.PathLike, *, base_dir: Path | None = None) -> Path:
    base = base_dir if base_dir is not None else cfg_path.parent
    return _resolve_relative_to(base, _expand(value))


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


class PLabelSourceCampaignHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["campaign_history"] = "campaign_history"


class PLabelSourceUSRSidecar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["usr_sidecar"]
    dataset: Optional[str] = None
    path: str = "_opal/observed_labels.parquet"


PLabelSource = Union[PLabelSourceCampaignHistory, PLabelSourceUSRSidecar]


class PLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: PLabelSource = Field(default_factory=PLabelSourceCampaignHistory)
    y_space: Optional[str] = None
    id_column: str = "id"
    round_column: str = "observed_round"
    batch_column: str = "batch_id"
    dedup_policy: Literal["latest_by_round", "all_events", "error_on_duplicate"] = "latest_by_round"


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


class PWriteback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_records: Literal["label_history", "ledger_only"] = "label_history"


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


class PRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign: PCampaign
    data: PData
    transforms_x: PPluginRef
    transforms_y: PPluginRef
    model: PPluginRef
    objectives: List[PPluginRef]
    selection: PPluginRef
    labels: PLabels = Field(default_factory=PLabels)
    writeback: Optional[PWriteback] = None
    training: PTraining = Field(default_factory=PTraining)
    ingest: PIngest = Field(default_factory=PIngest)
    scoring: PScoring = Field(default_factory=PScoring)
    safety: PSafety = Field(default_factory=PSafety)
    plot_config: Optional[str] = None
    plot_defaults: Dict[str, Any] = Field(default_factory=dict)
    plot_presets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Non-core, optional block used by the `opal plot` CLI.
    plots: List[Dict[str, Any]] = Field(default_factory=list)


def _require_registered_plugin(*, category: str, name: str, available: set[str]) -> None:
    if name in available:
        return
    avail = ", ".join(sorted(available))
    raise ConfigError(f"Unknown {category} plugin '{name}'. Available plugins: {avail}")


def _validate_registered_plugin_names(pyd: PRoot) -> None:
    from ..registries.models import list_models
    from ..registries.objectives import list_objectives
    from ..registries.selection import list_selections
    from ..registries.transforms_x import list_transforms_x
    from ..registries.transforms_y import list_transforms_y, list_y_ops

    _require_registered_plugin(
        category="transform_x",
        name=str(pyd.transforms_x.name),
        available=set(list_transforms_x()),
    )
    _require_registered_plugin(
        category="transform_y",
        name=str(pyd.transforms_y.name),
        available=set(list_transforms_y()),
    )
    _require_registered_plugin(
        category="model",
        name=str(pyd.model.name),
        available=set(list_models()),
    )
    _require_registered_plugin(
        category="selection",
        name=str(pyd.selection.name),
        available=set(list_selections()),
    )

    available_objectives = set(list_objectives())
    for obj in pyd.objectives:
        _require_registered_plugin(
            category="objective",
            name=str(obj.name),
            available=available_objectives,
        )

    available_y_ops = set(list_y_ops())
    for y_op in pyd.training.y_ops:
        _require_registered_plugin(
            category="training.y_ops",
            name=str(y_op.name),
            available=available_y_ops,
        )


def load_config(path: Path | str) -> RootConfig:
    cfg_path = Path(path).resolve()
    campaign_root = resolve_campaign_root(cfg_path)
    try:
        raw = yaml.load(cfg_path.read_text(), Loader=_StrictLoader)
    except (yaml.YAMLError, KeyError) as e:
        raise ConfigError(f"Invalid campaign.yaml: {e}") from e

    try:
        pyd = PRoot.model_validate(raw)

    except ValidationError as e:
        raise ConfigError(f"Invalid campaign.yaml: {e}")

    _validate_registered_plugin_names(pyd)

    # Validate params with schemas
    tx = pyd.transforms_x
    ty = pyd.transforms_y
    mdl = pyd.model
    sel = pyd.selection

    try:
        tx_params = validate_params("transform_x", tx.name, tx.params)
        ty_params = validate_params("transform_y", ty.name, ty.params)
        mdl_params = validate_params("model", mdl.name, mdl.params)
        sel_params = validate_params("selection", sel.name, sel.params)
    except Exception as e:
        raise ConfigError(f"Invalid campaign.yaml plugin params: {e}")

    if not pyd.objectives:
        raise ConfigError("Invalid campaign.yaml: objectives must contain at least one objective plugin entry.")
    obj_names = [o.name for o in pyd.objectives]
    if len(obj_names) != len(set(obj_names)):
        raise ConfigError("Invalid campaign.yaml: objective names must be unique in objectives.")
    try:
        obj_refs = [PluginRef(o.name, validate_params("objective", o.name, o.params)) for o in pyd.objectives]
    except Exception as e:
        raise ConfigError(f"Invalid campaign.yaml objective params: {e}")

    # Build dataclasses
    def _abs(v: str, *, base_dir: Path | None = None) -> str:
        return str(resolve_path_like(cfg_path, v, base_dir=base_dir))

    if isinstance(pyd.data.location, PLocationUSR):
        loc_dc = LocationUSR(
            kind="usr",
            dataset=pyd.data.location.dataset,
            path=_abs(pyd.data.location.path, base_dir=campaign_root),
        )
    else:
        loc_dc = LocationLocal(kind="local", path=_abs(pyd.data.location.path, base_dir=campaign_root))

    data_dc = DataBlock(
        location=loc_dc,
        x_column_name=pyd.data.x_column_name,
        y_column_name=pyd.data.y_column_name,
        y_expected_length=pyd.data.y_expected_length,
        transforms_x=PluginRef(tx.name, tx_params),
        transforms_y=PluginRef(ty.name, ty_params),
    )

    if isinstance(pyd.labels.source, PLabelSourceUSRSidecar):
        if not isinstance(pyd.data.location, PLocationUSR):
            raise ConfigError("Invalid campaign.yaml: labels.source.kind=usr_sidecar requires data.location.kind=usr.")
        if pyd.writeback is None:
            raise ConfigError(
                "Invalid campaign.yaml: labels.source.kind=usr_sidecar requires explicit "
                "writeback.prediction_records (use ledger_only unless records label-history prediction "
                "writeback is intentional)."
            )
        label_dataset = pyd.labels.source.dataset or pyd.data.location.dataset
        if label_dataset != pyd.data.location.dataset:
            raise ConfigError(
                "Invalid campaign.yaml: labels.source.dataset must target the same dataset as data.location.dataset."
            )
        if not pyd.labels.y_space or not str(pyd.labels.y_space).strip():
            raise ConfigError("Invalid campaign.yaml: labels.y_space is required for labels.source.kind=usr_sidecar.")
        if Path(str(pyd.labels.source.path)).is_absolute():
            raise ConfigError("Invalid campaign.yaml: labels.source.path must be relative to the USR dataset root.")
        label_source = LabelSourceUSRSidecar(
            kind="usr_sidecar",
            dataset=label_dataset,
            path=str(pyd.labels.source.path),
        )
    else:
        label_source = LabelSourceCampaignHistory()

    labels_dc = LabelsBlock(
        source=label_source,
        y_space=(str(pyd.labels.y_space).strip() if pyd.labels.y_space else None),
        id_column=str(pyd.labels.id_column),
        round_column=str(pyd.labels.round_column),
        batch_column=str(pyd.labels.batch_column),
        dedup_policy=str(pyd.labels.dedup_policy),
    )

    selection_dc = SelectionBlock(selection=PluginRef(sel.name, sel_params))
    objectives_dc = ObjectivesBlock(objectives=obj_refs)
    training_dc = TrainingBlock(
        policy=pyd.training.policy or {},
        y_ops=[PluginRef(t.name, t.params) for t in pyd.training.y_ops],
    )
    ingest_dc = IngestBlock(duplicate_policy=pyd.ingest.duplicate_policy)
    scoring_dc = ScoringBlock(score_batch_size=int(pyd.scoring.score_batch_size))
    writeback_dc = WritebackBlock(
        prediction_records=(pyd.writeback.prediction_records if pyd.writeback else "label_history")
    )
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
            workdir=str(resolve_path_like(cfg_path, pyd.campaign.workdir, base_dir=campaign_root)),
        ),
        data=data_dc,
        model=PluginRef(mdl.name, mdl_params),
        selection=selection_dc,
        objectives=objectives_dc,
        training=training_dc,
        ingest=ingest_dc,
        scoring=scoring_dc,
        safety=safety_dc,
        labels=labels_dc,
        writeback=writeback_dc,
        plot_config=(_abs(pyd.plot_config) if pyd.plot_config else None),
    )
    return root
