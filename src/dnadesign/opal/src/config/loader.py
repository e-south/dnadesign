"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/config/loader.py

Loads and validates OPAL campaign YAML into typed config objects. Resolves.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import Literal

from ..core.config_resolve import resolve_campaign_root
from ..core.utils import ConfigError
from .plugin_schemas import validate_params
from .types import (
    ArtifactRetentionBlock,
    CampaignBlock,
    CandidateEligibilityBlock,
    CandidateScope,
    DataBlock,
    IngestBlock,
    LabelsBlock,
    LabelSourceCampaignHistory,
    LabelSourceUSRSidecar,
    LocationLocal,
    LocationUSR,
    OwnershipBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBatchAllocationBlock,
    SelectionBatchBlock,
    SelectionView,
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


class PSelectionView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    objective: PPluginRef
    selection: PPluginRef

    @field_validator("id")
    @classmethod
    def _id_ok(cls, value: str) -> str:
        import re

        out = str(value).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", out):
            raise ValueError("selection view id must match ^[a-z0-9][a-z0-9_-]*$")
        return out


class PSelectionBatchAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["round_robin_next_best_unallocated"]
    view_priority: List[str] = Field(min_length=1)

    @field_validator("view_priority")
    @classmethod
    def _view_priority_valid(cls, value: List[str]) -> List[str]:
        out = [str(item).strip() for item in value]
        if any(not item for item in out):
            raise ValueError("selection_batch.allocation.view_priority entries must be non-empty")
        if len(out) != len(set(out)):
            raise ValueError("selection_batch.allocation.view_priority must not contain duplicates")
        return out


class PSelectionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deduplicate_by: Optional[str] = None
    expected_unique_count: Optional[int] = None
    allocation: Optional[PSelectionBatchAllocation] = None

    @field_validator("deduplicate_by")
    @classmethod
    def _deduplicate_by_nonempty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        out = str(value).strip()
        if not out:
            raise ValueError("selection_batch.deduplicate_by must be non-empty when provided")
        return out

    @field_validator("expected_unique_count")
    @classmethod
    def _expected_unique_count_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        out = int(value)
        if out <= 0:
            raise ValueError("selection_batch.expected_unique_count must be positive")
        return out


class PCandidateEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: List[PPluginRef] = Field(default_factory=list)


class PCandidateScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["id_list"] = "id_list"
    path: str
    id_column: str = "id"


class PData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: PLocation
    x_column_name: str
    y_column_name: str
    y_expected_length: Optional[int] = None
    candidate_scope: Optional[PCandidateScope] = None


class PLabelSourceCampaignHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["campaign_history"] = "campaign_history"


class PLabelSourceUSRSidecar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["usr_sidecar"]
    dataset: Optional[str] = None
    path: str = "_opal/observed_labels.parquet"
    manifest_path: Optional[str] = None


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
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    prediction_records: Literal["ledger_only"] = "ledger_only"


class PArtifactRetention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["audit_full", "production_review", "ephemeral_selection"] = "audit_full"
    prediction_ledger: Literal[
        "all_rounds_full",
        "latest_full_plus_selected_history",
        "selected_history_only",
    ] = "all_rounds_full"
    plot_tidy_data: Literal["full", "compact", "none"] = "full"
    model_artifacts: Literal["all", "latest"] = "all"
    tabular_format: Literal["parquet", "parquet_zstd"] = "parquet"
    max_estimated_bytes: int = 50_000_000_000
    fail_if_estimate_exceeds: bool = True
    final_round: Optional[int] = None

    @field_validator("max_estimated_bytes")
    @classmethod
    def _max_estimated_bytes_positive(cls, value: int) -> int:
        out = int(value)
        if out <= 0:
            raise ValueError("artifact_retention.max_estimated_bytes must be positive")
        return out

    @field_validator("final_round")
    @classmethod
    def _final_round_nonnegative(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        out = int(value)
        if out < 0:
            raise ValueError("artifact_retention.final_round must be non-negative")
        return out


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
    max_x_matrix_gib: float = 8.0

    @field_validator("max_x_matrix_gib")
    @classmethod
    def _max_x_matrix_gib_positive(cls, value: float) -> float:
        out = float(value)
        if out <= 0.0:
            raise ValueError("safety.max_x_matrix_gib must be > 0")
        return out


class POwnership(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_scope: Literal["opal_demo", "study_campaign"]
    study_id: Optional[str] = None
    dataset_id: Optional[str] = None
    portable: bool = True

    @model_validator(mode="after")
    def _scope_contract(self) -> "POwnership":
        study_id = str(self.study_id or "").strip()
        dataset_id = str(self.dataset_id or "").strip()
        if self.owner_scope == "opal_demo":
            if study_id:
                raise ValueError("opal_demo ownership must not declare study_id")
            if dataset_id:
                raise ValueError("opal_demo ownership must not declare dataset_id")
            if not self.portable:
                raise ValueError("opal_demo ownership must set portable=true")
            return self
        if not study_id:
            raise ValueError("study_campaign ownership requires study_id")
        if not dataset_id:
            raise ValueError("study_campaign ownership requires dataset_id")
        if self.portable:
            raise ValueError("study_campaign ownership must set portable=false")
        self.study_id = study_id
        self.dataset_id = dataset_id
        return self


class PRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["opal.campaign.v3"]
    campaign: PCampaign
    data: PData
    candidate_eligibility: PCandidateEligibility = Field(default_factory=PCandidateEligibility)
    transforms_x: PPluginRef
    transforms_y: PPluginRef
    model: PPluginRef
    selection_views: List[PSelectionView]
    selection_batch: PSelectionBatch = Field(default_factory=PSelectionBatch)
    labels: PLabels = Field(default_factory=PLabels)
    writeback: Optional[PWriteback] = None
    artifact_retention: PArtifactRetention = Field(default_factory=PArtifactRetention)
    training: PTraining = Field(default_factory=PTraining)
    ingest: PIngest = Field(default_factory=PIngest)
    scoring: PScoring = Field(default_factory=PScoring)
    safety: PSafety = Field(default_factory=PSafety)
    plot_config: Optional[str] = None
    ownership: POwnership


def _require_registered_plugin(*, category: str, name: str, available: set[str]) -> None:
    if name in available:
        return
    avail = ", ".join(sorted(available))
    raise ConfigError(f"Unknown {category} plugin '{name}'. Available plugins: {avail}")


def _dataset_relative_path(value: str, *, field: str) -> str:
    raw = str(value).strip()
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(raw)
    if not raw or "\\" in raw or posix_path.is_absolute() or windows_path.is_absolute() or ".." in posix_path.parts:
        raise ConfigError(f"Invalid campaign.yaml: labels.source.{field} must be relative to the USR dataset root.")
    return posix_path.as_posix()


def _validate_registered_plugin_names(pyd: PRoot) -> None:
    from ..registries.eligibility import list_candidate_eligibility_rules
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
    available_objectives = set(list_objectives())
    available_selections = set(list_selections())
    for view in pyd.selection_views:
        _require_registered_plugin(
            category="objective",
            name=str(view.objective.name),
            available=available_objectives,
        )
        _require_registered_plugin(
            category="selection",
            name=str(view.selection.name),
            available=available_selections,
        )

    available_y_ops = set(list_y_ops())
    for y_op in pyd.training.y_ops:
        _require_registered_plugin(
            category="training.y_ops",
            name=str(y_op.name),
            available=available_y_ops,
        )

    available_eligibility = set(list_candidate_eligibility_rules())
    for rule in pyd.candidate_eligibility.rules:
        _require_registered_plugin(
            category="candidate_eligibility",
            name=str(rule.name),
            available=available_eligibility,
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

    try:
        tx_params = validate_params("transform_x", tx.name, tx.params)
        ty_params = validate_params("transform_y", ty.name, ty.params)
        mdl_params = validate_params("model", mdl.name, mdl.params)
    except Exception as e:
        raise ConfigError(f"Invalid campaign.yaml plugin params: {e}")

    if not pyd.selection_views:
        raise ConfigError("Invalid campaign.yaml: selection_views must contain at least one entry.")
    view_ids = [view.id for view in pyd.selection_views]
    if len(view_ids) != len(set(view_ids)):
        raise ConfigError("Invalid campaign.yaml: selection view ids must be unique.")
    selection_views: list[SelectionView] = []
    try:
        for view in pyd.selection_views:
            objective = PluginRef(
                view.objective.name,
                validate_params("objective", view.objective.name, view.objective.params),
            )
            selection_params = validate_params("selection", view.selection.name, view.selection.params)
            for ref_key in ("score_ref", "uncertainty_ref"):
                ref = selection_params.get(ref_key)
                if ref is not None and "/" in str(ref):
                    raise ConfigError(
                        f"Invalid campaign.yaml: selection_views[{view.id}].selection.params.{ref_key} "
                        "must be an objective channel name, not a namespaced reference."
                    )
            selection_views.append(
                SelectionView(
                    id=view.id,
                    objective=objective,
                    selection=PluginRef(view.selection.name, selection_params),
                )
            )
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Invalid campaign.yaml selection view params: {e}")
    exclude_policies = {bool(view.selection.params.get("exclude_already_labeled", True)) for view in selection_views}
    if len(exclude_policies) != 1:
        raise ConfigError(
            "Invalid campaign.yaml: all selection views must use the same "
            "selection.params.exclude_already_labeled policy."
        )
    allocation_dc: SelectionBatchAllocationBlock | None = None
    if pyd.selection_batch.allocation is not None:
        allocation = pyd.selection_batch.allocation
        priority = list(allocation.view_priority)
        missing_priority = sorted(set(view_ids) - set(priority))
        unknown_priority = sorted(set(priority) - set(view_ids))
        if missing_priority or unknown_priority or len(priority) != len(view_ids):
            raise ConfigError(
                "Invalid campaign.yaml: selection_batch.allocation.view_priority must be an exact "
                f"permutation of selection view ids; missing={missing_priority}, unknown={unknown_priority}."
            )
        quota_total = 0
        for view in selection_views:
            params = dict(view.selection.params)
            if str(params.get("tie_handling", "")).strip() != "ordinal":
                raise ConfigError(
                    "Invalid campaign.yaml: selection_batch allocation requires "
                    f"selection_views[{view.id}].selection.params.tie_handling='ordinal'."
                )
            if not bool(params.get("require_exact_top_k", False)):
                raise ConfigError(
                    "Invalid campaign.yaml: selection_batch allocation requires "
                    f"selection_views[{view.id}].selection.params.require_exact_top_k=true."
                )
            try:
                top_k = int(params["top_k"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(
                    "Invalid campaign.yaml: selection_batch allocation requires a positive integer "
                    f"selection_views[{view.id}].selection.params.top_k."
                ) from exc
            if top_k <= 0:
                raise ConfigError("Invalid campaign.yaml: selection_batch allocation requires positive top_k values.")
            quota_total += top_k
        expected_unique_count = pyd.selection_batch.expected_unique_count
        if expected_unique_count is None:
            raise ConfigError("Invalid campaign.yaml: selection_batch allocation requires expected_unique_count.")
        if int(expected_unique_count) != quota_total:
            raise ConfigError(
                "Invalid campaign.yaml: selection_batch.expected_unique_count must equal the sum of "
                f"selection view top_k quotas ({quota_total}) when allocation is configured."
            )
        allocation_dc = SelectionBatchAllocationBlock(
            strategy=str(allocation.strategy),
            view_priority=priority,
        )
    try:
        eligibility_rules = [
            PluginRef(rule.name, validate_params("candidate_eligibility", rule.name, rule.params))
            for rule in pyd.candidate_eligibility.rules
        ]
    except Exception as e:
        raise ConfigError(f"Invalid campaign.yaml candidate_eligibility params: {e}")

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
        candidate_scope=(
            CandidateScope(
                kind=str(pyd.data.candidate_scope.kind),
                path=_abs(pyd.data.candidate_scope.path, base_dir=campaign_root),
                id_column=str(pyd.data.candidate_scope.id_column),
            )
            if pyd.data.candidate_scope is not None
            else None
        ),
    )

    if isinstance(pyd.labels.source, PLabelSourceUSRSidecar):
        if not isinstance(pyd.data.location, PLocationUSR):
            raise ConfigError("Invalid campaign.yaml: labels.source.kind=usr_sidecar requires data.location.kind=usr.")
        if pyd.writeback is None:
            raise ConfigError(
                "Invalid campaign.yaml: labels.source.kind=usr_sidecar requires explicit "
                "writeback.prediction_records=ledger_only."
            )
        label_dataset = pyd.labels.source.dataset or pyd.data.location.dataset
        if label_dataset != pyd.data.location.dataset:
            raise ConfigError(
                "Invalid campaign.yaml: labels.source.dataset must target the same dataset as data.location.dataset."
            )
        if not pyd.labels.y_space or not str(pyd.labels.y_space).strip():
            raise ConfigError("Invalid campaign.yaml: labels.y_space is required for labels.source.kind=usr_sidecar.")
        label_path = _dataset_relative_path(str(pyd.labels.source.path), field="path")
        manifest_path = (
            _dataset_relative_path(str(pyd.labels.source.manifest_path), field="manifest_path")
            if pyd.labels.source.manifest_path is not None
            else None
        )
        if manifest_path is not None and pyd.ownership.owner_scope != "study_campaign":
            raise ConfigError("Invalid campaign.yaml: labels.source.manifest_path requires study_campaign ownership.")
        label_source = LabelSourceUSRSidecar(
            kind="usr_sidecar",
            dataset=label_dataset,
            path=label_path,
            manifest_path=manifest_path,
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

    candidate_eligibility_dc = CandidateEligibilityBlock(rules=eligibility_rules)
    training_dc = TrainingBlock(
        policy=pyd.training.policy or {},
        y_ops=[PluginRef(t.name, t.params) for t in pyd.training.y_ops],
    )
    ingest_dc = IngestBlock(duplicate_policy=pyd.ingest.duplicate_policy)
    scoring_dc = ScoringBlock(score_batch_size=int(pyd.scoring.score_batch_size))
    writeback_dc = WritebackBlock(
        prediction_records=(pyd.writeback.prediction_records if pyd.writeback else "ledger_only")
    )
    artifact_retention_dc = ArtifactRetentionBlock(
        mode=str(pyd.artifact_retention.mode),
        prediction_ledger=str(pyd.artifact_retention.prediction_ledger),
        plot_tidy_data=str(pyd.artifact_retention.plot_tidy_data),
        model_artifacts=str(pyd.artifact_retention.model_artifacts),
        tabular_format=str(pyd.artifact_retention.tabular_format),
        max_estimated_bytes=int(pyd.artifact_retention.max_estimated_bytes),
        fail_if_estimate_exceeds=bool(pyd.artifact_retention.fail_if_estimate_exceeds),
        final_round=(None if pyd.artifact_retention.final_round is None else int(pyd.artifact_retention.final_round)),
    )
    safety_dc = SafetyBlock(
        fail_on_mixed_biotype_or_alphabet=pyd.safety.fail_on_mixed_biotype_or_alphabet,
        require_biotype_and_alphabet_on_init=pyd.safety.require_biotype_and_alphabet_on_init,
        conflict_policy_on_duplicate_ids=pyd.safety.conflict_policy_on_duplicate_ids,
        write_back_requires_columns_present=pyd.safety.write_back_requires_columns_present,
        accept_x_mismatch=pyd.safety.accept_x_mismatch,
        max_x_matrix_gib=float(pyd.safety.max_x_matrix_gib),
    )
    ownership_dc = OwnershipBlock(
        owner_scope=pyd.ownership.owner_scope,
        study_id=pyd.ownership.study_id,
        dataset_id=pyd.ownership.dataset_id,
        portable=bool(pyd.ownership.portable),
    )

    root = RootConfig(
        schema_version=str(pyd.schema_version),
        campaign=CampaignBlock(
            name=pyd.campaign.name,
            slug=pyd.campaign.slug,
            workdir=str(resolve_path_like(cfg_path, pyd.campaign.workdir, base_dir=campaign_root)),
            description=(str(pyd.campaign.description).strip() if pyd.campaign.description else None),
            metadata=dict(pyd.campaign.metadata or {}),
        ),
        data=data_dc,
        model=PluginRef(mdl.name, mdl_params),
        selection_views=selection_views,
        selection_batch=SelectionBatchBlock(
            deduplicate_by=pyd.selection_batch.deduplicate_by,
            expected_unique_count=pyd.selection_batch.expected_unique_count,
            allocation=allocation_dc,
        ),
        candidate_eligibility=candidate_eligibility_dc,
        training=training_dc,
        ingest=ingest_dc,
        scoring=scoring_dc,
        safety=safety_dc,
        labels=labels_dc,
        writeback=writeback_dc,
        artifact_retention=artifact_retention_dc,
        plot_config=(_abs(pyd.plot_config) if pyd.plot_config else None),
        ownership=ownership_dc,
    )
    return root
