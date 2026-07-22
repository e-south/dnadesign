"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/config/types.py

Configuration contracts for types OPAL config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---- Data location ----
@dataclass
class LocationUSR:
    kind: str  # "usr"
    dataset: str
    path: str


@dataclass
class LocationLocal:
    kind: str  # "local"
    path: str


DataLocation = Union[LocationUSR, LocationLocal]


@dataclass
class CandidateScope:
    kind: str  # "id_list"
    path: str
    id_column: str = "id"


# ---- Label sources ----
@dataclass
class LabelSourceCampaignHistory:
    kind: str = "campaign_history"


@dataclass
class LabelSourceUSRSidecar:
    kind: str
    dataset: str
    path: str
    manifest_path: Optional[str] = None


LabelSource = Union[LabelSourceCampaignHistory, LabelSourceUSRSidecar]


# ---- Generic plugin refs (name + params) ----
@dataclass
class PluginRef:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEligibilityBlock:
    rules: List[PluginRef] = field(default_factory=list)


# ---- Blocks ----
@dataclass
class DataBlock:
    location: DataLocation
    x_column_name: str
    y_column_name: str
    transforms_x: PluginRef
    transforms_y: PluginRef
    y_expected_length: Optional[int] = None
    candidate_scope: Optional[CandidateScope] = None


@dataclass(frozen=True)
class SelectionView:
    id: str
    objective: PluginRef
    selection: PluginRef


@dataclass(frozen=True)
class SelectionBatchAllocationBlock:
    strategy: str
    view_priority: List[str]


@dataclass(frozen=True)
class SelectionBatchBlock:
    deduplicate_by: Optional[str] = None
    expected_unique_count: Optional[int] = None
    allocation: Optional[SelectionBatchAllocationBlock] = None


@dataclass
class TrainingBlock:
    policy: Dict[str, Any] = field(default_factory=dict)
    # NEW: ephemeral per-round Y operations (fit/transform/inverse). Optional; default = [].
    y_ops: List[PluginRef] = field(default_factory=list)


@dataclass
class LabelsBlock:
    source: LabelSource = field(default_factory=LabelSourceCampaignHistory)
    y_space: Optional[str] = None
    id_column: str = "id"
    round_column: str = "observed_round"
    batch_column: str = "batch_id"
    dedup_policy: str = "latest_by_round"


@dataclass
class IngestBlock:
    duplicate_policy: str = "error"  # error | keep_first | keep_last


@dataclass
class ScoringBlock:
    score_batch_size: int = 10_000


@dataclass
class WritebackBlock:
    prediction_records: str = "ledger_only"


@dataclass
class ArtifactRetentionBlock:
    # audit_full | production_review | ephemeral_selection
    mode: str = "audit_full"
    # all_rounds_full | latest_full_plus_selected_history | selected_history_only
    prediction_ledger: str = "all_rounds_full"
    # full | compact | none
    plot_tidy_data: str = "full"
    # all | latest
    model_artifacts: str = "all"
    # parquet | parquet_zstd
    tabular_format: str = "parquet"
    max_estimated_bytes: int = 50_000_000_000
    fail_if_estimate_exceeds: bool = True
    final_round: Optional[int] = None


@dataclass
class SafetyBlock:
    fail_on_mixed_biotype_or_alphabet: bool = True
    require_biotype_and_alphabet_on_init: bool = True
    conflict_policy_on_duplicate_ids: str = "error"
    write_back_requires_columns_present: bool = True
    accept_x_mismatch: bool = False
    max_x_matrix_gib: float = 8.0


@dataclass
class OwnershipBlock:
    owner_scope: str
    study_id: Optional[str] = None
    dataset_id: Optional[str] = None
    portable: bool = True


@dataclass
class CampaignBlock:
    name: str
    slug: str
    workdir: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RootConfig:
    schema_version: str
    campaign: CampaignBlock
    data: DataBlock
    model: PluginRef
    selection_views: List[SelectionView]
    selection_batch: SelectionBatchBlock
    training: TrainingBlock
    ingest: IngestBlock
    scoring: ScoringBlock
    safety: SafetyBlock
    ownership: OwnershipBlock
    candidate_eligibility: CandidateEligibilityBlock = field(default_factory=CandidateEligibilityBlock)
    labels: LabelsBlock = field(default_factory=LabelsBlock)
    writeback: WritebackBlock = field(default_factory=WritebackBlock)
    artifact_retention: ArtifactRetentionBlock = field(default_factory=ArtifactRetentionBlock)
    plot_config: Optional[str] = None
