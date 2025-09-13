"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/types.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union


# ---- Data location ----
@dataclass
class LocationUSR:
    kind: str
    dataset: str
    usr_root: str


@dataclass
class LocationLocal:
    kind: str
    path: str


DataLocation = Union[LocationUSR, LocationLocal]


# ---- Generic plugin refs (name + params) ----
@dataclass
class PluginRef:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


# ---- Data block ----
@dataclass
class DataBlock:
    location: DataLocation
    # Accept x_column_name via the loader (aliases).
    representation_column_name: str
    # Accept y_column_name via the loader (aliases).
    label_source_column_name: str
    y_expected_length: Optional[int] = None
    # plugin-configured transforms
    transforms_x: PluginRef = field(default_factory=lambda: PluginRef("identity", {}))
    transforms_y: PluginRef = field(
        default_factory=lambda: PluginRef("logic5_from_tidy_v1", {})
    )


# ---- Selection & objective (plugin-driven) ----
@dataclass
class SelectionBlock:
    selection: PluginRef = field(default_factory=lambda: PluginRef("top_n", {}))
    # convenience surface for common params (auto-populated by loader)
    top_k_default: Optional[int] = None
    tie_handling: Optional[str] = None
    objective: Optional[str] = None  # "maximize" | "minimize"
    sort_stability: Optional[str] = None  # resolved stable sort key


@dataclass
class ObjectiveBlock:
    objective: PluginRef = field(default_factory=lambda: PluginRef("sfxi_v1", {}))


# ---- Training / Safety / Meta ----
@dataclass
class TargetScalerCfg:
    enable: bool = True
    method: str = "robust_iqr_per_target"
    minimum_labels_required: int = 5
    center_statistic: str = "median"
    scale_statistic: str = "iqr"
@dataclass
class TrainingBlock:
    # model plugin (e.g., random_forest)
    models: PluginRef = field(default_factory=lambda: PluginRef("random_forest", {}))
    policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "cumulative_training": True,
            "label_cross_round_deduplication_policy": "latest_only",
            "allow_resuggesting_candidates_until_labeled": True,
        }
    )
    target_scaler: TargetScalerCfg = field(default_factory=TargetScalerCfg)


@dataclass
class SafetyBlock:
    fail_on_mixed_biotype_or_alphabet: bool = True
    require_biotype_and_alphabet_on_init: bool = True
    conflict_policy_on_duplicate_ids: str = "error"
    write_back_requires_columns_present: bool = True
    accept_x_mismatch: bool = False


@dataclass
class CampaignBlock:
    name: str
    slug: str
    workdir: str


@dataclass
class MetadataBlock:
    notes: str = ""


@dataclass
class ScoringBlock:
    score_batch_size: int = 10_000
    sort_stability: str = (
        "(-opal__{slug}__r{round}__selection_score__{objective}, id)"
    )


@dataclass
class RootConfig:
    campaign: CampaignBlock
    data: DataBlock
    selection: SelectionBlock
    objective: ObjectiveBlock
    training: TrainingBlock
    scoring: ScoringBlock
    safety: SafetyBlock
    metadata: MetadataBlock

    def asdict(self):
        return asdict(self)
