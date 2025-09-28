"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/types.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


# ---- Data location ----
@dataclass
class LocationUSR:
    kind: str  # "usr"
    dataset: str
    path: str  # unified key (replaces former usr_root)


@dataclass
class LocationLocal:
    kind: str  # "local"
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
    x_column_name: str
    y_column_name: str
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
    top_k_default: Optional[int] = None
    tie_handling: Optional[str] = None
    objective: Optional[str] = None  # "maximize" | "minimize"


@dataclass
class ObjectiveBlock:
    objective: PluginRef = field(default_factory=lambda: PluginRef("sfxi_v1", {}))


# ---- Training / Safety / Meta ----
@dataclass
class TargetNormalizerCfg:
    """
    Per-target Y normalization used at model fit-time and inverted on prediction.
    """

    enable: bool = True
    method: str = "robust_iqr_per_target"
    minimum_labels_required: int = 5
    center_statistic: str = "median"
    scale_statistic: str = "iqr"  # keep name 'scale_statistic' in config for clarity


@dataclass
class TrainingBlock:
    models: PluginRef = field(default_factory=lambda: PluginRef("random_forest", {}))
    policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "cumulative_training": True,
            "label_cross_round_deduplication_policy": "latest_only",
            "allow_resuggesting_candidates_until_labeled": True,
        }
    )
    target_normalizer: TargetNormalizerCfg = field(default_factory=TargetNormalizerCfg)


@dataclass
class ScoringBlock:
    score_batch_size: int = 10_000


@dataclass
class SafetyBlock:
    fail_on_mixed_biotype_or_alphabet: bool = True
    require_biotype_and_alphabet_on_init: bool = True
    conflict_policy_on_duplicate_ids: str = "error"
    write_back_requires_columns_present: bool = True
    accept_x_mismatch: bool = False


@dataclass
class MetadataBlock:
    notes: str = ""


@dataclass
class CampaignBlock:
    name: str
    slug: str
    workdir: str


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
