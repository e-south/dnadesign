"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/types.py

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


# ---- Generic plugin refs (name + params) ----
@dataclass
class PluginRef:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


# ---- Blocks ----
@dataclass
class DataBlock:
    location: DataLocation
    x_column_name: str
    y_column_name: str
    transforms_x: PluginRef
    transforms_y: PluginRef
    y_expected_length: Optional[int] = None


@dataclass
class SelectionBlock:
    selection: PluginRef  # params must contain top_k + score_ref; objective/tie controls are explicit


@dataclass
class ObjectivesBlock:
    objectives: List[PluginRef]


@dataclass
class TrainingBlock:
    policy: Dict[str, Any] = field(default_factory=dict)
    # NEW: ephemeral per-round Y operations (fit/transform/inverse). Optional; default = [].
    y_ops: List[PluginRef] = field(default_factory=list)


@dataclass
class IngestBlock:
    duplicate_policy: str = "error"  # error | keep_first | keep_last


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
    model: PluginRef
    selection: SelectionBlock
    objectives: ObjectivesBlock
    training: TrainingBlock
    ingest: IngestBlock
    scoring: ScoringBlock
    safety: SafetyBlock
    metadata: MetadataBlock
    plot_config: Optional[str] = None
