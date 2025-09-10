"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config.py

Defines campaign configuration as dataclasses.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------- Data location ----------


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


# ---------- Ingest transform (CSV -> Y) ----------


@dataclass
class IngestTransformParams:
    schema: Dict[str, str] = field(default_factory=dict)
    enforce_single_timepoint: bool = True
    replicate_aggregation: str = "mean"
    replicate_warn_threshold: int = 3
    pre_processing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestTransform:
    name: str = "logic5_from_tidy_v1"
    params: IngestTransformParams = field(default_factory=IngestTransformParams)


# ---------- Representation transform (X) ----------


@dataclass
class RepTransform:
    name: str = "identity"
    params: Dict[str, Any] = field(default_factory=dict)


# ---------- Data block ----------


@dataclass
class DataBlock:
    location: DataLocation
    representation_column_name: str
    label_source_column_name: str
    y_expected_length: Optional[int] = None
    ingest: IngestTransform = field(default_factory=IngestTransform)
    representation_transform: RepTransform = field(default_factory=RepTransform)


# ---------- Selection ----------


@dataclass
class ObjectiveParams:
    setpoint_vector: List[float]
    weighting_between_logic_and_effect: Dict[str, float] = field(
        default_factory=lambda: {"logic_weight": 0.5, "effect_weight": 0.5}
    )
    combination_of_logic_and_effect: Dict[str, float] = field(
        default_factory=lambda: {
            "formula": "product",
            "logic_exponent_beta": 1.0,
            "effect_exponent_gamma": 1.0,
        }
    )
    logic_fidelity_measure: Dict[str, str] = field(
        default_factory=lambda: {
            "distance": "l2_to_setpoint",
            "normalization": "per_state_max_deviation_to_unit_interval",
        }
    )
    effect_size_scaling_for_selection: Dict[str, Any] = field(
        default_factory=lambda: {
            "scaling_method": "per_round_percentile_clip",
            "scaling_pool": "labeled_rows_only",
            "percentile": 95,
            "low_sample_fallback": {
                "min_labeled_count": 5,
                "fallback_percentile": 75,
                "epsilon": 1e-8,
            },
        }
    )


@dataclass
class Objective:
    name: str = "logic_plus_effect_v1"
    params: ObjectiveParams = field(
        default_factory=lambda: ObjectiveParams([0, 0, 0, 1])
    )


@dataclass
class SelectionBlock:
    strategy: str = "top_n"
    top_k_default: int = 12
    tie_handling: str = "competition_rank"
    objective: Objective = field(default_factory=Objective)


# ---------- Training ----------


@dataclass
class RFParams:
    n_estimators: int = 100
    criterion: str = "friedman_mse"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, float] = 1.0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = True
    random_state: int = 7
    n_jobs: int = -1


@dataclass
class TargetScalerCfg:
    enable: bool = True
    method: str = "robust_iqr_per_target"
    minimum_labels_required: int = 5
    center_statistic: str = "median"
    scale_statistic: str = "iqr"


@dataclass
class ModelBlock:
    name: str = "random_forest"
    params: RFParams = field(default_factory=RFParams)
    target_scaler: TargetScalerCfg = field(default_factory=TargetScalerCfg)


@dataclass
class TrainingBlock:
    model: ModelBlock = field(default_factory=ModelBlock)
    policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "cumulative_training": True,
            "label_cross_round_deduplication_policy": "latest_only",
            "allow_resuggesting_candidates_until_labeled": True,
        }
    )


# ---------- Scoring / Safety / Meta ----------


@dataclass
class ScoringBlock:
    score_batch_size: int = 10_000
    sort_stability: str = (
        "(-opal__{slug}__r{round}__selection_score__logic_plus_effect_v1, id)"
    )


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
    objective: str = "maximize"
    notes: str = ""


@dataclass
class RootConfig:
    campaign: CampaignBlock
    data: DataBlock
    selection: SelectionBlock
    training: TrainingBlock
    scoring: ScoringBlock
    safety: SafetyBlock
    metadata: MetadataBlock

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- Loader ----------


def _coerce_location(d: Dict[str, Any]) -> DataLocation:
    kind = d.get("kind")
    if kind == "usr":
        return LocationUSR(kind="usr", dataset=d["dataset"], usr_root=d["usr_root"])
    if kind == "local":
        return LocationLocal(kind="local", path=d["path"])
    raise ValueError(f"Unknown data.location.kind: {kind}")


def load_config(path: Path) -> RootConfig:
    raw = yaml.safe_load(Path(path).read_text())

    # campaign
    c = raw["campaign"]
    campaign = CampaignBlock(name=c["name"], slug=c["slug"], workdir=c["workdir"])

    # data
    d = raw["data"]
    loc = _coerce_location(d["location"])
    ingest_raw = d.get("ingest", {}) or {}
    ingest_tf = IngestTransform(
        name=(ingest_raw.get("transform", {}) or {}).get("name", "logic5_from_tidy_v1"),
        params=IngestTransformParams(
            **((ingest_raw.get("transform", {}) or {}).get("params", {}) or {})
        ),
    )
    rep_tf = RepTransform(**(d.get("representation_transform", {}) or {}))
    dblock = DataBlock(
        location=loc,
        representation_column_name=d["representation_column_name"],
        label_source_column_name=d["label_source_column_name"],
        y_expected_length=d.get("y_expected_length"),
        ingest=ingest_tf,
        representation_transform=rep_tf,
    )

    # selection
    s = raw.get("selection", {}) or {}
    obj = s.get("objective", {}) or {}
    op = ObjectiveParams(**(obj.get("params", {}) or {"setpoint_vector": [0, 0, 0, 1]}))
    selection = SelectionBlock(
        strategy=s.get("strategy", "top_n"),
        top_k_default=s.get("top_k_default", 12),
        tie_handling=s.get("tie_handling", "competition_rank"),
        objective=Objective(name=obj.get("name", "logic_plus_effect_v1"), params=op),
    )

    # training
    tr = raw.get("training", {}) or {}
    m = tr.get("model", {}) or {}
    model = ModelBlock(
        name=m.get("name", "random_forest"),
        params=RFParams(**(m.get("params", {}) or {})),
        target_scaler=TargetScalerCfg(**(m.get("target_scaler", {}) or {})),
    )
    training = TrainingBlock(model=model, policy=tr.get("policy", {}) or {})

    scoring = ScoringBlock(**(raw.get("scoring", {}) or {}))
    safety = SafetyBlock(**(raw.get("safety", {}) or {}))
    metadata = MetadataBlock(**(raw.get("metadata", {}) or {}))

    return RootConfig(
        campaign=campaign,
        data=dblock,
        selection=selection,
        training=training,
        scoring=scoring,
        safety=safety,
        metadata=metadata,
    )
