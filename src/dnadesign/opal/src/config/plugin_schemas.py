"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/plugin_schemas.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Registry: category -> name -> Pydantic model class
_SCHEMAS: Dict[str, Dict[str, Type[BaseModel]]] = {
    "transform_x": {},
    "transform_y": {},
    "model": {},
    "objective": {},
    "selection": {},
}


def register_param_schema(category: str, name: str):
    def _wrap(cls: Type[BaseModel]):
        _SCHEMAS.setdefault(category, {})
        if name in _SCHEMAS[category]:
            raise ValueError(f"Duplicate schema for {category}:{name}")
        _SCHEMAS[category][name] = cls
        return cls

    return _wrap


def validate_params(category: str, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate params against a registered schema when available. Unknown plugin names are allowed."""
    model = _SCHEMAS.get(category, {}).get(name)
    if not model:
        return params or {}
    return model.model_validate(params or {}).model_dump()


# ---------------------------
# Built-in schemas (examples)
# ---------------------------


# transform_x: identity (no params)
@register_param_schema("transform_x", "identity")
class _IdentityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


# transform_y: logic5_from_tidy_v1 (FLATTENED params)
class _ComputeRatio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_numerator_channel: str
    input_denominator_channel: str
    division_epsilon: float = 1e-8
    apply_log2_to_ratio: bool = True


class _BuildLogicVec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["log2_minmax_per_design"]
    expected_state_order: List[Literal["00", "10", "01", "11"]]
    minmax_epsilon: float = 1e-6
    equal_states_fallback: Literal[
        "uniform_quarters_and_warn", "uniform_quarters", "error"
    ] = "uniform_quarters_and_warn"


class _AggregateEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_signal_channel: str = "yfp"
    pre_aggregation_transform: Literal["log2", "none"] = "log2"
    on_state_weighting: Literal["setpoint_proportional", "uniform"] = (
        "setpoint_proportional"
    )
    aggregation_kind: Literal["geometric_mean", "arithmetic_mean"] = "geometric_mean"
    output_units: Literal["linear", "log2"] = "linear"


@register_param_schema("transform_y", "logic5_from_tidy_v1")
class _Logic5Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema: Dict[str, str] = Field(default_factory=dict)
    enforce_single_timepoint: bool = True
    replicate_aggregation: Literal["mean", "median"] = "mean"
    replicate_warn_threshold: int = 3

    # flat (no nested pre_processing)
    compute_per_state_ratio: Optional[_ComputeRatio] = None
    build_logic_intensity_vector: Optional[_BuildLogicVec] = None
    aggregate_effect_size_from_yfp: Optional[_AggregateEffect] = None


# ---- model: random_forest (with nested target_scaler) ----
class _TargetScalerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = True
    method: str = "robust_iqr_per_target"
    minimum_labels_required: int = 5
    center_statistic: str = "median"
    scale_statistic: str = "iqr"


@register_param_schema("model", "random_forest")
class _RFParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = 100
    criterion: str = "friedman_mse"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: float | str = 1.0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = True
    random_state: int = 7
    n_jobs: int = -1

    # model-local scaler config
    target_scaler: _TargetScalerParams = Field(default_factory=_TargetScalerParams)


@register_param_schema("objective", "sfxi_v1")
class _SFXIParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    setpoint_vector: List[float]
    logic_exponent_beta: float = 1.0
    intensity_exponent_gamma: float = 1.0
    intensity_log2_offset_delta: float = 0.0
    scaling: Dict[str, Any] = Field(
        default_factory=lambda: {
            "percentile": 95,
            "min_n": 5,
            "fallback_p": 75,
            "eps": 1e-8,
        }
    )

    @field_validator("setpoint_vector")
    @classmethod
    def _len4(cls, v):
        if len(v) != 4:
            raise ValueError(
                "objective.params.setpoint_vector must have length 4 (order: 00,10,01,11)"
            )
        return v


# selection: top_n
@register_param_schema("selection", "top_n")
class _TopNParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k_default: int = 12
    tie_handling: Literal["competition_rank", "dense_rank", "ordinal"] = (
        "competition_rank"
    )
    # NEW: direction + sort template live under selection jurisdiction
    objective: Literal["maximize", "minimize"] = "maximize"
    sort_stability: str = "(-opal__{slug}__r{round}__selection_score__{objective}, id)"

    @field_validator("top_k_default")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("top_k_default must be > 0")
        return v
