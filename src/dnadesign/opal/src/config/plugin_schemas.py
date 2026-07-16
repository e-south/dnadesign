"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/config/plugin_schemas.py

Configuration contracts for plugin schemas OPAL config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Registry: category -> name -> Pydantic model class
_SCHEMAS: Dict[str, Dict[str, Type[BaseModel]]] = {
    "transform_x": {},
    "transform_y": {},
    "model": {},
    "objective": {},
    "selection": {},
    "candidate_eligibility": {},
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
    if not isinstance(params, dict):
        raise TypeError(f"{category}:{name} params must be a mapping, got {type(params).__name__}.")
    model = _SCHEMAS.get(category, {}).get(name)
    if not model:
        return dict(params)
    return model.model_validate(params).model_dump()


# ---------------------------
# Built-in schemas
# ---------------------------


class _CandidateIdExclusionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    reason: str

    @field_validator("candidate_id", "reason")
    @classmethod
    def _text_non_empty(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("candidate exclusion entry fields must be non-empty")
        return value


@register_param_schema("candidate_eligibility", "candidate_id_exclusion")
class _CandidateIdExclusionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exclusion_set_id: str
    entries: List[_CandidateIdExclusionEntry] = Field(min_length=1)
    min_remaining_candidates: int = Field(ge=1)

    @field_validator("exclusion_set_id")
    @classmethod
    def _set_id_non_empty(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("exclusion_set_id must be non-empty")
        return value

    @model_validator(mode="after")
    def _candidate_ids_unique(self):
        ids = [entry.candidate_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate exclusion entries contain duplicate candidate IDs")
        return self


# transform_x schemas
@register_param_schema("transform_x", "identity")
class _IdentityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


# transform_y schemas
@register_param_schema("transform_y", "sfxi_vec8_from_table_v1")
class _Vec8TableParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_column: Optional[str] = None  # optional source id column name
    sequence_column: str = "sequence"
    logic_columns: List[Literal["v00", "v10", "v01", "v11"]] = Field(
        default_factory=lambda: ["v00", "v10", "v01", "v11"]
    )
    intensity_columns: List[str] = Field(default_factory=lambda: ["y00_star", "y10_star", "y01_star", "y11_star"])
    strict_bounds: bool = True
    clip_bounds_eps: float = 1e-6
    sfxi_log_json: Optional[str] = None
    expected_log2_offset_delta: float = 0.0
    enforce_log2_offset_match: bool = True

    @field_validator("id_column")
    @classmethod
    def _id_col_not_blank(cls, v):
        if v is None:
            return v
        if not str(v).strip():
            raise ValueError("id_column must be a non-empty string when provided.")
        return str(v)

    @field_validator("sfxi_log_json")
    @classmethod
    def _sfxi_log_json_not_blank(cls, v):
        if v is None:
            return v
        if not str(v).strip():
            raise ValueError("sfxi_log_json must be a non-empty string when provided.")
        return str(v)

    @field_validator("expected_log2_offset_delta")
    @classmethod
    def _expected_delta_valid(cls, v):
        if not np.isfinite(v) or float(v) < 0.0:
            raise ValueError("expected_log2_offset_delta must be >= 0.")
        return float(v)

    @field_validator("logic_columns")
    @classmethod
    def _logic_len4(cls, v):
        if len(v) != 4:
            raise ValueError("logic_columns must have length 4 in order [00,10,01,11]")
        return v

    @field_validator("intensity_columns")
    @classmethod
    def _intensity_len4(cls, v):
        if len(v) != 4:
            raise ValueError("intensity_columns must have length 4 in order [00,10,01,11]")
        return v


@register_param_schema("transform_y", "scalar_from_table_v1")
class _ScalarTableParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_column: Optional[str] = None  # must be exactly "id" if provided
    sequence_column: str = "sequence"
    y_column: str = "y"

    @field_validator("id_column")
    @classmethod
    def _id_col_must_be_lit_id(cls, v):
        if v is None:
            return v
        if v != "id":
            raise ValueError("id_column, if set, must be exactly 'id' (no aliases).")
        return v


@register_param_schema("transform_y", "vector_from_table_v1")
class _VectorTableParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_column: Optional[str] = None
    sequence_column: str = "sequence"
    value_columns: List[str]

    @field_validator("id_column")
    @classmethod
    def _id_col_not_blank(cls, v):
        if v is None:
            return v
        vv = str(v).strip()
        if not vv:
            raise ValueError("id_column must be a non-empty string when provided.")
        return vv

    @field_validator("sequence_column")
    @classmethod
    def _sequence_col_not_blank(cls, v):
        vv = str(v).strip()
        if not vv:
            raise ValueError("sequence_column must be a non-empty string.")
        return vv

    @field_validator("value_columns")
    @classmethod
    def _value_columns_valid(cls, v):
        if not v:
            raise ValueError("value_columns must contain at least one target column.")
        cols = [str(item).strip() for item in v]
        if any(not item for item in cols):
            raise ValueError("value_columns entries must be non-empty strings.")
        if len(set(cols)) != len(cols):
            raise ValueError("value_columns must not contain duplicates.")
        return cols


@register_param_schema("model", "random_forest")
class _RFParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = 100
    criterion: str = "friedman_mse"
    emit_feature_importance: bool = False
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


class _KernelBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lower: float = 1e-5
    upper: float = 1e5

    @field_validator("upper")
    @classmethod
    def _bounds_valid(cls, v, info):
        lower = float(info.data.get("lower", 1e-5))
        upper = float(v)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
            raise ValueError("kernel bounds must satisfy 0 < lower < upper.")
        return upper


class _GaussianProcessKernelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["rbf", "matern", "rational_quadratic", "dot_product"] = "rbf"
    length_scale: float = 1.0
    length_scale_bounds: _KernelBounds = Field(default_factory=_KernelBounds)
    nu: float = 1.5
    alpha: float = 1.0
    alpha_bounds: _KernelBounds = Field(default_factory=_KernelBounds)
    sigma_0: float = 1.0
    sigma_0_bounds: _KernelBounds = Field(default_factory=_KernelBounds)
    with_white_noise: bool = False
    noise_level: float = 1.0
    noise_level_bounds: _KernelBounds = Field(default_factory=_KernelBounds)

    @field_validator("length_scale", "nu", "alpha", "sigma_0", "noise_level")
    @classmethod
    def _positive_finite(cls, v: float) -> float:
        vv = float(v)
        if not np.isfinite(vv) or vv <= 0.0:
            raise ValueError("kernel scalar parameters must be positive finite values.")
        return vv


@register_param_schema("model", "gaussian_process")
class _GaussianProcessParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kernel: _GaussianProcessKernelParams | None = None
    alpha: float | List[float] = 1e-10
    normalize_y: bool = False
    random_state: int | None = None
    n_restarts_optimizer: int = 0
    optimizer: str | None = "fmin_l_bfgs_b"
    copy_X_train: bool = True

    @field_validator("alpha")
    @classmethod
    def _alpha_valid(cls, v: float | List[float]) -> float | List[float]:
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("alpha list must be non-empty.")
            vals = [float(x) for x in v]
            if any((not np.isfinite(x) or x <= 0.0) for x in vals):
                raise ValueError("alpha list entries must be positive finite values.")
            return vals
        vv = float(v)
        if not np.isfinite(vv) or vv <= 0.0:
            raise ValueError("alpha must be a positive finite value.")
        return vv

    @field_validator("n_restarts_optimizer")
    @classmethod
    def _restarts_non_negative(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("n_restarts_optimizer must be >= 0.")
        return int(v)

    @field_validator("optimizer")
    @classmethod
    def _optimizer_valid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        vv = str(v).strip()
        if not vv:
            raise ValueError("optimizer must be a non-empty string or null.")
        return vv


class _SFXIScaling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    percentile: int = 95
    min_n: int = 5
    eps: float = 1e-8

    @field_validator("percentile")
    @classmethod
    def _percentile_range(cls, v):
        if v < 1 or v > 100:
            raise ValueError("objective.params.scaling.percentile must be in [1, 100]")
        return v

    @field_validator("min_n")
    @classmethod
    def _min_n_range(cls, v):
        if v <= 0:
            raise ValueError("objective.params.scaling.min_n must be >= 1")
        return v

    @field_validator("eps")
    @classmethod
    def _eps_range(cls, v):
        if v <= 0:
            raise ValueError("objective.params.scaling.eps must be > 0")
        return v


@register_param_schema("objective", "sfxi_v1")
class _SFXIParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    setpoint_vector: List[float]
    logic_exponent_beta: float = 1.0
    intensity_exponent_gamma: float = 1.0
    intensity_log2_offset_delta: float = 0.0
    uncertainty_method: Optional[Literal["delta"]] = None
    scaling: _SFXIScaling = Field(default_factory=_SFXIScaling)

    @field_validator("setpoint_vector")
    @classmethod
    def _len4(cls, v):
        if len(v) != 4:
            raise ValueError("objective.params.setpoint_vector must have length 4 (order: 00,10,01,11)")
        if any((x < 0.0 or x > 1.0) for x in v):
            raise ValueError("objective.params.setpoint_vector entries must be in [0, 1]")
        return v


class _ResponseMagnitudeFeasibilityCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_separation_min: float
    on_magnitude_min: float
    off_magnitude_max: float
    response_separation_scale: float
    on_magnitude_scale: float
    off_magnitude_scale: float

    @field_validator(
        "response_separation_min",
        "on_magnitude_min",
        "off_magnitude_max",
        "response_separation_scale",
        "on_magnitude_scale",
        "off_magnitude_scale",
    )
    @classmethod
    def _finite(cls, v: float) -> float:
        value = float(v)
        if not np.isfinite(value):
            raise ValueError("RMF calibration values must be finite")
        return value

    @field_validator("response_separation_scale", "on_magnitude_scale", "off_magnitude_scale")
    @classmethod
    def _positive_scale(cls, v: float) -> float:
        value = float(v)
        if value <= 0.0:
            raise ValueError("RMF calibration scales must be > 0")
        return value


@register_param_schema("objective", "response_magnitude_feasibility_v1")
class _ResponseMagnitudeFeasibilityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state_ids: List[str]
    target_mask: List[Literal[0, 1]]
    calibration: _ResponseMagnitudeFeasibilityCalibration

    @model_validator(mode="after")
    def _state_contract_is_aligned(self):
        normalized = [str(value).strip() for value in self.state_ids]
        if len(normalized) < 2 or any(not value for value in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("objective.params.state_ids must contain at least two unique, non-empty values")
        if len(self.target_mask) != len(normalized):
            raise ValueError("objective.params.target_mask must align one-to-one with state_ids")
        on_count = int(sum(self.target_mask))
        if on_count <= 0 or on_count >= len(self.target_mask):
            raise ValueError("objective.params.target_mask must contain at least one ON and one OFF state")
        self.state_ids = normalized
        return self


@register_param_schema("objective", "scalar_identity_v1")
class _ScalarIdentityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_param_schema("objective", "spop_v1")
class _SpopParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_param_schema("objective", "vector_channel_v1")
class _VectorChannelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_index: int = 0
    channel_name: str = "channel"
    mode: Literal["maximize", "minimize"] = "maximize"

    @field_validator("channel_index")
    @classmethod
    def _channel_index_valid(cls, v):
        if int(v) < 0:
            raise ValueError("channel_index must be >= 0.")
        return int(v)

    @field_validator("channel_name")
    @classmethod
    def _channel_name_not_blank(cls, v):
        vv = str(v).strip()
        if not vv:
            raise ValueError("channel_name must be a non-empty string.")
        return vv


@register_param_schema("objective", "vector_target_similarity_v1")
class _VectorTargetSimilarityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_vector: List[float]

    @field_validator("target_vector")
    @classmethod
    def _target_vector_valid(cls, v):
        if not v:
            raise ValueError("target_vector must contain at least one numeric channel.")
        vals = [float(item) for item in v]
        if any(not np.isfinite(item) for item in vals):
            raise ValueError("target_vector entries must be finite.")
        return vals


@register_param_schema("selection", "top_n")
class _TopNParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int
    score_ref: str
    tie_handling: Literal["competition_rank", "dense_rank", "ordinal"]
    objective_mode: Literal["maximize", "minimize"]
    exclude_already_labeled: bool = True
    require_exact_top_k: bool = False

    @field_validator("top_k")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("top_k must be > 0")
        return v

    @field_validator("score_ref")
    @classmethod
    def _score_ref_non_empty(cls, v: str) -> str:
        vv = str(v).strip()
        if not vv:
            raise ValueError("score_ref must be a non-empty channel reference")
        return vv


@register_param_schema("selection", "expected_improvement")
class _ExpectedImprovementParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int
    score_ref: str
    uncertainty_ref: str
    tie_handling: Literal["competition_rank", "dense_rank", "ordinal"]
    objective_mode: Literal["maximize", "minimize"]
    exclude_already_labeled: bool = True
    require_exact_top_k: bool = False
    alpha: float = 1.0
    beta: float = 1.0

    @field_validator("top_k")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("top_k must be > 0")
        return v

    @field_validator("score_ref", "uncertainty_ref")
    @classmethod
    def _channel_ref_non_empty(cls, v: str) -> str:
        vv = str(v).strip()
        if not vv:
            raise ValueError("channel references must be non-empty strings")
        return vv

    @field_validator("alpha", "beta")
    @classmethod
    def _non_negative_finite_weight(cls, v: float) -> float:
        x = float(v)
        if not np.isfinite(x):
            raise ValueError("weights must be finite")
        if x < 0.0:
            raise ValueError("weights must be >= 0")
        return x


class _RestrictionSiteParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enzyme: str
    motif: str
    allowed_regions: List[Literal["left_flank", "core", "right_flank"]]

    @field_validator("enzyme")
    @classmethod
    def _enzyme_non_empty(cls, v: str) -> str:
        vv = str(v).strip()
        if not vv:
            raise ValueError("enzyme must be non-empty")
        return vv

    @field_validator("motif")
    @classmethod
    def _motif_upper_dna(cls, v: str) -> str:
        vv = str(v).strip().upper()
        if not vv or any(base not in {"A", "C", "G", "T"} for base in vv):
            raise ValueError("motif must be uppercase ACGT")
        return vv

    @field_validator("allowed_regions")
    @classmethod
    def _allowed_regions_non_empty(cls, v):
        if not v:
            raise ValueError("allowed_regions must contain at least one region")
        return v


class _RestrictionSiteExclusionRowFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    equals: str

    @field_validator("column", "equals")
    @classmethod
    def _filter_text_non_empty(cls, v: str) -> str:
        vv = str(v).strip()
        if not vv:
            raise ValueError("filter text fields must be non-empty")
        return vv


@register_param_schema("candidate_eligibility", "restriction_site_exclusion")
class _RestrictionSiteExclusionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_column: str = "sequence"
    scan_space: Literal["final_assembled_insert"] = "final_assembled_insert"
    assembly_strategy_ref: str
    left_flank: str
    right_flank: str
    expected_core_length: int
    min_remaining_candidates: Optional[int] = None
    on_violation: Literal["exclude"] = "exclude"
    forbidden_sites: List[_RestrictionSiteParams]
    exclude_rows_where: List[_RestrictionSiteExclusionRowFilter] = []

    @field_validator("sequence_column", "assembly_strategy_ref")
    @classmethod
    def _text_non_empty(cls, v: str) -> str:
        vv = str(v).strip()
        if not vv:
            raise ValueError("text fields must be non-empty")
        return vv

    @field_validator("left_flank", "right_flank")
    @classmethod
    def _flank_lower_dna(cls, v: str) -> str:
        vv = str(v).strip()
        if vv != vv.lower() or any(base not in {"a", "c", "g", "t"} for base in vv):
            raise ValueError("flanks must be lowercase acgt")
        return vv

    @field_validator("expected_core_length")
    @classmethod
    def _expected_core_length_positive(cls, v: int) -> int:
        out = int(v)
        if out <= 0:
            raise ValueError("expected_core_length must be positive")
        return out

    @field_validator("min_remaining_candidates")
    @classmethod
    def _min_remaining_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        out = int(v)
        if out <= 0:
            raise ValueError("min_remaining_candidates must be positive when provided")
        return out

    @field_validator("forbidden_sites")
    @classmethod
    def _forbidden_sites_non_empty(cls, v):
        if not v:
            raise ValueError("forbidden_sites must contain at least one site")
        return v
