"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/plugin_schemas.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type

import numpy as np
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
# Built-in schemas
# ---------------------------


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
    uncertainty_method: Optional[Literal["delta", "analytical"]] = None
    scaling: _SFXIScaling = Field(default_factory=_SFXIScaling)

    @field_validator("setpoint_vector")
    @classmethod
    def _len4(cls, v):
        if len(v) != 4:
            raise ValueError("objective.params.setpoint_vector must have length 4 (order: 00,10,01,11)")
        if any((x < 0.0 or x > 1.0) for x in v):
            raise ValueError("objective.params.setpoint_vector entries must be in [0, 1]")
        return v


@register_param_schema("objective", "scalar_identity_v1")
class _ScalarIdentityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_param_schema("selection", "top_n")
class _TopNParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int
    score_ref: str
    tie_handling: Literal["competition_rank", "dense_rank", "ordinal"]
    objective_mode: Literal["maximize", "minimize"]
    exclude_already_labeled: bool = True

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
