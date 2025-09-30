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
    id_column: Optional[str] = None  # must be exactly "id" if provided
    sequence_column: str = "sequence"
    logic_columns: List[Literal["v00", "v10", "v01", "v11"]] = Field(
        default_factory=lambda: ["v00", "v10", "v01", "v11"]
    )
    intensity_columns: List[str] = Field(
        default_factory=lambda: ["y00_star", "y10_star", "y01_star", "y11_star"]
    )
    strict_bounds: bool = True
    clip_bounds_eps: float = 1e-6

    @field_validator("id_column")
    @classmethod
    def _id_col_must_be_lit_id(cls, v):
        if v is None:
            return v
        if v != "id":
            raise ValueError("id_column, if set, must be exactly 'id' (no aliases).")
        return v

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
            raise ValueError(
                "intensity_columns must have length 4 in order [00,10,01,11]"
            )
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


@register_param_schema("selection", "top_n")
class _TopNParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = 12
    tie_handling: Literal["competition_rank", "dense_rank", "ordinal"] = (
        "competition_rank"
    )
    objective_mode: Optional[Literal["maximize", "minimize"]] = None
    exclude_already_labeled: bool = True

    @field_validator("top_k")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("top_k must be > 0")
        return v
