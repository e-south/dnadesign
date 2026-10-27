"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/logging.py

DenseGen logging configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Literal


class LoggingVisualsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tf_colors: dict[str, str]

    @field_validator("tf_colors")
    @classmethod
    def _tf_colors_nonempty(cls, v: dict[str, str]):
        if not v:
            raise ValueError("logging.visuals.tf_colors must be a non-empty mapping")
        cleaned: dict[str, str] = {}
        for key, value in v.items():
            tf = str(key).strip()
            color = str(value).strip()
            if not tf:
                raise ValueError("logging.visuals.tf_colors keys must be non-empty strings")
            if not color:
                raise ValueError("logging.visuals.tf_colors values must be non-empty strings")
            cleaned[tf] = color
        return cleaned


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_dir: str
    level: str = "INFO"
    suppress_solver_stderr: bool = True
    print_visual: bool = False
    progress_style: Literal["auto", "stream", "summary", "screen"] = "summary"
    progress_every: int = 1
    progress_refresh_seconds: float = 1.0
    show_tfbs: bool = False
    show_solutions: bool = False
    visuals: LoggingVisualsConfig | None = None

    @field_validator("log_dir")
    @classmethod
    def _log_dir_nonempty(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("logging.log_dir must be a non-empty string")
        return v

    @field_validator("level")
    @classmethod
    def _level_ok(cls, v: str):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        lv = (v or "").upper()
        if lv not in allowed:
            raise ValueError(f"logging.level must be one of {sorted(allowed)}")
        return lv

    @field_validator("progress_every")
    @classmethod
    def _progress_every_ok(cls, v: int):
        if v < 0:
            raise ValueError("logging.progress_every must be >= 0")
        return int(v)

    @field_validator("progress_refresh_seconds")
    @classmethod
    def _progress_refresh_ok(cls, v: float):
        if not isinstance(v, (int, float)) or float(v) <= 0:
            raise ValueError("logging.progress_refresh_seconds must be > 0")
        return float(v)

    @model_validator(mode="after")
    def _visuals_required_for_print_visual(self):
        if not self.print_visual:
            return self
        if self.visuals is None or not self.visuals.tf_colors:
            raise ValueError("logging.visuals.tf_colors must be set when logging.print_visual is true")
        return self
