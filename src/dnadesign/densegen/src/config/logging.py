"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/logging.py

DenseGen logging configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator
from typing_extensions import Literal


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_dir: str
    level: str = "INFO"
    suppress_solver_stderr: bool = True
    print_visual: bool = False
    progress_style: Literal["stream", "summary", "screen"] = "screen"
    progress_every: int = 1
    progress_refresh_seconds: float = 1.0
    show_tfbs: bool = False
    show_solutions: bool = False

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
