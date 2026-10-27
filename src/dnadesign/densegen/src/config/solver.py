"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/solver.py

DenseGen solver configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Literal


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Optional[str] = None
    strategy: Literal["iterate", "diverse", "optimal", "approximate"]
    strands: Literal["single", "double"] = "double"
    solver_attempt_timeout_seconds: float | None = None
    threads: int | None = None

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, v: str | None):
        if v is None:
            return v
        if not str(v).strip():
            raise ValueError("solver.backend must be a non-empty string")
        return v

    @field_validator("solver_attempt_timeout_seconds")
    @classmethod
    def _time_limit_ok(cls, v: float | None):
        if v is None:
            return v
        value = float(v)
        if value <= 0:
            raise ValueError("solver.solver_attempt_timeout_seconds must be > 0")
        return value

    @field_validator("threads")
    @classmethod
    def _threads_ok(cls, v: int | None):
        if v is None:
            return v
        value = int(v)
        if value <= 0:
            raise ValueError("solver.threads must be > 0")
        return value

    @model_validator(mode="after")
    def _strategy_backend_consistency(self):
        if self.strategy != "approximate" and not self.backend:
            raise ValueError("solver.backend is required unless strategy=approximate")
        if self.strategy == "approximate" and (
            self.solver_attempt_timeout_seconds is not None or self.threads is not None
        ):
            raise ValueError("solver.solver_attempt_timeout_seconds/threads are invalid when strategy=approximate")
        if self.threads is not None and self.backend:
            backend = str(self.backend).strip().upper()
            if backend == "CBC":
                raise ValueError("solver.threads is not supported for CBC backends.")
        return self
