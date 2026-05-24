"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/workspaces/contracts.py

Workspace configuration contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class WorkspaceMeta(BaseModel):
    id: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        ident = str(value or "").strip()
        if not _ID_RE.fullmatch(ident):
            raise ValueError(f"workspace id must be a compact identifier; got {value!r}")
        return ident


class WorkspaceRun(BaseModel):
    id: str
    protocol: str | None = None
    job: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        ident = str(value or "").strip()
        if not _ID_RE.fullmatch(ident):
            raise ValueError(f"run id must be a compact identifier; got {value!r}")
        return ident

    @model_validator(mode="after")
    def _protocol_or_job(self) -> "WorkspaceRun":
        if not self.protocol and not self.job:
            raise ValueError(f"run {self.id!r} must declare either protocol or job")
        if self.protocol and self.job:
            raise ValueError(f"run {self.id!r} must declare protocol or job, not both")
        return self


class WorkspaceConfig(BaseModel):
    workspace: WorkspaceMeta
    runs: list[WorkspaceRun] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_runs(self) -> "WorkspaceConfig":
        seen: set[str] = set()
        duplicates: list[str] = []
        for run in self.runs:
            if run.id in seen:
                duplicates.append(run.id)
            seen.add(run.id)
        if duplicates:
            raise ValueError(f"duplicate run id(s): {sorted(set(duplicates))}")
        return self
