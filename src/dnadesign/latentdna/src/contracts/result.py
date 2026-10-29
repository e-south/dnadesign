"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/result.py

Command result contracts for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    schema_version: Literal["latentdna.command_result.v1"] = "latentdna.command_result.v1"
    command: str
    workspace_id: str
    status: Literal["ok", "attention", "missing", "error"]
    run_id: str | None = None
    dry_run: bool = False
    artifact_kind: str | None = None
    artifact_id: str | None = None
    outputs: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    input_digests: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    freshness_known: bool | None = None
