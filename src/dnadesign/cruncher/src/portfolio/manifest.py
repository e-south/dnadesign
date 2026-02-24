"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/portfolio/manifest.py

Manifest and status models for Portfolio aggregation runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

PortfolioStatusLabel = Literal["running", "completed", "failed"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortfolioSourceRun(StrictBaseModel):
    source_id: str
    source_label: str
    workspace_name: str
    workspace_path: str
    run_dir: str
    run_name: str
    source_top_k: int
    selected_elites: int


class PortfolioPreparedSource(StrictBaseModel):
    source_id: str
    runbook_path: str
    step_ids: list[str] = Field(default_factory=list)


class PortfolioManifestV1(StrictBaseModel):
    schema_version: int = 1
    portfolio_name: str
    portfolio_id: str
    spec_path: str
    spec_sha256: str
    created_at: str
    execution_mode: str = "aggregate_only"
    source_runs: list[PortfolioSourceRun] = Field(default_factory=list)
    prepared_sources: list[PortfolioPreparedSource] = Field(default_factory=list)
    table_paths: list[str] = Field(default_factory=list)
    plot_paths: list[str] = Field(default_factory=list)


class PortfolioStatusV1(StrictBaseModel):
    schema_version: int = 1
    portfolio_name: str
    portfolio_id: str
    status: PortfolioStatusLabel = "running"
    n_sources: int = 0
    n_selected_elites: int = 0
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None


def write_portfolio_manifest(path: Path, manifest: PortfolioManifestV1) -> None:
    atomic_write_json(path, manifest.model_dump(mode="json"), indent=2, sort_keys=False, allow_nan=False)


def write_portfolio_status(path: Path, status: PortfolioStatusV1) -> None:
    atomic_write_json(path, status.model_dump(mode="json"), indent=2, sort_keys=False, allow_nan=False)


def load_portfolio_manifest(path: Path) -> PortfolioManifestV1:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio manifest not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Portfolio manifest must be a JSON object: {path}")
    return PortfolioManifestV1.model_validate(payload)


def load_portfolio_status(path: Path) -> PortfolioStatusV1:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio status not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Portfolio status must be a JSON object: {path}")
    return PortfolioStatusV1.model_validate(payload)
