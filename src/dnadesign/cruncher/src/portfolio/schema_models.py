"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/portfolio/schema_models.py

Defines strict schemas for Portfolio aggregation specifications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PortfolioArtifacts(StrictBaseModel):
    table_format: Literal["parquet", "csv"] = "parquet"
    write_csv: bool = False


class PortfolioEliteShowcaseSelector(StrictBaseModel):
    elite_ids: list[str] | None = None
    elite_ranks: list[int] | None = None

    @field_validator("elite_ids")
    @classmethod
    def _check_elite_ids(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("portfolio.plots.elite_showcase.source_selectors.*.elite_ids must be non-empty")
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            token = str(value).strip()
            if not token:
                raise ValueError(
                    "portfolio.plots.elite_showcase.source_selectors.*.elite_ids entries must be non-empty"
                )
            if token in seen:
                raise ValueError("portfolio.plots.elite_showcase.source_selectors.*.elite_ids entries must be unique")
            cleaned.append(token)
            seen.add(token)
        return cleaned

    @field_validator("elite_ranks")
    @classmethod
    def _check_elite_ranks(cls, values: list[int] | None) -> list[int] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("portfolio.plots.elite_showcase.source_selectors.*.elite_ranks must be non-empty")
        cleaned: list[int] = []
        seen: set[int] = set()
        for value in values:
            rank = int(value)
            if rank < 1:
                raise ValueError("portfolio.plots.elite_showcase.source_selectors.*.elite_ranks entries must be >= 1")
            if rank in seen:
                raise ValueError("portfolio.plots.elite_showcase.source_selectors.*.elite_ranks entries must be unique")
            cleaned.append(rank)
            seen.add(rank)
        return cleaned

    @model_validator(mode="after")
    def _check_selector_mode(self) -> "PortfolioEliteShowcaseSelector":
        has_ids = bool(self.elite_ids)
        has_ranks = bool(self.elite_ranks)
        if has_ids == has_ranks:
            raise ValueError(
                "portfolio.plots.elite_showcase.source_selectors.* must provide exactly one of elite_ids or elite_ranks"
            )
        return self


class PortfolioEliteShowcasePlot(StrictBaseModel):
    enabled: bool = True
    top_n_per_source: int | None = None
    ncols: int = 5
    plot_format: Literal["pdf", "png"] = "pdf"
    dpi: int = 250
    source_selectors: dict[str, PortfolioEliteShowcaseSelector] = Field(default_factory=dict)

    @field_validator("top_n_per_source")
    @classmethod
    def _check_top_n_per_source(cls, value: int | None) -> int | None:
        if value is None:
            return None
        top_n = int(value)
        if top_n < 1:
            raise ValueError("portfolio.plots.elite_showcase.top_n_per_source must be >= 1")
        return top_n

    @field_validator("ncols")
    @classmethod
    def _check_ncols(cls, value: int) -> int:
        ncols = int(value)
        if ncols < 1:
            raise ValueError("portfolio.plots.elite_showcase.ncols must be >= 1")
        return ncols

    @field_validator("dpi")
    @classmethod
    def _check_dpi(cls, value: int) -> int:
        dpi = int(value)
        if dpi < 72:
            raise ValueError("portfolio.plots.elite_showcase.dpi must be >= 72")
        return dpi

    @field_validator("source_selectors")
    @classmethod
    def _check_source_selectors(
        cls, values: dict[str, PortfolioEliteShowcaseSelector]
    ) -> dict[str, PortfolioEliteShowcaseSelector]:
        cleaned: dict[str, PortfolioEliteShowcaseSelector] = {}
        for source_id, selector in values.items():
            token = str(source_id).strip()
            if not token:
                raise ValueError("portfolio.plots.elite_showcase.source_selectors keys must be non-empty")
            if not _SOURCE_ID_RE.match(token):
                raise ValueError(
                    "portfolio.plots.elite_showcase.source_selectors keys must be slug-safe "
                    "([A-Za-z0-9][A-Za-z0-9._-]*)"
                )
            cleaned[token] = selector
        return cleaned


class PortfolioPlots(StrictBaseModel):
    elite_showcase: PortfolioEliteShowcasePlot = PortfolioEliteShowcasePlot()


class PortfolioSourceBase(StrictBaseModel):
    id: str
    workspace: Path
    run_dir: Path
    label: str | None = None

    @field_validator("id")
    @classmethod
    def _check_source_id(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("portfolio.sources[].id must be non-empty")
        if not _SOURCE_ID_RE.match(text):
            raise ValueError("portfolio.sources[].id must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @field_validator("workspace", "run_dir")
    @classmethod
    def _check_non_empty_path(cls, value: Path, info) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError(f"portfolio.sources[].{info.field_name} must be non-empty")
        return Path(text)

    @field_validator("label")
    @classmethod
    def _check_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("portfolio.sources[].label must be non-empty when provided")
        return text


class PortfolioSourcePrepare(StrictBaseModel):
    runbook: Path
    step_ids: list[str]

    @field_validator("runbook")
    @classmethod
    def _check_runbook(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("portfolio.sources[].prepare.runbook must be non-empty")
        return Path(text)

    @field_validator("step_ids")
    @classmethod
    def _check_step_ids(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("portfolio.sources[].prepare.step_ids must be non-empty")
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            token = str(value).strip()
            if not token:
                raise ValueError("portfolio.sources[].prepare.step_ids entries must be non-empty")
            if not _SOURCE_ID_RE.match(token):
                raise ValueError(
                    "portfolio.sources[].prepare.step_ids entries must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)"
                )
            if token in seen:
                raise ValueError("portfolio.sources[].prepare.step_ids entries must be unique")
            cleaned.append(token)
            seen.add(token)
        return cleaned


class PortfolioSource(PortfolioSourceBase):
    prepare: PortfolioSourcePrepare | None = None
    study_spec: Path | None = None

    @field_validator("study_spec")
    @classmethod
    def _check_study_spec(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("portfolio.sources[].study_spec must be non-empty when provided")
        return Path(text)


class PortfolioSequenceLengthTable(StrictBaseModel):
    enabled: bool = False
    study_spec: Path = Path("configs/studies/length_vs_score.study.yaml")
    top_n_lengths: int = 6

    @field_validator("study_spec")
    @classmethod
    def _check_study_spec(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("portfolio.studies.sequence_length_table.study_spec must be non-empty")
        return Path(text)

    @field_validator("top_n_lengths")
    @classmethod
    def _check_top_n_lengths(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("portfolio.studies.sequence_length_table.top_n_lengths must be >= 1")
        return int(value)


class PortfolioStudies(StrictBaseModel):
    ensure_specs: list[Path] = Field(default_factory=list)
    sequence_length_table: PortfolioSequenceLengthTable = PortfolioSequenceLengthTable()

    @field_validator("ensure_specs")
    @classmethod
    def _check_ensure_specs(cls, values: list[Path]) -> list[Path]:
        cleaned: list[Path] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                raise ValueError("portfolio.studies.ensure_specs entries must be non-empty")
            token = str(Path(text))
            if token in seen:
                raise ValueError("portfolio.studies.ensure_specs entries must be unique")
            seen.add(token)
            cleaned.append(Path(text))
        return cleaned

    @model_validator(mode="after")
    def _check_sequence_length_table_contract(self) -> "PortfolioStudies":
        if not self.sequence_length_table.enabled:
            return self
        table_spec = str(self.sequence_length_table.study_spec)
        declared = {str(item) for item in self.ensure_specs}
        if table_spec not in declared:
            raise ValueError(
                "portfolio.studies.sequence_length_table.study_spec must be listed in portfolio.studies.ensure_specs"
            )
        return self


class PortfolioExecution(StrictBaseModel):
    mode: Literal["aggregate_only", "prepare_then_aggregate"] = "aggregate_only"
    max_parallel_sources: int = 4

    @field_validator("max_parallel_sources")
    @classmethod
    def _check_max_parallel_sources(cls, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("portfolio.execution.max_parallel_sources must be >= 1")
        return int(value)


class PortfolioSpec(StrictBaseModel):
    schema_version: int = 3
    name: str
    execution: PortfolioExecution = PortfolioExecution()
    studies: PortfolioStudies = PortfolioStudies()
    artifacts: PortfolioArtifacts = PortfolioArtifacts()
    plots: PortfolioPlots = PortfolioPlots()
    sources: list[PortfolioSource]

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        if int(value) != 3:
            raise ValueError("Portfolio schema v3 required (portfolio.schema_version: 3)")
        return int(value)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("portfolio.name must be non-empty")
        if not _SOURCE_ID_RE.match(text):
            raise ValueError("portfolio.name must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, values: list[PortfolioSource]) -> list[PortfolioSource]:
        if not values:
            raise ValueError("portfolio.sources must be non-empty")
        seen: set[str] = set()
        for source in values:
            source_id = str(source.id)
            if source_id in seen:
                raise ValueError("portfolio.sources source ids must be unique")
            seen.add(source_id)
        return values

    @model_validator(mode="after")
    def _check_prepare_contracts(self) -> "PortfolioSpec":
        if self.execution.mode == "prepare_then_aggregate":
            missing = [source.id for source in self.sources if source.prepare is None]
            if missing:
                raise ValueError(
                    f"portfolio.execution.mode=prepare_then_aggregate requires prepare for every source: {missing}"
                )
        known_ids = {str(source.id) for source in self.sources}
        selector_ids = set(self.plots.elite_showcase.source_selectors.keys())
        unknown_ids = sorted(selector_ids - known_ids)
        if unknown_ids:
            raise ValueError(
                f"portfolio.plots.elite_showcase.source_selectors contains unknown source ids: {unknown_ids}"
            )
        return self


class PortfolioRoot(StrictBaseModel):
    portfolio: PortfolioSpec
