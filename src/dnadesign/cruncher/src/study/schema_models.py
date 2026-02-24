"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/schema_models.py

Defines the strict schema for Study sweep specifications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

_TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_STUDY_FACTOR_KEYS = {
    "sample.sequence_length",
    "sample.elites.select.diversity",
    "sample.optimizer.chains",
    "sample.optimizer.cooling.stages",
    "sample.moves.overrides.proposal_adapt.freeze_after_beta",
    "sample.moves.overrides.gibbs_inertia.p_stay_end",
    "sample.moves.overrides.move_schedule.end",
}


def _normalize_factor_key(raw_key: object, *, field_label: str) -> str:
    key = str(raw_key).strip()
    if not key:
        raise ValueError(f"{field_label} keys must be non-empty dot paths")
    if key.startswith(".") or key.endswith(".") or ".." in key or " " in key:
        raise ValueError(f"{field_label} key is invalid: {raw_key!r}")
    if key not in _ALLOWED_STUDY_FACTOR_KEYS:
        raise ValueError(f"{field_label} study factor key is not supported: {raw_key!r}")
    return key


class RegulatorSetTarget(StrictBaseModel):
    kind: Literal["regulator_set"] = "regulator_set"
    set_index: int

    @field_validator("set_index")
    @classmethod
    def _check_set_index(cls, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("study.target.set_index must be >= 1")
        return int(value)


StudyTarget = Annotated[
    RegulatorSetTarget,
    Field(discriminator="kind"),
]


class StudyExecution(StrictBaseModel):
    parallelism: int = 6
    on_trial_error: Literal["continue", "abort"] = "continue"
    exit_code_policy: Literal["nonzero_if_any_error", "always_zero"] = "nonzero_if_any_error"
    summarize_after_run: bool = True

    @field_validator("parallelism")
    @classmethod
    def _check_parallelism(cls, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("study.execution.parallelism must be >= 1")
        return int(value)


class StudyArtifacts(StrictBaseModel):
    trial_output_profile: Literal["minimal", "analysis_ready"] = "minimal"


class StudyReplicates(StrictBaseModel):
    seed_path: str = "sample.seed"
    seeds: list[int]

    @field_validator("seed_path")
    @classmethod
    def _check_seed_path(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("study.replicates.seed_path must be a non-empty dot path")
        if "." not in text:
            raise ValueError("study.replicates.seed_path must be a dot path (e.g., sample.seed)")
        return text

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("study.replicates.seeds must be non-empty")
        normalized: list[int] = []
        seen: set[int] = set()
        for value in values:
            if not isinstance(value, int) or value < 0:
                raise ValueError("study.replicates.seeds must contain non-negative integers")
            seed = int(value)
            if seed in seen:
                raise ValueError("study.replicates.seeds must be unique")
            seen.add(seed)
            normalized.append(seed)
        return normalized


class StudyTrial(StrictBaseModel):
    id: str
    factors: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _check_trial_id(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("study.trials[].id must be non-empty")
        if not _TRIAL_ID_RE.match(text):
            raise ValueError("study.trials[].id must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @field_validator("factors")
    @classmethod
    def _check_factor_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _normalize_factor_key(raw_key, field_label="study.trials[].factors")
            normalized[key] = raw_value
        return normalized


class StudyTrialGrid(StrictBaseModel):
    id_prefix: str
    factors: dict[str, list[Any]]

    @field_validator("id_prefix")
    @classmethod
    def _check_id_prefix(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("study.trial_grids[].id_prefix must be non-empty")
        if not _TRIAL_ID_RE.match(text):
            raise ValueError("study.trial_grids[].id_prefix must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @field_validator("factors")
    @classmethod
    def _check_grid_factors(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not value:
            raise ValueError("study.trial_grids[].factors must be non-empty")
        normalized: dict[str, list[Any]] = {}
        for raw_key, raw_values in value.items():
            key = _normalize_factor_key(raw_key, field_label="study.trial_grids[].factors")
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError(f"study.trial_grids[].factors['{key}'] must be a non-empty list")
            normalized[key] = list(raw_values)
        return normalized

    @model_validator(mode="after")
    def _check_grid_size(self) -> "StudyTrialGrid":
        combinations = 1
        for values in self.factors.values():
            combinations *= len(values)
        if combinations > 500:
            raise ValueError("study.trial_grids[] expands to too many combinations (>500)")
        return self


class StudyMmrSweepReplayConfig(StrictBaseModel):
    enabled: bool = False
    pool_size_values: list[Union[Literal["auto", "all"], int]] = Field(default_factory=lambda: ["auto"])
    diversity_values: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0])

    @field_validator("pool_size_values")
    @classmethod
    def _check_pool_sizes(
        cls,
        values: list[Union[Literal["auto", "all"], int]],
    ) -> list[Union[Literal["auto", "all"], int]]:
        if not values:
            raise ValueError("study.replays.mmr_sweep.pool_size_values must contain at least one value")
        normalized: list[Union[Literal["auto", "all"], int]] = []
        for value in values:
            if value in {"auto", "all"}:
                normalized.append(value)
                continue
            if not isinstance(value, int) or value < 1:
                raise ValueError("study.replays.mmr_sweep.pool_size_values entries must be 'auto', 'all', or >= 1")
            normalized.append(int(value))
        return normalized

    @field_validator("diversity_values")
    @classmethod
    def _check_diversities(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("study.replays.mmr_sweep.diversity_values must contain at least one value")
        normalized: list[float] = []
        for value in values:
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError("study.replays.mmr_sweep.diversity_values entries must be in [0, 1]")
            normalized.append(float(value))
        return normalized

    @model_validator(mode="after")
    def _check_grid_size(self) -> "StudyMmrSweepReplayConfig":
        n_items = len(self.pool_size_values) * len(self.diversity_values)
        if n_items > 500:
            raise ValueError("study.replays.mmr_sweep grid is too large (>500 combinations)")
        return self


class StudyReplays(StrictBaseModel):
    mmr_sweep: StudyMmrSweepReplayConfig = StudyMmrSweepReplayConfig()


class StudySpec(StrictBaseModel):
    schema_version: int = 3
    name: str
    base_config: Path
    target: StudyTarget
    execution: StudyExecution = StudyExecution()
    artifacts: StudyArtifacts = StudyArtifacts()
    replicates: StudyReplicates
    trials: list[StudyTrial]
    trial_grids: list[StudyTrialGrid] = Field(default_factory=list)
    replays: StudyReplays = StudyReplays()

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        if value != 3:
            raise ValueError("Study schema v3 required (study.schema_version: 3)")
        return int(value)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("study.name must be non-empty")
        if not _TRIAL_ID_RE.match(text):
            raise ValueError("study.name must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @field_validator("base_config")
    @classmethod
    def _check_base_config(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("study.base_config must be non-empty")
        return Path(text)

    @field_validator("trials")
    @classmethod
    def _check_trials(cls, values: list[StudyTrial]) -> list[StudyTrial]:
        ids = [trial.id for trial in values]
        if len(ids) != len(set(ids)):
            raise ValueError("study.trials trial ids must be unique")
        return values

    @field_validator("trial_grids")
    @classmethod
    def _check_trial_grids(cls, values: list[StudyTrialGrid]) -> list[StudyTrialGrid]:
        prefixes = [grid.id_prefix for grid in values]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("study.trial_grids id_prefix values must be unique")
        return values

    @model_validator(mode="after")
    def _check_trial_sources(self) -> "StudySpec":
        if not self.trials and not self.trial_grids:
            raise ValueError("study must define at least one trial via trials or trial_grids")
        return self


class StudyRoot(StrictBaseModel):
    study: StudySpec
