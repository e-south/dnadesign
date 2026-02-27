"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/root.py

DenseGen root configuration schema and loader.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .base import (
    SUPPORTED_SCHEMA_VERSIONS,
    ConfigError,
    LoadedConfig,
    RunConfig,
    _is_relative_to,
    _StrictLoader,
    resolve_outputs_scoped_path,
    resolve_run_root,
)
from .generation import GenerationConfig, expand_generation_plans, normalize_motif_sets
from .inputs import InputConfig
from .logging import LoggingConfig
from .output import OutputConfig
from .plots import PlotConfig
from .postprocess import PostprocessConfig
from .runtime import RuntimeConfig
from .solver import SolverConfig


class DenseGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    run: RunConfig
    inputs: List[InputConfig]
    motif_sets: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    output: OutputConfig
    generation: GenerationConfig
    solver: SolverConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    logging: LoggingConfig

    @field_validator("schema_version")
    @classmethod
    def _schema_version_supported(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("densegen.schema_version must be a non-empty string")
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported densegen.schema_version: {v!r}. Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return v

    @model_validator(mode="after")
    def _inputs_nonempty(self):
        if not self.inputs:
            raise ValueError("At least one input is required")
        names = [i.name for i in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("Input names must be unique")
        return self

    @model_validator(mode="after")
    def _normalize_motif_sets(self):
        self.motif_sets = normalize_motif_sets(self.motif_sets)
        return self

    @model_validator(mode="after")
    def _expand_plan(self):
        self.generation.plan = expand_generation_plans(
            plan=list(self.generation.plan),
            motif_sets=dict(self.motif_sets or {}),
            sequence_length=int(self.generation.sequence_length),
            max_plans=int(self.generation.expansion.max_plans),
        )
        return self

    @model_validator(mode="after")
    def _sequence_constraints_motif_sets(self):
        constraints = self.generation.sequence_constraints
        if constraints is None:
            return self
        known_sets = set(self.motif_sets)
        for rule in list(constraints.forbid_kmers or []):
            missing = [name for name in list(rule.patterns_from_motif_sets or []) if name not in known_sets]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(
                    f"generation.sequence_constraints.forbid_kmers references unknown motif_sets: {preview}."
                )
        return self

    @model_validator(mode="after")
    def _background_forbid_kmers_motif_sets(self):
        known_sets = set(self.motif_sets)
        for inp in self.inputs:
            if getattr(inp, "type", None) != "background_pool":
                continue
            filters = getattr(inp.sampling, "filters", None)
            if filters is None:
                continue
            for entry in list(getattr(filters, "forbid_kmers", []) or []):
                if isinstance(entry, str):
                    continue
                patterns = list(getattr(entry, "patterns_from_motif_sets", []) or [])
                missing = [name for name in patterns if name not in known_sets]
                if missing:
                    preview = ", ".join(missing[:10])
                    raise ValueError(
                        f"background_pool.sampling.filters.forbid_kmers references unknown motif_sets: {preview}."
                    )
        return self

    @model_validator(mode="after")
    def _plan_include_inputs(self):
        input_names = {i.name for i in self.inputs}
        for plan in self.generation.plan or []:
            include_inputs = list(plan.sampling.include_inputs or [])
            missing = [name for name in include_inputs if name not in input_names]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(
                    "generation.plan[].sampling.include_inputs includes unknown inputs: "
                    f"{preview}. Known inputs: {sorted(input_names)}"
                )
        return self

    @model_validator(mode="after")
    def _background_pool_inputs(self):
        input_by_name = {i.name: i for i in self.inputs}
        pwm_inputs = {
            "pwm_meme",
            "pwm_meme_set",
            "pwm_jaspar",
            "pwm_matrix_csv",
            "pwm_artifact",
            "pwm_artifact_set",
        }
        for inp in self.inputs:
            if getattr(inp, "type", None) != "background_pool":
                continue
            filters = getattr(inp.sampling, "filters", None)
            fimo_cfg = getattr(filters, "fimo_exclude", None) if filters is not None else None
            if fimo_cfg is None:
                continue
            names = list(getattr(fimo_cfg, "pwms_input", []) or [])
            missing = [name for name in names if name not in input_by_name]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(
                    f"background_pool.sampling.filters.fimo_exclude.pwms_input includes unknown inputs: {preview}."
                )
            non_pwm = [name for name in names if getattr(input_by_name[name], "type", None) not in pwm_inputs]
            if non_pwm:
                preview = ", ".join(non_pwm[:10])
                raise ValueError(
                    f"background_pool.sampling.filters.fimo_exclude.pwms_input must reference PWM inputs: {preview}."
                )
        return self


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    densegen: DenseGenConfig
    plots: Optional[PlotConfig] = None

    @model_validator(mode="after")
    def _plots_source_required(self):
        targets = self.densegen.output.targets
        if len(targets) > 1:
            if self.plots is None or self.plots.source is None:
                raise ValueError("plots.source must be set when output.targets has multiple sinks")
            if self.plots.source not in targets:
                raise ValueError("plots.source must be one of output.targets")
        return self


def _validate_run_scoped_paths(cfg_path: Path, root_cfg: RootConfig) -> None:
    run_cfg = root_cfg.densegen.run
    run_root = resolve_run_root(cfg_path, run_cfg.root)
    if not _is_relative_to(cfg_path, run_root):
        raise ConfigError(f"Config file must live inside densegen.run.root ({run_root}), got: {cfg_path}")

    out_cfg = root_cfg.densegen.output
    if out_cfg.parquet is not None:
        resolve_outputs_scoped_path(
            cfg_path,
            run_root,
            out_cfg.parquet.path,
            label="output.parquet.path",
        )
    if out_cfg.usr is not None:
        resolve_outputs_scoped_path(
            cfg_path,
            run_root,
            out_cfg.usr.root,
            label="output.usr.root",
        )

    log_dir = root_cfg.densegen.logging.log_dir
    resolve_outputs_scoped_path(cfg_path, run_root, log_dir, label="logging.log_dir")

    sampling_cfg = root_cfg.densegen.generation.sampling
    if getattr(sampling_cfg, "library_source", None) == "artifact" and getattr(
        sampling_cfg, "library_artifact_path", None
    ):
        resolve_outputs_scoped_path(
            cfg_path,
            run_root,
            sampling_cfg.library_artifact_path,
            label="sampling.library_artifact_path",
        )

    if root_cfg.plots is not None:
        resolve_outputs_scoped_path(
            cfg_path,
            run_root,
            root_cfg.plots.out_dir,
            label="plots.out_dir",
        )


def _reject_removed_solver_options(raw: object) -> None:
    if not isinstance(raw, dict):
        return
    densegen = raw.get("densegen")
    if not isinstance(densegen, dict):
        return
    solver = densegen.get("solver")
    if not isinstance(solver, dict):
        return
    if "options" in solver:
        raise ConfigError(
            "solver.options has been removed. Use solver.solver_attempt_timeout_seconds or solver.threads instead."
        )
    if "allow_unknown_options" in solver:
        raise ConfigError("solver.allow_unknown_options has been removed.")


def _reject_removed_postprocess_options(raw: object) -> None:
    if not isinstance(raw, dict):
        return
    densegen = raw.get("densegen")
    if not isinstance(densegen, dict):
        return
    postprocess = densegen.get("postprocess")
    if not isinstance(postprocess, dict):
        return
    if "validate_final_sequence" in postprocess:
        raise ConfigError("postprocess.validate_final_sequence has been removed. Use generation.sequence_constraints.")


def load_config(path: Path | str) -> LoadedConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.load(cfg_path.read_text(), Loader=_StrictLoader)
    _reject_removed_solver_options(raw)
    _reject_removed_postprocess_options(raw)
    try:
        root = RootConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid DenseGen config: {e}")
    _validate_run_scoped_paths(cfg_path, root)
    return LoadedConfig(path=cfg_path, root=root)
