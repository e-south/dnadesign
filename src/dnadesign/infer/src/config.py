"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .bootstrap import initialize_registry
from .errors import ConfigError
from .registry import resolve_fn

Precision = Literal["fp32", "fp16", "bf16"]
Alphabet = Literal["dna", "protein"]
Format = Literal["float", "list", "numpy", "tensor"]
Operation = Literal["extract", "generate"]
ParallelismStrategy = Literal["single_device", "multi_gpu_vortex"]


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelParallelismConfig(StrictConfigModel):
    strategy: ParallelismStrategy = "single_device"
    min_gpus: int = Field(default=1, ge=1)
    gpu_ids: Optional[List[int]] = None

    @field_validator("gpu_ids")
    @classmethod
    def _validate_gpu_ids(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if len(value) == 0:
            raise ConfigError("model.parallelism.gpu_ids must be non-empty when provided")
        if any(idx < 0 for idx in value):
            raise ConfigError("model.parallelism.gpu_ids must contain non-negative integers")
        if len(set(value)) != len(value):
            raise ConfigError("model.parallelism.gpu_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_parallelism_contract(self) -> "ModelParallelismConfig":
        if self.strategy == "single_device":
            if self.min_gpus != 1:
                raise ConfigError("model.parallelism.min_gpus must be 1 when strategy='single_device'")
            if self.gpu_ids is not None and len(self.gpu_ids) != 1:
                raise ConfigError("model.parallelism.gpu_ids must have exactly one id for strategy='single_device'")
            return self

        if self.min_gpus < 2:
            raise ConfigError("model.parallelism.min_gpus must be >= 2 when strategy='multi_gpu_vortex'")
        if self.gpu_ids is not None:
            if len(self.gpu_ids) < 2:
                raise ConfigError("model.parallelism.gpu_ids must include at least two ids for multi_gpu_vortex")
            if len(self.gpu_ids) < self.min_gpus:
                raise ConfigError("model.parallelism.gpu_ids must include at least min_gpus ids")
        return self


class ModelConfig(StrictConfigModel):
    id: str
    device: str = Field(..., description="e.g., cuda:0 or cpu")
    precision: Precision = Field(...)
    alphabet: Alphabet
    batch_size: Optional[int] = None
    parallelism: ModelParallelismConfig = Field(default_factory=ModelParallelismConfig)


class IngestConfig(StrictConfigModel):
    source: Literal["sequences", "records", "pt_file", "usr"]
    path: Optional[str] = None
    field: Optional[str] = "sequence"
    dataset: Optional[str] = None
    root: Optional[str] = None
    ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def _validate_by_source(self) -> "IngestConfig":
        if self.source in {"records", "pt_file"} and not self.field:
            raise ConfigError("ingest.field is required for records/pt_file sources")
        if self.source == "usr":
            if not self.dataset:
                raise ConfigError("ingest.dataset is required for source='usr'")
            if self.path:
                raise ConfigError("ingest.path is not allowed for source='usr'")
            if not self.field:
                self.field = "sequence"
        return self


class OutputSpec(StrictConfigModel):
    id: str
    fn: str  # namespaced e.g. evo2.logits
    params: Dict[str, Any] = Field(default_factory=dict)
    format: Format

    @field_validator("fn")
    @classmethod
    def _known_fn(cls, v: str) -> str:
        initialize_registry()
        resolve_fn(v)
        return v


class CheckpointConfig(StrictConfigModel):
    enabled: bool = False
    every_n: int = 100


class IOConfig(StrictConfigModel):
    write_back: bool = False
    overwrite: bool = False
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class JobConfig(StrictConfigModel):
    id: str
    operation: Operation
    ingest: IngestConfig
    # extract
    outputs: Optional[List[OutputSpec]] = None
    # generate
    fn: Optional[str] = None  # NEW: optional namespaced fn for generation
    params: Optional[Dict[str, Any]] = None
    returns: Optional[List[Dict[str, str]]] = None
    # io (write-back only applies to records/pt_file/usr)
    io: IOConfig = Field(default_factory=IOConfig)

    @model_validator(mode="after")
    def _by_kind(self) -> "JobConfig":
        if self.operation == "extract":
            if not self.outputs:
                raise ConfigError("extract job requires 'outputs'")
        elif self.operation == "generate":
            if not self.params:
                raise ConfigError("generate job requires 'params'")
        return self


class RootConfig(StrictConfigModel):
    model: ModelConfig
    jobs: List[JobConfig]
