"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import ConfigError
from .registry import resolve_fn

Precision = Literal["fp32", "fp16", "bf16"]
Alphabet = Literal["dna", "protein"]
Format = Literal["float", "list", "numpy", "tensor"]
Operation = Literal["extract", "generate"]


class ModelConfig(BaseModel):
    id: str
    device: str = Field(..., description="e.g., cuda:0 or cpu")
    precision: Precision = Field(...)
    alphabet: Alphabet
    batch_size: Optional[int] = None


class IngestConfig(BaseModel):
    source: Literal["sequences", "records", "pt_file", "usr"]
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
            if not self.field:
                self.field = "sequence"
        return self


class OutputSpec(BaseModel):
    id: str
    fn: str  # namespaced e.g. evo2.logits
    params: Dict[str, Any] = Field(default_factory=dict)
    format: Format

    @field_validator("fn")
    @classmethod
    def _known_fn(cls, v: str) -> str:
        resolve_fn(v)  # ensures it was registered by adapters/__init__.py
        return v


class CheckpointConfig(BaseModel):
    enabled: bool = False
    every_n: int = 100


class IOConfig(BaseModel):
    write_back: bool = False
    overwrite: bool = False
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class JobConfig(BaseModel):
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


class RootConfig(BaseModel):
    model: ModelConfig
    jobs: List[JobConfig]
