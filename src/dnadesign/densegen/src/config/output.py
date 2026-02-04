"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/output.py

DenseGen output schemas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal


class OutputSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bio_type: str = "dna"
    alphabet: str = "dna_4"

    @field_validator("bio_type", "alphabet")
    @classmethod
    def _nonempty(cls, v: str, info):
        if not v or not str(v).strip():
            raise ValueError(f"output.schema.{info.field_name} must be a non-empty string")
        return v


class OutputUSRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    root: str
    chunk_size: int = 128
    allow_overwrite: bool = False


class OutputParquetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    deduplicate: bool = True
    chunk_size: int = 2048

    @field_validator("path")
    @classmethod
    def _path_is_file(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("output.parquet.path must be a non-empty string")
        if not str(v).endswith(".parquet"):
            raise ValueError("output.parquet.path must point to a .parquet file")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_ok(cls, v: int):
        if v <= 0:
            raise ValueError("chunk_size must be > 0")
        return v


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    targets: List[Literal["usr", "parquet"]]
    output_schema: OutputSchemaConfig = Field(alias="schema")
    usr: Optional[OutputUSRConfig] = None
    parquet: Optional[OutputParquetConfig] = None

    @model_validator(mode="after")
    def _require_targets(self):
        if not self.targets:
            raise ValueError("output.targets must contain at least one sink.")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("output.targets must not contain duplicates.")
        if "usr" in self.targets and self.usr is None:
            raise ValueError("output.usr is required when output.targets includes 'usr'")
        if "parquet" in self.targets and self.parquet is None:
            raise ValueError("output.parquet is required when output.targets includes 'parquet'")
        return self
