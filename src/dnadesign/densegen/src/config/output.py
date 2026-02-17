"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/output.py

DenseGen output schemas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
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
    health_event_interval_seconds: float = 60.0
    npz_fields: List[str] = Field(default_factory=list)
    npz_root: Optional[str] = None

    @field_validator("dataset")
    @classmethod
    def _dataset_valid(cls, v: str):
        value = str(v).strip().replace("\\", "/")
        if not value:
            raise ValueError("output.usr.dataset must be a non-empty string")
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.usr.dataset must be a relative path")
        if any(part in {".", ".."} for part in path.parts):
            raise ValueError("output.usr.dataset must not contain '.' or '..'")
        return Path(*path.parts).as_posix()

    @field_validator("root")
    @classmethod
    def _root_nonempty(cls, v: str):
        value = str(v).strip()
        if not value:
            raise ValueError("output.usr.root must be a non-empty string")
        return value

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("output.usr.chunk_size must be > 0")
        return int(v)

    @field_validator("npz_fields")
    @classmethod
    def _npz_fields_valid(cls, v: List[str]):
        cleaned = [str(item).strip() for item in v if str(item).strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("output.usr.npz_fields must not contain duplicates")
        return cleaned

    @field_validator("health_event_interval_seconds")
    @classmethod
    def _health_event_interval_positive(cls, v: float):
        value = float(v)
        if value <= 0:
            raise ValueError("output.usr.health_event_interval_seconds must be > 0")
        return value


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
