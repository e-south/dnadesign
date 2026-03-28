"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/catalogs.py

Catalog, output, and ligation support schemas for YIU.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.common import _validate_slug

LigationCompatibilityMode = Literal["exact_complement", "partial_complement", "bulged"]


class YiuEnzymeCatalogEntry(StrictBaseModel):
    id: str
    recognition_sequence: str
    top_cut_offset: int | None = None
    bottom_cut_offset: int | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="catalog_entry.id")

    @field_validator("recognition_sequence")
    @classmethod
    def _validate_recognition_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuEnzymeCatalogSpec(StrictBaseModel):
    entries: list[YiuEnzymeCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuEnzymeCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog enzyme ids must be unique")
        return self


class YiuRestrictionCatalogDocument(StrictBaseModel):
    restriction_enzymes: YiuEnzymeCatalogSpec


class YiuNickaseCatalogDocument(StrictBaseModel):
    nickases: YiuEnzymeCatalogSpec


class YiuAdapterCatalogEntry(StrictBaseModel):
    id: str
    sequence: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="adapter_catalog_entry.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuAdapterCatalogSpec(StrictBaseModel):
    entries: list[YiuAdapterCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuAdapterCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog adapter ids must be unique")
        return self


class YiuAdapterCatalogDocument(StrictBaseModel):
    adapters: YiuAdapterCatalogSpec


class YiuOligoPartCatalogEntry(StrictBaseModel):
    id: str
    part_kind: Literal["primer", "adapter", "backbone", "other"] = "other"
    sequence: str
    phosphorylated_5p: bool = False

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="oligo_part_catalog_entry.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuOligoPartCatalogSpec(StrictBaseModel):
    entries: list[YiuOligoPartCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuOligoPartCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog oligo-part ids must be unique")
        return self


class YiuOligoPartCatalogDocument(StrictBaseModel):
    oligo_parts: YiuOligoPartCatalogSpec


class YiuBackboneCatalogEntry(StrictBaseModel):
    id: str
    sequence: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="backbone_catalog_entry.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)


class YiuBackboneCatalogSpec(StrictBaseModel):
    entries: list[YiuBackboneCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuBackboneCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog backbone ids must be unique")
        return self


class YiuBackboneCatalogDocument(StrictBaseModel):
    backbones: YiuBackboneCatalogSpec


class YiuGenericEnzymeCatalogDocument(StrictBaseModel):
    enzymes: YiuEnzymeCatalogSpec


class OutputSpec(StrictBaseModel):
    run_dir: Path = Path("outputs/yiu/explicit")
    emit_view_contracts: bool = True

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be relative to the workspace root")
        if ".." in path.parts:
            raise ValueError("output.run_dir must stay inside the workspace root")
        return path


class OutputSpecV2(OutputSpec):
    emit_baserender_jobs: bool = False
    publish_contract_version: int = 3

    @field_validator("publish_contract_version")
    @classmethod
    def _validate_publish_contract_version(cls, value: int) -> int:
        if int(value) not in {2, 3}:
            raise ValueError("output.publish_contract_version must be 2 or 3")
        return int(value)

    @model_validator(mode="after")
    def _validate_visual_output_dependencies(self) -> "OutputSpecV2":
        if self.emit_baserender_jobs and not self.emit_view_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_view_contracts=true.")
        return self


class PartialComplementRule(StrictBaseModel):
    min_paired_nt: int = Field(ge=1)
    allow_left_tail: bool = True
    allow_right_tail: bool = True


class BulgedCompatibilityRule(StrictBaseModel):
    min_left_paired_nt: int = Field(default=1, ge=1)
    min_right_paired_nt: int = Field(default=1, ge=1)
    max_bulge_nt: int = Field(default=1, ge=0)
    allow_terminal_tails: bool = True


class LigationRuleSpec(StrictBaseModel):
    mode: LigationCompatibilityMode = "exact_complement"
    min_contiguous_core_bp: int = Field(default=1, ge=1)
    max_left_tail_nt: int = Field(default=0, ge=0)
    max_right_tail_nt: int = Field(default=0, ge=0)
    max_bulge_nt: int = Field(default=0, ge=0)
    min_left_flank_bp: int = Field(default=0, ge=0)
    min_right_flank_bp: int = Field(default=0, ge=0)
    bulge_owner: Literal["primary", "complement", "either"] = "either"
