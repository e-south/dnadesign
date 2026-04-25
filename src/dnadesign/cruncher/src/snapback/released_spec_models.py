"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_spec_models.py

Explicit spec-side contracts for released-product snapback.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import normalize_dna
from dnadesign.cruncher.snapback.models import EFFECTIVE_CAP_LOOP_NT, CatalogSources, StrictSnapbackModel
from dnadesign.cruncher.snapback.released_route_policy import (
    _DEFAULT_DISALLOWED_NICKASE_WARNING_CODES,
    normalize_release_catalog_path_list,
    normalize_warning_code_list,
)


class ReleaseCatalogSources(StrictSnapbackModel):
    preset: str | None = None
    additional_presets: list[str] = Field(default_factory=list)
    additional_paths: list[Path] = Field(default_factory=list)

    @field_validator("preset")
    @classmethod
    def _validate_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("release_sources.preset must be non-empty when provided.")
        return text

    @field_validator("additional_presets")
    @classmethod
    def _validate_additional_presets(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("release_sources.additional_presets must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("release_sources.additional_presets must not repeat values.")
        return normalized

    @field_validator("additional_paths")
    @classmethod
    def _validate_additional_paths(cls, value: list[Path]) -> list[Path]:
        return normalize_release_catalog_path_list(value, label="release_sources.additional_paths")

    @model_validator(mode="after")
    def _validate_sources(self) -> "ReleaseCatalogSources":
        if self.preset is None and not self.additional_presets and not self.additional_paths:
            raise ValueError("release sources must define a preset, an additional preset, or an additional path.")
        preset_ids = self.resolved_preset_ids()
        if len(set(preset_ids)) != len(preset_ids):
            raise ValueError("release sources presets must not repeat values across preset and additional_presets.")
        return self

    def resolved_preset_ids(self) -> list[str]:
        preset_ids: list[str] = []
        if self.preset is not None:
            preset_ids.append(self.preset)
        preset_ids.extend(self.additional_presets)
        return preset_ids


class ReleasedSnapbackHeader(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    kind: Literal["single_nick_released_snapback_v1"] = "single_nick_released_snapback_v1"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("released_snapback.name must be non-empty.")
        return text


class ReleasedSnapbackInputSpec(StrictSnapbackModel):
    precursor_top_strand: str

    @field_validator("precursor_top_strand")
    @classmethod
    def _validate_precursor_top_strand(cls, value: str) -> str:
        return normalize_dna(value)


class ReleasedNickStageSpec(StrictSnapbackModel):
    nickase_variant_id: str
    catalog: CatalogSources
    intended_site_sequence: str | None = None

    @field_validator("nickase_variant_id")
    @classmethod
    def _validate_variant_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("nick_stage.nickase_variant_id must be non-empty.")
        return text

    @field_validator("intended_site_sequence")
    @classmethod
    def _validate_intended_site_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_dna(value)


class ReleasedReleaseStageSpec(StrictSnapbackModel):
    release_variant_id: str
    catalog: ReleaseCatalogSources
    intended_site_sequence: str | None = None
    retained_side: Literal["upstream"] = "upstream"
    stage_order: Literal["nick_then_release"] = "nick_then_release"

    @field_validator("release_variant_id")
    @classmethod
    def _validate_variant_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("release_stage.release_variant_id must be non-empty.")
        return text

    @field_validator("intended_site_sequence")
    @classmethod
    def _validate_intended_site_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_dna(value)


class ReleasedFinalTargetGeometry(StrictSnapbackModel):
    nick_boundary_from_left: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_cap_nt(self) -> "ReleasedFinalTargetGeometry":
        if self.cap_nt != EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(
                f"final_target.cap_nt must equal the fixed snapback effective cap loop size of {EFFECTIVE_CAP_LOOP_NT}."
            )
        return self


class ReleasedSnapbackConstraintsSpec(StrictSnapbackModel):
    allow_post_release_loss_of_nickase_site: bool = True
    allow_post_release_loss_of_release_site: bool = True
    require_release_site_downstream_of_nick: bool = True
    require_complete_downstream_fragment_separation: bool = True
    disallowed_nickase_warning_codes: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES)
    )

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return normalize_warning_code_list(value, label="constraints.disallowed_nickase_warning_codes")


class ReleasedSnapbackOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/released_design")

    @field_validator("run_dir", mode="before")
    @classmethod
    def _validate_run_dir(cls, value: Path | str) -> Path | str:
        raw_text = str(value or "").strip()
        if not raw_text:
            raise ValueError("output.run_dir must be non-empty.")
        path = Path(raw_text)
        if path.is_absolute():
            raise ValueError("output.run_dir must be a relative path inside the workspace.")
        if any(part == ".." for part in path.parts):
            raise ValueError("output.run_dir must not traverse outside the workspace.")
        return raw_text


class SingleNickReleasedSnapbackSpec(StrictSnapbackModel):
    released_snapback: ReleasedSnapbackHeader
    input: ReleasedSnapbackInputSpec
    nick_stage: ReleasedNickStageSpec
    release_stage: ReleasedReleaseStageSpec
    final_target: ReleasedFinalTargetGeometry
    constraints: ReleasedSnapbackConstraintsSpec = Field(default_factory=ReleasedSnapbackConstraintsSpec)
    output: ReleasedSnapbackOutputConfig = Field(default_factory=ReleasedSnapbackOutputConfig)

    @property
    def name(self) -> str:
        return self.released_snapback.name


__all__ = [
    "ReleaseCatalogSources",
    "ReleasedFinalTargetGeometry",
    "ReleasedNickStageSpec",
    "ReleasedReleaseStageSpec",
    "ReleasedSnapbackConstraintsSpec",
    "ReleasedSnapbackHeader",
    "ReleasedSnapbackInputSpec",
    "ReleasedSnapbackOutputConfig",
    "SingleNickReleasedSnapbackSpec",
]
