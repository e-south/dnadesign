"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/spec_input_models.py

Input-facing YIU spec models and validation rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.errors import (
    YIU_INPUT_KIND_UNKNOWN,
    YIU_INPUT_MUTUALLY_EXCLUSIVE,
)
from dnadesign.cruncher.yiu.spec_common import (
    normalize_optional_text,
    normalize_yiu_sequence,
    require_non_empty_text,
)


class YiuSpecRoot(StrictBaseModel):
    schema_version: Literal[1] = 1
    contract: Literal["split_yiu_payload_rendering_v4"] = "split_yiu_payload_rendering_v4"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return require_non_empty_text(value, field_name="yiu.name")


class UserSequenceInput(StrictBaseModel):
    sequence: str

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_yiu_sequence(value, ctx="input.user_sequence.sequence")


class SampleHitInput(StrictBaseModel):
    hit_id: str
    sample_name: str
    payload_sequence: str | None = None
    source_artifact_path: str | None = None
    source_artifact: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hit_id", "sample_name")
    @classmethod
    def _validate_required_text(cls, value: str, info) -> str:
        return require_non_empty_text(value, field_name=str(info.field_name))

    @field_validator("payload_sequence")
    @classmethod
    def _validate_payload_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_yiu_sequence(value, ctx="input.sample_hit.payload_sequence")

    @field_validator("source_artifact_path", "source_artifact")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        return normalize_optional_text(value)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @model_validator(mode="after")
    def _validate_resolution_paths(self) -> "SampleHitInput":
        source_workspace = self.metadata.get("source_workspace")
        has_workspace_ref = isinstance(source_workspace, str) and str(source_workspace).strip() != ""
        has_artifact_ref = self.source_artifact_path is not None or self.source_artifact is not None
        if self.payload_sequence is None and not has_artifact_ref and not has_workspace_ref:
            raise ValueError(
                "sample_hit requires payload_sequence or a resolvable source artifact reference "
                "(source_artifact_path, source_artifact, or metadata.source_workspace)."
            )
        return self


class InputSpec(StrictBaseModel):
    kind: Literal["user_sequence", "sample_hit"]
    user_sequence: UserSequenceInput | None = None
    sample_hit: SampleHitInput | None = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        kind = str(value).strip()
        if kind not in {"user_sequence", "sample_hit"}:
            raise ValueError(f"{YIU_INPUT_KIND_UNKNOWN}: unsupported input.kind={value!r}")
        return kind

    @model_validator(mode="after")
    def _validate_subtype(self) -> "InputSpec":
        declared = {
            "user_sequence": self.user_sequence is not None,
            "sample_hit": self.sample_hit is not None,
        }
        if sum(declared.values()) != 1:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: exactly one input subtype must be populated")
        if self.kind == "user_sequence" and self.user_sequence is None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.kind=user_sequence requires input.user_sequence")
        if self.kind == "sample_hit" and self.sample_hit is None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.kind=sample_hit requires input.sample_hit")
        if self.kind != "user_sequence" and self.user_sequence is not None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.user_sequence is incompatible with input.kind")
        if self.kind != "sample_hit" and self.sample_hit is not None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.sample_hit is incompatible with input.kind")
        return self


__all__ = [
    "InputSpec",
    "SampleHitInput",
    "UserSequenceInput",
    "YiuSpecRoot",
]
