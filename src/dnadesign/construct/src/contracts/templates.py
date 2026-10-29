"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/templates.py

Template source contracts for construct configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .base import StrictConfigModel


class TemplateLiteralSourceConfig(StrictConfigModel):
    kind: Literal["literal"]
    sequence: str
    label: Optional[str] = None

    @field_validator("sequence")
    @classmethod
    def _sequence_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.sequence cannot be empty when kind='literal'.")
        return text

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.label cannot be empty when provided.")
        return text


class TemplatePathSourceConfig(StrictConfigModel):
    kind: Literal["path"]
    path: str
    label: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _path_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.path cannot be empty when kind='path'.")
        return text

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.label cannot be empty when provided.")
        return text


class TemplateUSRSourceConfig(StrictConfigModel):
    kind: Literal["usr"]
    dataset: str
    root: Optional[str] = None
    record_id: str
    field: str = "sequence"
    label: Optional[str] = None

    @field_validator("dataset", "record_id", "field")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.dataset, record_id, and field cannot be empty.")
        return text

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.label cannot be empty when provided.")
        return text


TemplateSourceConfig = Annotated[
    TemplateLiteralSourceConfig | TemplatePathSourceConfig | TemplateUSRSourceConfig,
    Field(discriminator="kind"),
]


class TemplateConfig(StrictConfigModel):
    id: str
    source: TemplateSourceConfig
    circular: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [
            field for field in ("kind", "sequence", "path", "dataset", "root", "record_id", "field") if field in data
        ]
        if isinstance(data.get("source"), str):
            legacy_fields.append("source")
        if legacy_fields:
            joined = ", ".join(f"template.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Move template locator fields under template.source.* "
                "and keep any human-readable provenance text in template.source.label."
            )
        return data

    @field_validator("id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.id cannot be empty.")
        return text
