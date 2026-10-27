"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/parts.py

Part sequence, placement locator, and placement guard contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .base import StrictConfigModel


class PartSequenceConfig(StrictConfigModel):
    source: Literal["input_field", "literal"]
    field: Optional[str] = None
    literal: Optional[str] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "PartSequenceConfig":
        if self.source == "input_field":
            if not str(self.field or "").strip():
                raise ValueError("part.sequence.field is required when source='input_field'.")
            if self.literal is not None:
                raise ValueError("part.sequence.literal is not allowed when source='input_field'.")
        if self.source == "literal":
            if not str(self.literal or "").strip():
                raise ValueError("part.sequence.literal is required when source='literal'.")
            if self.field is not None:
                raise ValueError("part.sequence.field is not allowed when source='literal'.")
        return self


class CoordinatePlacementLocatorConfig(StrictConfigModel):
    kind: Literal["coordinates"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class FlankPlacementLocatorConfig(StrictConfigModel):
    kind: Literal["flanks"]
    upstream_sequence: str
    downstream_sequence: str

    @field_validator("upstream_sequence", "downstream_sequence")
    @classmethod
    def _sequence_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("placement.locator flank sequences cannot be empty.")
        return text


PlacementLocatorConfig = Annotated[
    CoordinatePlacementLocatorConfig | FlankPlacementLocatorConfig,
    Field(discriminator="kind"),
]


class PlacementGuardsConfig(StrictConfigModel):
    replaced_sequence: Optional[str] = None
    upstream_sequence: Optional[str] = None
    downstream_sequence: Optional[str] = None
    replaced_span_bp: Optional[int] = Field(default=None, ge=0)
    require_unique_forward_matches: bool = False

    @field_validator("replaced_sequence", "upstream_sequence", "downstream_sequence")
    @classmethod
    def _optional_sequence_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("placement.guards sequences cannot be empty when provided.")
        return text

    @model_validator(mode="after")
    def _validate_meaningful_guard(self) -> "PlacementGuardsConfig":
        has_guard = any(
            value is not None
            for value in (
                self.replaced_sequence,
                self.upstream_sequence,
                self.downstream_sequence,
                self.replaced_span_bp,
            )
        )
        if self.require_unique_forward_matches and not any(
            value is not None for value in (self.replaced_sequence, self.upstream_sequence, self.downstream_sequence)
        ):
            raise ValueError("placement.guards.require_unique_forward_matches requires at least one guard sequence.")
        if not has_guard and not self.require_unique_forward_matches:
            raise ValueError("placement.guards must declare at least one guard.")
        return self


class PlacementConfig(StrictConfigModel):
    kind: Literal["insert", "replace"]
    orientation: Literal["forward", "reverse_complement"] = "forward"
    locator: PlacementLocatorConfig
    guards: Optional[PlacementGuardsConfig] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "PlacementConfig":
        if isinstance(self.locator, CoordinatePlacementLocatorConfig):
            if self.locator.end < self.locator.start:
                raise ValueError("placement.locator.end must be >= placement.locator.start.")
            if self.kind == "insert" and self.locator.end != self.locator.start:
                raise ValueError(
                    "insert placement requires placement.locator.end == placement.locator.start "
                    "when locator.kind='coordinates'."
                )
            if self.kind == "replace" and self.locator.end == self.locator.start:
                raise ValueError(
                    "replace placement requires placement.locator.end > placement.locator.start "
                    "when locator.kind='coordinates'."
                )
        if self.kind == "insert" and self.guards is not None and self.guards.replaced_sequence is not None:
            raise ValueError("placement.guards.replaced_sequence is only allowed when kind='replace'.")
        if isinstance(self.locator, FlankPlacementLocatorConfig) and self.guards is not None:
            if self.guards.upstream_sequence is not None or self.guards.downstream_sequence is not None:
                raise ValueError(
                    "placement.guards.upstream_sequence/downstream_sequence are not allowed when "
                    "locator.kind='flanks'. Use placement.locator.upstream_sequence/downstream_sequence instead."
                )
        return self


class PartConfig(StrictConfigModel):
    name: str
    role: str = "part"
    sequence: PartSequenceConfig
    placement: PlacementConfig

    @field_validator("name", "role")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("part name/role cannot be empty.")
        return text
