"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/composition_review_svg_v1.py

Manifest for generated composition review SVG panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import VisualContractModel


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


PanelKindV1 = Literal["secondary_structure", "component_span"]


class CompositionReviewSvgSourcesV1(VisualContractModel):
    structure_svg: str
    component_span_svg: str
    visual_contract: str
    bundle_manifest: str | None = None

    @field_validator("structure_svg", "component_span_svg", "visual_contract")
    @classmethod
    def _source_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="source ref")

    @field_validator("bundle_manifest")
    @classmethod
    def _optional_source_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="bundle_manifest")


class CompositionReviewSvgArtifactsV1(VisualContractModel):
    review_svg: str
    review_png: str

    @field_validator("review_svg", "review_png")
    @classmethod
    def _artifact_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="review artifact")


class CompositionReviewSvgLayoutV1(VisualContractModel):
    row_count: Literal[2] = 2
    panel_order: list[PanelKindV1] = Field(default_factory=lambda: ["secondary_structure", "component_span"])
    component_nucleotide_font_size_px: float = Field(gt=0)
    structure_fit_policy: Literal["balanced_visual_weight"] = "balanced_visual_weight"
    structure_scale: float = Field(gt=0)
    component_scale: float = Field(gt=0)
    structure_width_px: float = Field(gt=0)
    component_width_px: float = Field(gt=0)
    structure_effective_nucleotide_font_size_px: float = Field(gt=0)
    component_effective_nucleotide_font_size_px: float = Field(gt=0)
    component_panel_emphasis: Literal["bold_glyph_review"] = "bold_glyph_review"
    component_source_title_policy: Literal["omit_redundant_source_title"] = "omit_redundant_source_title"
    structure_to_component_width_ratio: float = Field(gt=0)
    vertical_gap_px: float = Field(ge=0)
    review_png_scale: float = Field(gt=0)
    review_png_ppi: float = Field(gt=0)

    @model_validator(mode="after")
    def _validate_panel_order(self) -> "CompositionReviewSvgLayoutV1":
        if self.panel_order != ["secondary_structure", "component_span"]:
            raise ValueError("panel_order must be ['secondary_structure', 'component_span'].")
        return self


class CompositionReviewSvgQaV1(VisualContractModel):
    subplot_visual_weight_balanced: bool
    component_panel_emphasis_applied: bool
    component_source_title_omitted: bool
    component_source_title_omitted_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CompositionReviewSvgV1(VisualContractModel):
    contract_kind: Literal["composition_review_svg_v1"] = "composition_review_svg_v1"
    schema_version: Literal[1] = 1
    review_id: str
    composition_id: str
    sequence_id: str
    sequence_sha256: str
    length: int = Field(ge=1)
    sources: CompositionReviewSvgSourcesV1
    artifacts: CompositionReviewSvgArtifactsV1
    layout: CompositionReviewSvgLayoutV1
    qa: CompositionReviewSvgQaV1

    @field_validator("review_id", "composition_id", "sequence_id", "sequence_sha256")
    @classmethod
    def _required_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="review field")


__all__ = ["CompositionReviewSvgV1"]
