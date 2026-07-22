"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/visual/viennarna_secondary_structure_svg_v1.py

Manifest for ViennaRNA-native secondary-structure SVG artifacts.

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


class ViennaRNAStructureSvgArtifactsV1(VisualContractModel):
    native_svg: str
    annotated_svg: str
    annotation_manifest: str

    @field_validator("native_svg", "annotated_svg", "annotation_manifest")
    @classmethod
    def _artifact_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="artifact ref")


class ViennaRNAStructureSvgQaV1(VisualContractModel):
    nucleotide_node_count: int = Field(ge=0)
    basepair_node_count: int = Field(ge=0)
    cross_copy_pair_count: int = Field(ge=0)
    length_matches_svg_nodes: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ViennaRNAStructureSvgV1(VisualContractModel):
    contract_kind: Literal["viennarna_secondary_structure_svg_v1"] = "viennarna_secondary_structure_svg_v1"
    schema_version: Literal[1] = 1
    plot_id: str
    prediction_id: str
    sequence_id: str
    sequence_sha256: str
    length: int = Field(ge=1)
    backend_name: str
    backend_version: str
    layout_algorithm: str
    command: list[str] = Field(min_length=1)
    source_prediction: str
    source_visual_contract: str | None = None
    artifacts: ViennaRNAStructureSvgArtifactsV1
    qa: ViennaRNAStructureSvgQaV1

    @field_validator(
        "plot_id",
        "prediction_id",
        "sequence_id",
        "sequence_sha256",
        "backend_name",
        "backend_version",
        "layout_algorithm",
        "source_prediction",
    )
    @classmethod
    def _required_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="plot field")

    @field_validator("source_visual_contract")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="source_visual_contract")

    @field_validator("command")
    @classmethod
    def _command_not_blank(cls, value: list[str]) -> list[str]:
        return [_not_blank(item, label="command item") for item in value]

    @model_validator(mode="after")
    def _validate_qa(self) -> "ViennaRNAStructureSvgV1":
        if self.qa.length_matches_svg_nodes and self.qa.nucleotide_node_count != self.length:
            raise ValueError("length_matches_svg_nodes=true requires nucleotide_node_count to equal length.")
        return self


__all__ = ["ViennaRNAStructureSvgV1"]
