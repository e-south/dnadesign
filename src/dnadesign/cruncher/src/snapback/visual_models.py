"""
Typed contracts for visual-only single-nick snapback examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import (
    motif_matches,
    normalize_dna,
    normalize_iupac,
    reverse_complement,
    reverse_complement_iupac,
)
from dnadesign.cruncher.snapback.models import CoordinateSpan, StrictSnapbackModel
from dnadesign.cruncher.snapback.publication_support import complement_sequence


class SnapbackVisualHeader(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    contract: Literal["single_nick_snapback_visual_v1"] = "single_nick_snapback_visual_v1"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("snapback_visual.name must be non-empty.")
        return text


class SnapbackVisualInputSpec(StrictSnapbackModel):
    precursor_top_strand: str

    @field_validator("precursor_top_strand")
    @classmethod
    def _validate_precursor_top_strand(cls, value: str) -> str:
        return normalize_dna(value)


class SnapbackVisualNickSpec(StrictSnapbackModel):
    label: str
    site_sequence: str
    site_span: CoordinateSpan
    nick_boundary: int = Field(ge=0)
    nicked_strand: Literal["top"] = "top"

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("nick.label must be non-empty.")
        return text

    @field_validator("site_sequence")
    @classmethod
    def _validate_site_sequence(cls, value: str) -> str:
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_site_span_width(self) -> "SnapbackVisualNickSpec":
        if self.site_span.length != len(self.site_sequence):
            raise ValueError("nick.site_span length must equal nick.site_sequence length.")
        return self


class SnapbackVisualProductSpec(StrictSnapbackModel):
    active_strand: Literal["bottom"] = "bottom"
    active_label: str = "Retained Bottom"
    upstream_context_nt: int = Field(ge=0)
    stem_sequence: str
    cap_sequence: str
    foldback_sequence: str

    @field_validator("active_label")
    @classmethod
    def _validate_active_label(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("product.active_label must be non-empty.")
        return text

    @field_validator("stem_sequence", "cap_sequence", "foldback_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value)

    @model_validator(mode="after")
    def _validate_foldback(self) -> "SnapbackVisualProductSpec":
        expected = reverse_complement(self.stem_sequence)
        if self.foldback_sequence != expected:
            raise ValueError(
                f"product.foldback_sequence must be the reverse complement of product.stem_sequence ({expected})."
            )
        return self


class SnapbackVisualOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/visual")
    render_format: Literal["png", "svg", "pdf"] = "pdf"

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


class SingleNickSnapbackVisualSpec(StrictSnapbackModel):
    snapback_visual: SnapbackVisualHeader
    input: SnapbackVisualInputSpec
    nick: SnapbackVisualNickSpec
    product: SnapbackVisualProductSpec
    output: SnapbackVisualOutputConfig = Field(default_factory=SnapbackVisualOutputConfig)

    @model_validator(mode="after")
    def _validate_precursor_geometry(self) -> "SingleNickSnapbackVisualSpec":
        precursor_top = self.input.precursor_top_strand
        if self.nick.site_span.end > len(precursor_top):
            raise ValueError("nick.site_span must stay inside input.precursor_top_strand.")
        observed_site = precursor_top[self.nick.site_span.start : self.nick.site_span.end]
        reverse_site = reverse_complement_iupac(self.nick.site_sequence)
        if not motif_matches(observed_site, self.nick.site_sequence) and not motif_matches(observed_site, reverse_site):
            raise ValueError(
                "nick.site_sequence must match input.precursor_top_strand at nick.site_span "
                f"({observed_site}) in either forward or reverse-complement IUPAC orientation."
            )
        if self.nick.nick_boundary > len(precursor_top):
            raise ValueError("nick.nick_boundary must stay inside input.precursor_top_strand.")
        if self.product.upstream_context_nt != self.nick.nick_boundary:
            raise ValueError("product.upstream_context_nt must equal nick.nick_boundary for visual-only v1.")

        bottom = complement_sequence(precursor_top)
        expected_product = (
            bottom[: self.product.upstream_context_nt]
            + self.product.stem_sequence
            + self.product.cap_sequence
            + self.product.foldback_sequence
        )
        observed_product = bottom[: len(expected_product)]
        if expected_product != observed_product:
            raise ValueError(
                "product upstream/stem/cap/foldback decomposition must match the precursor bottom-strand "
                f"prefix ({observed_product})."
            )
        if len(expected_product) != len(bottom):
            raise ValueError(
                "visual-only v1 requires product upstream/stem/cap/foldback to cover the full precursor complement."
            )
        return self

    @property
    def name(self) -> str:
        return self.snapback_visual.name

    @property
    def active_product_sequence(self) -> str:
        return (
            complement_sequence(self.input.precursor_top_strand)[: self.product.upstream_context_nt]
            + self.product.stem_sequence
            + self.product.cap_sequence
            + self.product.foldback_sequence
        )

    @property
    def effective_stem_bp(self) -> int:
        return self.product.upstream_context_nt + len(self.product.stem_sequence)


class SnapbackVisualReport(StrictSnapbackModel):
    kind: Literal["snapback_visual_report_v1"] = "snapback_visual_report_v1"
    status: Literal["rendered"] = "rendered"
    spec_name: str
    workspace_root: str
    spec_path: str
    run_dir: str
    plot_data_path: str
    plot_path: str
    precursor_top_strand: str
    precursor_bottom_strand: str
    nick_label: str
    nick_boundary_from_left: int
    active_product_sequence: str
    upstream_context_nt: int
    effective_stem_bp: int
    stem_sequence: str
    cap_sequence: str
    foldback_sequence: str


__all__ = [
    "SingleNickSnapbackVisualSpec",
    "SnapbackVisualHeader",
    "SnapbackVisualInputSpec",
    "SnapbackVisualNickSpec",
    "SnapbackVisualOutputConfig",
    "SnapbackVisualProductSpec",
    "SnapbackVisualReport",
]
