"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/scar_nick_visual_v1.py

Shared scar-nick visual contract for terminal scar-nick QA rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import JsonMap, PositiveLengthSpan, VisualContractModel

_IUPAC_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")
_DNA_BASES = frozenset("ACGT")
_IUPAC_DNA_BASES = frozenset("ACGTRYSWKMBDHVN")
_MIN_NICKASE_RECOGNITION_NT = 4


def _complement_3to5(sequence: str) -> str:
    return sequence.translate(_IUPAC_COMPLEMENT).upper()


def _validate_alphabet_symbols(*, label: str, sequence: str, alphabet: str) -> None:
    allowed = _DNA_BASES if alphabet == "dna" else _IUPAC_DNA_BASES
    invalid = sorted({base.upper() for base in sequence if base.upper() not in allowed})
    if invalid:
        raise ValueError(f"{label} contains symbols outside {alphabet}: {', '.join(invalid)}")


def _iupac_bases_for_symbol(symbol: str) -> set[str]:
    mapping: dict[str, set[str]] = {
        "A": {"A"},
        "C": {"C"},
        "G": {"G"},
        "T": {"T"},
        "R": {"A", "G"},
        "Y": {"C", "T"},
        "S": {"G", "C"},
        "W": {"A", "T"},
        "K": {"G", "T"},
        "M": {"A", "C"},
        "B": {"C", "G", "T"},
        "D": {"A", "G", "T"},
        "H": {"A", "C", "T"},
        "V": {"A", "C", "G"},
        "N": {"A", "C", "G", "T"},
    }
    text = str(symbol or "").upper()
    if text not in mapping:
        raise ValueError(f"Unknown IUPAC nucleotide symbol: {symbol!r}")
    return set(mapping[text])


def _iupac_symbols_overlap(left_symbol: str, right_symbol: str) -> bool:
    return bool(_iupac_bases_for_symbol(left_symbol) & _iupac_bases_for_symbol(right_symbol))


def _recognition_nt(motif: str) -> int:
    return sum(1 for symbol in motif if _iupac_bases_for_symbol(symbol) != set(_DNA_BASES))


class ScarNickRectangularFillV1(VisualContractModel):
    fill_id: str
    semantic: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    cover_rows: Literal["primary", "complement", "both"] = "both"
    fill: str
    alpha: float = Field(ge=0.0, le=1.0)
    corner_radius: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_fill(self) -> "ScarNickRectangularFillV1":
        if self.end <= self.start:
            raise ValueError("rectangular fill end must be > start")
        if not self.fill.strip():
            raise ValueError("rectangular fill color must be non-empty")
        if self.semantic == "retained_type_iis_scar" and self.corner_radius != 0.0:
            raise ValueError("retained Type IIS scar fill must be rectangular")
        return self


class ScarNickSourceSpanV1(VisualContractModel):
    start: int
    end: int

    @model_validator(mode="after")
    def _validate_span(self) -> "ScarNickSourceSpanV1":
        if self.end <= self.start:
            raise ValueError("source span end must be > start")
        return self


class ScarNickFragmentSpanV1(VisualContractModel):
    row: Literal["primary", "complement"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_fragment_span(self) -> "ScarNickFragmentSpanV1":
        if self.end <= self.start:
            raise ValueError("fragment span end must be > start")
        return self


class ScarNickPanelV1(VisualContractModel):
    panel_id: Literal["pre_release", "post_release"]
    title: str
    state_kind: Literal["pre_terminal_nick", "post_terminal_nick"]
    nick_state: Literal["intact", "nicked"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    terminal_boundary: int = Field(ge=0)
    nick_boundary: int = Field(ge=0)
    retained_product_span: PositiveLengthSpan
    release_site_span: PositiveLengthSpan
    type_iis_offset_span: PositiveLengthSpan | None = None
    retained_scar_span: PositiveLengthSpan
    nickase_site_span: PositiveLengthSpan
    fragment_spans: list[ScarNickFragmentSpanV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_panel(self) -> "ScarNickPanelV1":
        if self.end <= self.start:
            raise ValueError("panel end must be > start")
        if not str(self.title or "").strip():
            raise ValueError("panel title must be non-empty")
        if self.panel_id == "pre_release" and self.state_kind != "pre_terminal_nick":
            raise ValueError("pre_release panel requires pre_terminal_nick state")
        if self.panel_id == "post_release" and self.state_kind != "post_terminal_nick":
            raise ValueError("post_release panel requires post_terminal_nick state")
        if self.state_kind == "pre_terminal_nick" and self.nick_state != "intact":
            raise ValueError("pre_terminal_nick panel requires nick_state='intact'")
        if self.state_kind == "post_terminal_nick" and self.nick_state != "nicked":
            raise ValueError("post_terminal_nick panel requires nick_state='nicked'")
        if self.nick_boundary != self.terminal_boundary:
            raise ValueError("scar-nick panel requires nick_boundary == terminal_boundary")
        for label, span in (
            ("retained_product_span", self.retained_product_span),
            ("release_site_span", self.release_site_span),
            ("type_iis_offset_span", self.type_iis_offset_span),
            ("retained_scar_span", self.retained_scar_span),
            ("nickase_site_span", self.nickase_site_span),
        ):
            if span is None:
                continue
            if span.start < self.start or span.end > self.end:
                raise ValueError(f"{label} must lie within its panel bounds")
        if self.retained_scar_span.end - self.retained_scar_span.start != 4:
            raise ValueError("panel retained_scar_span must mark the 4-nt terminal Type IIS scar")
        if self.terminal_boundary != self.retained_scar_span.end:
            raise ValueError("panel terminal_boundary must equal retained_scar_span.end")
        if self.retained_product_span.start != self.retained_scar_span.start:
            raise ValueError("panel retained_product_span must start at retained_scar_span.start")
        if self.retained_product_span.end != self.retained_scar_span.end:
            raise ValueError("panel retained_product_span must terminate at retained_scar_span.end")
        for span in self.fragment_spans:
            if span.start < self.start or span.end > self.end:
                raise ValueError("fragment span must lie within its panel bounds")
        if self.panel_id == "pre_release" and self.fragment_spans:
            raise ValueError("pre_release panel must not contain replacement fragment spans")
        if self.panel_id == "post_release" and not self.fragment_spans:
            raise ValueError("post_release panel requires at least one replacement fragment span")
        return self


class ScarNickReleasePlacementV1(VisualContractModel):
    variant_id: str
    orientation: Literal["forward"]
    recognition_sequence: str
    source_catalog_id: str
    source_url: str
    commercial_confidence: str
    warning_codes: list[str] = Field(default_factory=list)
    recognition_site_start: int
    recognition_site_end: int
    top_cut_boundary: int
    bottom_cut_boundary: int
    retained_scar_start: int
    retained_scar_end: int
    retained_scar_nt: int = Field(ge=1)
    recognition_site_excised: bool

    @model_validator(mode="after")
    def _validate_release(self) -> "ScarNickReleasePlacementV1":
        for label, value in (
            ("variant_id", self.variant_id),
            ("recognition_sequence", self.recognition_sequence),
            ("source_catalog_id", self.source_catalog_id),
            ("source_url", self.source_url),
            ("commercial_confidence", self.commercial_confidence),
        ):
            if not str(value or "").strip():
                raise ValueError(f"release_placement.{label} must be non-empty")
        if self.recognition_site_end <= self.recognition_site_start:
            raise ValueError("release_placement recognition site span must be positive length")
        if self.retained_scar_end <= self.retained_scar_start:
            raise ValueError("release_placement retained scar span must be positive length")
        if self.retained_scar_end - self.retained_scar_start != self.retained_scar_nt:
            raise ValueError("release_placement retained_scar_nt must match retained scar span")
        return self


class ScarNickNickasePlacementV1(VisualContractModel):
    variant_id: str
    specificity_id: str
    orientation: Literal["forward", "reverse"]
    canonical_read_row: Literal["primary", "complement"]
    site: str
    motif_top_5to3: str
    recognition_nt: int = Field(ge=1)
    vendor: str
    source_url: str
    source_family: Literal["nicking_endonuclease"]
    commercial_confidence: str
    warning_codes: list[str] = Field(default_factory=list)
    source_site_start: int
    source_site_end: int
    strand: Literal["top", "bottom"]
    boundary: int
    terminal_boundary: int
    display_boundary: int = Field(ge=0)
    display_site_span: PositiveLengthSpan
    exact_terminal: bool

    @model_validator(mode="after")
    def _validate_nickase(self) -> "ScarNickNickasePlacementV1":
        for label, value in (
            ("variant_id", self.variant_id),
            ("specificity_id", self.specificity_id),
            ("site", self.site),
            ("motif_top_5to3", self.motif_top_5to3),
            ("vendor", self.vendor),
            ("source_url", self.source_url),
            ("commercial_confidence", self.commercial_confidence),
        ):
            if not str(value or "").strip():
                raise ValueError(f"nickase.{label} must be non-empty")
        if self.orientation == "reverse" and self.canonical_read_row != "complement":
            raise ValueError("reverse nickase placements must mark complement as the canonical read row")
        if self.orientation == "forward" and self.canonical_read_row != "primary":
            raise ValueError("forward nickase placements must mark primary as the canonical read row")
        if self.source_site_end <= self.source_site_start:
            raise ValueError("nickase source site span must be positive length")
        return self


class ScarNickVisualV1(VisualContractModel):
    contract_kind: Literal["scar_nick_visual_v1"] = "scar_nick_visual_v1"
    state_id: str
    state_kind: Literal["pre_post_terminal_nick"]
    event_scope: Literal["terminal_nick"] = "terminal_nick"
    alphabet: Literal["dna", "iupac_dna"] = "dna"
    title: str | None = None
    primary_sequence: str
    complement_sequence: str
    primary_row_label: str
    complement_row_label: str
    terminal_boundary: int = Field(ge=0)
    nick_boundary: int = Field(ge=0)
    retained_product_span: PositiveLengthSpan
    release_site_span: PositiveLengthSpan
    type_iis_offset_span: PositiveLengthSpan | None = None
    retained_scar_span: PositiveLengthSpan
    junction_partner_span: PositiveLengthSpan | None = None
    nickase_site_span: PositiveLengthSpan
    nickase_site_source_span: ScarNickSourceSpanV1 | None = None
    nickase_site_span_clipped: bool = False
    nick_state: Literal["pre_post"]
    retained_scar: str
    left_base: str
    right_base: str
    nicked_strand: Literal["top", "bottom"]
    surviving_strand: Literal["top", "bottom"]
    profile_s3s2s1s0: str
    profile_payload_outward: str
    pair_classes: list[JsonMap] = Field(default_factory=list)
    panels: list[ScarNickPanelV1]
    rectangular_fills: list[ScarNickRectangularFillV1] = Field(default_factory=list)
    release_placement: ScarNickReleasePlacementV1
    nickase: ScarNickNickasePlacementV1
    meta: JsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> "ScarNickVisualV1":
        if not self.primary_sequence:
            raise ValueError("primary_sequence must be non-empty")
        if len(self.complement_sequence) != len(self.primary_sequence):
            raise ValueError("complement_sequence must match primary_sequence length")
        _validate_alphabet_symbols(
            label="primary_sequence",
            sequence=self.primary_sequence,
            alphabet=self.alphabet,
        )
        _validate_alphabet_symbols(
            label="complement_sequence",
            sequence=self.complement_sequence,
            alphabet=self.alphabet,
        )
        if len(self.left_base) != 4 or any(base.upper() not in _DNA_BASES for base in self.left_base):
            raise ValueError("left_base must contain exactly four A/C/G/T bases")
        if len(self.right_base) != 4 or any(base.upper() not in _DNA_BASES for base in self.right_base):
            raise ValueError("right_base must contain exactly four A/C/G/T bases")
        if self.nicked_strand == self.surviving_strand:
            raise ValueError("nicked_strand and surviving_strand must differ")
        sequence_length = len(self.primary_sequence)
        if self.terminal_boundary > sequence_length:
            raise ValueError("terminal_boundary must lie within primary_sequence bounds")
        if self.nick_boundary > sequence_length:
            raise ValueError("nick_boundary must lie within primary_sequence bounds")
        if self.nick_boundary != self.terminal_boundary:
            raise ValueError("scar-nick visual requires nick_boundary == terminal_boundary")
        for label, span in (
            ("retained_product_span", self.retained_product_span),
            ("release_site_span", self.release_site_span),
            ("type_iis_offset_span", self.type_iis_offset_span),
            ("retained_scar_span", self.retained_scar_span),
            ("junction_partner_span", self.junction_partner_span),
            ("nickase_site_span", self.nickase_site_span),
        ):
            if span is not None and span.end > sequence_length:
                raise ValueError(f"{label} must lie within primary_sequence bounds")
        if self.retained_scar_span.end - self.retained_scar_span.start != 4:
            raise ValueError("retained_scar_span must mark the 4-nt terminal Type IIS scar")
        if self.terminal_boundary != self.retained_scar_span.end:
            raise ValueError("terminal_boundary must equal retained_scar_span.end")
        if self.junction_partner_span is not None:
            raise ValueError("scar-nick visual must not place partner sequence downstream of the nick")
        product_covers_scar = self.retained_product_span.start == self.retained_scar_span.start
        product_terminates_at_scar = self.retained_product_span.end == self.retained_scar_span.end
        if not product_covers_scar or not product_terminates_at_scar:
            raise ValueError("retained_product_span must terminate at retained_scar_span.end")
        if self.retained_scar != self.primary_sequence[self.retained_scar_span.start : self.retained_scar_span.end]:
            raise ValueError("retained_scar must match retained_scar_span on primary_sequence")
        if self.retained_scar != self.left_base.upper():
            raise ValueError("retained_scar must match left_base")
        if self.release_placement.retained_scar_nt != 4:
            raise ValueError("scar-nick visual requires a 4-nt Type IIS retained scar")
        if (
            self.release_placement.retained_scar_end - self.release_placement.retained_scar_start
            != self.release_placement.retained_scar_nt
        ):
            raise ValueError("release_placement retained scar span must match retained_scar_nt")
        release_sequence = self.release_placement.recognition_sequence
        if len(release_sequence) != self.release_site_span.end - self.release_site_span.start:
            raise ValueError("release recognition_sequence length must match release_site_span")
        observed_release = self.primary_sequence[self.release_site_span.start : self.release_site_span.end]
        for observed_symbol, expected_symbol in zip(observed_release, release_sequence, strict=True):
            if not _iupac_symbols_overlap(observed_symbol, expected_symbol):
                raise ValueError("release_site_span must match release_placement recognition_sequence")
        if self.release_placement.recognition_site_excised is not True:
            raise ValueError("scar-nick visual requires an excised Type IIS recognition site")
        if self.nickase.exact_terminal is not True:
            raise ValueError("scar-nick visual requires an exact terminal nickase placement")
        if self.nickase.strand != self.nicked_strand:
            raise ValueError("nickase strand must match nicked_strand")
        if self.nickase.display_boundary != self.nick_boundary:
            raise ValueError("nickase display_boundary must equal nick_boundary")
        if self.nickase.display_site_span.start != self.nickase_site_span.start:
            raise ValueError("nickase display_site_span must match nickase_site_span")
        if self.nickase.display_site_span.end != self.nickase_site_span.end:
            raise ValueError("nickase display_site_span must match nickase_site_span")
        nickase_motif = self.nickase.motif_top_5to3
        observed_recognition_nt = _recognition_nt(nickase_motif)
        if self.nickase.recognition_nt != observed_recognition_nt:
            raise ValueError("nickase recognition_nt must match motif_top_5to3")
        if observed_recognition_nt < _MIN_NICKASE_RECOGNITION_NT:
            raise ValueError("scar-nick visual requires a nickase recognition site of at least 4 nt")
        if len(nickase_motif) != self.nickase_site_span.end - self.nickase_site_span.start:
            raise ValueError("nickase motif length must match nickase_site_span")
        observed_nickase = self.primary_sequence[self.nickase_site_span.start : self.nickase_site_span.end]
        for observed_symbol, expected_symbol in zip(observed_nickase, nickase_motif, strict=True):
            if not _iupac_symbols_overlap(observed_symbol, expected_symbol):
                raise ValueError("nickase_site_span must match nickase motif")
        if len(self.profile_s3s2s1s0) != 4 or any(char not in {"M", "W", "X"} for char in self.profile_s3s2s1s0):
            raise ValueError("profile_s3s2s1s0 must contain exactly four M/W/X symbols")
        if self.profile_payload_outward != self.profile_s3s2s1s0[::-1]:
            raise ValueError("profile_payload_outward must be the reverse of profile_s3s2s1s0")
        if len(self.pair_classes) != 4:
            raise ValueError("pair_classes must contain exactly four entries")
        expected_positions = list(range(4))
        observed_positions = [entry.get("position") for entry in self.pair_classes]
        if observed_positions != expected_positions:
            raise ValueError("pair_classes positions must be ordered 0..3")
        observed_profile = "".join(str(entry.get("class_label") or "") for entry in self.pair_classes)
        if observed_profile != self.profile_s3s2s1s0:
            raise ValueError("pair_classes class labels must match profile_s3s2s1s0")
        if self.nickase_site_source_span is None:
            raise ValueError("scar-nick visual requires nickase_site_source_span")
        if self.nickase_site_span_clipped:
            raise ValueError("scar-nick visual requires the full nickase site span to be visible")
        if len(self.panels) != 2:
            raise ValueError("scar-nick visual requires exactly two panels")
        panel_ids = [panel.panel_id for panel in self.panels]
        if panel_ids != ["pre_release", "post_release"]:
            raise ValueError("scar-nick visual panels must be ordered pre_release, post_release")
        right_display = self.right_base.upper()[::-1]
        perfect_complement = _complement_3to5(self.primary_sequence)
        allowed_non_complement_indices: set[int] = set()
        expected_fragment_row = "primary" if self.nicked_strand == "top" else "complement"
        for panel in self.panels:
            if panel.end > sequence_length:
                raise ValueError("panel must lie within primary_sequence bounds")
            panel_release = self.primary_sequence[panel.release_site_span.start : panel.release_site_span.end]
            for observed_symbol, expected_symbol in zip(panel_release, release_sequence, strict=True):
                if not _iupac_symbols_overlap(observed_symbol, expected_symbol):
                    raise ValueError(
                        f"{panel.panel_id} release_site_span must match release_placement recognition_sequence"
                    )
            panel_nickase = self.primary_sequence[panel.nickase_site_span.start : panel.nickase_site_span.end]
            for observed_symbol, expected_symbol in zip(panel_nickase, nickase_motif, strict=True):
                if not _iupac_symbols_overlap(observed_symbol, expected_symbol):
                    raise ValueError(f"{panel.panel_id} nickase_site_span must match nickase motif")
            if panel.panel_id == "pre_release":
                observed_panel_complement = self.complement_sequence[panel.start : panel.end]
                expected_panel_complement = perfect_complement[panel.start : panel.end]
                if observed_panel_complement.upper() != expected_panel_complement.upper():
                    raise ValueError("pre_release panel must be Watson-Crick paired before adapter annealing")
            if panel.panel_id == "post_release":
                for fragment_span in panel.fragment_spans:
                    if fragment_span.row != expected_fragment_row:
                        raise ValueError("post_release fragment spans must be on the nicked strand")
                    if fragment_span.end > panel.retained_scar_span.start:
                        raise ValueError("post_release fragment spans must stop before the retained scar")
            downstream = self.primary_sequence[panel.terminal_boundary : panel.end]
            if any(symbol.upper() != "N" for symbol in downstream):
                raise ValueError("scar-nick visual allows only degenerate N symbols downstream of each terminal nick")
            if self.primary_sequence[panel.retained_scar_span.start : panel.retained_scar_span.end] != self.left_base:
                raise ValueError("each panel retained_scar_span must match left_base")
            if panel.panel_id == "post_release":
                observed_right = self.complement_sequence[panel.retained_scar_span.start : panel.retained_scar_span.end]
                if observed_right != right_display:
                    raise ValueError(
                        "post_release retained_scar_span on complement row must display right_base in S-order"
                    )
                allowed_non_complement_indices.update(
                    range(panel.retained_scar_span.start, panel.retained_scar_span.end)
                )
        for index, (observed, expected) in enumerate(zip(self.complement_sequence, perfect_complement, strict=True)):
            if index in allowed_non_complement_indices:
                continue
            if observed.upper() != expected.upper():
                raise ValueError(
                    "complement_sequence may differ from complement only inside post_release retained scar"
                )
        for fill in self.rectangular_fills:
            if fill.end > sequence_length:
                raise ValueError("rectangular fill must lie within primary_sequence bounds")
        release_fills = [fill for fill in self.rectangular_fills if fill.semantic == "type_iis_release_site"]
        nickase_fills = [fill for fill in self.rectangular_fills if fill.semantic == "nickase_footprint"]
        scar_fills = [fill for fill in self.rectangular_fills if fill.semantic == "retained_type_iis_scar"]
        if len(release_fills) != len(self.panels):
            raise ValueError("scar-nick visual requires one type_iis_release_site fill per panel")
        if len(nickase_fills) != 1:
            raise ValueError("scar-nick visual requires one nickase_footprint fill on the pre_release panel only")
        if len(scar_fills) != len(self.panels):
            raise ValueError("scar-nick visual requires one retained_type_iis_scar fill per panel")
        pre_panel = self.panels[0]
        nickase_fill = nickase_fills[0]
        if (
            nickase_fill.start != pre_panel.nickase_site_span.start
            or nickase_fill.end != pre_panel.nickase_site_span.end
        ):
            raise ValueError("nickase_footprint fill must cover the pre_release panel only")
        for panel in self.panels:
            if not any(
                fill.start == panel.release_site_span.start and fill.end == panel.release_site_span.end
                for fill in release_fills
            ):
                raise ValueError("Type IIS release site fills must cover every panel release_site_span")
            if not any(
                fill.start == panel.retained_scar_span.start and fill.end == panel.retained_scar_span.end
                for fill in scar_fills
            ):
                raise ValueError("retained Type IIS scar fills must cover every panel retained_scar_span")
        return self


__all__ = [
    "ScarNickFragmentSpanV1",
    "ScarNickNickasePlacementV1",
    "ScarNickPanelV1",
    "ScarNickRectangularFillV1",
    "ScarNickReleasePlacementV1",
    "ScarNickSourceSpanV1",
    "ScarNickVisualV1",
]
