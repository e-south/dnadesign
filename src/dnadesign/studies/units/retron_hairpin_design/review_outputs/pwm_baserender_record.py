"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm_baserender_record.py

BaseRender record assembly for bidirectional TetR PWM trim panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import dnadesign.baserender as baserender

from .plan import PwmTrimPanel
from .pwm_visual_layers import sequence_row_visual_meta

TOP_RETAINED_FEATURE_ID = "tetO_retained_payload_span"
BOTTOM_RETAINED_FEATURE_ID = "tetO_retained_payload_span_complement"


@dataclass(frozen=True)
class PwmLogoColumn:
    parent_position_0: int
    parent_base: str
    information_bits: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class PwmLogoLayer:
    motif_instance_id: str
    strand: str
    start_0: int
    end_0: int
    occurrence_rank: int
    matrix: tuple[tuple[float, float, float, float], ...]


def build_pwm_baserender_record(
    columns: Sequence[PwmLogoColumn],
    *,
    parent_sequence: str,
    panel: PwmTrimPanel,
    logo_layers: Sequence[PwmLogoLayer],
):
    site_start = columns[0].parent_position_0
    site_end = columns[-1].parent_position_0 + 1
    included = [
        index for index in range(len(parent_sequence)) if panel.retained_start_0 <= index < panel.retained_end_0
    ]
    trimmed = [index for index in range(len(parent_sequence)) if index not in included]
    retained_sequence = parent_sequence[panel.retained_start_0 : panel.retained_end_0]
    return baserender.Record(
        id=panel.payload_trim_id,
        alphabet="DNA",
        sequence=parent_sequence,
        features=(
            _retained_feature(
                feature_id=TOP_RETAINED_FEATURE_ID,
                retained_sequence=retained_sequence,
                panel=panel,
                strand="fwd",
            ),
            _retained_feature(
                feature_id=BOTTOM_RETAINED_FEATURE_ID,
                retained_sequence=_reverse_complement(retained_sequence),
                panel=panel,
                strand="rev",
            ),
        ),
        effects=tuple(
            _motif_logo_effect(layer, columns=columns, panel=panel, site_start=site_start, site_end=site_end)
            for layer in logo_layers
        ),
        display=baserender.Display(tag_labels={"tf:tetR_trim": "retained TetR payload"}),
        meta=sequence_row_visual_meta(
            complement_sequence=_complement(parent_sequence),
            included=included,
            trimmed=trimmed,
            panel=panel,
            site_start=site_start,
            site_end=site_end,
        ),
    )


def observed_sequence_for_panel(columns: Sequence[PwmLogoColumn], panel: PwmTrimPanel) -> str:
    return "".join(
        column.parent_base if panel.retained_start_0 <= column.parent_position_0 < panel.retained_end_0 else "N"
        for column in columns
    )


def _retained_feature(
    *,
    feature_id: str,
    retained_sequence: str,
    panel: PwmTrimPanel,
    strand: str,
):
    return baserender.Feature(
        id=feature_id,
        kind="kmer",
        span=baserender.Span(start=panel.retained_start_0, end=panel.retained_end_0, strand=strand),
        label=retained_sequence,
        tags=("tf:tetR_trim",),
        attrs={"style_token": "tf:tetR_trim"},
        render={},
    )


def _motif_logo_effect(
    layer: PwmLogoLayer,
    *,
    columns: Sequence[PwmLogoColumn],
    panel: PwmTrimPanel,
    site_start: int,
    site_end: int,
):
    observed = observed_sequence_for_panel(columns, panel)
    target_feature_id = TOP_RETAINED_FEATURE_ID
    if layer.strand == "-":
        observed = _complement(observed)[::-1]
        target_feature_id = BOTTOM_RETAINED_FEATURE_ID
    return baserender.Effect(
        kind="motif_logo",
        target={"feature_id": target_feature_id},
        params={
            "matrix": [[float(value) for value in row] for row in layer.matrix],
            "render_span": {"start": site_start, "end": site_end},
            "observed_sequence_5to3": observed,
        },
        render={"priority": 20 + layer.occurrence_rank},
    )


def _complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan")).upper()


def _reverse_complement(sequence: str) -> str:
    return _complement(sequence)[::-1]


__all__ = [
    "PwmLogoColumn",
    "PwmLogoLayer",
    "build_pwm_baserender_record",
    "observed_sequence_for_panel",
]
