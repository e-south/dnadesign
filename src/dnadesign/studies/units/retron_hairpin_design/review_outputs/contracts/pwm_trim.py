"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/pwm_trim.py

PWM trim-panel parsing for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...compiler.exceptions import RetronMsdCompilerError
from ..pwm.retention import PwmMotifOccurrence, validate_declared_trim_windows


@dataclass(frozen=True)
class PwmTrimPanel:
    payload_trim_id: str
    label: str
    retained_start_0: int
    retained_end_0: int
    trim_5p_nt: int
    trim_3p_nt: int
    retained_information_fraction: float


@dataclass(frozen=True)
class PwmTrimContext:
    panels: tuple[PwmTrimPanel, ...]
    parent_payload_sequence: str
    motif_occurrences: tuple[PwmMotifOccurrence, ...]


def parse_pwm_trim_context(
    *,
    pwm_family: Mapping[str, Any],
    design_set: Mapping[str, Any],
    meme_pwm_path: Path,
) -> PwmTrimContext:
    payload_trims = _require_mapping(design_set.get("payload_trims"), "design-set payload_trims")
    panels = tuple(
        _parse_pwm_panel(panel, payload_trims=payload_trims)
        for panel in _require_sequence(pwm_family.get("panels"), "PWM panels")
    )
    unknown_trim_ids = {panel.payload_trim_id for panel in panels} - set(payload_trims)
    if unknown_trim_ids:
        raise RetronMsdCompilerError(
            "Retron PWM triptych payload_trim_id values must be declared in design-set payload_trims: "
            f"{sorted(unknown_trim_ids)}"
        )
    parent_sequence, motif_occurrences = _parse_parent_payload(design_set, payload_trims=payload_trims)
    validate_declared_trim_windows(
        panels,
        parent_length=len(parent_sequence),
        motif_occurrences=motif_occurrences,
        meme_pwm_path=meme_pwm_path,
    )
    return PwmTrimContext(
        panels=panels,
        parent_payload_sequence=parent_sequence,
        motif_occurrences=motif_occurrences,
    )


def _parse_pwm_panel(raw: object, *, payload_trims: Mapping[str, Any]) -> PwmTrimPanel:
    panel = _require_mapping(raw, "PWM panel")
    span = _require_mapping(panel.get("retained_parent_span_0"), "PWM panel retained_parent_span_0")
    payload_trim_id = str(panel.get("payload_trim_id") or "").strip()
    trim = _require_mapping(payload_trims.get(payload_trim_id), f"design-set {payload_trim_id} payload trim")
    trim_span = _require_mapping(
        trim.get("retained_parent_span_0"), f"design-set {payload_trim_id} retained_parent_span_0"
    )
    if (int(span.get("start")), int(span.get("end"))) != (int(trim_span.get("start")), int(trim_span.get("end"))):
        raise RetronMsdCompilerError(
            f"Retron PWM panel retained_parent_span_0 must match the design-set payload trim span for {payload_trim_id}"
        )
    return PwmTrimPanel(
        payload_trim_id=payload_trim_id,
        label=str(panel.get("label") or "").strip(),
        retained_start_0=int(span.get("start")),
        retained_end_0=int(span.get("end")),
        trim_5p_nt=int(trim.get("trim_5p_nt")),
        trim_3p_nt=int(trim.get("trim_3p_nt")),
        retained_information_fraction=float(trim.get("retained_information_fraction")),
    )


def _parse_parent_payload(
    design_set: Mapping[str, Any],
    *,
    payload_trims: Mapping[str, Any],
) -> tuple[str, tuple[PwmMotifOccurrence, ...]]:
    parent_payload = _require_mapping(design_set.get("parent_payload"), "design-set parent_payload")
    parent_sequence = str(parent_payload.get("source_sequence_5to3") or "").strip().upper()
    if not parent_sequence:
        raise RetronMsdCompilerError("Retron review parent_payload.source_sequence_5to3 cannot be blank")
    full_payload = _find_full_payload_trim(payload_trims)
    if (
        full_payload is not None
        and parent_sequence != str(full_payload.get("exact_sequence_5to3") or "").strip().upper()
    ):
        raise RetronMsdCompilerError(
            "Retron review parent_payload.source_sequence_5to3 must match the untrimmed payload exact_sequence_5to3"
        )
    motif_occurrences = tuple(
        PwmMotifOccurrence.from_mapping(_require_mapping(raw, "parent_payload motif_occurrence"))
        for raw in _require_sequence(parent_payload.get("motif_occurrences"), "parent_payload motif_occurrences")
    )
    return parent_sequence, motif_occurrences


def _find_full_payload_trim(payload_trims: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = []
    for trim_id, raw_trim in payload_trims.items():
        trim = _require_mapping(raw_trim, f"design-set {trim_id} payload trim")
        span = _require_mapping(trim.get("retained_parent_span_0"), f"design-set {trim_id} retained_parent_span_0")
        start = int(span.get("start"))
        end = int(span.get("end"))
        sequence = str(trim.get("exact_sequence_5to3") or "")
        if start == 0 and end == len(sequence):
            candidates.append(trim)
    if len(candidates) > 1:
        raise RetronMsdCompilerError(
            f"Retron design set must declare at most one untrimmed payload, found {len(candidates)}"
        )
    return candidates[0] if candidates else None


def _require_mapping(raw: object, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RetronMsdCompilerError(f"Retron review output expected mapping for {label}")
    return raw


def _require_sequence(raw: object, label: str) -> list[object]:
    if not isinstance(raw, list):
        raise RetronMsdCompilerError(f"Retron review output expected list for {label}")
    return raw


__all__ = ["PwmTrimContext", "PwmTrimPanel", "parse_pwm_trim_context"]
