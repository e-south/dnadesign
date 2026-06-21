"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/retention.py

PWM information-retention helpers for Retron trim review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ...compiler.exceptions import RetronMsdCompilerError


@dataclass(frozen=True)
class PwmMotifOccurrence:
    motif_instance_id: str
    start_0: int
    end_0: int
    strand: str
    occurrence_rank: int

    @property
    def width(self) -> int:
        return self.end_0 - self.start_0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PwmMotifOccurrence":
        try:
            occurrence = cls(
                motif_instance_id=str(raw["motif_instance_id"]).strip(),
                start_0=int(raw["start"]),
                end_0=int(raw["end"]),
                strand=str(raw["strand"]).strip(),
                occurrence_rank=int(raw["occurrence_rank"]),
            )
        except KeyError as exc:
            raise RetronMsdCompilerError(f"Retron PWM motif occurrence is missing {exc.args[0]!r}") from exc
        occurrence.validate()
        return occurrence

    def validate(self) -> None:
        if not self.motif_instance_id:
            raise RetronMsdCompilerError("Retron PWM motif occurrence id cannot be blank")
        if self.strand not in {"+", "-"}:
            raise RetronMsdCompilerError(f"Retron PWM motif occurrence strand must be '+' or '-': {self.strand!r}")
        if self.end_0 <= self.start_0:
            raise RetronMsdCompilerError(f"Retron PWM motif occurrence span must be non-empty: {self}")


@dataclass(frozen=True)
class PwmRetainedSpan:
    start_0: int
    end_0: int
    retained_bits_by_occurrence: tuple[float, ...]
    parent_bits_by_occurrence: tuple[float, ...]

    @property
    def retained_information_bits(self) -> float:
        return sum(self.retained_bits_by_occurrence)

    @property
    def parent_information_bits(self) -> float:
        return sum(self.parent_bits_by_occurrence)

    @property
    def retained_information_fraction(self) -> float:
        parent = self.parent_information_bits
        return 0.0 if parent <= 0 else self.retained_information_bits / parent

    def sequence_from(self, parent_sequence: str) -> str:
        return parent_sequence[self.start_0 : self.end_0]


def load_meme_probability_matrix(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    if not path.is_file():
        raise RetronMsdCompilerError(f"Retron PWM source not found: {path}")
    rows: list[tuple[float, float, float, float]] = []
    in_matrix = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("letter-probability matrix:"):
            in_matrix = True
            continue
        if not in_matrix:
            continue
        parts = stripped.split()
        if len(parts) != 4:
            break
        try:
            a_prob, c_prob, g_prob, t_prob = (float(part) for part in parts)
        except ValueError:
            break
        rows.append((a_prob, c_prob, g_prob, t_prob))
    if not rows:
        raise RetronMsdCompilerError(f"No MEME probability matrix found in Retron PWM source: {path}")
    return tuple(rows)


def load_meme_information_bits(path: Path) -> tuple[float, ...]:
    return tuple(_information_bits(row) for row in load_meme_probability_matrix(path))


def select_best_retained_span(
    *,
    parent_length: int,
    retained_length: int,
    motif_bits: Sequence[float],
    occurrences: Sequence[PwmMotifOccurrence],
) -> PwmRetainedSpan:
    _validate_selector_inputs(parent_length, retained_length, motif_bits, occurrences)
    parent_bits_by_occurrence = tuple(sum(motif_bits) for _occurrence in occurrences)
    best: tuple[float, float, int, PwmRetainedSpan] | None = None
    parent_center = parent_length / 2.0
    for start in range(0, parent_length - retained_length + 1):
        end = start + retained_length
        retained_by_occurrence = tuple(
            _retained_bits_for_occurrence(start, end, motif_bits=motif_bits, occurrence=occurrence)
            for occurrence in occurrences
        )
        retained = PwmRetainedSpan(
            start_0=start,
            end_0=end,
            retained_bits_by_occurrence=retained_by_occurrence,
            parent_bits_by_occurrence=parent_bits_by_occurrence,
        )
        center_penalty = abs(((start + end) / 2.0) - parent_center)
        candidate_key = (retained.retained_information_bits, -center_penalty, -start, retained)
        if best is None or candidate_key[:3] > best[:3]:
            best = candidate_key
    if best is None:
        raise RetronMsdCompilerError("Retron PWM retention selector found no candidate spans")
    return best[3]


def validate_declared_trim_windows(
    panels: Sequence[object],
    *,
    parent_length: int,
    motif_occurrences: Sequence[PwmMotifOccurrence],
    meme_pwm_path: Path,
) -> None:
    motif_bits = load_meme_information_bits(meme_pwm_path)
    for panel in panels:
        retained_start = int(getattr(panel, "retained_start_0"))
        retained_end = int(getattr(panel, "retained_end_0"))
        retained_length = retained_end - retained_start
        selected = select_best_retained_span(
            parent_length=parent_length,
            retained_length=retained_length,
            motif_bits=motif_bits,
            occurrences=motif_occurrences,
        )
        if (selected.start_0, selected.end_0) != (retained_start, retained_end):
            raise RetronMsdCompilerError(
                f"Retron PWM panel {getattr(panel, 'payload_trim_id')} declares retained span "
                f"[{retained_start},{retained_end}), but dual-site IC selector chooses "
                f"[{selected.start_0},{selected.end_0}) for retained length {retained_length}"
            )


def _validate_selector_inputs(
    parent_length: int,
    retained_length: int,
    motif_bits: Sequence[float],
    occurrences: Sequence[PwmMotifOccurrence],
) -> None:
    if parent_length <= 0:
        raise RetronMsdCompilerError("Retron PWM parent length must be positive")
    if retained_length <= 0 or retained_length > parent_length:
        raise RetronMsdCompilerError(
            f"Retron PWM retained length must be in [1, {parent_length}], got {retained_length}"
        )
    if not motif_bits:
        raise RetronMsdCompilerError("Retron PWM motif_bits cannot be empty")
    if not occurrences:
        raise RetronMsdCompilerError("Retron PWM retention selector requires at least one motif occurrence")
    for occurrence in occurrences:
        occurrence.validate()
        if occurrence.width != len(motif_bits):
            raise RetronMsdCompilerError(
                f"Retron PWM occurrence {occurrence.motif_instance_id} width {occurrence.width} "
                f"does not match PWM width {len(motif_bits)}"
            )
        if occurrence.start_0 < 0 or occurrence.end_0 > parent_length:
            raise RetronMsdCompilerError(
                f"Retron PWM occurrence {occurrence.motif_instance_id} is outside parent length {parent_length}"
            )


def _retained_bits_for_occurrence(
    start: int,
    end: int,
    *,
    motif_bits: Sequence[float],
    occurrence: PwmMotifOccurrence,
) -> float:
    retained = 0.0
    for motif_index, bits in enumerate(motif_bits):
        parent_position = _parent_position_for_motif_index(occurrence, motif_index)
        if start <= parent_position < end:
            retained += float(bits)
    return retained


def _parent_position_for_motif_index(occurrence: PwmMotifOccurrence, motif_index: int) -> int:
    if occurrence.strand == "-":
        return occurrence.end_0 - 1 - motif_index
    return occurrence.start_0 + motif_index


def _information_bits(row: Sequence[float]) -> float:
    return max(0.0, sum(value * math.log2(value / 0.25) for value in row if value > 0))


__all__ = [
    "PwmMotifOccurrence",
    "PwmRetainedSpan",
    "load_meme_information_bits",
    "load_meme_probability_matrix",
    "select_best_retained_span",
    "validate_declared_trim_windows",
]
