"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/contracts.py

Contracts for selected OPAL candidates and cloning strategies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Mapping

from dnadesign.opal import RestrictionSiteSpec

_LOWER_DNA = re.compile(r"[acgt]+")
_UPPER_DNA = re.compile(r"[ACGT]+")
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")


def _require_non_empty_text(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def require_nonnegative_integer(value: object, *, field: str) -> int:
    """Parse one exact integer without accepting booleans or lossy coercion."""

    try:
        parsed = _exact_integer(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


def require_positive_integer(value: object, *, field: str) -> int:
    """Parse one exact positive integer without accepting lossy coercion."""

    try:
        parsed = _exact_integer(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def optional_nonnegative_integer(value: object, *, field: str) -> int | None:
    """Parse an optional exact non-negative integer; blank text means absent."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return require_nonnegative_integer(value, field=field)


def _exact_integer(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer value")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if _INTEGER_TEXT.fullmatch(text) is not None:
            return int(text)
    raise ValueError("value is not an exact integer")


@dataclass(frozen=True)
class CloningStrategy:
    """Versioned transform that wraps promoter cores for synthesis ordering."""

    name: str
    version: str
    left_flank: str
    right_flank: str
    expected_core_length: int
    restriction_sites: tuple[RestrictionSiteSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_text(self.name, field="name"))
        object.__setattr__(self, "version", _require_non_empty_text(self.version, field="version"))
        object.__setattr__(self, "left_flank", _require_non_empty_text(self.left_flank, field="left_flank"))
        object.__setattr__(self, "right_flank", _require_non_empty_text(self.right_flank, field="right_flank"))
        if not _LOWER_DNA.fullmatch(self.left_flank):
            raise ValueError("left_flank must be lowercase acgt")
        if not _LOWER_DNA.fullmatch(self.right_flank):
            raise ValueError("right_flank must be lowercase acgt")
        object.__setattr__(
            self,
            "expected_core_length",
            require_positive_integer(self.expected_core_length, field="expected_core_length"),
        )
        object.__setattr__(
            self,
            "restriction_sites",
            tuple(
                site if isinstance(site, RestrictionSiteSpec) else RestrictionSiteSpec.from_mapping(site)
                for site in self.restriction_sites
            ),
        )

    @property
    def strategy_id(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def expected_final_length(self) -> int:
        return len(self.left_flank) + self.expected_core_length + len(self.right_flank)


@dataclass(frozen=True)
class SelectionMembership:
    """One target view's reason for including a candidate in the logical batch."""

    selection_view_id: str
    rank: int
    score: float | None = None
    score_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_view_id",
            _require_non_empty_text(self.selection_view_id, field="selection_view_id"),
        )
        object.__setattr__(
            self,
            "rank",
            require_positive_integer(self.rank, field="selection membership rank"),
        )
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        if self.score_ref is not None:
            object.__setattr__(self, "score_ref", _require_non_empty_text(self.score_ref, field="score_ref"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SelectionMembership:
        return cls(
            selection_view_id=str(value["selection_view_id"]),
            rank=require_positive_integer(value["rank"], field="selection membership rank"),
            score=None if value.get("score") is None else float(value["score"]),
            score_ref=None if value.get("score_ref") is None else str(value["score_ref"]),
        )


@dataclass(frozen=True)
class SelectedCandidate:
    """Selected OPAL promoter candidate plus study-owned order alias."""

    campaign_slug: str
    selection_memberships: tuple[SelectionMembership, ...]
    as_of_round: int
    run_id: str
    selection_rank: int
    id: str
    sequence: str
    synthesis_name: str
    selection_source: str = "selected_csv"
    selection_epoch: str = "opal_model_round"
    assay_batch_index: int | None = None
    model_as_of_round: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_slug", _require_non_empty_text(self.campaign_slug, field="campaign_slug"))
        memberships = tuple(
            row if isinstance(row, SelectionMembership) else SelectionMembership.from_mapping(row)
            for row in self.selection_memberships
        )
        if not memberships:
            raise ValueError("selection_memberships must contain at least one target view")
        view_ids = [row.selection_view_id for row in memberships]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("selection_memberships contains duplicate selection_view_id values")
        object.__setattr__(self, "selection_memberships", memberships)
        object.__setattr__(self, "run_id", _require_non_empty_text(self.run_id, field="run_id"))
        object.__setattr__(self, "id", _require_non_empty_text(self.id, field="id"))
        object.__setattr__(self, "synthesis_name", _require_non_empty_text(self.synthesis_name, field="synthesis_name"))
        object.__setattr__(
            self,
            "selection_source",
            _require_non_empty_text(self.selection_source, field="selection_source"),
        )
        object.__setattr__(
            self,
            "selection_epoch",
            _require_non_empty_text(self.selection_epoch, field="selection_epoch"),
        )
        object.__setattr__(self, "sequence", _require_non_empty_text(self.sequence, field="sequence"))
        object.__setattr__(
            self,
            "as_of_round",
            require_nonnegative_integer(self.as_of_round, field="as_of_round"),
        )
        object.__setattr__(
            self,
            "selection_rank",
            require_positive_integer(self.selection_rank, field="selection_rank"),
        )
        if self.assay_batch_index is not None:
            object.__setattr__(
                self,
                "assay_batch_index",
                require_nonnegative_integer(self.assay_batch_index, field="assay_batch_index"),
            )
        if self.model_as_of_round is None and self.selection_epoch == "opal_model_round":
            object.__setattr__(self, "model_as_of_round", self.as_of_round)
        elif self.model_as_of_round is not None:
            object.__setattr__(
                self,
                "model_as_of_round",
                require_nonnegative_integer(self.model_as_of_round, field="model_as_of_round"),
            )

    @property
    def selection_view_ids(self) -> tuple[str, ...]:
        return tuple(row.selection_view_id for row in self.selection_memberships)


def validate_promoter_core(sequence: str, *, expected_length: int, candidate_id: str) -> str:
    core = _require_non_empty_text(sequence, field="sequence")
    if not _UPPER_DNA.fullmatch(core):
        raise ValueError(f"candidate {candidate_id} core sequence must be uppercase ACGT")
    if len(core) != expected_length:
        raise ValueError(f"candidate {candidate_id} core sequence expected length {expected_length}, got {len(core)}")
    return core
