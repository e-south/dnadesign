"""Contracts for selected OPAL candidates and cloning strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dnadesign.opal import RestrictionSiteSpec

_LOWER_DNA = re.compile(r"[acgt]+")
_UPPER_DNA = re.compile(r"[ACGT]+")


def _require_non_empty_text(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


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
        if int(self.expected_core_length) <= 0:
            raise ValueError("expected_core_length must be positive")
        object.__setattr__(self, "expected_core_length", int(self.expected_core_length))
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
class SelectedCandidate:
    """Selected OPAL promoter candidate plus study-owned order alias."""

    campaign_slug: str
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
        if int(self.as_of_round) < 0:
            raise ValueError("as_of_round must be non-negative")
        if int(self.selection_rank) <= 0:
            raise ValueError("selection_rank must be positive")
        object.__setattr__(self, "as_of_round", int(self.as_of_round))
        object.__setattr__(self, "selection_rank", int(self.selection_rank))
        if self.assay_batch_index is not None:
            if int(self.assay_batch_index) < 0:
                raise ValueError("assay_batch_index must be non-negative when provided")
            object.__setattr__(self, "assay_batch_index", int(self.assay_batch_index))
        if self.model_as_of_round is None and self.selection_epoch == "opal_model_round":
            object.__setattr__(self, "model_as_of_round", int(self.as_of_round))
        elif self.model_as_of_round is not None:
            if int(self.model_as_of_round) < 0:
                raise ValueError("model_as_of_round must be non-negative when provided")
            object.__setattr__(self, "model_as_of_round", int(self.model_as_of_round))


def validate_promoter_core(sequence: str, *, expected_length: int, candidate_id: str) -> str:
    core = _require_non_empty_text(sequence, field="sequence")
    if not _UPPER_DNA.fullmatch(core):
        raise ValueError(f"candidate {candidate_id} core sequence must be uppercase ACGT")
    if len(core) != expected_length:
        raise ValueError(f"candidate {candidate_id} core sequence expected length {expected_length}, got {len(core)}")
    return core
