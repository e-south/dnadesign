"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_metadata.py

Stage-A TFBS metadata assembly for pool rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .stage_a_types import SelectionMeta


@dataclass(frozen=True)
class TFBSMeta:
    best_hit_score: float
    rank_within_regulator: int
    tier: int
    fimo_start: int
    fimo_stop: int
    fimo_strand: str
    tfbs_core: str
    fimo_matched_sequence: Optional[str]
    selection_meta: SelectionMeta
    selection_policy: str
    selection_alpha: Optional[float]
    selection_similarity: Optional[str]
    selection_shortlist_min: Optional[int]
    selection_shortlist_factor: Optional[int]
    selection_shortlist_max: Optional[int]
    selection_tier_fraction_used: Optional[float]
    selection_tier_limit: Optional[int]
    shortlist_k: Optional[int]
    selection_pool_source: Optional[str]
    tier_target_fraction: Optional[float]
    tier_target_required_unique: Optional[int]
    tier_target_met: Optional[bool]
    tier_target_eligible_unique: int

    def __post_init__(self) -> None:
        if int(self.rank_within_regulator) <= 0:
            raise ValueError("rank_within_regulator must be >= 1.")
        if int(self.tier) < 0:
            raise ValueError("tier must be >= 0.")
        if int(self.tier_target_eligible_unique) < 0:
            raise ValueError("tier_target_eligible_unique must be >= 0.")
        if not self.fimo_strand:
            raise ValueError("fimo_strand is required for TFBS metadata.")

    def to_dict(self) -> dict[str, object]:
        base = {
            "best_hit_score": float(self.best_hit_score),
            "rank_within_regulator": int(self.rank_within_regulator),
            "tier": int(self.tier),
            "fimo_start": int(self.fimo_start),
            "fimo_stop": int(self.fimo_stop),
            "fimo_strand": str(self.fimo_strand),
            "tfbs_core": str(self.tfbs_core),
            "selection_policy": str(self.selection_policy),
            "selection_alpha": self.selection_alpha,
            "selection_similarity": self.selection_similarity,
            "selection_shortlist_min": self.selection_shortlist_min,
            "selection_shortlist_factor": self.selection_shortlist_factor,
            "selection_shortlist_max": self.selection_shortlist_max,
            "selection_tier_fraction_used": self.selection_tier_fraction_used,
            "selection_tier_limit": self.selection_tier_limit,
            "shortlist_k": self.shortlist_k,
            "selection_pool_source": self.selection_pool_source,
            "tier_target_fraction": self.tier_target_fraction,
            "tier_target_required_unique": self.tier_target_required_unique,
            "tier_target_met": self.tier_target_met,
            "tier_target_eligible_unique": int(self.tier_target_eligible_unique),
        }
        if self.fimo_matched_sequence:
            base["fimo_matched_sequence"] = str(self.fimo_matched_sequence)
        base.update(self.selection_meta.to_dict())
        return base
