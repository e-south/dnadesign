"""
Ranking and dedupe policy for released-product target-search hits.
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.snapback.released_search_models import ReleasedTargetSearchHit
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry

_SNAPBACK_TIER_RANK = {
    "tier1": 0,
    "tier2": 1,
    "tier3": 2,
    None: 3,
}
_COMMERCIAL_CONFIDENCE_RANK = {
    "primary_vendor_current": 0,
    "secondary_vendor_current": 1,
    "legacy_vendor_page": 2,
    None: 3,
}


@dataclass(frozen=True)
class ReleasedRankingPolicy:
    """Named ranking and dedupe policy for released-product target-search hits."""

    target: ReleasedFinalTargetGeometry | None = None

    @staticmethod
    def upstream_retained_duplex_bp(hit: ReleasedTargetSearchHit) -> int:
        return min(hit.projection.retained_partner_length_nt, hit.nick_boundary_from_left)

    @classmethod
    def effective_stem_bp(cls, hit: ReleasedTargetSearchHit) -> int:
        return cls.upstream_retained_duplex_bp(hit) + hit.final_candidate.paired_bp

    def dedupe_key(self, hit: ReleasedTargetSearchHit) -> tuple[str, str, str]:
        active_product_sequence = hit.projection.active_product_sequence
        effective_stem_bp = self.effective_stem_bp(hit)
        cap_nt = hit.final_candidate.cap_nt
        return (
            hit.active_strand,
            active_product_sequence[:effective_stem_bp],
            active_product_sequence[effective_stem_bp : effective_stem_bp + cap_nt],
        )

    def dedupe_ranked_hits(self, ranked_hits: list[ReleasedTargetSearchHit]) -> list[ReleasedTargetSearchHit]:
        deduped_hits: list[ReleasedTargetSearchHit] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for hit in ranked_hits:
            key = self.dedupe_key(hit)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_hits.append(hit)
        return deduped_hits

    def exact_hit_key(self, hit: ReleasedTargetSearchHit) -> tuple[object, ...]:
        nick_selection = hit.nickase.selection
        nick_warning_codes = nick_selection.warning_codes if nick_selection is not None else []
        return (
            hit.extra_target_strand_nick_count,
            hit.extra_nick_event_count,
            (
                _SNAPBACK_TIER_RANK[nick_selection.snapback_tier if nick_selection is not None else None],
                (
                    0
                    if nick_selection is not None and nick_selection.outside_site is True
                    else 1
                    if nick_selection is not None
                    else 2
                ),
                -(hit.nickase.motif_len or len(hit.nickase.motif_top_5to3)),
                _COMMERCIAL_CONFIDENCE_RANK[
                    nick_selection.commercial_confidence if nick_selection is not None else None
                ],
                len(nick_warning_codes),
                hit.nickase.variant_id,
            ),
            (
                _COMMERCIAL_CONFIDENCE_RANK[hit.release_enzyme.commercial_confidence],
                len(hit.release_enzyme.warning_codes),
                min(
                    abs(hit.release_enzyme.top_cut_offset - hit.release_enzyme.recognition_len),
                    abs(hit.release_enzyme.bottom_cut_offset - hit.release_enzyme.recognition_len),
                ),
                hit.release_enzyme.recognition_len,
                hit.release_enzyme.variant_id,
            ),
            hit.projection.retained_partner_length_nt,
            hit.active_product_length_nt,
            hit.precursor_length_nt,
            hit.sacrificial_downstream_tail_nt,
            hit.precursor_top_strand,
            hit.nickase_variant_id,
            hit.release_variant_id,
        )

    def near_hit_key(self, hit: ReleasedTargetSearchHit) -> tuple[object, ...]:
        if self.target is None:
            raise ValueError("ReleasedRankingPolicy.near_hit_key requires a target geometry.")
        target_effective_stem_bp = self.target.nick_boundary_from_left + self.target.paired_bp
        target_input_length = target_effective_stem_bp + self.target.cap_nt
        return (
            abs(self.effective_stem_bp(hit) - target_effective_stem_bp),
            abs(hit.nick_boundary_from_left - self.target.nick_boundary_from_left),
            abs(hit.active_product_input_length_nt - target_input_length),
            *self.exact_hit_key(hit),
        )

    def rank_hits(self, hits: list[ReleasedTargetSearchHit], *, exact: bool) -> list[ReleasedTargetSearchHit]:
        ranked = sorted(hits, key=self.exact_hit_key if exact else self.near_hit_key)
        deduped = self.dedupe_ranked_hits(ranked)
        return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(deduped, start=1)]


def rank_hits(
    hits: list[ReleasedTargetSearchHit],
    *,
    target: ReleasedFinalTargetGeometry,
    exact: bool,
) -> list[ReleasedTargetSearchHit]:
    return ReleasedRankingPolicy(target=target).rank_hits(hits, exact=exact)


__all__ = ["ReleasedRankingPolicy", "rank_hits"]
