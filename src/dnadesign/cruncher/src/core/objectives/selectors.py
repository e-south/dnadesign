"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/objectives/selectors.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import BestHitSelectorSpec, TopKDistinctSelectorSpec, WindowHit

_EPS = 1.0e-12


class HitSelector(Protocol):
    def select(self, hits: Sequence[WindowHit], spec: object) -> tuple[WindowHit, ...]: ...


class BestHitSelector:
    def select(self, hits: Sequence[WindowHit], spec: BestHitSelectorSpec) -> tuple[WindowHit, ...]:
        _ = spec
        if not hits:
            return ()
        best = max(
            hits,
            key=lambda hit: (
                hit.scaled_score,
                -hit.start,
                1 if hit.strand == "+" else 0,
            ),
        )
        return (best,)


@dataclass(frozen=True)
class _SelectionState:
    count: int
    bottleneck: float
    total: float
    indices: tuple[int, ...]
    signature: tuple[tuple[int, int, int], ...]


def _hit_key(hit: WindowHit) -> tuple[int, int, int]:
    return (hit.start, hit.end, 0 if hit.strand == "+" else 1)


def _insert_signature(
    signature: tuple[tuple[int, int, int], ...],
    key: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    if not signature:
        return (key,)
    inserted = False
    merged: list[tuple[int, int, int]] = []
    for item in signature:
        if not inserted and key < item:
            merged.append(key)
            inserted = True
        merged.append(item)
    if not inserted:
        merged.append(key)
    return tuple(merged)


def _is_better(
    candidate: _SelectionState | None,
    current: _SelectionState | None,
    *,
    compare_count: bool = False,
) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    if compare_count and candidate.count != current.count:
        return candidate.count > current.count
    if candidate.bottleneck > current.bottleneck + _EPS:
        return True
    if current.bottleneck > candidate.bottleneck + _EPS:
        return False
    if candidate.total > current.total + _EPS:
        return True
    if current.total > candidate.total + _EPS:
        return False
    return candidate.signature < current.signature


class TopKDistinctSelector:
    def select(self, hits: Sequence[WindowHit], spec: TopKDistinctSelectorSpec) -> tuple[WindowHit, ...]:
        if spec.distinctness.strand_rule != "collapse_same_locus":
            raise ValueError(f"Unsupported strand rule: {spec.distinctness.strand_rule!r}")
        if spec.copies < 1:
            raise ValueError("top_k_distinct selector requires copies >= 1")
        if not hits:
            return ()
        if spec.distinctness.mode == "interval":
            return self._select_interval_distinct(hits, spec)
        if spec.distinctness.mode == "offset":
            return self._select_offset_distinct(hits, spec)
        raise ValueError(f"Unsupported distinctness mode: {spec.distinctness.mode!r}")

    def _select_interval_distinct(
        self,
        hits: Sequence[WindowHit],
        spec: TopKDistinctSelectorSpec,
    ) -> tuple[WindowHit, ...]:
        ordered = sorted(
            hits,
            key=lambda hit: (hit.end, hit.start, 0 if hit.strand == "+" else 1, -hit.scaled_score),
        )
        predecessors = self._predecessors(ordered, min_gap=int(spec.distinctness.min_gap))
        k = int(spec.copies)
        n = len(ordered)
        dp: list[list[_SelectionState | None]] = [[None for _ in range(k + 1)] for _ in range(n + 1)]
        dp[0][0] = _SelectionState(count=0, bottleneck=float("inf"), total=0.0, indices=(), signature=())

        for j in range(1, n + 1):
            hit = ordered[j - 1]
            hit_key = _hit_key(hit)
            for c in range(k + 1):
                best = dp[j - 1][c]
                if c > 0:
                    prev = dp[predecessors[j - 1] + 1][c - 1]
                    if prev is not None:
                        candidate = _SelectionState(
                            count=prev.count + 1,
                            bottleneck=(
                                min(prev.bottleneck, hit.scaled_score) if prev.count else float(hit.scaled_score)
                            ),
                            total=float(prev.total + hit.scaled_score),
                            indices=prev.indices + (j - 1,),
                            signature=_insert_signature(prev.signature, hit_key),
                        )
                        if _is_better(candidate, best):
                            best = candidate
                dp[j][c] = best

        best_state: _SelectionState | None = None
        for c in range(k, -1, -1):
            candidate = dp[n][c]
            if _is_better(candidate, best_state, compare_count=True):
                best_state = candidate
            if best_state is not None and best_state.count == k:
                break

        if best_state is None:
            return ()
        return tuple(
            sorted(
                (ordered[idx] for idx in best_state.indices),
                key=lambda hit: (hit.start, hit.end, hit.strand),
            )
        )

    def _select_offset_distinct(
        self,
        hits: Sequence[WindowHit],
        spec: TopKDistinctSelectorSpec,
    ) -> tuple[WindowHit, ...]:
        collapsed = self._collapse_same_locus_hits(hits)
        ordered = sorted(
            collapsed,
            key=lambda hit: (hit.start, 0 if hit.strand == "+" else 1, -hit.scaled_score),
        )
        predecessors = self._offset_predecessors(ordered, min_gap=int(spec.distinctness.min_gap))
        return self._select_with_dp(ordered, predecessors=predecessors, copies=int(spec.copies))

    @staticmethod
    def _collapse_same_locus_hits(hits: Sequence[WindowHit]) -> tuple[WindowHit, ...]:
        best_by_locus: dict[tuple[int, int], WindowHit] = {}
        for hit in hits:
            locus = (int(hit.start), int(hit.end))
            current = best_by_locus.get(locus)
            if current is None:
                best_by_locus[locus] = hit
                continue
            if hit.scaled_score > current.scaled_score + _EPS or (
                abs(hit.scaled_score - current.scaled_score) <= _EPS
                and (hit.start, hit.end, 0 if hit.strand == "+" else 1)
                < (current.start, current.end, 0 if current.strand == "+" else 1)
            ):
                best_by_locus[locus] = hit
        return tuple(best_by_locus.values())

    def _select_with_dp(
        self,
        hits: Sequence[WindowHit],
        *,
        predecessors: Sequence[int],
        copies: int,
    ) -> tuple[WindowHit, ...]:
        k = int(copies)
        n = len(hits)
        dp: list[list[_SelectionState | None]] = [[None for _ in range(k + 1)] for _ in range(n + 1)]
        dp[0][0] = _SelectionState(count=0, bottleneck=float("inf"), total=0.0, indices=(), signature=())

        for j in range(1, n + 1):
            hit = hits[j - 1]
            hit_key = _hit_key(hit)
            for c in range(k + 1):
                best = dp[j - 1][c]
                if c > 0:
                    prev = dp[predecessors[j - 1] + 1][c - 1]
                    if prev is not None:
                        candidate = _SelectionState(
                            count=prev.count + 1,
                            bottleneck=(
                                min(prev.bottleneck, hit.scaled_score) if prev.count else float(hit.scaled_score)
                            ),
                            total=float(prev.total + hit.scaled_score),
                            indices=prev.indices + (j - 1,),
                            signature=_insert_signature(prev.signature, hit_key),
                        )
                        if _is_better(candidate, best):
                            best = candidate
                dp[j][c] = best

        best_state: _SelectionState | None = None
        for c in range(k, -1, -1):
            candidate = dp[n][c]
            if _is_better(candidate, best_state, compare_count=True):
                best_state = candidate
            if best_state is not None and best_state.count == k:
                break

        if best_state is None:
            return ()
        return tuple(
            sorted(
                (hits[idx] for idx in best_state.indices),
                key=lambda hit: (hit.start, hit.end, hit.strand),
            )
        )

    @staticmethod
    def _predecessors(hits: Sequence[WindowHit], *, min_gap: int) -> list[int]:
        predecessors: list[int] = []
        for j, hit in enumerate(hits):
            predecessor = -1
            for i in range(j - 1, -1, -1):
                earlier = hits[i]
                if earlier.end + min_gap <= hit.start:
                    predecessor = i
                    break
            predecessors.append(predecessor)
        return predecessors

    @staticmethod
    def _offset_predecessors(hits: Sequence[WindowHit], *, min_gap: int) -> list[int]:
        predecessors: list[int] = []
        for j, hit in enumerate(hits):
            predecessor = -1
            for i in range(j - 1, -1, -1):
                earlier = hits[i]
                if earlier.start + min_gap < hit.start:
                    predecessor = i
                    break
            predecessors.append(predecessor)
        return predecessors
