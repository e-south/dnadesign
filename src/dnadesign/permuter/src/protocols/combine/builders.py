"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/combine/builders.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


def _key_for_combo(events: List[Tuple[int, str, str, float]]) -> str:
    # Canonical AA key like "K12D|G45A|…", sorted by position ascending
    parts = [
        f"{wt}{pos}{alt}" for (pos, wt, alt, _) in sorted(events, key=lambda x: x[0])
    ]
    return "|".join(parts)


def _rank_score_for(
    picks: List[Tuple[int, str, str, float]],
    *,
    objective: str,
) -> float:
    """
    Compute the sampler's ranking objective for a combo.
    Implemented objectives:
      • 'sum_of_singles'  → sum of single-event scores (current default)
    """
    if objective == "sum_of_singles":
        return float(sum(x[3] for x in picks))
    # No silent fallbacks: be explicit about unsupported objectives.
    raise ValueError(f"combine_aa.random_sample: unknown rank_objective={objective!r}")


def random_sample(
    elite: List[Tuple[int, str, str, float]],
    combine_cfg: Dict,
    rng: np.random.Generator,
) -> List[Tuple[List[Tuple[int, str, str, float]], float]]:
    """
    Sample combos without position collisions, with per-k caps and a global budget.
    Returns a list of (events, proposal_score) sorted by (score desc, key asc).
    """
    comb = (combine_cfg or {}).get("combine", {})
    k_min = int(comb.get("k_min", 0))
    k_max = int(comb.get("k_max", 0))
    strategy = str(comb.get("strategy", "random"))
    if strategy != "random":
        raise ValueError("combine_aa: only strategy='random' is implemented in v0.1")
    budget_total = int(comb.get("budget_total", 0))
    if k_min < 1 or k_max < k_min:
        raise ValueError(
            "combine_aa: combine.k_min/k_max must be positive and k_max ≥ k_min"
        )
    if budget_total <= 0:
        raise ValueError("combine_aa: combine.budget_total must be > 0")
    rnd = comb.get("random", {}) or {}
    per_k = rnd.get("samples_per_k", {})
    rank_objective = str(rnd.get("rank_objective", "sum_of_singles"))
    # Normalize keys to int
    per_k_int: Dict[int, int] = {
        int(k): int(v) for k, v in (per_k.items() if per_k else [])
    }
    # Fallback: even split of budget across k when per‑k targets aren’t specified
    if not per_k_int:
        k_vals = list(range(k_min, k_max + 1))
        if k_vals:
            base = max(0, budget_total // len(k_vals))
            extra = max(0, budget_total % len(k_vals))
            per_k_int = {
                k: base + (1 if i < extra else 0) for i, k in enumerate(k_vals)
            }

    results: List[Tuple[List[Tuple[int, str, str, float]], float]] = []
    seen_keys: set[str] = set()
    remaining_budget = budget_total

    # Convenience: array of candidate indices; allow multiple alts per position
    n = len(elite)
    if n == 0:
        return []

    for k in range(k_min, k_max + 1):
        target = int(per_k_int.get(k, 0))
        if target <= 0:
            continue
        count_k = 0
        # Guard: if even choosing distinct positions is impossible, skip fast
        if len(set(pos for pos, *_ in elite)) < k:
            continue

        # Try to draw up to 'target' combos at this k
        attempts = 0
        # Soft attempt cap avoids spinning when target is too ambitious
        max_attempts = max(10_000, target * 50)
        while count_k < target and remaining_budget > 0 and attempts < max_attempts:
            attempts += 1
            # Sample k distinct indices; then check for position collisions
            idxs = rng.choice(n, size=k, replace=False).tolist()
            picks = [elite[i] for i in idxs]
            positions = [p[0] for p in picks]
            if len(set(positions)) != len(positions):
                continue  # collision → resample

            key = _key_for_combo(picks)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            score = _rank_score_for(picks, objective=rank_objective)
            results.append((sorted(picks, key=lambda x: x[0]), float(score)))
            count_k += 1
            remaining_budget -= 1
            if remaining_budget <= 0:
                break

    # Stable ordering
    results.sort(key=lambda t: (-float(t[1]), _key_for_combo(t[0])))
    return results


def enumerate_all(
    elite: List[Tuple[int, str, str, float]],
    combine_cfg: Dict,
) -> List[Tuple[List[Tuple[int, str, str, float]], float]]:
    """
    Exhaustively enumerate all non-colliding combos for k in [k_min..k_max].
    Deterministic ordering by (sum_of_singles desc, canonical key asc).
    WARNING: combinatorial explosion — caller must keep p (positions) modest.
    """
    comb = (combine_cfg or {}).get("combine", {})
    k_min = int(comb.get("k_min", 0))
    k_max = int(comb.get("k_max", 0))
    if k_min < 1 or k_max < k_min:
        raise ValueError(
            "combine_aa: combine.k_min/k_max must be positive and k_max ≥ k_min"
        )
    # Ranking objective parity with sampling; default to sum_of_singles
    rnd = comb.get("random", {}) or {}
    objective = str(rnd.get("rank_objective", "sum_of_singles"))

    n = len(elite)
    results: List[Tuple[List[Tuple[int, str, str, float]], float]] = []
    seen_keys: set[str] = set()
    for k in range(k_min, k_max + 1):
        for idxs in combinations(range(n), k):
            picks = [elite[i] for i in idxs]
            positions = [p[0] for p in picks]
            if len(set(positions)) != len(positions):
                continue  # collision on a position
            key = _key_for_combo(picks)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            score = _rank_score_for(picks, objective=objective)
            results.append((sorted(picks, key=lambda x: x[0]), float(score)))
    results.sort(key=lambda t: (-float(t[1]), _key_for_combo(t[0])))
    return results
