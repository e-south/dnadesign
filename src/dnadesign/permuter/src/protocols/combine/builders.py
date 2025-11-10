"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/combine/builders.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _key_for_combo(events: List[Tuple[int, str, str, float]]) -> str:
    # Canonical AA key like "K12D|G45A|…", sorted by position ascending
    parts = [
        f"{wt}{pos}{alt}" for (pos, wt, alt, _) in sorted(events, key=lambda x: x[0])
    ]
    return "|".join(parts)


def random_sample(
    elite: List[Tuple[int, str, str, float]],
    combine_cfg: Dict,
    rng: np.random.Generator,
) -> List[Tuple[List[Tuple[int, str, str, float]], float]]:
    """
    Sample combos without position collisions, per-k caps and global budget.
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
    per_k = (comb.get("random", {}) or {}).get("samples_per_k", {})
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

    # Build an index of events by position to prevent collisions
    pos_to_idx: Dict[int, List[int]] = {}
    for i, (pos, wt, alt, sc) in enumerate(elite):
        pos_to_idx.setdefault(pos, []).append(i)

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
            score = float(sum(x[3] for x in picks))
            results.append((sorted(picks, key=lambda x: x[0]), score))
            count_k += 1
            remaining_budget -= 1
            if remaining_budget <= 0:
                break

    # Stable ordering
    results.sort(key=lambda t: (-t[1], _key_for_combo(t[0])))
    return results


def beam_search(*_, **__):
    raise NotImplementedError("combine_aa: beam_search not implemented in v0.1")
