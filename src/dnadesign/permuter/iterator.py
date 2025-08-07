"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/iterator.py

Perform iterative permute/evaluate/select rounds.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import Dict, List, Tuple

from evaluator import evaluate
from permute_record import permute_record
from selector import select


def iterate(
    elites: List[Dict],
    iterator_cfg: Dict,
    protocol: str,
    params: Dict,
    regions: List,
    lookup_tables: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    max_rounds = iterator_cfg["total_rounds"]
    current_elites = elites
    all_variants = []
    for rnd in range(2, max_rounds + 1):
        new_variants = []
        for ref in current_elites:
            new_variants.extend(
                permute_record(ref, protocol, params, regions, lookup_tables)
            )
        scores = evaluate(new_variants, ref["evaluator"], ref["metric"])
        for v, s in zip(new_variants, scores):
            v["score"] = s
            v["round"] = rnd
        current_elites = select(new_variants, iterator_cfg)
        all_variants = new_variants
    return all_variants, current_elites
