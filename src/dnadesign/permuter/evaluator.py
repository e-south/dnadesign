"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluator.py

Evaluate variants using specified metric (e.g., via evo_wrapper).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import Dict, List

from evo_wrapper import compute_llr, score_log_likelihood


def evaluate(variants: List[Dict], evaluator: str, metric: str) -> List[float]:
    assert metric in (
        "log_likelihood",
        "log_likelihood_ratio",
    ), f"Unsupported metric: {metric}"
    ref_seq = (
        variants[0]["sequence"]
        if variants[0]["meta_type"] == "reference"
        else variants[0]["sequence"]
    )
    # compute reference LL
    if metric == "log_likelihood":
        return [score_log_likelihood(v["sequence"], evaluator) for v in variants]
    # log-likelihood ratio
    return [compute_llr(ref_seq, v["sequence"], evaluator) for v in variants]
