"""Generic statistical helpers for enrichment tables."""

from __future__ import annotations

import math


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeometric_survival(*, observed: int, population: int, successes: int, draws: int) -> float:
    """One-sided hypergeometric survival probability for enrichment tests."""

    if population <= 0:
        return float("nan")
    if draws < 0 or successes < 0 or draws > population or successes > population:
        return float("nan")
    if observed <= 0:
        return 1.0
    upper = min(successes, draws)
    if observed > upper:
        return 0.0
    denominator = log_comb(population, draws)
    log_terms = [
        log_comb(successes, hits) + log_comb(population - successes, draws - hits) - denominator
        for hits in range(observed, upper + 1)
    ]
    maximum = max(log_terms)
    if not math.isfinite(maximum):
        return float("nan")
    probability = math.exp(maximum) * sum(math.exp(term - maximum) for term in log_terms)
    return float(min(1.0, max(0.0, probability)))


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    denominator = b * c
    numerator = a * d
    if denominator == 0:
        if numerator > 0:
            return float("inf")
        return float("nan")
    return float(numerator / denominator)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    finite_pairs = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    output = [float("nan")] * len(p_values)
    if not finite_pairs:
        return output
    ordered = sorted(finite_pairs, key=lambda item: item[1])
    m = len(ordered)
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        adjusted = min(1.0, p_value * m / float(rank))
        running = min(running, adjusted)
        output[index] = float(running)
    return output
