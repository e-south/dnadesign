"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/cli/stage_a_summary_rows.py

Build and format Stage-A sampling summary rows for CLI recaps.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from ..core.artifacts.pool import PoolData
from ..core.stage_a.stage_a_summary import PWMSamplingSummary
from .sampling import format_selection_label


def _format_sampling_ratio(value: int, target: int | None) -> str:
    if target is None or target <= 0:
        return str(int(value))
    return f"{int(value)}/{int(target)}"


def _format_count(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_ratio(count: int | None, total: int | None) -> str:
    if count is None:
        return "-"
    if total is None or int(total) <= 0:
        return _format_count(count)
    pct = 100.0 * float(count) / float(total)
    return f"{_format_count(count)} ({pct:.0f}%)"


def _format_sampling_lengths(
    *,
    min_len: int | None,
    median_len: float | None,
    mean_len: float | None,
    max_len: int | None,
    count: int | None,
) -> str:
    if count is None:
        return "-"
    if min_len is None or median_len is None or mean_len is None or max_len is None:
        return f"{int(count)}/-/-/-/-"
    return f"{int(count)}/{int(min_len)}/{median_len:.1f}/{mean_len:.1f}/{int(max_len)}"


def _format_score_stats(
    *,
    min_score: float | None,
    median_score: float | None,
    mean_score: float | None,
    max_score: float | None,
) -> str:
    if min_score is None or median_score is None or mean_score is None or max_score is None:
        return "-"
    return f"{min_score:.2f}/{median_score:.2f}/{mean_score:.2f}/{max_score:.2f}"


def _format_diversity_value(value: float | None, *, show_sign: bool = False) -> str:
    if value is None:
        return "-"
    if show_sign:
        return f"{float(value):+.2f}"
    return f"{float(value):.2f}"


def _format_score_norm_summary(summary) -> str:
    if summary is None:
        return "-"
    top = getattr(summary, "top_candidates", None)
    diversified = getattr(summary, "diversified_candidates", None)
    if top is None or diversified is None:
        return "-"
    return (
        f"top {float(top.min):.2f}/{float(top.median):.2f}/{float(top.max):.2f} | "
        f"div {float(diversified.min):.2f}/{float(diversified.median):.2f}/{float(diversified.max):.2f}"
    )


def _format_score_norm_triplet(summary, *, label: str) -> str:
    if summary is None:
        return "-"
    block = getattr(summary, label, None)
    if block is None:
        return "-"
    return f"{float(block.min):.2f}/{float(block.median):.2f}/{float(block.max):.2f}"


def _format_tier_counts(eligible: list[int] | None, retained: list[int] | None) -> str:
    if not eligible or not retained:
        raise ValueError("Stage-A tier counts are required.")
    if len(eligible) != len(retained):
        raise ValueError("Stage-A tier counts length mismatch.")
    parts = []
    for idx in range(len(eligible)):
        parts.append(f"t{idx} {int(eligible[idx])}/{int(retained[idx])}")
    return " | ".join(parts)


def _format_tier_fraction_label(fraction: float) -> str:
    return f"{float(fraction) * 100:.3f}%"


def _stage_a_row_from_pwm_summary(*, summary: PWMSamplingSummary, pool_name: str) -> dict[str, object]:
    input_name = summary.input_name or pool_name
    if summary.generated is None:
        raise ValueError("Stage-A summary missing generated count.")
    if not summary.regulator:
        raise ValueError("Stage-A summary missing regulator.")
    regulator = summary.regulator
    if summary.candidates_with_hit is None:
        raise ValueError("Stage-A summary missing candidates_with_hit.")
    if summary.eligible_raw is None:
        raise ValueError("Stage-A summary missing eligible_raw.")
    if summary.eligible_unique is None:
        raise ValueError("Stage-A summary missing eligible_unique.")
    if summary.retained is None:
        raise ValueError("Stage-A summary missing retained count.")
    if summary.eligible_tier_counts is None or summary.retained_tier_counts is None:
        raise ValueError("Stage-A summary missing tier counts.")
    if len(summary.eligible_tier_counts) != len(summary.retained_tier_counts):
        raise ValueError("Stage-A summary tier counts length mismatch.")
    if summary.tier_fractions is None:
        raise ValueError("Stage-A summary missing tier fractions.")
    tier_fractions = list(summary.tier_fractions)
    if len(summary.retained_tier_counts) not in {len(tier_fractions), len(tier_fractions) + 1}:
        raise ValueError("Stage-A summary tier fraction/count length mismatch.")
    if summary.selection_policy is None:
        raise ValueError("Stage-A summary missing selection policy.")
    if summary.diversity is None:
        raise ValueError("Stage-A diversity summary missing.")
    candidates = _format_count(summary.generated)
    has_hit = _format_ratio(summary.candidates_with_hit, summary.generated)
    eligible_raw = _format_ratio(summary.eligible_raw, summary.generated)
    eligible_unique = _format_ratio(summary.eligible_unique, summary.eligible_raw)
    retained = _format_count(summary.retained)
    tier_target = "-"
    if summary.tier_target_fraction is not None:
        frac = float(summary.tier_target_fraction)
        frac_label = f"{frac:.3%}"
        if summary.tier_target_met is True:
            tier_target = f"{frac_label} met"
        elif summary.tier_target_met is False:
            tier_target = f"{frac_label} unmet"
    selection_label = format_selection_label(
        policy=str(summary.selection_policy),
        alpha=summary.selection_alpha,
        relevance_norm=summary.selection_relevance_norm or "minmax_raw_score",
    )
    tier_counts = _format_tier_counts(summary.eligible_tier_counts, summary.retained_tier_counts)
    tier_fill = "-"
    if summary.retained_tier_counts:
        tier_labels = [_format_tier_fraction_label(frac) for frac in tier_fractions]
        if len(summary.retained_tier_counts) == len(tier_fractions) + 1:
            tier_labels.append("rest")
        last_idx = None
        for idx, val in enumerate(summary.retained_tier_counts):
            if int(val) > 0:
                last_idx = idx
        if last_idx is not None:
            tier_fill = tier_labels[last_idx]
    length_label = _format_sampling_lengths(
        min_len=summary.retained_len_min,
        median_len=summary.retained_len_median,
        mean_len=summary.retained_len_mean,
        max_len=summary.retained_len_max,
        count=summary.retained,
    )
    score_label = _format_score_stats(
        min_score=summary.retained_score_min,
        median_score=summary.retained_score_median,
        mean_score=summary.retained_score_mean,
        max_score=summary.retained_score_max,
    )
    diversity = summary.diversity
    core_hamming = diversity.core_hamming
    pairwise = core_hamming.pairwise
    if pairwise is None:
        raise ValueError("Stage-A diversity missing pairwise summary.")
    top_pairwise = pairwise.top_candidates
    diversified_pairwise = pairwise.diversified_candidates
    if int(diversified_pairwise.n_pairs) <= 0 or int(top_pairwise.n_pairs) <= 0:
        pairwise_top_label = "n/a"
        pairwise_div_label = "n/a"
    else:
        pairwise_top_label = _format_diversity_value(top_pairwise.median)
        pairwise_div_label = _format_diversity_value(diversified_pairwise.median)
    score_block = diversity.score_quantiles
    if score_block.top_candidates is None or score_block.diversified_candidates is None:
        raise ValueError("Stage-A diversity missing top/diversified score quantiles.")
    score_norm_top = _format_score_norm_triplet(diversity.score_norm_summary, label="top_candidates")
    score_norm_div = _format_score_norm_triplet(diversity.score_norm_summary, label="diversified_candidates")
    if diversity.set_overlap_fraction is None or diversity.set_overlap_swaps is None:
        raise ValueError("Stage-A diversity missing overlap stats.")
    diversity_overlap = f"{float(diversity.set_overlap_fraction) * 100:.1f}%"
    diversity_swaps = str(int(diversity.set_overlap_swaps))
    if diversity.candidate_pool_size is None:
        raise ValueError("Stage-A diversity missing pool size.")
    pool_label = str(int(diversity.candidate_pool_size))
    pool_source = "-"
    if summary.selection_pool_capped:
        pool_label = f"{pool_label}*"
        if summary.selection_pool_cap_value is not None:
            pool_source = f"cap={int(summary.selection_pool_cap_value)}"
    elif summary.selection_pool_rung_fraction_used is not None:
        pool_source = f"rung={float(summary.selection_pool_rung_fraction_used) * 100:.3f}%"
    diversity_pool = pool_label
    return {
        "input_name": str(input_name),
        "regulator": str(regulator),
        "generated": candidates,
        "has_hit": has_hit,
        "eligible_raw": eligible_raw,
        "eligible_unique": eligible_unique,
        "retained": retained,
        "tier_fill": tier_fill,
        "tier_counts": tier_counts,
        "tier_target": tier_target,
        "selection": selection_label,
        "score": score_label,
        "length": length_label,
        "pairwise_top": pairwise_top_label,
        "pairwise_div": pairwise_div_label,
        "score_norm_top": score_norm_top,
        "score_norm_div": score_norm_div,
        "set_overlap": diversity_overlap,
        "set_swaps": diversity_swaps,
        "diversity_pool": diversity_pool,
        "diversity_pool_source": pool_source,
        "tier0_score": summary.tier0_score,
        "tier1_score": summary.tier1_score,
        "tier2_score": summary.tier2_score,
    }


def _stage_a_row_from_sequence_pool(*, pool: PoolData) -> dict[str, object]:
    total = len(pool.sequences)
    lengths = [len(seq) for seq in pool.sequences]
    if lengths:
        arr = np.asarray(lengths, dtype=float)
        length_label = _format_sampling_lengths(
            min_len=int(arr.min()),
            median_len=float(np.median(arr)),
            mean_len=float(arr.mean()),
            max_len=int(arr.max()),
            count=int(total),
        )
    else:
        length_label = _format_sampling_lengths(
            min_len=None,
            median_len=None,
            mean_len=None,
            max_len=None,
            count=int(total),
        )
    return {
        "input_name": str(pool.name),
        "regulator": "-",
        "generated": _format_count(total),
        "has_hit": _format_ratio(total, total),
        "eligible_raw": _format_ratio(total, total),
        "eligible_unique": _format_ratio(total, total),
        "retained": _format_count(total),
        "tier_fill": "-",
        "tier_counts": "-",
        "tier_target": "-",
        "selection": "-",
        "score": "-",
        "length": length_label,
        "pairwise_top": "-",
        "pairwise_div": "-",
        "score_norm_top": "-",
        "score_norm_div": "-",
        "set_overlap": "-",
        "set_swaps": "-",
        "diversity_pool": "-",
        "diversity_pool_source": "-",
        "tier0_score": None,
        "tier1_score": None,
        "tier2_score": None,
    }


def _stage_a_sampling_rows(
    pool_data: dict[str, PoolData],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in pool_data.values():
        summaries = pool.summaries or []
        if summaries:
            for summary in summaries:
                if not isinstance(summary, PWMSamplingSummary):
                    continue
                rows.append(_stage_a_row_from_pwm_summary(summary=summary, pool_name=str(pool.name)))
            continue
        rows.append(_stage_a_row_from_sequence_pool(pool=pool))
    rows.sort(key=lambda row: (row["input_name"], row["regulator"]))
    return rows
