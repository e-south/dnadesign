"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/diagnostics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.move_stats import move_stats_frame
from dnadesign.cruncher.analysis.overlap import compute_overlap_tables
from dnadesign.cruncher.core.optimizers.kinds import resolve_optimizer_kind
from dnadesign.cruncher.core.sequence import identity_key


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_score_array(idata: Any) -> np.ndarray | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not hasattr(posterior, "get"):
        return None
    score = posterior.get("score")
    if score is None:
        return None
    try:
        return np.asarray(score)
    except (TypeError, ValueError):
        return None


def _score_cols(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _sequence_frame(draw_df: pd.DataFrame) -> pd.DataFrame:
    if "phase" in draw_df.columns:
        draw_df = draw_df[draw_df["phase"] == "draw"].copy()
    return draw_df


def _acceptance_tail_from_move_stats(
    move_stats: list[dict[str, object]],
    *,
    phase: str = "draw",
    mh_only: bool = True,
    window_fraction: float = 0.2,
    min_window: int = 20,
    max_window: int = 200,
) -> tuple[float | None, int | None, int | None]:
    tail_rows, window, total_sweeps = _tail_move_window_records(
        move_stats,
        phase=phase,
        window_fraction=window_fraction,
        min_window=min_window,
        max_window=max_window,
    )
    if not tail_rows:
        return None, window, total_sweeps
    kinds = {"B", "M", "L", "W", "I"} if mh_only else None
    rate, _ = _acceptance_rate(tail_rows, move_kinds=kinds)
    return rate, window, total_sweeps


def _tail_move_window_records(
    move_stats: list[dict[str, object]],
    *,
    phase: str = "draw",
    window_fraction: float = 0.2,
    min_window: int = 20,
    max_window: int = 200,
) -> tuple[list[dict[str, object]], int | None, int | None]:
    cleaned = move_stats_frame(move_stats, phase=phase)
    if cleaned.empty:
        return [], None, None

    sweeps = sorted(set(cleaned["sweep_idx"].tolist()))
    if not sweeps:
        return [], None, None
    total_sweeps = len(sweeps)
    window = int(round(window_fraction * total_sweeps))
    window = max(min_window, min(max_window, window))
    window = min(window, total_sweeps)
    cutoff = sweeps[-window]
    tail = cleaned.loc[cleaned["sweep_idx"] >= cutoff, :]
    rows = [
        {
            "sweep_idx": int(row["sweep_idx"]),
            "attempted": int(row["attempted"]),
            "accepted": int(row["accepted"]),
            "move_kind": str(row["move_kind"]),
            "delta": row.get("delta"),
            "delta_hamming": row.get("delta_hamming"),
            "gibbs_changed": row.get("gibbs_changed"),
        }
        for row in tail.to_dict(orient="records")
    ]
    return rows, window, total_sweeps


def _acceptance_rate(
    rows: list[dict[str, object]],
    *,
    move_kinds: set[str] | None = None,
    downhill_only: bool = False,
) -> tuple[float | None, int]:
    attempted_total = 0
    accepted_total = 0
    for row in rows:
        move_kind = str(row.get("move_kind"))
        if move_kinds is not None and move_kind not in move_kinds:
            continue
        delta = _safe_float(row.get("delta"))
        if downhill_only and (delta is None or delta >= 0):
            continue
        attempted = _safe_int(row.get("attempted"))
        accepted = _safe_int(row.get("accepted"))
        if attempted is None or accepted is None:
            continue
        attempted_total += attempted
        accepted_total += accepted
    if attempted_total <= 0:
        return None, 0
    return accepted_total / float(attempted_total), attempted_total


def _delta_std(rows: list[dict[str, object]], *, move_kinds: set[str] | None = None) -> float | None:
    deltas: list[float] = []
    for row in rows:
        move_kind = str(row.get("move_kind"))
        if move_kinds is not None and move_kind not in move_kinds:
            continue
        delta = _safe_float(row.get("delta"))
        if delta is None:
            continue
        deltas.append(delta)
    if not deltas:
        return None
    return float(np.std(np.asarray(deltas, dtype=float)))


def _gibbs_flip_rate(rows: list[dict[str, object]]) -> tuple[float | None, int]:
    attempts = 0
    changed = 0
    for row in rows:
        if str(row.get("move_kind")) != "S":
            continue
        attempted = _safe_int(row.get("attempted"))
        if attempted is None or attempted <= 0:
            continue
        marker = row.get("gibbs_changed")
        if marker is None:
            hamming = _safe_float(row.get("delta_hamming"))
            if hamming is None:
                continue
            is_changed = hamming > 0
        else:
            is_changed = bool(marker)
        attempts += attempted
        if is_changed:
            changed += attempted
    if attempts <= 0:
        return None, 0
    return changed / float(attempts), attempts


def _tail_hamming_mean(rows: list[dict[str, object]]) -> tuple[float | None, int]:
    values: list[float] = []
    for row in rows:
        val = _safe_float(row.get("delta_hamming"))
        if val is None:
            continue
        values.append(val)
    if not values:
        return None, 0
    return float(np.mean(np.asarray(values, dtype=float))), len(values)


@dataclass
class _DiagnosticsState:
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"

    def mark(self, level: str) -> None:
        if level == "fail":
            self.status = "fail"
        elif level == "warn" and self.status == "ok":
            self.status = "warn"

    def warn(self, message: str, *, level: str = "warn") -> None:
        self.warnings.append(message)
        self.mark(level)


def _diagnostic_thresholds() -> dict[str, float]:
    return {
        "rhat_warn": 1.1,
        "rhat_fail": 1.5,
        "ess_ratio_warn": 0.10,
        "ess_ratio_fail": 0.02,
        "acceptance_low": 0.05,
        "acceptance_high": 0.95,
        "rugged_acceptance_low": 0.10,
        "rugged_acceptance_high": 0.45,
        "rugged_downhill_acceptance_high": 0.05,
        "unique_fraction_warn": 0.20,
        "balance_index_warn": 0.50,
    }


def _collect_pvalue_cache_metrics(sample_meta: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(sample_meta, dict):
        return None
    pvalue_cache = sample_meta.get("pvalue_cache")
    if not isinstance(pvalue_cache, dict):
        return None
    cache_payload = {
        "hits": _safe_int(pvalue_cache.get("hits")),
        "misses": _safe_int(pvalue_cache.get("misses")),
        "maxsize": _safe_int(pvalue_cache.get("maxsize")),
        "currsize": _safe_int(pvalue_cache.get("currsize")),
    }
    if any(value is not None for value in cache_payload.values()):
        return cache_payload
    return None


def _warn_trace_missing_score(state: _DiagnosticsState) -> None:
    state.warn("Trace missing score array; skipping R-hat/ESS.")


def _clean_trace_score_array(
    *,
    score_arr: np.ndarray,
    trace_metrics: dict[str, object],
    state: _DiagnosticsState,
) -> np.ndarray | None:
    cleaned = np.asarray(score_arr, dtype=float)
    if np.isfinite(cleaned).all():
        return cleaned
    state.warn("Trace contains non-finite scores; dropping invalid draws for diagnostics.")
    cleaned_chains: list[np.ndarray] = []
    counts: list[int] = []
    for chain in cleaned:
        valid = chain[np.isfinite(chain)]
        if valid.size == 0:
            continue
        cleaned_chains.append(valid)
        counts.append(int(valid.size))
    if not cleaned_chains:
        state.warn("Trace has only NaN-padded chains; skipping diagnostics.")
        return None
    dropped = int(cleaned.shape[0] - len(cleaned_chains))
    if dropped > 0:
        state.warn(f"Trace diagnostics dropped {dropped} empty chains.")
    min_draws = min(counts)
    max_draws = max(counts)
    if min_draws != max_draws:
        state.warn("Trace chains have unequal lengths; diagnostics computed on shortest chain length.")
        trace_metrics["draws_min"] = int(min_draws)
        trace_metrics["draws_max"] = int(max_draws)
    return np.vstack([chain[:min_draws] for chain in cleaned_chains])


def _resolve_trace_score_array(
    *,
    trace_idata: Any | None,
    trace_required: bool,
    trace_metrics: dict[str, object],
    state: _DiagnosticsState,
) -> np.ndarray | None:
    if trace_idata is None:
        if trace_required:
            state.warn("Trace missing: trace-based diagnostics unavailable.")
        return None
    score_arr = _trace_score_array(trace_idata)
    if score_arr is None:
        _warn_trace_missing_score(state)
        return None
    score_arr = np.asarray(score_arr, dtype=float)
    if score_arr.ndim < 2:
        _warn_trace_missing_score(state)
        return None
    cleaned = _clean_trace_score_array(score_arr=score_arr, trace_metrics=trace_metrics, state=state)
    if cleaned is None or cleaned.size == 0:
        state.warn("Trace missing usable score array; skipping R-hat/ESS.")
        return None
    return cleaned


def _compute_trace_convergence(
    *,
    score_arr: np.ndarray,
) -> tuple[float | None, float | None]:
    try:
        import arviz as az

        score_idata = az.from_dict(posterior={"score": score_arr})
        score = score_idata.posterior["score"]
        rhat = _safe_float(az.rhat(score)["score"].item())
        ess = _safe_float(az.ess(score)["score"].item())
    except Exception as exc:
        raise ValueError(f"Trace diagnostics failed: {exc}") from exc
    return rhat, ess


def _append_trace_convergence_metrics(
    *,
    trace_metrics: dict[str, object],
    score_arr: np.ndarray,
    mode: str | None,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    n_chains, n_draws = int(score_arr.shape[0]), int(score_arr.shape[1])
    trace_metrics["chains"] = n_chains
    trace_metrics["draws"] = n_draws
    if n_chains < 2:
        state.warn(f"R-hat requires ≥2 chains (got {n_chains}).")
    if n_draws < 4:
        state.warn(f"ESS requires ≥4 draws (got {n_draws}).")
    if n_chains < 2 or n_draws < 4:
        return
    rhat, ess = _compute_trace_convergence(score_arr=score_arr)
    if rhat is not None:
        trace_metrics["rhat"] = rhat
        _apply_rhat_warning(rhat=rhat, mode=mode, thresholds=thresholds, state=state)
    if ess is not None:
        trace_metrics["ess"] = ess
        denom = max(1, n_chains * n_draws)
        ess_ratio = float(ess) / float(denom)
        trace_metrics["ess_ratio"] = ess_ratio
        _apply_ess_warning(ess_ratio=ess_ratio, mode=mode, thresholds=thresholds, state=state)


def _append_trace_distribution_metrics(*, trace_metrics: dict[str, object], score_arr: np.ndarray) -> None:
    flat = score_arr.reshape(-1)
    trace_metrics["score_mean"] = _safe_float(np.mean(flat))
    trace_metrics["score_std"] = _safe_float(np.std(flat))
    trace_metrics["score_min"] = _safe_float(np.min(flat))
    trace_metrics["score_max"] = _safe_float(np.max(flat))
    q = max(1, int(score_arr.shape[1] * 0.2))
    start_mean = _safe_float(np.mean(score_arr[:, :q]))
    end_mean = _safe_float(np.mean(score_arr[:, -q:]))
    if start_mean is None or end_mean is None:
        return
    delta = end_mean - start_mean
    trace_metrics["score_delta"] = _safe_float(delta)
    denom = max(abs(start_mean), 1e-12)
    trace_metrics["score_delta_pct"] = _safe_float(delta / denom)


def _apply_rhat_warning(
    *,
    rhat: float,
    mode: str | None,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    if mode != "sample":
        return
    if rhat >= thresholds["rhat_fail"]:
        state.warn(f"Directional R-hat={rhat:.3f} suggests failed mixing.", level="fail")
        return
    if rhat > thresholds["rhat_warn"]:
        state.warn(f"Directional R-hat={rhat:.3f} indicates weak mixing.")


def _apply_ess_warning(
    *,
    ess_ratio: float,
    mode: str | None,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    if mode != "sample":
        return
    if ess_ratio <= thresholds["ess_ratio_fail"]:
        state.warn(f"Directional ESS ratio {ess_ratio:.3f} suggests failed mixing.", level="fail")
        return
    if ess_ratio < thresholds["ess_ratio_warn"]:
        state.warn(f"Directional ESS ratio {ess_ratio:.3f} is low; consider longer runs.")


def _build_trace_metrics(
    *,
    trace_idata: Any | None,
    trace_required: bool,
    mode: str | None,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> dict[str, object]:
    trace_metrics: dict[str, object] = {}
    score_arr = _resolve_trace_score_array(
        trace_idata=trace_idata,
        trace_required=trace_required,
        trace_metrics=trace_metrics,
        state=state,
    )
    if score_arr is None:
        return trace_metrics
    _append_trace_convergence_metrics(
        trace_metrics=trace_metrics,
        score_arr=score_arr,
        mode=mode,
        thresholds=thresholds,
        state=state,
    )
    _append_trace_distribution_metrics(trace_metrics=trace_metrics, score_arr=score_arr)
    return trace_metrics


def _build_sequence_metrics(
    *,
    sequences_df: pd.DataFrame | None,
    has_canonical: bool,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> dict[str, object]:
    seq_metrics: dict[str, object] = {}
    if sequences_df is not None and "sequence" in sequences_df.columns:
        draw_df = _sequence_frame(sequences_df)
        total = int(len(draw_df))
        unique = None
        if has_canonical:
            if "canonical_sequence" in draw_df.columns:
                unique = int(draw_df["canonical_sequence"].astype(str).str.strip().str.upper().nunique())
            else:
                source = draw_df["sequence"].astype(str)
                keys = source.map(lambda seq: identity_key(seq, bidirectional=True))
                unique = int(keys.nunique())
            seq_metrics["unique_sequences_canonical"] = unique
            seq_metrics["unique_sequences_raw"] = int(draw_df["sequence"].astype(str).nunique())
        else:
            unique = int(draw_df["sequence"].astype(str).nunique())
        seq_metrics["n_sequences"] = total
        if unique is not None:
            seq_metrics["unique_sequences"] = unique
        if total > 0 and unique is not None:
            unique_fraction = unique / float(total)
            seq_metrics["unique_fraction"] = unique_fraction
            if unique_fraction < thresholds["unique_fraction_warn"]:
                state.warn(f"Unique sequence fraction {unique_fraction:.2f} is low; sampler may be stuck.")
    elif sequences_df is not None:
        state.warn("Sequences table missing 'sequence' column; diversity check skipped.")
    return seq_metrics


def _build_elites_metrics(
    *,
    elites_df: pd.DataFrame | None,
    elites_hits_df: pd.DataFrame | None,
    tf_names: list[str],
    sample_meta: dict[str, object] | None,
    has_canonical: bool,
    overlap_summary: dict[str, float | None] | None,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> dict[str, object]:
    elites_metrics: dict[str, object] = {}
    if elites_df is not None and elites_df.empty:
        state.warn("No elites matched filters; consider relaxing elite thresholds.")
    if elites_df is None or elites_df.empty:
        return elites_metrics
    top_k = _safe_int(sample_meta.get("top_k")) if sample_meta else None
    subset = _resolve_elite_subset(
        elites_df=elites_df,
        top_k=top_k,
        sample_meta=sample_meta,
        state=state,
        elites_metrics=elites_metrics,
    )
    _append_elite_balance_metrics(
        subset=subset,
        tf_names=tf_names,
        thresholds=thresholds,
        state=state,
        elites_metrics=elites_metrics,
    )
    _append_elite_normalized_balance_metrics(
        subset=subset,
        tf_names=tf_names,
        elites_metrics=elites_metrics,
    )
    _append_elite_canonical_uniques(
        subset=subset,
        has_canonical=has_canonical,
        elites_metrics=elites_metrics,
    )
    _append_elite_overlap_metrics(
        elites_df=elites_df,
        elites_hits_df=elites_hits_df,
        tf_names=tf_names,
        overlap_summary=overlap_summary,
        elites_metrics=elites_metrics,
    )
    return elites_metrics


def _resolve_elite_subset(
    *,
    elites_df: pd.DataFrame,
    top_k: int | None,
    sample_meta: dict[str, object] | None,
    state: _DiagnosticsState,
    elites_metrics: dict[str, object],
) -> pd.DataFrame:
    elites_metrics["n_elites"] = int(len(elites_df))
    if sample_meta:
        elites_metrics["top_k"] = top_k
    if top_k and len(elites_df) < top_k:
        state.warn(f"Elite count {len(elites_df)} < top_k={top_k}; diversity constraint may be tight.")
    if top_k and "rank" in elites_df.columns:
        return elites_df.nsmallest(top_k, "rank")
    if top_k:
        return elites_df.head(top_k)
    return elites_df


def _append_elite_balance_metrics(
    *,
    subset: pd.DataFrame,
    tf_names: list[str],
    thresholds: dict[str, float],
    state: _DiagnosticsState,
    elites_metrics: dict[str, object],
) -> None:
    score_cols = _score_cols(tf_names)
    missing = [col for col in score_cols if col not in subset.columns]
    if missing:
        raise ValueError(f"Elites parquet missing score columns: {missing}")
    scores = subset[score_cols].to_numpy(dtype=float)
    joint_min = scores.min(axis=1)
    joint_mean = scores.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        balance = np.divide(
            joint_min,
            joint_mean,
            out=np.full_like(joint_min, np.nan, dtype=float),
            where=joint_mean != 0,
        )
    if not balance.size:
        return
    balance_max = _safe_float(np.nanmax(balance))
    balance_median = _safe_float(np.nanmedian(balance))
    elites_metrics["balance_index_max"] = balance_max
    elites_metrics["balance_index_median"] = balance_median
    if balance_max is not None and balance_max < thresholds["balance_index_warn"]:
        state.warn(f"Balance index max {balance_max:.2f} suggests one TF dominates scores.")


def _append_elite_normalized_balance_metrics(
    *,
    subset: pd.DataFrame,
    tf_names: list[str],
    elites_metrics: dict[str, object],
) -> None:
    norm_cols = [f"norm_{tf}" for tf in tf_names]
    if not all(col in subset.columns for col in norm_cols):
        return
    norm_scores = subset[norm_cols].to_numpy(dtype=float)
    norm_min = norm_scores.min(axis=1)
    norm_mean = norm_scores.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_balance = np.divide(
            norm_min,
            norm_mean,
            out=np.full_like(norm_min, np.nan, dtype=float),
            where=norm_mean != 0,
        )
    elites_metrics["normalized_balance_median"] = _safe_float(np.nanmedian(norm_balance))
    elites_metrics["normalized_min_median"] = _safe_float(np.nanmedian(norm_min))


def _append_elite_canonical_uniques(
    *,
    subset: pd.DataFrame,
    has_canonical: bool,
    elites_metrics: dict[str, object],
) -> None:
    if not has_canonical:
        return
    canonical_col = "canonical_sequence" if "canonical_sequence" in subset.columns else "sequence"
    source = subset[canonical_col].astype(str)
    keys = source.map(lambda seq: identity_key(seq, bidirectional=True))
    elites_metrics["unique_elites_canonical"] = int(keys.nunique())


def _append_elite_overlap_metrics(
    *,
    elites_df: pd.DataFrame,
    elites_hits_df: pd.DataFrame | None,
    tf_names: list[str],
    overlap_summary: dict[str, float | None] | None,
    elites_metrics: dict[str, object],
) -> None:
    summary = overlap_summary
    if summary is None:
        if elites_hits_df is None:
            raise ValueError("elites_hits.parquet is required for overlap metrics.")
        _, _, summary = compute_overlap_tables(elites_df, elites_hits_df, tf_names)
    if not isinstance(summary, dict):
        return
    overlap_rate_median = summary.get("overlap_rate_median")
    overlap_total_bp_median = summary.get("overlap_total_bp_median")
    if overlap_rate_median is not None:
        elites_metrics["overlap_rate_median"] = _safe_float(overlap_rate_median)
    if overlap_total_bp_median is not None:
        elites_metrics["overlap_total_bp_median"] = _safe_float(overlap_total_bp_median)


def _append_optimizer_acceptance_metrics(
    *,
    optimizer_metrics: dict[str, object],
    optimizer_stats: dict[str, object],
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    acc = optimizer_stats.get("acceptance_rate") or {}
    acc_b = _safe_float(acc.get("B"))
    acc_m = _safe_float(acc.get("M"))
    if acc_b is not None:
        optimizer_metrics.setdefault("acceptance_rate", {})["B"] = acc_b
        if acc_b < thresholds["acceptance_low"] or acc_b > thresholds["acceptance_high"]:
            state.warn(f"Block-move acceptance {acc_b:.2f} is outside typical bounds.")
    if acc_m is not None:
        optimizer_metrics.setdefault("acceptance_rate", {})["M"] = acc_m
        if acc_m < thresholds["acceptance_low"] or acc_m > thresholds["acceptance_high"]:
            state.warn(f"Multi-site acceptance {acc_m:.2f} is outside typical bounds.")
    acc_mh = _safe_float(optimizer_stats.get("acceptance_rate_mh"))
    acc_all = _safe_float(optimizer_stats.get("acceptance_rate_all"))
    if acc_mh is not None:
        optimizer_metrics["acceptance_rate_mh"] = acc_mh
    if acc_all is not None:
        optimizer_metrics["acceptance_rate_all"] = acc_all


def _append_tail_window_context(
    *,
    optimizer_metrics: dict[str, object],
    tail_window: int | None,
    tail_total: int | None,
) -> None:
    if tail_window is not None:
        optimizer_metrics["acceptance_rate_non_s_tail_window"] = tail_window
    if tail_total is not None:
        optimizer_metrics["acceptance_rate_non_s_tail_sweeps"] = tail_total


def _append_tail_non_s_metrics(
    *,
    optimizer_metrics: dict[str, object],
    tail_rows: list[dict[str, object]],
    tail_window: int | None,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    tail_non_s, tail_non_s_attempts = _acceptance_rate(tail_rows, move_kinds={"B", "M", "L", "W", "I"})
    if tail_non_s is None:
        return
    optimizer_metrics["acceptance_rate_non_s_tail"] = tail_non_s
    optimizer_metrics["acceptance_rate_non_s_tail_attempts"] = tail_non_s_attempts
    if tail_non_s < thresholds["acceptance_low"] or tail_non_s > thresholds["acceptance_high"]:
        state.warn(f"Tail non-S acceptance {tail_non_s:.2f} is outside typical bounds (window={tail_window}).")


def _append_tail_move_kind_metrics(
    *,
    optimizer_metrics: dict[str, object],
    tail_rows: list[dict[str, object]],
) -> None:
    for move_kind in ("B", "M", "L", "W", "I"):
        rate, attempts = _acceptance_rate(tail_rows, move_kinds={move_kind})
        if rate is None:
            continue
        optimizer_metrics[f"acceptance_tail_{move_kind}"] = rate
        optimizer_metrics[f"acceptance_tail_{move_kind}_attempts"] = attempts
    tail_mh_all, attempts_mh_all = _acceptance_rate(tail_rows, move_kinds={"B", "M", "L", "W", "I"})
    if tail_mh_all is not None:
        optimizer_metrics["acceptance_tail_mh_all"] = tail_mh_all
        optimizer_metrics["acceptance_tail_mh_all_attempts"] = attempts_mh_all


def _append_tail_rugged_metrics(
    *,
    optimizer_metrics: dict[str, object],
    tail_rows: list[dict[str, object]],
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    tail_rugged, attempts_rugged = _acceptance_rate(tail_rows, move_kinds={"B", "M"})
    if tail_rugged is not None:
        optimizer_metrics["acceptance_tail_rugged"] = tail_rugged
        optimizer_metrics["acceptance_tail_rugged_attempts"] = attempts_rugged
        if attempts_rugged >= 10 and (
            tail_rugged < thresholds["rugged_acceptance_low"] or tail_rugged > thresholds["rugged_acceptance_high"]
        ):
            state.warn(
                f"Tail rugged acceptance {tail_rugged:.2f} is outside target band "
                f"[{thresholds['rugged_acceptance_low']:.2f}, {thresholds['rugged_acceptance_high']:.2f}]."
            )
    downhill_rugged, attempts_downhill_rugged = _acceptance_rate(
        tail_rows,
        move_kinds={"B", "M"},
        downhill_only=True,
    )
    if downhill_rugged is not None:
        optimizer_metrics["downhill_accept_tail_rugged"] = downhill_rugged
        optimizer_metrics["downhill_accept_tail_rugged_attempts"] = attempts_downhill_rugged
        if attempts_downhill_rugged >= 10 and downhill_rugged > thresholds["rugged_downhill_acceptance_high"]:
            state.warn(
                f"Tail rugged downhill acceptance {downhill_rugged:.2f} exceeds "
                f"{thresholds['rugged_downhill_acceptance_high']:.2f}."
            )


def _append_tail_variability_metrics(
    *,
    optimizer_metrics: dict[str, object],
    tail_rows: list[dict[str, object]],
) -> None:
    gibbs_flip_rate, gibbs_flip_attempts = _gibbs_flip_rate(tail_rows)
    if gibbs_flip_rate is not None:
        optimizer_metrics["gibbs_flip_rate_tail"] = gibbs_flip_rate
        optimizer_metrics["gibbs_flip_rate_tail_attempts"] = gibbs_flip_attempts
    tail_hamming_mean, tail_hamming_n = _tail_hamming_mean(tail_rows)
    if tail_hamming_mean is not None:
        optimizer_metrics["tail_step_hamming_mean"] = tail_hamming_mean
        optimizer_metrics["tail_step_hamming_n"] = tail_hamming_n
    delta_std_tail = _delta_std(tail_rows)
    if delta_std_tail is not None:
        optimizer_metrics["score_delta_std_tail"] = delta_std_tail
    delta_std_tail_mh = _delta_std(tail_rows, move_kinds={"B", "M", "L", "W", "I"})
    if delta_std_tail_mh is not None:
        optimizer_metrics["score_delta_std_tail_mh"] = delta_std_tail_mh
    delta_std_tail_rugged = _delta_std(tail_rows, move_kinds={"B", "M"})
    if delta_std_tail_rugged is not None:
        optimizer_metrics["score_delta_std_tail_rugged"] = delta_std_tail_rugged


def _append_tail_move_metrics(
    *,
    optimizer_metrics: dict[str, object],
    move_stats: list[dict[str, object]],
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> None:
    tail_rows, tail_window, tail_total = _tail_move_window_records(move_stats)
    _append_tail_window_context(
        optimizer_metrics=optimizer_metrics,
        tail_window=tail_window,
        tail_total=tail_total,
    )
    _append_tail_non_s_metrics(
        optimizer_metrics=optimizer_metrics,
        tail_rows=tail_rows,
        tail_window=tail_window,
        thresholds=thresholds,
        state=state,
    )
    _append_tail_move_kind_metrics(
        optimizer_metrics=optimizer_metrics,
        tail_rows=tail_rows,
    )
    _append_tail_rugged_metrics(
        optimizer_metrics=optimizer_metrics,
        tail_rows=tail_rows,
        thresholds=thresholds,
        state=state,
    )
    _append_tail_variability_metrics(
        optimizer_metrics=optimizer_metrics,
        tail_rows=tail_rows,
    )


def _append_unique_successes_metric(
    *,
    optimizer_metrics: dict[str, object],
    optimizer_stats: dict[str, object],
    sample_meta: dict[str, object] | None,
    state: _DiagnosticsState,
) -> None:
    unique_successes = _safe_int(optimizer_stats.get("unique_successes"))
    if unique_successes is None:
        return
    optimizer_metrics["unique_successes"] = unique_successes
    if not sample_meta:
        return
    early_stop = sample_meta.get("early_stop") if isinstance(sample_meta, dict) else None
    if not isinstance(early_stop, dict) or not early_stop.get("require_min_unique"):
        return
    min_unique = _safe_int(early_stop.get("min_unique"))
    if min_unique is not None and unique_successes < min_unique:
        state.warn(f"unique successes {unique_successes} below minimum {min_unique} required by early stop.")


def _build_optimizer_metrics(
    *,
    optimizer: dict[str, object] | None,
    optimizer_stats: dict[str, object] | None,
    sample_meta: dict[str, object] | None,
    resolved_optimizer_kind: str,
    thresholds: dict[str, float],
    state: _DiagnosticsState,
) -> dict[str, object]:
    optimizer_metrics: dict[str, object] = {}
    optimizer_kind_for_stats = resolved_optimizer_kind
    if optimizer:
        optimizer_kind_for_stats = resolve_optimizer_kind(
            optimizer.get("kind"),
            context="Sampling diagnostics field 'optimizer.kind'",
        )
    optimizer_metrics["kind"] = optimizer_kind_for_stats
    if not optimizer_stats:
        state.warn("Optimizer stats missing from run manifest; acceptance checks skipped.")
        return optimizer_metrics
    _append_optimizer_acceptance_metrics(
        optimizer_metrics=optimizer_metrics,
        optimizer_stats=optimizer_stats,
        thresholds=thresholds,
        state=state,
    )
    move_stats = optimizer_stats.get("move_stats")
    if isinstance(move_stats, list):
        _append_tail_move_metrics(
            optimizer_metrics=optimizer_metrics,
            move_stats=move_stats,
            thresholds=thresholds,
            state=state,
        )
    _append_unique_successes_metric(
        optimizer_metrics=optimizer_metrics,
        optimizer_stats=optimizer_stats,
        sample_meta=sample_meta,
        state=state,
    )
    return optimizer_metrics


def summarize_sampling_diagnostics(
    *,
    trace_idata: Any | None,
    sequences_df: pd.DataFrame | None,
    elites_df: pd.DataFrame | None,
    elites_hits_df: pd.DataFrame | None,
    tf_names: list[str],
    optimizer: dict[str, object] | None,
    optimizer_stats: dict[str, object] | None,
    mode: str | None = None,
    optimizer_kind: str | None = None,
    sample_meta: dict[str, object] | None = None,
    trace_required: bool = True,
    overlap_summary: dict[str, float | None] | None = None,
) -> dict[str, object]:
    """
    Build a diagnostics summary for sample/analyze/report.

    This function is intentionally conservative: it records metrics whenever
    available and emits warnings for common failure modes, without auto-tuning
    or silently changing behavior.
    """
    state = _DiagnosticsState()
    if optimizer is not None and not isinstance(optimizer, dict):
        raise ValueError("Sampling diagnostics field 'optimizer' must be an object when provided.")
    if optimizer_stats is not None and not isinstance(optimizer_stats, dict):
        raise ValueError("Sampling diagnostics field 'optimizer_stats' must be an object when provided.")

    thresholds = _diagnostic_thresholds()

    metrics: dict[str, object] = {}
    if mode is None and sample_meta:
        mode = str(sample_meta.get("mode") or "")
    raw_optimizer_kind = optimizer_kind
    if raw_optimizer_kind is None and sample_meta:
        raw_optimizer_kind = sample_meta.get("optimizer_kind") or raw_optimizer_kind
    if raw_optimizer_kind is None and isinstance(optimizer, dict):
        raw_optimizer_kind = optimizer.get("kind") if isinstance(optimizer.get("kind"), str) else raw_optimizer_kind
    resolved_optimizer_kind = resolve_optimizer_kind(raw_optimizer_kind, context="Sampling diagnostics")
    has_canonical = bool(sample_meta.get("dsdna_canonicalize")) if sample_meta else False

    pvalue_cache_metrics = _collect_pvalue_cache_metrics(sample_meta)
    if pvalue_cache_metrics is not None:
        metrics["pvalue_cache"] = pvalue_cache_metrics

    if mode:
        metrics["mode"] = mode
    metrics["optimizer_kind"] = resolved_optimizer_kind

    metrics["trace"] = _build_trace_metrics(
        trace_idata=trace_idata,
        trace_required=trace_required,
        mode=mode,
        thresholds=thresholds,
        state=state,
    )
    metrics["sequences"] = _build_sequence_metrics(
        sequences_df=sequences_df,
        has_canonical=has_canonical,
        thresholds=thresholds,
        state=state,
    )
    metrics["elites"] = _build_elites_metrics(
        elites_df=elites_df,
        elites_hits_df=elites_hits_df,
        tf_names=tf_names,
        sample_meta=sample_meta,
        has_canonical=has_canonical,
        overlap_summary=overlap_summary,
        thresholds=thresholds,
        state=state,
    )
    metrics["optimizer"] = _build_optimizer_metrics(
        optimizer=optimizer,
        optimizer_stats=optimizer_stats,
        sample_meta=sample_meta,
        resolved_optimizer_kind=resolved_optimizer_kind,
        thresholds=thresholds,
        state=state,
    )

    # ---- sample meta ------------------------------------------------------
    if sample_meta:
        metrics["sample"] = sample_meta

    return {
        "status": state.status,
        "warnings": state.warnings,
        "metrics": metrics,
        "thresholds": thresholds,
    }
