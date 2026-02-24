"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/diagnostics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
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
    warnings: list[str] = []
    status = "ok"
    if optimizer is not None and not isinstance(optimizer, dict):
        raise ValueError("Sampling diagnostics field 'optimizer' must be an object when provided.")
    if optimizer_stats is not None and not isinstance(optimizer_stats, dict):
        raise ValueError("Sampling diagnostics field 'optimizer_stats' must be an object when provided.")

    thresholds = {
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

    def _mark(level: str) -> None:
        nonlocal status
        if level == "fail":
            status = "fail"
        elif level == "warn" and status == "ok":
            status = "warn"

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

    if isinstance(sample_meta, dict):
        pvalue_cache = sample_meta.get("pvalue_cache")
        if isinstance(pvalue_cache, dict):
            cache_payload = {
                "hits": _safe_int(pvalue_cache.get("hits")),
                "misses": _safe_int(pvalue_cache.get("misses")),
                "maxsize": _safe_int(pvalue_cache.get("maxsize")),
                "currsize": _safe_int(pvalue_cache.get("currsize")),
            }
            if any(value is not None for value in cache_payload.values()):
                metrics["pvalue_cache"] = cache_payload

    if mode:
        metrics["mode"] = mode
    metrics["optimizer_kind"] = resolved_optimizer_kind

    # ---- trace diagnostics -------------------------------------------------
    trace_metrics: dict[str, object] = {}
    rhat = None
    ess = None
    score_arr = None
    if trace_idata is None:
        if trace_required:
            warnings.append("Trace missing: trace-based diagnostics unavailable.")
            _mark("warn")
    else:
        score_arr = _trace_score_array(trace_idata)
        if score_arr is None:
            warnings.append("Trace missing score array; skipping R-hat/ESS.")
            _mark("warn")
        else:
            score_arr = np.asarray(score_arr, dtype=float)
            if score_arr.ndim < 2:
                warnings.append("Trace missing score array; skipping R-hat/ESS.")
                _mark("warn")
                score_arr = None
        if score_arr is not None:
            if not np.isfinite(score_arr).all():
                warnings.append("Trace contains non-finite scores; dropping invalid draws for diagnostics.")
                _mark("warn")
                cleaned_chains: list[np.ndarray] = []
                counts: list[int] = []
                for chain in score_arr:
                    valid = chain[np.isfinite(chain)]
                    if valid.size == 0:
                        continue
                    cleaned_chains.append(valid)
                    counts.append(int(valid.size))
                if not cleaned_chains:
                    warnings.append("Trace has only NaN-padded chains; skipping diagnostics.")
                    _mark("warn")
                    score_arr = None
                else:
                    dropped = int(score_arr.shape[0] - len(cleaned_chains))
                    if dropped > 0:
                        warnings.append(f"Trace diagnostics dropped {dropped} empty chains.")
                        _mark("warn")
                    min_draws = min(counts)
                    max_draws = max(counts)
                    if min_draws != max_draws:
                        warnings.append(
                            "Trace chains have unequal lengths; diagnostics computed on shortest chain length."
                        )
                        _mark("warn")
                        trace_metrics["draws_min"] = int(min_draws)
                        trace_metrics["draws_max"] = int(max_draws)
                    score_arr = np.vstack([chain[:min_draws] for chain in cleaned_chains])

            if score_arr is None or score_arr.size == 0:
                warnings.append("Trace missing usable score array; skipping R-hat/ESS.")
                _mark("warn")
                score_arr = None
            if score_arr is not None:
                n_chains, n_draws = int(score_arr.shape[0]), int(score_arr.shape[1])
                trace_metrics["chains"] = n_chains
                trace_metrics["draws"] = n_draws
                if n_chains < 2:
                    warnings.append(f"R-hat requires ≥2 chains (got {n_chains}).")
                    _mark("warn")
                if n_draws < 4:
                    warnings.append(f"ESS requires ≥4 draws (got {n_draws}).")
                    _mark("warn")
                if n_chains >= 2 and n_draws >= 4:
                    try:
                        import arviz as az

                        score_idata = az.from_dict(posterior={"score": score_arr})
                        score = score_idata.posterior["score"]
                        rhat = _safe_float(az.rhat(score)["score"].item())
                        ess = _safe_float(az.ess(score)["score"].item())
                    except Exception as exc:
                        raise ValueError(f"Trace diagnostics failed: {exc}") from exc
                if rhat is not None:
                    trace_metrics["rhat"] = rhat
                    if mode == "sample":
                        if rhat >= thresholds["rhat_fail"]:
                            warnings.append(f"Directional R-hat={rhat:.3f} suggests failed mixing.")
                            _mark("fail")
                        elif rhat > thresholds["rhat_warn"]:
                            warnings.append(f"Directional R-hat={rhat:.3f} indicates weak mixing.")
                            _mark("warn")
                if ess is not None:
                    trace_metrics["ess"] = ess
                    denom = max(1, n_chains * n_draws)
                    ess_ratio = float(ess) / float(denom)
                    trace_metrics["ess_ratio"] = ess_ratio
                    if mode == "sample":
                        if ess_ratio <= thresholds["ess_ratio_fail"]:
                            warnings.append(f"Directional ESS ratio {ess_ratio:.3f} suggests failed mixing.")
                            _mark("fail")
                        elif ess_ratio < thresholds["ess_ratio_warn"]:
                            warnings.append(f"Directional ESS ratio {ess_ratio:.3f} is low; consider longer runs.")
                            _mark("warn")

                flat = score_arr.reshape(-1)
                trace_metrics["score_mean"] = _safe_float(np.mean(flat))
                trace_metrics["score_std"] = _safe_float(np.std(flat))
                trace_metrics["score_min"] = _safe_float(np.min(flat))
                trace_metrics["score_max"] = _safe_float(np.max(flat))
                q = max(1, int(score_arr.shape[1] * 0.2))
                start_mean = _safe_float(np.mean(score_arr[:, :q]))
                end_mean = _safe_float(np.mean(score_arr[:, -q:]))
                if start_mean is not None and end_mean is not None:
                    delta = end_mean - start_mean
                    trace_metrics["score_delta"] = _safe_float(delta)
                    denom = max(abs(start_mean), 1e-12)
                    trace_metrics["score_delta_pct"] = _safe_float(delta / denom)

    metrics["trace"] = trace_metrics

    # ---- sequence diversity ----------------------------------------------
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
                warnings.append(f"Unique sequence fraction {unique_fraction:.2f} is low; sampler may be stuck.")
                _mark("warn")
    elif sequences_df is not None:
        warnings.append("Sequences table missing 'sequence' column; diversity check skipped.")
        _mark("warn")
    metrics["sequences"] = seq_metrics

    # ---- elite balance ----------------------------------------------------
    elites_metrics: dict[str, object] = {}
    if elites_df is not None and elites_df.empty:
        warnings.append("No elites matched filters; consider relaxing elite thresholds.")
        _mark("warn")
    if elites_df is not None and not elites_df.empty:
        elites_metrics["n_elites"] = int(len(elites_df))
        if sample_meta:
            top_k = _safe_int(sample_meta.get("top_k"))
            elites_metrics["top_k"] = top_k
            if top_k and len(elites_df) < top_k:
                warnings.append(f"Elite count {len(elites_df)} < top_k={top_k}; diversity constraint may be tight.")
                _mark("warn")
        score_cols = _score_cols(tf_names)
        missing = [col for col in score_cols if col not in elites_df.columns]
        if missing:
            raise ValueError(f"Elites parquet missing score columns: {missing}")
        subset = elites_df
        if sample_meta:
            top_k = _safe_int(sample_meta.get("top_k"))
            if top_k and "rank" in elites_df.columns:
                subset = elites_df.nsmallest(top_k, "rank")
            elif top_k:
                subset = elites_df.head(top_k)
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
        if balance.size:
            balance_max = _safe_float(np.nanmax(balance))
            balance_median = _safe_float(np.nanmedian(balance))
            elites_metrics["balance_index_max"] = balance_max
            elites_metrics["balance_index_median"] = balance_median
            if balance_max is not None and balance_max < thresholds["balance_index_warn"]:
                warnings.append(f"Balance index max {balance_max:.2f} suggests one TF dominates scores.")
                _mark("warn")
        norm_cols = [f"norm_{tf}" for tf in tf_names]
        if all(col in subset.columns for col in norm_cols):
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
        if has_canonical:
            source = (
                subset["canonical_sequence"] if "canonical_sequence" in subset.columns else subset["sequence"]
            ).astype(str)
            keys = source.map(lambda seq: identity_key(seq, bidirectional=True))
            elites_metrics["unique_elites_canonical"] = int(keys.nunique())
        if overlap_summary is not None:
            overlap_rate_median = overlap_summary.get("overlap_rate_median")
            overlap_total_bp_median = overlap_summary.get("overlap_total_bp_median")
            if overlap_rate_median is not None:
                elites_metrics["overlap_rate_median"] = _safe_float(overlap_rate_median)
            if overlap_total_bp_median is not None:
                elites_metrics["overlap_total_bp_median"] = _safe_float(overlap_total_bp_median)
        else:
            if elites_hits_df is None:
                raise ValueError("elites_hits.parquet is required for overlap metrics.")
            _, _, overlap_summary = compute_overlap_tables(elites_df, elites_hits_df, tf_names)
            overlap_rate_median = overlap_summary.get("overlap_rate_median")
            overlap_total_bp_median = overlap_summary.get("overlap_total_bp_median")
            if overlap_rate_median is not None:
                elites_metrics["overlap_rate_median"] = _safe_float(overlap_rate_median)
            if overlap_total_bp_median is not None:
                elites_metrics["overlap_total_bp_median"] = _safe_float(overlap_total_bp_median)
    metrics["elites"] = elites_metrics

    # ---- optimizer stats --------------------------------------------------
    optimizer_metrics: dict[str, object] = {}
    optimizer_kind_for_stats = resolved_optimizer_kind
    if optimizer:
        optimizer_kind_for_stats = resolve_optimizer_kind(
            optimizer.get("kind"),
            context="Sampling diagnostics field 'optimizer.kind'",
        )
    optimizer_metrics["kind"] = optimizer_kind_for_stats
    if optimizer_stats:
        acc = optimizer_stats.get("acceptance_rate") or {}
        acc_b = _safe_float(acc.get("B"))
        acc_m = _safe_float(acc.get("M"))
        if acc_b is not None:
            optimizer_metrics.setdefault("acceptance_rate", {})["B"] = acc_b
            if acc_b < thresholds["acceptance_low"] or acc_b > thresholds["acceptance_high"]:
                warnings.append(f"Block-move acceptance {acc_b:.2f} is outside typical bounds.")
                _mark("warn")
        if acc_m is not None:
            optimizer_metrics.setdefault("acceptance_rate", {})["M"] = acc_m
            if acc_m < thresholds["acceptance_low"] or acc_m > thresholds["acceptance_high"]:
                warnings.append(f"Multi-site acceptance {acc_m:.2f} is outside typical bounds.")
                _mark("warn")
        acc_mh = _safe_float(optimizer_stats.get("acceptance_rate_mh"))
        acc_all = _safe_float(optimizer_stats.get("acceptance_rate_all"))
        if acc_mh is not None:
            optimizer_metrics["acceptance_rate_mh"] = acc_mh
        if acc_all is not None:
            optimizer_metrics["acceptance_rate_all"] = acc_all
        move_stats = optimizer_stats.get("move_stats")
        if isinstance(move_stats, list):
            tail_rows, tail_window, tail_total = _tail_move_window_records(move_stats)
            if tail_window is not None:
                optimizer_metrics["acceptance_rate_non_s_tail_window"] = tail_window
            if tail_total is not None:
                optimizer_metrics["acceptance_rate_non_s_tail_sweeps"] = tail_total

            tail_non_s, tail_non_s_attempts = _acceptance_rate(tail_rows, move_kinds={"B", "M", "L", "W", "I"})
            if tail_non_s is not None:
                optimizer_metrics["acceptance_rate_non_s_tail"] = tail_non_s
                optimizer_metrics["acceptance_rate_non_s_tail_attempts"] = tail_non_s_attempts
                if tail_non_s < thresholds["acceptance_low"] or tail_non_s > thresholds["acceptance_high"]:
                    warnings.append(
                        f"Tail non-S acceptance {tail_non_s:.2f} is outside typical bounds (window={tail_window})."
                    )
                    _mark("warn")

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

            tail_rugged, attempts_rugged = _acceptance_rate(tail_rows, move_kinds={"B", "M"})
            if tail_rugged is not None:
                optimizer_metrics["acceptance_tail_rugged"] = tail_rugged
                optimizer_metrics["acceptance_tail_rugged_attempts"] = attempts_rugged
                if attempts_rugged >= 10 and (
                    tail_rugged < thresholds["rugged_acceptance_low"]
                    or tail_rugged > thresholds["rugged_acceptance_high"]
                ):
                    warnings.append(
                        f"Tail rugged acceptance {tail_rugged:.2f} is outside target band "
                        f"[{thresholds['rugged_acceptance_low']:.2f}, {thresholds['rugged_acceptance_high']:.2f}]."
                    )
                    _mark("warn")

            downhill_rugged, attempts_downhill_rugged = _acceptance_rate(
                tail_rows,
                move_kinds={"B", "M"},
                downhill_only=True,
            )
            if downhill_rugged is not None:
                optimizer_metrics["downhill_accept_tail_rugged"] = downhill_rugged
                optimizer_metrics["downhill_accept_tail_rugged_attempts"] = attempts_downhill_rugged
                if attempts_downhill_rugged >= 10 and downhill_rugged > thresholds["rugged_downhill_acceptance_high"]:
                    warnings.append(
                        f"Tail rugged downhill acceptance {downhill_rugged:.2f} exceeds "
                        f"{thresholds['rugged_downhill_acceptance_high']:.2f}."
                    )
                    _mark("warn")

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
        unique_successes = _safe_int(optimizer_stats.get("unique_successes"))
        if unique_successes is not None:
            optimizer_metrics["unique_successes"] = unique_successes
            if sample_meta:
                early_stop = sample_meta.get("early_stop") if isinstance(sample_meta, dict) else None
                if isinstance(early_stop, dict) and early_stop.get("require_min_unique"):
                    min_unique = _safe_int(early_stop.get("min_unique"))
                    if min_unique is not None and unique_successes < min_unique:
                        warnings.append(
                            f"unique successes {unique_successes} below minimum {min_unique} required by early stop."
                        )
                        _mark("warn")
    else:
        warnings.append("Optimizer stats missing from run manifest; acceptance checks skipped.")
        _mark("warn")
    metrics["optimizer"] = optimizer_metrics

    # ---- sample meta ------------------------------------------------------
    if sample_meta:
        metrics["sample"] = sample_meta

    return {
        "status": status,
        "warnings": warnings,
        "metrics": metrics,
        "thresholds": thresholds,
    }
