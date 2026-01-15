"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/diagnostics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


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
    except Exception:
        return None


def _score_cols(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _sequence_frame(draw_df: pd.DataFrame) -> pd.DataFrame:
    if "phase" in draw_df.columns:
        draw_df = draw_df[draw_df["phase"] == "draw"].copy()
    return draw_df


def summarize_sampling_diagnostics(
    *,
    trace_idata: Any | None,
    sequences_df: pd.DataFrame | None,
    elites_df: pd.DataFrame | None,
    tf_names: list[str],
    optimizer: dict[str, object] | None,
    optimizer_stats: dict[str, object] | None,
    mode: str | None = None,
    optimizer_kind: str | None = None,
    sample_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build a diagnostics summary for sample/analyze/report.

    This function is intentionally conservative: it records metrics whenever
    available and emits warnings for common failure modes, without auto-tuning
    or silently changing behavior.
    """
    warnings: list[str] = []
    status = "ok"

    thresholds = {
        "rhat_warn": 1.1,
        "rhat_fail": 1.5,
        "ess_ratio_warn": 0.10,
        "ess_ratio_fail": 0.02,
        "acceptance_low": 0.05,
        "acceptance_high": 0.95,
        "swap_low": 0.05,
        "swap_high": 0.90,
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
    if optimizer_kind is None and sample_meta:
        optimizer_kind = sample_meta.get("optimizer_kind") or optimizer_kind
    if optimizer_kind is None and isinstance(optimizer, dict):
        optimizer_kind = optimizer.get("kind") if isinstance(optimizer.get("kind"), str) else optimizer_kind
    optimizer_kind = str(optimizer_kind).lower() if optimizer_kind else None
    if mode:
        metrics["mode"] = mode
    if optimizer_kind:
        metrics["optimizer_kind"] = optimizer_kind

    # ---- trace diagnostics -------------------------------------------------
    trace_metrics: dict[str, object] = {}
    rhat = None
    ess = None
    score_arr = None
    if trace_idata is None:
        warnings.append("Trace missing: trace-based diagnostics unavailable.")
        _mark("warn")
    else:
        score_arr = _trace_score_array(trace_idata)
        if score_arr is None or score_arr.ndim < 2:
            warnings.append("Trace missing score array; skipping R-hat/ESS.")
            _mark("warn")
        else:
            n_chains, n_draws = int(score_arr.shape[0]), int(score_arr.shape[1])
            trace_metrics["chains"] = n_chains
            trace_metrics["draws"] = n_draws
            skip_mixing = optimizer_kind == "pt"
            if skip_mixing:
                trace_metrics["mixing_note"] = "PT ladder: R-hat/ESS not computed across temperatures."
            else:
                if n_chains < 2:
                    warnings.append(f"R-hat requires ≥2 chains (got {n_chains}).")
                    _mark("warn")
                if n_draws < 4:
                    warnings.append(f"ESS requires ≥4 draws (got {n_draws}).")
                    _mark("warn")
                if n_chains >= 2 and n_draws >= 4:
                    try:
                        import arviz as az

                        score = trace_idata.posterior["score"]
                        rhat = _safe_float(az.rhat(score)["score"].item())
                        ess = _safe_float(az.ess(score)["score"].item())
                    except Exception as exc:
                        warnings.append(f"Trace diagnostics failed: {exc}")
                        _mark("warn")
                if rhat is not None:
                    trace_metrics["rhat"] = rhat
                    if mode == "sample":
                        if rhat >= thresholds["rhat_fail"]:
                            warnings.append(f"R-hat={rhat:.3f} suggests failed mixing.")
                            _mark("fail")
                        elif rhat > thresholds["rhat_warn"]:
                            warnings.append(f"R-hat={rhat:.3f} indicates weak mixing.")
                            _mark("warn")
                if ess is not None:
                    trace_metrics["ess"] = ess
                    denom = max(1, n_chains * n_draws)
                    ess_ratio = float(ess) / float(denom)
                    trace_metrics["ess_ratio"] = ess_ratio
                    if mode == "sample":
                        if ess_ratio <= thresholds["ess_ratio_fail"]:
                            warnings.append(f"ESS ratio {ess_ratio:.3f} suggests failed mixing.")
                            _mark("fail")
                        elif ess_ratio < thresholds["ess_ratio_warn"]:
                            warnings.append(f"ESS ratio {ess_ratio:.3f} is low; consider longer runs.")
                            _mark("warn")

            if score_arr is not None and score_arr.size > 0:
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
                best_so_far = np.maximum.accumulate(score_arr, axis=1)
                bsf_start = _safe_float(np.mean(best_so_far[:, 0]))
                bsf_end = _safe_float(np.mean(best_so_far[:, -1]))
                if bsf_start is not None and bsf_end is not None:
                    bsf_delta = bsf_end - bsf_start
                    trace_metrics["best_so_far_start"] = bsf_start
                    trace_metrics["best_so_far_end"] = bsf_end
                    trace_metrics["best_so_far_delta"] = _safe_float(bsf_delta)
                    denom = max(1, int(score_arr.shape[1]) - 1)
                    trace_metrics["best_so_far_slope"] = _safe_float(bsf_delta / denom)

    metrics["trace"] = trace_metrics

    # ---- sequence diversity ----------------------------------------------
    seq_metrics: dict[str, object] = {}
    if sequences_df is not None and "sequence" in sequences_df.columns:
        draw_df = _sequence_frame(sequences_df)
        total = int(len(draw_df))
        unique = int(draw_df["sequence"].nunique())
        seq_metrics["n_sequences"] = total
        seq_metrics["unique_sequences"] = unique
        if total > 0:
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
        if "sequence" in subset.columns:
            seqs = subset["sequence"].astype(str).tolist()
            if len(seqs) >= 2:
                total = 0.0
                count = 0
                for i in range(len(seqs)):
                    for j in range(i + 1, len(seqs)):
                        s0, s1 = seqs[i], seqs[j]
                        if len(s0) != len(s1):
                            continue
                        dist = sum(c0 != c1 for c0, c1 in zip(s0, s1)) / float(len(s0))
                        total += dist
                        count += 1
                if count:
                    elites_metrics["diversity_hamming"] = _safe_float(total / count)
    metrics["elites"] = elites_metrics

    # ---- optimizer stats --------------------------------------------------
    optimizer_metrics: dict[str, object] = {}
    optimizer_kind = None
    if optimizer:
        optimizer_kind = optimizer.get("kind")
        optimizer_metrics["kind"] = optimizer_kind
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
        swap_rate = _safe_float(optimizer_stats.get("swap_acceptance_rate"))
        if swap_rate is not None:
            optimizer_metrics["swap_acceptance_rate"] = swap_rate
            if swap_rate < thresholds["swap_low"] or swap_rate > thresholds["swap_high"]:
                warnings.append(f"Swap acceptance {swap_rate:.2f} suggests poor PT mixing.")
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
