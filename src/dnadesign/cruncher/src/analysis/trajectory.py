"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/trajectory.py

Build deterministic optimization trajectory points for plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer

_DNA_BASE_TO_INT: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _validate_numeric(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"Trajectory objective column '{column}' contains non-numeric values.")
    return values.astype(float)


def _softmin(values: np.ndarray, beta: float) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float("nan")
    scaled = -beta * finite_values
    max_scaled = float(np.max(scaled))
    if not np.isfinite(max_scaled):
        return float("nan")
    logsum = max_scaled + float(np.log(np.exp(scaled - max_scaled).sum()))
    return float(-logsum / beta)


def resolve_cold_chain(
    *,
    beta_ladder: list[float] | None,
    chain_ids: Iterable[int],
    beta_by_chain: dict[int, float] | None = None,
) -> int:
    ids = sorted({int(v) for v in chain_ids})
    if not ids:
        raise ValueError("Cannot resolve cold chain from an empty chain set.")
    if len(ids) == 1:
        return ids[0]
    if beta_ladder is None:
        if beta_by_chain is None:
            raise ValueError("Missing optimizer beta ladder; cannot resolve cold chain for multi-chain trajectory.")
        missing = [chain_id for chain_id in ids if chain_id not in beta_by_chain]
        if missing:
            raise ValueError(f"Missing beta values for chain IDs: {missing}")
        beta_arr = np.asarray([beta_by_chain[chain_id] for chain_id in ids], dtype=float)
        if not np.isfinite(beta_arr).all():
            raise ValueError("Trajectory chain beta values must contain only finite values.")
        max_beta = float(np.max(beta_arr))
        present_candidates = [
            chain_id for chain_id in ids if np.isclose(beta_by_chain[chain_id], max_beta, rtol=0.0, atol=1e-12)
        ]
        if len(present_candidates) > 1:
            raise ValueError(
                "optimizer beta ladder is ambiguous; multiple chains share the maximum beta "
                f"(chains={present_candidates}, beta={max_beta})."
            )
        return int(present_candidates[0])
    if not beta_ladder:
        raise ValueError("optimizer beta ladder is empty; cannot resolve cold chain.")
    if max(ids) >= len(beta_ladder):
        raise ValueError(
            "optimizer beta ladder does not cover all chain IDs "
            f"(max chain={max(ids)}, ladder length={len(beta_ladder)})."
        )
    beta_arr = np.asarray(beta_ladder, dtype=float)
    if not np.isfinite(beta_arr).all():
        raise ValueError("optimizer beta ladder must contain finite values.")
    max_beta = float(np.max(beta_arr))
    cold_candidates = np.where(np.isclose(beta_arr, max_beta, rtol=0.0, atol=1e-12))[0]
    present_candidates = [int(idx) for idx in cold_candidates if int(idx) in ids]
    if len(present_candidates) > 1:
        raise ValueError(
            "optimizer beta ladder is ambiguous; multiple chains share the maximum beta "
            f"(chains={present_candidates}, beta={max_beta})."
        )
    if len(present_candidates) == 0:
        cold_candidates_list = [int(idx) for idx in cold_candidates]
        raise ValueError(
            "optimizer beta ladder cold index is not present in trajectory chain IDs "
            f"(cold_candidates={cold_candidates_list}, chains={ids})."
        )
    cold_idx = int(present_candidates[0])
    return cold_idx


def _resolve_objective_scalar(
    df: pd.DataFrame,
    tf_names: Iterable[str],
    *,
    objective_config: dict[str, object] | None,
) -> pd.Series:
    direct_columns = ("objective_scalar", "score_aggregate", "combined_score_final", "combined_score")
    for column in direct_columns:
        if column in df.columns:
            return _validate_numeric(df[column], column=column)

    score_cols = _score_columns(tf_names)
    missing = [col for col in score_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot reconstruct trajectory objective scalar; missing score columns: {missing}")

    objective_cfg = objective_config if isinstance(objective_config, dict) else {}
    combine = str(objective_cfg.get("combine") or "min").strip().lower()
    scores = df[score_cols].to_numpy(dtype=float)
    if combine == "sum":
        return pd.Series(scores.sum(axis=1), index=df.index, dtype=float)
    if combine != "min":
        raise ValueError(
            f"Unsupported objective combine mode '{combine}' while reconstructing trajectory objective scalar."
        )

    softmin_cfg = objective_cfg.get("softmin")
    softmin_enabled = isinstance(softmin_cfg, dict) and bool(softmin_cfg.get("enabled"))
    beta_value = objective_cfg.get("softmin_final_beta_used")
    beta_final: float | None = None
    if beta_value is not None:
        try:
            beta_final = float(beta_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("objective.softmin_final_beta_used must be numeric when provided.") from exc
    if softmin_enabled and beta_final is not None and beta_final > 0:
        objective_vals = np.asarray([_softmin(row, beta_final) for row in scores], dtype=float)
        return pd.Series(objective_vals, index=df.index, dtype=float)
    return pd.Series(scores.min(axis=1), index=df.index, dtype=float)


def _encode_sequence(seq: object, *, context: str) -> np.ndarray:
    clean = str(seq).strip().upper()
    if not clean:
        raise ValueError(f"{context} sequence is empty.")
    try:
        return np.asarray([_DNA_BASE_TO_INT[base] for base in clean], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"{context} sequence contains invalid base(s): {seq!r}") from exc


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _subsample_indices_early_priority(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    early_span = max(1, int(round(0.15 * n)))
    early_budget = min(max(1, int(round(0.40 * max_points))), early_span, max_points)
    early_idx = np.arange(early_budget, dtype=int)
    tail_budget = max_points - len(early_idx)
    if tail_budget <= 0:
        return early_idx
    tail_idx = np.unique(np.linspace(early_budget, n - 1, tail_budget).round().astype(int))
    return np.unique(np.concatenate([early_idx, tail_idx]))


def _subsample_chainwise(out: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(out) <= max_points:
        return out
    if "chain" not in out.columns:
        idx = _subsample_indices_early_priority(len(out), max_points)
        return out.iloc[idx].reset_index(drop=True)

    grouped = list(out.groupby("chain", sort=True, dropna=False))
    if not grouped:
        return out
    chain_count = len(grouped)
    budget_floor = max(1, max_points // chain_count)
    budget_remainder = max_points - (budget_floor * chain_count)

    sampled_parts: list[pd.DataFrame] = []
    for idx, (_, chain_df) in enumerate(grouped):
        budget = budget_floor + (1 if idx < budget_remainder else 0)
        budget = min(len(chain_df), max(1, budget))
        sampled_idx = _subsample_indices_early_priority(len(chain_df), budget)
        sampled_parts.append(chain_df.iloc[sampled_idx])

    sampled = pd.concat(sampled_parts).sort_values(["chain", "sweep"]).reset_index(drop=True)
    if len(sampled) <= max_points:
        return sampled
    keep_idx = _subsample_indices_early_priority(len(sampled), max_points)
    return sampled.iloc[keep_idx].reset_index(drop=True)


def project_scores(
    df: pd.DataFrame,
    tf_names: Iterable[str],
) -> Tuple[pd.Series, pd.Series, str, str, pd.Series, pd.Series]:
    tf_list = list(tf_names)
    score_cols = _score_columns(tf_list)
    missing = [col for col in score_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns for trajectory projection: {missing}")
    scores = df[score_cols].to_numpy(dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    worst = pd.Series(sorted_scores[:, 0], index=df.index)
    second = pd.Series(sorted_scores[:, 1] if sorted_scores.shape[1] > 1 else sorted_scores[:, 0], index=df.index)
    if len(tf_list) == 2:
        x = df[score_cols[0]].astype(float)
        y = df[score_cols[1]].astype(float)
        return x, y, score_cols[0], score_cols[1], worst, second
    return worst, second, "worst_tf_score", "second_worst_tf_score", worst, second


def build_trajectory_points(
    sequences_df: pd.DataFrame,
    tf_names: Iterable[str],
    *,
    max_points: int,
    objective_config: dict[str, object] | None = None,
    beta_ladder: list[float] | None = None,
) -> pd.DataFrame:
    if sequences_df is None or sequences_df.empty:
        return pd.DataFrame()

    df = sequences_df.copy()
    if "chain" not in df.columns:
        raise ValueError("Trajectory points are missing required column 'chain'.")
    chain_series = pd.to_numeric(df["chain"], errors="coerce")
    if chain_series.isna().any():
        raise ValueError("Trajectory chain values must be numeric.")
    df["chain"] = chain_series.astype(int)
    if "sweep_idx" in df.columns:
        sweep_series = pd.to_numeric(df["sweep_idx"], errors="coerce")
        if sweep_series.isna().any():
            raise ValueError("Trajectory sweep_idx values must be numeric.")
        df["sweep_idx"] = sweep_series.astype(int)
        df = df.sort_values(["chain", "sweep_idx"])
    elif "draw" in df.columns:
        draw_series = pd.to_numeric(df["draw"], errors="coerce")
        if draw_series.isna().any():
            raise ValueError("Trajectory draw values must be numeric.")
        df["draw"] = draw_series.astype(int)
        df = df.sort_values(["chain", "draw"])
    else:
        df = df.sort_values(["chain"])

    x, y, x_metric, y_metric, worst, second = project_scores(df, tf_names)
    score_cols = _score_columns(tf_names)
    cols = [c for c in ("draw", "sweep_idx", "phase", "chain", "beta", "sequence") if c in df.columns]
    cols += score_cols
    out = df[cols].copy()
    if "sweep_idx" in out.columns:
        out["sweep"] = out["sweep_idx"].astype(int)
    elif "draw" in out.columns:
        out["sweep"] = out["draw"].astype(int)
    else:
        out["sweep"] = out.groupby("chain").cumcount()
    out["sweep_idx"] = out["sweep"].astype(int)
    if "beta" not in out.columns:
        if beta_ladder is not None:
            out["beta"] = (
                out["chain"]
                .astype(int)
                .map(lambda chain: float(beta_ladder[int(chain)]) if int(chain) < len(beta_ladder) else np.nan)
            )
        else:
            out["beta"] = np.nan
    else:
        out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
        if out["beta"].notna().any() and out["beta"].isna().any():
            raise ValueError("Trajectory beta values must be numeric when provided.")
    out["x"] = x
    out["y"] = y
    out["x_metric"] = x_metric
    out["y_metric"] = y_metric
    out["worst_tf_score"] = worst
    out["second_worst_tf_score"] = second
    out["objective_scalar"] = _resolve_objective_scalar(df, tf_names, objective_config=objective_config)
    return _subsample_chainwise(out.reset_index(drop=True), max_points)


def add_raw_llr_objective(
    trajectory_df: pd.DataFrame,
    tf_names: Iterable[str],
    *,
    pwms: dict[str, PWM],
    objective_config: dict[str, object] | None,
    bidirectional: bool,
    pwm_pseudocounts: float,
    log_odds_clip: float | None,
) -> pd.DataFrame:
    if trajectory_df is None or trajectory_df.empty:
        return pd.DataFrame()
    if "sequence" not in trajectory_df.columns:
        raise ValueError("Trajectory points missing required column 'sequence' for raw-LLR objective.")
    tf_list = [str(tf) for tf in tf_names]
    if not tf_list:
        raise ValueError("Cannot compute raw-LLR objective without TF names.")
    missing_pwms = [tf for tf in tf_list if tf not in pwms]
    if missing_pwms:
        raise ValueError(f"Cannot compute raw-LLR objective; missing PWMs for TFs: {missing_pwms}")
    scorer_raw = Scorer(
        {tf: pwms[tf] for tf in tf_list},
        bidirectional=bool(bidirectional),
        scale="llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )
    scorer_norm = Scorer(
        {tf: pwms[tf] for tf in tf_list},
        bidirectional=bool(bidirectional),
        scale="normalized-llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )
    out = trajectory_df.copy()
    raw_payload: list[dict[str, float]] = []
    norm_payload: list[dict[str, float]] = []
    for row_idx, seq in out["sequence"].items():
        seq_arr = _encode_sequence(seq, context=f"trajectory row {row_idx}")
        raw_payload.append(scorer_raw.compute_all_per_pwm(seq_arr, int(seq_arr.size)))
        norm_payload.append(scorer_norm.compute_all_per_pwm(seq_arr, int(seq_arr.size)))
    raw_df = pd.DataFrame(raw_payload, index=out.index)
    norm_df = pd.DataFrame(norm_payload, index=out.index)
    raw_score_df = pd.DataFrame(index=out.index)
    norm_score_df = pd.DataFrame(index=out.index)
    for tf in tf_list:
        if tf not in raw_df.columns:
            raise ValueError(f"Raw-LLR scorer output missing TF '{tf}'.")
        if tf not in norm_df.columns:
            raise ValueError(f"Normalized-LLR scorer output missing TF '{tf}'.")
        raw_values = pd.to_numeric(raw_df[tf], errors="coerce")
        norm_values = pd.to_numeric(norm_df[tf], errors="coerce")
        if raw_values.isna().any():
            raise ValueError(f"Raw-LLR scorer output for TF '{tf}' contains non-numeric values.")
        if norm_values.isna().any():
            raise ValueError(f"Normalized-LLR scorer output for TF '{tf}' contains non-numeric values.")
        out[f"raw_llr_{tf}"] = raw_values.astype(float)
        out[f"norm_llr_{tf}"] = norm_values.astype(float)
        raw_score_df[f"score_{tf}"] = raw_values.astype(float)
        norm_score_df[f"score_{tf}"] = norm_values.astype(float)
    out["raw_llr_objective"] = _resolve_objective_scalar(raw_score_df, tf_list, objective_config=objective_config)
    out["norm_llr_objective"] = _resolve_objective_scalar(norm_score_df, tf_list, objective_config=objective_config)
    return out


def build_chain_trajectory_points(
    trajectory_df: pd.DataFrame,
    *,
    max_points: int,
) -> pd.DataFrame:
    if trajectory_df is None or trajectory_df.empty:
        return pd.DataFrame()
    required = {
        "chain",
        "sweep",
        "phase",
        "beta",
        "x",
        "y",
        "x_metric",
        "y_metric",
        "objective_scalar",
        "raw_llr_objective",
        "norm_llr_objective",
    }
    missing = [name for name in sorted(required) if name not in trajectory_df.columns]
    if missing:
        raise ValueError(f"Trajectory points missing required columns for chain lineage: {missing}")

    out = trajectory_df.copy()
    out["chain"] = pd.to_numeric(out["chain"], errors="coerce")
    if out["chain"].isna().any():
        raise ValueError("Trajectory chain values must be numeric.")
    out["chain"] = out["chain"].astype(int)
    out["sweep_idx"] = pd.to_numeric(out["sweep"], errors="coerce")
    if out["sweep_idx"].isna().any():
        raise ValueError("Trajectory sweep values must be numeric for chain lineage.")
    out["sweep_idx"] = out["sweep_idx"].astype(int)
    out = out.sort_values(["chain", "sweep_idx"]).drop_duplicates(["chain", "sweep_idx"], keep="last")
    out["x_tf"] = out["x"].astype(float)
    out["y_tf"] = out["y"].astype(float)

    if max_points > 0 and len(out) > max_points:
        grouped = list(out.groupby("chain", sort=True, dropna=False))
        if grouped:
            budget_floor = max(1, max_points // len(grouped))
            budget_remainder = max_points - (budget_floor * len(grouped))
            sampled_parts: list[pd.DataFrame] = []
            for idx, (_, chain_df) in enumerate(grouped):
                budget = budget_floor + (1 if idx < budget_remainder else 0)
                budget = min(len(chain_df), max(1, budget))
                sampled_idx = _subsample_indices_early_priority(len(chain_df), budget)
                sampled_parts.append(chain_df.iloc[sampled_idx])
            out = pd.concat(sampled_parts).sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
        else:
            out = out.iloc[_subsample_indices_early_priority(len(out), max_points)].reset_index(drop=True)

    return out.reset_index(drop=True)
