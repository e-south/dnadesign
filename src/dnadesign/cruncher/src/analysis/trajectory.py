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


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _validate_numeric(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"Trajectory objective column '{column}' contains non-numeric values.")
    return values.astype(float)


def _softmin(values: np.ndarray, beta: float) -> float:
    scaled = -beta * values
    max_scaled = float(np.max(scaled))
    logsum = max_scaled + float(np.log(np.exp(scaled - max_scaled).sum()))
    return float(-logsum / beta)


def resolve_cold_chain(*, beta_ladder: list[float] | None, chain_ids: Iterable[int]) -> int:
    ids = sorted({int(v) for v in chain_ids})
    if not ids:
        raise ValueError("Cannot resolve cold chain from an empty chain set.")
    if len(ids) == 1:
        return ids[0]
    if beta_ladder is None:
        raise ValueError("Missing optimizer beta ladder; cannot resolve cold chain for multi-chain trajectory.")
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
    cold_idx = int(np.argmax(beta_arr))
    if cold_idx not in ids:
        raise ValueError(
            f"optimizer beta ladder cold index is not present in trajectory chain IDs (cold={cold_idx}, chains={ids})."
        )
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

    overflow = len(sampled) - max_points
    non_cold_idx = sampled.index[sampled["is_cold_chain"] == 0]
    drop_idx = list(non_cold_idx[-overflow:])
    if len(drop_idx) < overflow:
        remain = overflow - len(drop_idx)
        cold_idx = sampled.index[sampled["is_cold_chain"] == 1]
        drop_idx.extend(list(cold_idx[-remain:]))
    return sampled.drop(index=drop_idx).reset_index(drop=True)


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
        df["chain"] = 0
    if "draw" in df.columns:
        df = df.sort_values(["chain", "draw"])
    else:
        df = df.sort_values(["chain"])

    x, y, x_metric, y_metric, worst, second = project_scores(df, tf_names)
    score_cols = _score_columns(tf_names)
    cols = [c for c in ("draw", "phase", "chain") if c in df.columns] + score_cols
    out = df[cols].copy()
    if "draw" in out.columns:
        out["sweep"] = out["draw"].astype(int)
    else:
        out["sweep"] = out.groupby("chain").cumcount()
    out["x"] = x
    out["y"] = y
    out["x_metric"] = x_metric
    out["y_metric"] = y_metric
    out["worst_tf_score"] = worst
    out["second_worst_tf_score"] = second
    out["objective_scalar"] = _resolve_objective_scalar(df, tf_names, objective_config=objective_config)
    cold_chain = resolve_cold_chain(beta_ladder=beta_ladder, chain_ids=out["chain"].astype(int).unique())
    out["is_cold_chain"] = (out["chain"].astype(int) == int(cold_chain)).astype(int)

    return _subsample_chainwise(out.reset_index(drop=True), max_points)


def compute_best_so_far_path(
    trajectory_df: pd.DataFrame,
    *,
    objective_col: str = "objective_scalar",
    sweep_col: str = "sweep",
) -> pd.DataFrame:
    if trajectory_df is None or trajectory_df.empty:
        raise ValueError("Cannot compute best-so-far path from empty trajectory data.")
    if objective_col not in trajectory_df.columns:
        raise ValueError(f"Missing objective column for best-so-far path: {objective_col}")
    if sweep_col not in trajectory_df.columns:
        raise ValueError(f"Missing sweep column for best-so-far path: {sweep_col}")
    if "x" not in trajectory_df.columns or "y" not in trajectory_df.columns:
        raise ValueError("Trajectory data must include x/y columns for best-so-far path.")

    df = trajectory_df.copy()
    df[objective_col] = pd.to_numeric(df[objective_col], errors="coerce")
    if df[objective_col].isna().any():
        raise ValueError(f"Trajectory objective column '{objective_col}' contains non-numeric values.")
    df[sweep_col] = pd.to_numeric(df[sweep_col], errors="coerce")
    if df[sweep_col].isna().any():
        raise ValueError(f"Trajectory sweep column '{sweep_col}' contains non-numeric values.")
    df[sweep_col] = df[sweep_col].astype(int)

    winners = (
        df.sort_values([sweep_col, objective_col], ascending=[True, False])
        .groupby(sweep_col, sort=True, as_index=False)
        .first()
        .sort_values(sweep_col)
        .reset_index(drop=True)
    )
    if winners.empty:
        raise ValueError("Cannot compute best-so-far path from empty per-sweep winners.")

    best_rows: list[dict[str, float | int]] = []
    best_idx = 0
    best_value = float(winners[objective_col].iloc[0])
    for idx, row in winners.iterrows():
        score = float(row[objective_col])
        if score >= best_value:
            best_value = score
            best_idx = idx
        best_row = winners.iloc[best_idx]
        best_rows.append(
            {
                str(sweep_col): int(row[sweep_col]),
                "x": float(best_row["x"]),
                "y": float(best_row["y"]),
                str(objective_col): float(best_row[objective_col]),
            }
        )
    return pd.DataFrame(best_rows)
