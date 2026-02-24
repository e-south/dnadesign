"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/score_space.py

Resolve score-space projection modes and TF-axis selections for analysis plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd


def _select_tf_pair(tf_names: list[str], pairwise: object) -> tuple[str, str] | None:
    if pairwise == "off":
        return None
    if pairwise == "all_pairs_grid":
        if len(tf_names) < 2:
            return None
        return (tf_names[0], tf_names[1])
    if isinstance(pairwise, list) and len(pairwise) == 2:
        a, b = str(pairwise[0]), str(pairwise[1])
        if a not in tf_names or b not in tf_names:
            raise ValueError("analysis.pairwise TFs must be present in the run.")
        return (a, b)
    if len(tf_names) >= 2:
        return (tf_names[0], tf_names[1])
    return None


def _resolve_trajectory_tf_pair(tf_names: list[str], pairwise: object) -> tuple[str, str]:
    selected = _select_tf_pair(tf_names, pairwise)
    if selected is not None:
        return selected
    if len(tf_names) == 1:
        tf_name = str(tf_names[0])
        return (tf_name, tf_name)
    raise ValueError("Trajectory scatter plot requires at least one TF.")


def _all_tf_pairs(tf_names: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for idx, tf_name in enumerate(tf_names):
        for tf_other in tf_names[idx + 1 :]:
            pairs.append((str(tf_name), str(tf_other)))
    return pairs


def _resolve_score_space_spec(tf_names: list[str], pairwise: object) -> dict[str, object]:
    if isinstance(pairwise, list):
        if len(pairwise) != 2:
            raise ValueError("analysis.pairwise must be 'off', 'auto', 'all_pairs_grid', or a list of two TF names")
        tf_a = str(pairwise[0]).strip()
        tf_b = str(pairwise[1]).strip()
        if tf_a not in tf_names or tf_b not in tf_names:
            raise ValueError("analysis.pairwise TFs must be present in the run.")
        return {
            "mode": "pair",
            "pairs": [(tf_a, tf_b)],
            "x_metric": f"score_{tf_a}",
            "y_metric": f"score_{tf_b}",
        }

    mode = str(pairwise).strip().lower()
    if mode == "all_pairs_grid":
        if len(tf_names) < 2:
            raise ValueError("analysis.pairwise=all_pairs_grid requires at least two TFs.")
        return {
            "mode": "all_pairs_grid",
            "pairs": _all_tf_pairs(tf_names),
            "x_metric": None,
            "y_metric": None,
        }
    if mode == "off":
        if len(tf_names) == 1:
            tf_name = str(tf_names[0]).strip()
            return {
                "mode": "pair",
                "pairs": [(tf_name, tf_name)],
                "x_metric": f"score_{tf_name}",
                "y_metric": f"score_{tf_name}",
            }
        raise ValueError("analysis.pairwise=off disables score-space projection for multi-TF runs.")
    if mode != "auto":
        raise ValueError("analysis.pairwise must be 'off', 'auto', 'all_pairs_grid', or a list of two TF names")

    if len(tf_names) >= 3:
        return {
            "mode": "worst_vs_second_worst",
            "pairs": [],
            "x_metric": "worst_tf_score",
            "y_metric": "second_worst_tf_score",
        }
    if len(tf_names) == 2:
        tf_a, tf_b = str(tf_names[0]), str(tf_names[1])
        return {
            "mode": "pair",
            "pairs": [(tf_a, tf_b)],
            "x_metric": f"score_{tf_a}",
            "y_metric": f"score_{tf_b}",
        }
    tf_name = str(tf_names[0])
    return {
        "mode": "pair",
        "pairs": [(tf_name, tf_name)],
        "x_metric": f"score_{tf_name}",
        "y_metric": f"score_{tf_name}",
    }


def _resolve_worst_second_tf_pair(
    *,
    elites_df: pd.DataFrame,
    tf_names: list[str],
    score_prefix: str,
) -> tuple[str, str]:
    if elites_df is None or elites_df.empty:
        raise ValueError("Cannot resolve worst/second TF axis pair without elite rows.")
    if len(tf_names) < 2:
        raise ValueError("Cannot resolve worst/second TF axis pair with fewer than two TFs.")
    score_cols = [f"{score_prefix}{tf_name}" for tf_name in tf_names]
    missing = [column for column in score_cols if column not in elites_df.columns]
    if missing:
        raise ValueError(f"Cannot resolve worst/second TF axis pair; missing score columns: {missing}")
    score_frame = elites_df[score_cols].apply(pd.to_numeric, errors="coerce")
    if score_frame.isna().any().any():
        raise ValueError("Cannot resolve worst/second TF axis pair; elite score columns must be numeric.")
    worst_cols = score_frame.idxmin(axis=1).astype(str)
    second_cols = score_frame.apply(lambda row: row.nsmallest(2).index[-1], axis=1).astype(str)
    if worst_cols.empty or second_cols.empty:
        raise ValueError("Cannot resolve worst/second TF axis pair from empty elite score rankings.")
    for column in pd.concat([worst_cols, second_cols], axis=0):
        if not str(column).startswith(score_prefix):
            raise ValueError(f"Invalid score column while resolving worst/second TF pair: {column!r}")
    worst_tfs = [str(column).removeprefix(score_prefix) for column in worst_cols.tolist()]
    second_tfs = [str(column).removeprefix(score_prefix) for column in second_cols.tolist()]

    def _select(values: list[str], *, exclude: set[str] | None = None) -> str:
        blocked = exclude or set()
        counts: dict[str, int] = {}
        for value in values:
            if value in blocked:
                continue
            counts[value] = counts.get(value, 0) + 1
        if not counts:
            raise ValueError("Cannot resolve worst/second TF axis pair after applying exclusions.")
        return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]

    x_tf = _select(worst_tfs)
    y_tf = _select(second_tfs, exclude={x_tf})
    return str(x_tf), str(y_tf)
