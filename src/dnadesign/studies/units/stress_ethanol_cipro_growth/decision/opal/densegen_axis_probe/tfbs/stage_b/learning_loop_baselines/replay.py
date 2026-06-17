"""Deterministic replay primitives for frozen learning-loop baselines."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd


def frozen_rank_chunks(
    scores: pd.DataFrame,
    *,
    selection_k: int,
    rounds: int,
    excluded_ids: Collection[str],
    id_column: str = "id",
    score_column: str = "score",
) -> pd.DataFrame:
    """Return deterministic top-ranked chunks from a frozen score table."""

    if selection_k <= 0:
        raise ValueError("frozen replay requires positive selection_k")
    if rounds <= 0:
        raise ValueError("frozen replay requires positive rounds")
    required = {id_column, score_column}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"frozen replay score table missing column(s): {missing}")

    frame = scores.loc[:, [id_column, score_column]].copy()
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].drop_duplicates().head(5).tolist()
        raise ValueError(f"frozen replay score table contains duplicate id(s): {duplicates}")

    excluded = {str(value) for value in excluded_ids}
    frame = frame.loc[~frame[id_column].isin(excluded)].copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="raise")
    if not np.isfinite(frame[score_column].to_numpy(dtype=float)).all():
        raise ValueError("frozen replay score table contains non-finite scores")

    required_count = int(selection_k) * int(rounds)
    if len(frame) < required_count:
        raise ValueError(
            f"frozen replay has insufficient candidates: need {required_count}, found {len(frame)} after exclusions"
        )

    frame["_ordinal"] = np.arange(len(frame), dtype=int)
    ranked = frame.sort_values([score_column, "_ordinal"], ascending=[False, True], kind="mergesort").head(
        required_count
    )
    ranked = ranked.reset_index(drop=True)
    ranked["round"] = ranked.index // int(selection_k)
    ranked["frozen_rank"] = ranked.index + 1
    ranked["selection_source"] = "frozen_round0"
    return ranked.rename(columns={id_column: "id", score_column: "score"})[
        ["round", "id", "frozen_rank", "score", "selection_source"]
    ]


def known_label_rank_chunks(
    labels: pd.DataFrame,
    *,
    label_name: str,
    selection_k: int,
    rounds: int,
    excluded_ids: Collection[str],
    id_column: str = "id",
) -> pd.DataFrame:
    """Return deterministic chunks selected by the known label under the same budget."""

    if selection_k <= 0:
        raise ValueError("known-label ranking replay requires positive selection_k")
    if rounds <= 0:
        raise ValueError("known-label ranking replay requires positive rounds")
    required = {id_column, label_name}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"known-label ranking table missing column(s): {missing}")

    frame = labels.loc[:, [id_column, label_name]].copy()
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].drop_duplicates().head(5).tolist()
        raise ValueError(f"known-label ranking table contains duplicate id(s): {duplicates}")

    excluded = {str(value) for value in excluded_ids}
    frame = frame.loc[~frame[id_column].isin(excluded)].copy()
    frame[label_name] = pd.to_numeric(frame[label_name], errors="raise")
    if not np.isfinite(frame[label_name].to_numpy(dtype=float)).all():
        raise ValueError("known-label ranking table contains non-finite label values")

    required_count = int(selection_k) * int(rounds)
    if len(frame) < required_count:
        raise ValueError(
            f"known-label ranking replay has insufficient candidates: need {required_count}, "
            f"found {len(frame)} after exclusions"
        )

    frame["_ordinal"] = np.arange(len(frame), dtype=int)
    ranked = frame.sort_values([label_name, "_ordinal"], ascending=[False, True], kind="mergesort").head(required_count)
    ranked = ranked.reset_index(drop=True)
    ranked["round"] = ranked.index // int(selection_k)
    ranked["known_label_rank"] = ranked.index + 1
    ranked["selection_source"] = "known_label_ranking"
    return ranked.rename(columns={id_column: "id", label_name: "label_value"})[
        ["round", "id", "known_label_rank", "label_value", "selection_source"]
    ]
