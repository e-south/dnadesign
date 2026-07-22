"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/observed_sfxi_replay.py

Replay canonical SFXI on its historical observed vec8 label corpus.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dnadesign.opal import SFXIScoringConfig, score_vec8, score_vec8_with_denom

SFXI_VEC8_COLUMNS = (
    "v00",
    "v10",
    "v01",
    "v11",
    "y00_star",
    "y10_star",
    "y01_star",
    "y11_star",
)
_SOURCE_IDENTITY_COLUMNS = ("id", "design_id", "reader_experiment_id")
_LABEL_COLUMNS = ("id", "sequence", "y_obs")
_ACTIVE_IDENTITY_COLUMNS = ("id", "sequence")
_ES_DESIGN_PATTERN = r"^pDual-10-ES[0-9]+p$"


@dataclass(frozen=True)
class ObservedSfxiViewContext:
    """Persisted SFXI settings needed for one historical observed-label replay."""

    selection_view_id: str
    target_mask: tuple[float, float, float, float]
    denom: float
    scaling_percentile: int
    scaling_min_n: int
    scaling_eps: float
    intensity_log2_offset_delta: float
    source_campaign_slug: str
    source_run_id: str

    def __post_init__(self) -> None:
        if not self.selection_view_id or not self.source_campaign_slug or not self.source_run_id:
            raise ValueError("Observed SFXI view identity fields must be non-empty.")
        if len(self.target_mask) != 4 or set(self.target_mask) - {0.0, 1.0}:
            raise ValueError("Observed SFXI target masks must contain four binary values.")
        if not any(self.target_mask) or all(self.target_mask):
            raise ValueError("Observed SFXI target masks must contain at least one ON and one OFF state.")
        if not np.isfinite(self.denom) or self.denom <= 0.0:
            raise ValueError("Observed SFXI persisted denominators must be positive and finite.")
        if not 1 <= self.scaling_percentile <= 100 or self.scaling_min_n < 1:
            raise ValueError("Observed SFXI scaling settings are invalid.")
        if not np.isfinite(self.scaling_eps) or self.scaling_eps <= 0.0:
            raise ValueError("Observed SFXI scaling epsilon must be positive and finite.")
        if not np.isfinite(self.intensity_log2_offset_delta) or self.intensity_log2_offset_delta < 0.0:
            raise ValueError("Observed SFXI intensity offset must be finite and nonnegative.")


def build_observed_sfxi_decomposition(
    source_rows: pd.DataFrame,
    label_rows: pd.DataFrame,
    *,
    view_contexts: tuple[ObservedSfxiViewContext, ...],
    active_identities: pd.DataFrame | None,
    top_k: int = 6,
) -> pd.DataFrame:
    """Score the exact historical SFXI source vectors under each persisted view."""

    if not view_contexts:
        raise ValueError("Observed SFXI replay requires at least one view context.")
    view_ids = [context.selection_view_id for context in view_contexts]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("Observed SFXI replay view IDs must be unique.")
    source = _validated_source_rows(source_rows)
    labels = _validated_label_rows(label_rows)
    active = _validated_active_identities(active_identities) if active_identities is not None else None
    if top_k < 1 or top_k > len(source):
        raise ValueError(f"Observed SFXI top_k must be between 1 and {len(source)}; got {top_k}.")

    label_ids = labels["id"].tolist()
    _require_same_id_universe(source["id"], labels["id"], context="historical SFXI source and label ledger")
    source = source.set_index("id").loc[label_ids].reset_index()
    source_vec8 = source.loc[:, SFXI_VEC8_COLUMNS].to_numpy(dtype=float)
    label_vec8 = _stack_label_vectors(labels["y_obs"])
    if not np.allclose(source_vec8, label_vec8, rtol=0.0, atol=1.0e-12):
        maximum_error = float(np.max(np.abs(source_vec8 - label_vec8)))
        raise ValueError(
            f"Explicit SFXI source vec8 does not match the historical label ledger; maximum error={maximum_error}."
        )
    if active is not None:
        _assert_active_identity_parity(active, labels)

    identity = source.loc[:, _SOURCE_IDENTITY_COLUMNS].merge(
        labels.loc[:, [column for column in ("id", "sequence", "observed_round") if column in labels]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    active_ids = set(active["id"]) if active is not None else set()
    rows: list[pd.DataFrame] = []
    for context in view_contexts:
        config = _scoring_config(context)
        recomputed = score_vec8(source_vec8, config)
        if not np.isclose(recomputed.denom_used, context.denom, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{context.selection_view_id}: persisted denominator {context.denom} does not match "
                f"the historical label pool recomputation {recomputed.denom_used}."
            )
        scored = score_vec8_with_denom(source_vec8, config, denom=context.denom)
        frame = identity.copy()
        frame["selection_view_id"] = context.selection_view_id
        frame["target_mask"] = "|".join(str(int(value)) for value in context.target_mask)
        frame["source_campaign_slug"] = context.source_campaign_slug
        frame["source_run_id"] = context.source_run_id
        frame["source_y_contract"] = "sfxi_vec8"
        frame["denom_persisted"] = float(context.denom)
        frame["denom_recomputed"] = float(recomputed.denom_used)
        frame["logic_fidelity"] = scored.logic_fidelity
        frame["effect_raw"] = scored.effect_raw
        frame["effect_scaled"] = scored.effect_scaled
        frame["sfxi"] = scored.sfxi
        frame["logic_rank"] = _ordinal_rank(frame, "logic_fidelity")
        frame["effect_rank"] = _ordinal_rank(frame, "effect_scaled")
        frame["sfxi_rank"] = _ordinal_rank(frame, "sfxi")
        frame["is_highest_observed_sfxi"] = frame["sfxi_rank"].le(top_k)
        frame["control_role"] = frame["design_id"].map(_control_role)
        frame["is_sensor_control"] = frame["control_role"].ne("")
        frame["is_es_design"] = frame["design_id"].str.fullmatch(_ES_DESIGN_PATTERN).fillna(False)
        if active is None:
            frame["in_promoted_response_window_corpus"] = pd.array([pd.NA] * len(frame), dtype="boolean")
            frame["promoted_response_window_corpus_status"] = "not_available"
        else:
            frame["in_promoted_response_window_corpus"] = frame["id"].isin(active_ids)
            frame["promoted_response_window_corpus_status"] = "verified"
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["selection_view_id", "sfxi_rank"], kind="mergesort").reset_index(drop=True)


def summarize_observed_sfxi_decomposition(detail_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize full-corpus, source-deletion, and ES-design rank associations."""

    required = {
        "selection_view_id",
        "reader_experiment_id",
        "design_id",
        "sfxi",
        "logic_fidelity",
        "effect_scaled",
    }
    if missing := sorted(required - set(detail_rows.columns)):
        raise ValueError(f"Observed SFXI decomposition summary lacks required columns: {missing}")
    if detail_rows.empty:
        raise ValueError("Observed SFXI decomposition summary requires at least one row.")
    rows: list[dict[str, object]] = []
    for selection_view_id, view in detail_rows.groupby("selection_view_id", sort=True):
        rows.append(_correlation_row(view, selection_view_id=str(selection_view_id), scope="all_observed_labels"))
        for experiment_id in sorted(view["reader_experiment_id"].astype(str).unique()):
            subset = view.loc[view["reader_experiment_id"].astype(str).ne(experiment_id)]
            rows.append(
                _correlation_row(
                    subset,
                    selection_view_id=str(selection_view_id),
                    scope="leave_one_experiment_out",
                    excluded_reader_experiment_id=experiment_id,
                )
            )
        es_only = view.loc[view["design_id"].astype(str).str.fullmatch(_ES_DESIGN_PATTERN).fillna(False)]
        rows.append(_correlation_row(es_only, selection_view_id=str(selection_view_id), scope="es_designs_only"))
    return pd.DataFrame.from_records(rows)


def _validated_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*_SOURCE_IDENTITY_COLUMNS, *SFXI_VEC8_COLUMNS}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Historical source must provide explicit SFXI vec8 columns; missing={missing}.")
    result = frame.loc[:, [*_SOURCE_IDENTITY_COLUMNS, *SFXI_VEC8_COLUMNS]].copy()
    _validate_unique_nonempty_ids(result, context="historical SFXI source")
    for column in ("design_id", "reader_experiment_id"):
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"Historical SFXI source column {column!r} contains empty values.")
    values = result.loc[:, SFXI_VEC8_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Historical SFXI source vec8 contains non-finite values.")
    return result


def _validated_label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := sorted(set(_LABEL_COLUMNS) - set(frame.columns)):
        raise ValueError(f"Historical SFXI label ledger lacks required columns: {missing}")
    columns = [*_LABEL_COLUMNS, *(column for column in ("observed_round",) if column in frame)]
    result = frame.loc[:, columns].copy()
    _validate_unique_nonempty_ids(result, context="historical SFXI label ledger")
    result["sequence"] = result["sequence"].astype(str)
    if result["sequence"].str.strip().eq("").any() or result["sequence"].duplicated().any():
        raise ValueError("Historical SFXI label sequences must be non-empty and unique.")
    _stack_label_vectors(result["y_obs"])
    return result


def _validated_active_identities(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := sorted(set(_ACTIVE_IDENTITY_COLUMNS) - set(frame.columns)):
        raise ValueError(f"Promoted response-window identities lack required columns: {missing}")
    result = frame.loc[:, _ACTIVE_IDENTITY_COLUMNS].copy()
    _validate_unique_nonempty_ids(result, context="promoted response-window identities")
    result["sequence"] = result["sequence"].astype(str)
    if result["sequence"].str.strip().eq("").any() or result["sequence"].duplicated().any():
        raise ValueError("Promoted response-window sequences must be non-empty and unique.")
    return result


def _validate_unique_nonempty_ids(frame: pd.DataFrame, *, context: str) -> None:
    frame["id"] = frame["id"].astype(str)
    if frame["id"].str.strip().eq("").any() or frame["id"].duplicated().any():
        raise ValueError(f"{context} IDs must be non-empty and unique.")


def _require_same_id_universe(left: pd.Series, right: pd.Series, *, context: str) -> None:
    left_ids = set(left.astype(str))
    right_ids = set(right.astype(str))
    if left_ids != right_ids:
        raise ValueError(
            f"{context} candidate IDs disagree; missing={sorted(right_ids - left_ids)[:5]}, "
            f"extra={sorted(left_ids - right_ids)[:5]}."
        )


def _assert_active_identity_parity(active: pd.DataFrame, labels: pd.DataFrame) -> None:
    historical = labels.set_index("id")["sequence"].astype(str)
    missing = sorted(set(active["id"]) - set(historical.index))
    if missing:
        raise ValueError(
            f"Promoted response-window candidate IDs are absent from historical SFXI labels: {missing[:5]}"
        )
    for row in active.itertuples(index=False):
        if str(row.sequence) != historical.loc[str(row.id)]:
            raise ValueError(f"Promoted response-window sequence does not match historical SFXI label for {row.id!r}.")


def _stack_label_vectors(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=float).ravel() for value in values]
    if not rows or {row.size for row in rows} != {8}:
        raise ValueError("Historical SFXI label y_obs vectors must all have length eight.")
    matrix = np.vstack(rows)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Historical SFXI label y_obs contains non-finite values.")
    return matrix


def _scoring_config(context: ObservedSfxiViewContext) -> SFXIScoringConfig:
    return SFXIScoringConfig(
        setpoint_vector=context.target_mask,
        scaling_percentile=context.scaling_percentile,
        scaling_min_n=context.scaling_min_n,
        scaling_eps=context.scaling_eps,
        logic_exponent_beta=1.0,
        intensity_exponent_gamma=1.0,
        intensity_log2_offset_delta=context.intensity_log2_offset_delta,
    )


def _ordinal_rank(frame: pd.DataFrame, value_column: str) -> pd.Series:
    ordered = frame.sort_values([value_column, "id"], ascending=[False, True], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index, dtype=int)
    return ranks.reindex(frame.index)


def _control_role(design_id: object) -> str:
    value = str(design_id).lower()
    if value.endswith("-spyp"):
        return "SpyP"
    if value.endswith("-sulap"):
        return "sulAp"
    return ""


def _correlation_row(
    frame: pd.DataFrame,
    *,
    selection_view_id: str,
    scope: str,
    excluded_reader_experiment_id: str = "",
) -> dict[str, object]:
    logic = _spearman(frame["sfxi"], frame["logic_fidelity"])
    effect = _spearman(frame["sfxi"], frame["effect_scaled"])
    return {
        "selection_view_id": selection_view_id,
        "sensitivity_scope": scope,
        "excluded_reader_experiment_id": excluded_reader_experiment_id,
        "candidate_count": int(len(frame)),
        "sfxi_vs_logic_spearman": logic,
        "sfxi_vs_effect_spearman": effect,
        "correlation_defined": bool(np.isfinite(logic) and np.isfinite(effect)),
    }


def _spearman(left: pd.Series, right: pd.Series) -> float:
    x = pd.Series(left, dtype=float)
    y = pd.Series(right, dtype=float)
    if len(x) < 2 or x.nunique(dropna=False) < 2 or y.nunique(dropna=False) < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


__all__ = [
    "ObservedSfxiViewContext",
    "SFXI_VEC8_COLUMNS",
    "build_observed_sfxi_decomposition",
    "summarize_observed_sfxi_decomposition",
]
