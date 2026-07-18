"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/contracts.py

Identity and row contracts for response-window label promotion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)

CAMPAIGN_SLUG = "secg_msrb_greedy"
STUDY_ID = "stress_ethanol_cipro_growth"
Y_SPACE = "reader_response_window_vector_v1"
PROVENANCE_SCHEMA_ID = "stress_ethanol_cipro_growth.response_window_label_promotion.v5"
DEFAULT_OUTPUT_DIRECTORY = "_opal/response_window_labels_v5"
LABEL_FILENAME = "observed_labels.parquet"
PROVENANCE_FILENAME = "study_provenance.json"
PROMOTION_FILENAME = "promotion.manifest.json"
SOURCE_OBSERVATION_MANIFEST_FILENAME = "source_observation.manifest.json"
LABEL_COLUMNS = (
    "id",
    "display_label",
    "observed_round",
    "batch_id",
    "y_space",
    "y_obs",
)


class ResponseWindowLabelPromotionError(ValueError):
    """Raised when study observations cannot satisfy OPAL's immutable label contract."""


@dataclass(frozen=True)
class ResponseWindowLabelPromotionResult:
    output_directory: Path
    label_path: Path
    study_provenance_path: Path
    promotion_manifest_path: Path
    label_event_count: int
    unique_candidate_count: int


def build_label_frame(observations: pd.DataFrame, *, observed_round: int, batch_id: str) -> pd.DataFrame:
    values = observations.loc[:, VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != len(VALUE_COLUMNS) or not np.isfinite(values).all():
        raise ResponseWindowLabelPromotionError("candidate observations must contain finite eight-value vectors.")
    ids = observations["candidate_id"].astype(str)
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ResponseWindowLabelPromotionError("candidate observations require unique non-empty IDs.")
    if "display_label" not in observations.columns:
        raise ResponseWindowLabelPromotionError("candidate observations require study-issued display labels.")
    labels = observations["display_label"]
    label_strings = labels.map(lambda value: None if pd.isna(value) else str(value))
    canonical_labels = label_strings.map(lambda value: None if value is None else value.strip())
    invalid_labels = canonical_labels.isna() | canonical_labels.eq("") | label_strings.ne(canonical_labels)
    if invalid_labels.any():
        raise ResponseWindowLabelPromotionError("candidate observations require canonical non-empty display labels.")
    return (
        pd.DataFrame(
            {
                "id": ids.tolist(),
                "display_label": label_strings.tolist(),
                "observed_round": [observed_round] * len(ids),
                "batch_id": [batch_id] * len(ids),
                "y_space": [Y_SPACE] * len(ids),
                "y_obs": [row.tolist() for row in values],
            }
        )
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )


def validate_label_frame(frame: pd.DataFrame, *, context: str) -> None:
    """Validate the complete study label-event row contract."""

    if frame.columns.tolist() != list(LABEL_COLUMNS) or frame.empty:
        raise ResponseWindowLabelPromotionError(
            f"{context} labels must be nonempty with columns {list(LABEL_COLUMNS)}."
        )
    ids = frame["id"].astype(str)
    labels = frame["display_label"].astype(str)
    batches = frame["batch_id"].astype(str)
    if (
        frame[list(LABEL_COLUMNS)].isna().any().any()
        or ids.str.strip().ne(ids).any()
        or ids.str.strip().eq("").any()
        or labels.str.strip().ne(labels).any()
        or labels.str.strip().eq("").any()
        or batches.str.strip().ne(batches).any()
        or batches.str.strip().eq("").any()
        or frame["y_space"].astype(str).ne(Y_SPACE).any()
    ):
        raise ResponseWindowLabelPromotionError(f"{context} labels contain noncanonical row metadata.")
    rounds = pd.to_numeric(frame["observed_round"], errors="coerce")
    if rounds.isna().any() or (rounds < 0).any() or (rounds != rounds.astype(int)).any():
        raise ResponseWindowLabelPromotionError(f"{context} labels contain invalid observed rounds.")
    if frame.duplicated(subset=["id", "observed_round"], keep=False).any():
        raise ResponseWindowLabelPromotionError(f"{context} labels contain duplicate candidate/round events.")
    try:
        vectors = [np.asarray(value, dtype=float) for value in frame["y_obs"]]
    except (TypeError, ValueError) as exc:
        raise ResponseWindowLabelPromotionError(f"{context} labels contain nonnumeric y_obs values.") from exc
    if any(
        vector.ndim != 1 or vector.size != len(VALUE_COLUMNS) or not np.isfinite(vector).all() for vector in vectors
    ):
        raise ResponseWindowLabelPromotionError(
            f"{context} labels require finite one-dimensional y_obs vectors of width {len(VALUE_COLUMNS)}."
        )


def verify_candidate_identity(observations: pd.DataFrame, *, records: pd.DataFrame) -> None:
    if not {"id", "sequence"}.issubset(records.columns):
        raise ResponseWindowLabelPromotionError("OPAL candidate records require id and sequence columns.")
    candidates = records.loc[:, ["id", "sequence"]].copy()
    candidates["id"] = candidates["id"].astype(str)
    if candidates["id"].duplicated().any():
        raise ResponseWindowLabelPromotionError("OPAL candidate records contain duplicate IDs.")
    observed = observations.loc[:, ["candidate_id", "sequence_sha256"]].rename(columns={"candidate_id": "id"})
    observed["id"] = observed["id"].astype(str)
    merged = observed.merge(candidates, on="id", how="left", validate="one_to_one")
    if merged["sequence"].isna().any():
        missing = sorted(merged.loc[merged["sequence"].isna(), "id"].tolist())
        raise ResponseWindowLabelPromotionError(
            f"candidate observation IDs are absent from OPAL records: {missing[:10]}"
        )
    actual = merged["sequence"].astype(str).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    mismatch = actual.ne(merged["sequence_sha256"].astype(str))
    if mismatch.any():
        mismatched_ids = sorted(merged.loc[mismatch, "id"].tolist())
        raise ResponseWindowLabelPromotionError(
            f"candidate observation sequence digests disagree with OPAL records: {mismatched_ids[:10]}"
        )


def require_observation_contract(contract: object) -> dict[str, object]:
    if not isinstance(contract, dict):
        raise ResponseWindowLabelPromotionError("observation bundle lacks its observation contract.")
    if contract.get("y_space") != Y_SPACE or contract.get("value_order") != list(VALUE_COLUMNS):
        raise ResponseWindowLabelPromotionError("observation bundle Y-space contract disagrees with this promotion.")
    return contract


def confined_relative_directory(value: str) -> PurePosixPath:
    raw = str(value).strip()
    path = PurePosixPath(raw)
    if not raw or "\\" in raw or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ResponseWindowLabelPromotionError("label output directory must be a confined relative POSIX path.")
    return path


__all__ = [
    "CAMPAIGN_SLUG",
    "DEFAULT_OUTPUT_DIRECTORY",
    "LABEL_FILENAME",
    "LABEL_COLUMNS",
    "PROMOTION_FILENAME",
    "PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA_ID",
    "SOURCE_OBSERVATION_MANIFEST_FILENAME",
    "STUDY_ID",
    "Y_SPACE",
    "ResponseWindowLabelPromotionError",
    "ResponseWindowLabelPromotionResult",
    "build_label_frame",
    "confined_relative_directory",
    "require_observation_contract",
    "validate_label_frame",
    "verify_candidate_identity",
]
