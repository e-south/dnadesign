"""Verify and extend immutable response-window label promotions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    read_json_object,
)

from .contracts import (
    PROMOTION_FILENAME,
    ResponseWindowLabelPromotionError,
    validate_label_frame,
)
from .exclusions import (
    build_candidate_selection_exclusion_provenance,
    validate_candidate_selection_exclusion_provenance,
)
from .publication import verify_label_bundle


@dataclass(frozen=True)
class PriorPromotion:
    """One verified immutable promotion used as cumulative input."""

    labels: pd.DataFrame
    candidate_exclusions: list[dict[str, str]]
    reference: dict[str, object]


def load_prior_promotion(
    manifest_path: Path | None,
    *,
    dataset_root: Path,
    expected_width: int,
) -> PriorPromotion | None:
    """Verify an explicitly named prior promotion against the live candidate snapshot."""

    if manifest_path is None:
        return None
    root = Path(dataset_root).expanduser().resolve()
    raw_path = Path(manifest_path).expanduser()
    resolved = (root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ResponseWindowLabelPromotionError(
            "prior promotion manifest must remain within the dataset root."
        ) from exc
    if resolved.name != PROMOTION_FILENAME or not resolved.is_file():
        raise ResponseWindowLabelPromotionError(f"prior promotion manifest not found: {resolved}")
    relative_dir = PurePosixPath(relative.as_posix()).parent
    snapshot = verify_label_bundle(
        root,
        relative_dir=relative_dir,
        expected_width=expected_width,
    )
    if snapshot.promotion.manifest_path != resolved:
        raise ResponseWindowLabelPromotionError("verified prior promotion does not match the requested manifest path.")
    labels = pd.read_parquet(snapshot.promotion.label_path)
    validate_label_frame(labels, context="prior promotion")
    provenance = read_json_object(
        snapshot.promotion.study_provenance_path,
        label="prior response-window label study provenance",
    )
    exclusions = validate_candidate_selection_exclusion_provenance(provenance["candidate_selection_exclusions"])
    return PriorPromotion(
        labels=labels,
        candidate_exclusions=exclusions,
        reference={
            "label_path": snapshot.promotion.label_path.relative_to(root).as_posix(),
            "label_sha256": snapshot.promotion.label_sha256,
            "manifest_path": resolved.relative_to(root).as_posix(),
            "manifest_sha256": snapshot.promotion.manifest_sha256,
            "label_event_count": snapshot.promotion.row_count,
            "unique_candidate_count": int(labels["id"].astype(str).nunique()),
            "max_observed_round": int(labels["observed_round"].astype(int).max()),
        },
    )


def extend_label_frame(prior: PriorPromotion | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """Carry prior rows unchanged and append exactly one strictly later batch."""

    validate_label_frame(incoming, context="incoming promotion")
    if prior is None:
        return incoming.sort_values(["observed_round", "batch_id", "id"], kind="mergesort").reset_index(drop=True)
    prior_labels = prior.labels
    prior_round = int(prior_labels["observed_round"].max())
    incoming_rounds = incoming["observed_round"].astype(int).unique().tolist()
    incoming_batches = incoming["batch_id"].astype(str).unique().tolist()
    if len(incoming_rounds) != 1 or len(incoming_batches) != 1:
        raise ResponseWindowLabelPromotionError("incoming promotion must contain exactly one study-issued batch.")
    prior_events = set(zip(prior_labels["id"].astype(str), prior_labels["observed_round"].astype(int), strict=True))
    incoming_events = set(zip(incoming["id"].astype(str), incoming["observed_round"].astype(int), strict=True))
    if duplicates := sorted(prior_events & incoming_events):
        raise ResponseWindowLabelPromotionError(
            f"cumulative labels contain duplicate candidate/round events: {duplicates[:10]}"
        )
    if incoming_rounds[0] <= prior_round:
        raise ResponseWindowLabelPromotionError(
            "incoming observed round must be strictly later than every prior promotion round."
        )
    prior_batches = set(prior_labels["batch_id"].astype(str))
    if incoming_batches[0] in prior_batches:
        raise ResponseWindowLabelPromotionError("incoming batch_id reuses an existing promoted batch.")

    cumulative = pd.concat([prior_labels.copy(), incoming.copy()], ignore_index=True)
    duplicates = cumulative.duplicated(subset=["id", "observed_round"], keep=False)
    if duplicates.any():
        sample = cumulative.loc[duplicates, ["id", "observed_round"]].head(10).to_dict(orient="records")
        raise ResponseWindowLabelPromotionError(f"cumulative labels contain duplicate candidate/round events: {sample}")
    return cumulative.sort_values(["observed_round", "batch_id", "id"], kind="mergesort").reset_index(drop=True)


def merge_candidate_exclusions(
    prior_entries: Sequence[Mapping[str, object]],
    incoming_entries: Sequence[Mapping[str, object]],
    *,
    cumulative_labels: pd.DataFrame,
    incoming_labels: pd.DataFrame,
) -> list[dict[str, str]]:
    """Union study exclusions without dropping or silently rewording prior decisions."""

    prior = build_candidate_selection_exclusion_provenance(prior_entries)["entries"]
    incoming = build_candidate_selection_exclusion_provenance(incoming_entries)["entries"]
    incoming_label_ids = set(incoming_labels["id"].astype(str))
    cumulative_label_ids = set(cumulative_labels["id"].astype(str))
    by_id = {
        entry["candidate_id"]: entry["reason"] for entry in prior if entry["candidate_id"] not in incoming_label_ids
    }
    for entry in incoming:
        candidate_id = entry["candidate_id"]
        reason = entry["reason"]
        if candidate_id in cumulative_label_ids:
            raise ResponseWindowLabelPromotionError(
                f"incoming candidate exclusion conflicts with a promoted label: {candidate_id!r}."
            )
        if candidate_id in by_id and by_id[candidate_id] != reason:
            raise ResponseWindowLabelPromotionError(
                f"candidate exclusion reason drift for candidate ID {candidate_id!r}."
            )
        by_id[candidate_id] = reason
    conflicts = sorted(set(by_id) & cumulative_label_ids)
    if conflicts:
        raise ResponseWindowLabelPromotionError(f"candidate exclusions conflict with promoted labels: {conflicts[:10]}")
    return [{"candidate_id": candidate_id, "reason": by_id[candidate_id]} for candidate_id in sorted(by_id)]


__all__ = [
    "PriorPromotion",
    "extend_label_frame",
    "load_prior_promotion",
    "merge_candidate_exclusions",
]
