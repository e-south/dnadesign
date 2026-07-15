"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/observed_labels.py

Public verification API for immutable observed-label snapshots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from ..src.core.utils import OpalError
from ..src.storage.candidate_snapshot import candidate_snapshot_record
from ..src.storage.label_sources import ObservedLabelStore
from ..src.storage.observed_label_promotion import (
    OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
    ObservedLabelPromotionBinding,
    VerifiedObservedLabelPromotion,
)

OBSERVED_LABELS_API_VERSION = "1"


class ObservedLabelVerificationError(ValueError):
    """Raised when an immutable observed-label snapshot violates its contract."""


@dataclass(frozen=True)
class VerifiedObservedLabelSnapshot:
    """Verified promotion identity and its canonical candidate-label rows."""

    promotion: VerifiedObservedLabelPromotion
    labels: pd.DataFrame


def verify_observed_label_snapshot(
    binding: ObservedLabelPromotionBinding,
    *,
    expected_y_width: int,
) -> VerifiedObservedLabelSnapshot:
    """Verify and materialize one immutable candidate-label snapshot.

    Candidate IDs must occur exactly once across the complete snapshot. Each
    label must be a finite one-dimensional vector of ``expected_y_width``
    values. Manifest, provenance, artifact digest, and row-count verification
    are delegated to OPAL's storage contract.
    """

    if isinstance(expected_y_width, bool) or not isinstance(expected_y_width, Integral) or expected_y_width < 1:
        raise ObservedLabelVerificationError("expected_y_width must be a positive integer.")

    try:
        store = ObservedLabelStore(
            path=(Path(binding.dataset_root) / binding.label_path).resolve(),
            y_space=binding.y_space,
            dedup_policy="error_on_duplicate",
            promotion=binding,
        )
        frame = store._validated_frame()
        duplicate_ids = frame[store.id_column].duplicated(keep=False)
        if duplicate_ids.any():
            sample = sorted(frame.loc[duplicate_ids, store.id_column].astype(str).unique().tolist())[:10]
            raise OpalError(f"Duplicate labels for ids at multiple rounds (sample={sample}).")
        frame = frame.sort_values([store.id_column, store.round_column, "_row_order"])
        labels = pd.DataFrame(
            {
                "id": frame[store.id_column].astype(str).tolist(),
                "y": frame["y"].tolist(),
                "r": frame[store.round_column].astype(int).tolist(),
            }
        )
        promotion = store.verified_promotion()
    except OpalError as exc:
        raise ObservedLabelVerificationError(str(exc)) from exc

    if promotion is None:  # pragma: no cover - construction above always binds a manifest
        raise ObservedLabelVerificationError("observed-label snapshot requires a promotion manifest.")
    if labels.empty:
        raise ObservedLabelVerificationError("observed-label snapshot must contain at least one candidate label.")
    bad_widths = labels.loc[labels["y"].map(len) != int(expected_y_width), ["id", "y"]]
    if not bad_widths.empty:
        sample = bad_widths.head(5).to_dict(orient="records")
        raise ObservedLabelVerificationError(
            f"Observed label source y_obs length mismatch: expected {int(expected_y_width)} values per label "
            f"(sample={sample})."
        )
    return VerifiedObservedLabelSnapshot(promotion=promotion, labels=labels)


__all__ = [
    "OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION",
    "OBSERVED_LABELS_API_VERSION",
    "ObservedLabelPromotionBinding",
    "ObservedLabelVerificationError",
    "VerifiedObservedLabelPromotion",
    "VerifiedObservedLabelSnapshot",
    "candidate_snapshot_record",
    "verify_observed_label_snapshot",
]
