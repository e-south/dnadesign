"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/label_truth.py

Resolve the campaign's configured, manifest-pinned observed-label truth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dnadesign.opal import (
    ObservedLabelPromotionBinding,
    candidate_exclusion_sets_from_config,
    load_config,
    verify_observed_label_snapshot,
)

LABEL_TRUTH_SOURCE = "stress_ethanol_cipro_growth.response_window_observations"


class LabelTruthError(ValueError):
    """Raised when the campaign cannot identify its required label source."""


@dataclass(frozen=True)
class LabelTruthState:
    """Observed-label readiness derived from OPAL's configured source contract."""

    state: Literal["not_ready", "promoted"]
    label_source_state: Literal["not_verified", "verified"]
    observed_label_promotion_manifest: dict[str, object] | None

    @property
    def ready(self) -> bool:
        return self.state == "promoted"


def resolve_configured_label_truth(campaign_config_path: Path) -> LabelTruthState:
    """Verify the configured label publication when it exists.

    An absent promotion manifest is an honest pre-publication state. Once the
    configured manifest exists, its complete OPAL snapshot must verify.
    """

    config = load_config(Path(campaign_config_path).resolve())
    location = config.data.location
    source = config.labels.source
    if getattr(location, "kind", None) != "usr" or getattr(source, "kind", None) != "usr_sidecar":
        raise LabelTruthError("stress RMF label truth requires a USR sidecar source.")
    manifest_path = getattr(source, "manifest_path", None)
    y_space = config.labels.y_space
    expected_width = config.data.y_expected_length
    if not isinstance(manifest_path, str) or not manifest_path or not isinstance(y_space, str) or not y_space:
        raise LabelTruthError("stress RMF label truth requires a manifest-pinned Y-space.")
    if isinstance(expected_width, bool) or not isinstance(expected_width, int) or expected_width < 1:
        raise LabelTruthError("stress RMF label truth requires a positive expected Y width.")

    dataset_root = (Path(location.path) / str(source.dataset)).resolve()
    resolved_manifest = (dataset_root / manifest_path).resolve()
    if not resolved_manifest.is_relative_to(dataset_root):
        raise LabelTruthError("stress RMF promotion manifest escapes the configured USR dataset.")
    if not resolved_manifest.is_file():
        return LabelTruthState(
            state="not_ready",
            label_source_state="not_verified",
            observed_label_promotion_manifest=None,
        )

    snapshot = verify_observed_label_snapshot(
        ObservedLabelPromotionBinding(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            label_path=str(source.path),
            campaign_slug=str(config.campaign.slug),
            study_id=str(config.ownership.study_id),
            y_space=y_space,
            candidate_id_column=str(config.labels.id_column),
            candidate_x_column=str(config.data.x_column_name),
            candidate_exclusion_sets=candidate_exclusion_sets_from_config(config),
        ),
        expected_y_width=expected_width,
    )
    return LabelTruthState(
        state="promoted",
        label_source_state="verified",
        observed_label_promotion_manifest={
            "path": manifest_path,
            "sha256": snapshot.promotion.manifest_sha256,
        },
    )


__all__ = [
    "LABEL_TRUTH_SOURCE",
    "LabelTruthError",
    "LabelTruthState",
    "resolve_configured_label_truth",
]
