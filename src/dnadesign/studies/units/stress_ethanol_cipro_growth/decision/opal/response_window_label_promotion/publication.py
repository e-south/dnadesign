"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/publication.py

Manifest construction and immutable publication helpers for OPAL labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from dnadesign.opal import (
    OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
    ObservedLabelPromotionBinding,
    ObservedLabelVerificationError,
    VerifiedObservedLabelSnapshot,
    build_candidate_exclusion_projection,
    candidate_snapshot_record,
    load_config,
    verify_observed_label_snapshot,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
    read_json_object,
)

from .contracts import (
    CAMPAIGN_SLUG,
    LABEL_FILENAME,
    PROMOTION_FILENAME,
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_ID,
    STUDY_ID,
    Y_SPACE,
    ResponseWindowLabelPromotionError,
)
from .exclusions import (
    CANDIDATE_EXCLUSION_SET_ID,
    require_campaign_candidate_exclusion_parity,
)
from .provenance_validation import validate_study_provenance


def build_promotion_manifest(
    *,
    relative_dir: PurePosixPath,
    label_path: Path,
    provenance_path: Path,
    candidate_path: Path,
    label_event_count: int,
    candidate_exclusion_entries: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
        "campaign_slug": CAMPAIGN_SLUG,
        "study_id": STUDY_ID,
        "y_space": Y_SPACE,
        "study_provenance": {
            "schema_id": PROVENANCE_SCHEMA_ID,
            "path": (relative_dir / PROVENANCE_FILENAME).as_posix(),
            "sha256": file_sha256(provenance_path),
        },
        "candidate_exclusion_projection": build_candidate_exclusion_projection(
            exclusion_set_id=CANDIDATE_EXCLUSION_SET_ID,
            entries=candidate_exclusion_entries,
        ),
        "candidate_artifact": candidate_snapshot_record(candidate_path),
        "label_artifact": {
            "path": (relative_dir / LABEL_FILENAME).as_posix(),
            "sha256": file_sha256(label_path),
            "row_count": label_event_count,
        },
    }


def verify_label_bundle(
    dataset_root: Path,
    *,
    relative_dir: PurePosixPath,
    expected_width: int,
    candidate_root: Path | None = None,
) -> VerifiedObservedLabelSnapshot:
    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset_root,
        manifest_path=(relative_dir / PROMOTION_FILENAME).as_posix(),
        label_path=(relative_dir / LABEL_FILENAME).as_posix(),
        campaign_slug=CAMPAIGN_SLUG,
        study_id=STUDY_ID,
        y_space=Y_SPACE,
        candidate_path="records.parquet",
        candidate_id_column="id",
        candidate_x_column=None,
        candidate_root=candidate_root,
    )
    try:
        snapshot = verify_observed_label_snapshot(binding, expected_y_width=expected_width)
        provenance = read_json_object(
            snapshot.promotion.study_provenance_path,
            label="response-window label study provenance",
        )
    except (ObservedLabelVerificationError, OSError, UnicodeError, ValueError) as exc:
        raise ResponseWindowLabelPromotionError(f"published OPAL label contract failed: {exc}") from exc
    validate_study_provenance(
        provenance,
        bundle_root=dataset_root,
        candidate_root=dataset_root if candidate_root is None else candidate_root,
        label_path=snapshot.promotion.label_path,
        candidate_sha256=snapshot.promotion.candidate_sha256,
        candidate_row_count=snapshot.promotion.candidate_row_count,
    )
    return snapshot


def verify_campaign_binding(
    dataset_root: Path,
    *,
    relative_dir: PurePosixPath,
    expected_width: int,
    campaign_config_path: Path,
) -> VerifiedObservedLabelSnapshot:
    """Verify one campaign's explicit projection of a verified study bundle."""

    study_snapshot = verify_label_bundle(
        dataset_root,
        relative_dir=relative_dir,
        expected_width=expected_width,
    )
    config = load_config(campaign_config_path)
    _require_campaign_dataset(config, dataset_root, relative_dir=relative_dir)
    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset_root,
        manifest_path=(relative_dir / PROMOTION_FILENAME).as_posix(),
        label_path=(relative_dir / LABEL_FILENAME).as_posix(),
        campaign_slug=CAMPAIGN_SLUG,
        study_id=STUDY_ID,
        y_space=Y_SPACE,
        candidate_path="records.parquet",
        candidate_id_column=config.labels.id_column,
        candidate_x_column=config.data.x_column_name,
    )
    try:
        snapshot = verify_observed_label_snapshot(binding, expected_y_width=expected_width)
    except ObservedLabelVerificationError as exc:
        raise ResponseWindowLabelPromotionError(f"campaign-bound OPAL label contract failed: {exc}") from exc
    if snapshot.promotion.manifest_sha256 != study_snapshot.promotion.manifest_sha256:
        raise ResponseWindowLabelPromotionError("campaign binding resolved a different promotion manifest.")
    provenance = read_json_object(
        snapshot.promotion.study_provenance_path,
        label="response-window label study provenance",
    )
    authoritative_exclusions = validate_study_provenance(
        provenance,
        bundle_root=dataset_root,
        candidate_root=dataset_root,
        label_path=snapshot.promotion.label_path,
        candidate_sha256=snapshot.promotion.candidate_sha256,
        candidate_row_count=snapshot.promotion.candidate_row_count,
    )
    require_campaign_candidate_exclusion_parity(config, authoritative_entries=authoritative_exclusions)
    return snapshot


def _require_campaign_dataset(config, dataset_root: Path, *, relative_dir: PurePosixPath) -> None:
    location = config.data.location
    configured_root = (Path(location.path) / str(location.dataset)).resolve()
    if configured_root != Path(dataset_root).resolve():
        raise ResponseWindowLabelPromotionError(
            "campaign config and label publisher must reference the same candidate dataset root."
        )
    source = config.labels.source
    if (
        config.campaign.slug != CAMPAIGN_SLUG
        or config.ownership.study_id != STUDY_ID
        or config.labels.y_space != Y_SPACE
        or getattr(source, "path", None) != (relative_dir / LABEL_FILENAME).as_posix()
        or getattr(source, "manifest_path", None) != (relative_dir / PROMOTION_FILENAME).as_posix()
    ):
        raise ResponseWindowLabelPromotionError("campaign config disagrees with the response-window label contract.")


def publish_new_directory(*, staged_dir: Path, output_dir: Path) -> None:
    """Atomically publish without replacing an existing scientific artifact."""

    if output_dir.exists():
        raise FileExistsError(f"immutable label promotion already exists: {output_dir}")
    os.rename(staged_dir, output_dir)


__all__ = [
    "build_promotion_manifest",
    "publish_new_directory",
    "verify_campaign_binding",
    "verify_label_bundle",
]
