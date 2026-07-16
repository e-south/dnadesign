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
from datetime import UTC, datetime
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
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact import (
    SCHEMA_ID as OBSERVATION_SCHEMA_ID,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
    read_json_object,
)

from .contracts import (
    CAMPAIGN_SLUG,
    DEFAULT_CAMPAIGN_CONFIG_PATH,
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
    build_candidate_selection_exclusion_provenance,
    require_campaign_candidate_exclusion_parity,
    validate_candidate_selection_exclusion_provenance,
)

_PROVENANCE_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "created_at",
    "observation_bundle",
    "candidate_table",
    "candidate_selection_exclusions",
    "label_contract",
}


def build_study_provenance(
    *,
    observation_manifest: dict[str, object],
    observation_manifest_sha256: str,
    candidate_records_sha256: str,
    candidate_record_count: int,
    label_count: int,
    observed_round: int,
    batch_id: str,
    candidate_exclusion_entries: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_id": PROVENANCE_SCHEMA_ID,
        "schema_version": "1",
        "study_id": STUDY_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "observation_bundle": {
            "schema_id": OBSERVATION_SCHEMA_ID,
            "manifest_sha256": observation_manifest_sha256,
            "policy_id": observation_manifest["policy"]["policy_id"],
            "source_manifests": observation_manifest["source_manifests"],
        },
        "candidate_table": {
            "records_sha256": candidate_records_sha256,
            "record_count": candidate_record_count,
        },
        "candidate_selection_exclusions": build_candidate_selection_exclusion_provenance(candidate_exclusion_entries),
        "label_contract": {
            "y_space": Y_SPACE,
            "value_order": list(VALUE_COLUMNS),
            "observed_round": observed_round,
            "batch_id": batch_id,
            "row_count": label_count,
        },
    }


def build_promotion_manifest(
    *,
    relative_dir: PurePosixPath,
    label_path: Path,
    provenance_path: Path,
    candidate_path: Path,
    label_count: int,
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
            "row_count": label_count,
        },
    }


def verify_label_bundle(
    dataset_root: Path,
    *,
    relative_dir: PurePosixPath,
    expected_width: int,
    campaign_config_path: Path | None = DEFAULT_CAMPAIGN_CONFIG_PATH,
    candidate_root: Path | None = None,
) -> VerifiedObservedLabelSnapshot:
    config = None if campaign_config_path is None else load_config(campaign_config_path)
    if config is not None:
        _require_campaign_dataset(
            config,
            dataset_root if candidate_root is None else candidate_root,
            relative_dir=relative_dir,
        )
    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset_root,
        manifest_path=(relative_dir / PROMOTION_FILENAME).as_posix(),
        label_path=(relative_dir / LABEL_FILENAME).as_posix(),
        campaign_slug=CAMPAIGN_SLUG,
        study_id=STUDY_ID,
        y_space=Y_SPACE,
        candidate_path="records.parquet",
        candidate_id_column="id" if config is None else config.labels.id_column,
        candidate_x_column=None if config is None else config.data.x_column_name,
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
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ResponseWindowLabelPromotionError("published study provenance fields disagree.")
    if provenance["schema_id"] != PROVENANCE_SCHEMA_ID or provenance["study_id"] != STUDY_ID:
        raise ResponseWindowLabelPromotionError("published study provenance identity disagrees.")
    authoritative_exclusions = validate_candidate_selection_exclusion_provenance(
        provenance["candidate_selection_exclusions"]
    )
    if config is not None:
        require_campaign_candidate_exclusion_parity(
            config,
            authoritative_entries=authoritative_exclusions,
        )
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
    "build_study_provenance",
    "publish_new_directory",
    "verify_label_bundle",
]
