"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/publisher.py

Publish verified response-window observations through OPAL's label contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact import (
    RECORD_FILES,
    ResponseWindowObservationArtifactError,
    verify_response_window_observations,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
    read_json_object,
)

from .contracts import (
    DEFAULT_CAMPAIGN_CONFIG_PATH,
    DEFAULT_OUTPUT_DIRECTORY,
    LABEL_FILENAME,
    PROMOTION_FILENAME,
    PROVENANCE_FILENAME,
    ResponseWindowLabelPromotionError,
    ResponseWindowLabelPromotionResult,
    build_label_frame,
    confined_relative_directory,
    require_observation_contract,
    verify_candidate_identity,
)
from .exclusions import (
    derive_candidate_selection_exclusions,
    require_exclusion_candidates_in_records,
)
from .publication import (
    build_promotion_manifest,
    build_study_provenance,
    publish_new_directory,
    verify_label_bundle,
)


def publish_response_window_labels(
    *,
    observation_bundle_dir: Path,
    dataset_root: Path,
    output_relative_directory: str = DEFAULT_OUTPUT_DIRECTORY,
    campaign_config_path: Path | None = DEFAULT_CAMPAIGN_CONFIG_PATH,
) -> ResponseWindowLabelPromotionResult:
    """Publish labels only from one verified, approved study-observation bundle."""

    first_verification = verify_response_window_observations(observation_bundle_dir)
    observation_manifest = read_json_object(
        first_verification.manifest_json,
        label="response-window observation manifest",
    )
    contract = require_observation_contract(observation_manifest.get("observation_contract"))
    observations = pd.read_parquet(first_verification.observations_parquet)
    contributions = pd.read_parquet(first_verification.manifest_json.parent / RECORD_FILES["contributions"])
    candidate_exclusions = derive_candidate_selection_exclusions(observations, contributions)
    if candidate_exclusions and campaign_config_path is None:
        raise ResponseWindowLabelPromotionError("publication requires a campaign config for nonempty exclusions.")
    _require_stable_observation_read(
        observation_bundle_dir,
        expected_manifest_sha256=first_verification.manifest_sha256,
    )

    root = Path(dataset_root).expanduser().resolve()
    records_path = root / "records.parquet"
    if not records_path.is_file():
        raise ResponseWindowLabelPromotionError(f"OPAL candidate records not found: {records_path}")
    candidate_records_sha256 = file_sha256(records_path)
    records = pd.read_parquet(records_path, columns=["id", "sequence"])
    if file_sha256(records_path) != candidate_records_sha256:
        raise ResponseWindowLabelPromotionError("OPAL candidate records changed while labels were being prepared.")
    verify_candidate_identity(observations, records=records)
    require_exclusion_candidates_in_records(candidate_exclusions, records=records)
    labels = build_label_frame(
        observations,
        observed_round=int(contract["observed_round"]),
        batch_id=str(contract["batch_id"]),
    )

    relative_dir = confined_relative_directory(output_relative_directory)
    output = (root / relative_dir).resolve()
    _require_new_confined_output(output, root=root)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".response-window-labels-staging-", dir=output.parent) as temporary:
        staging_root = Path(temporary)
        staged_output = staging_root / relative_dir
        staged_output.mkdir(parents=True, exist_ok=False)
        label_path = staged_output / LABEL_FILENAME
        provenance_path = staged_output / PROVENANCE_FILENAME
        promotion_path = staged_output / PROMOTION_FILENAME
        labels.to_parquet(label_path, index=False)
        provenance = build_study_provenance(
            observation_manifest=observation_manifest,
            observation_manifest_sha256=first_verification.manifest_sha256,
            candidate_records_sha256=candidate_records_sha256,
            candidate_record_count=len(records),
            label_count=len(labels),
            observed_round=int(contract["observed_round"]),
            batch_id=str(contract["batch_id"]),
            candidate_exclusion_entries=candidate_exclusions,
        )
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        promotion = build_promotion_manifest(
            relative_dir=relative_dir,
            label_path=label_path,
            provenance_path=provenance_path,
            candidate_path=records_path,
            label_count=len(labels),
            candidate_exclusion_entries=candidate_exclusions,
        )
        promotion_path.write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_label_bundle(
            staging_root,
            relative_dir=relative_dir,
            expected_width=len(VALUE_COLUMNS),
            campaign_config_path=campaign_config_path,
            candidate_root=root,
        )
        if file_sha256(records_path) != candidate_records_sha256:
            raise ResponseWindowLabelPromotionError("OPAL candidate records changed before label publication.")
        try:
            publish_new_directory(staged_dir=staged_output, output_dir=output)
        except OSError as exc:
            raise ResponseWindowLabelPromotionError(f"could not publish complete label promotion: {exc}") from exc

    verify_label_bundle(
        root,
        relative_dir=relative_dir,
        expected_width=len(VALUE_COLUMNS),
        campaign_config_path=campaign_config_path,
    )
    return ResponseWindowLabelPromotionResult(
        output_directory=output,
        label_path=output / LABEL_FILENAME,
        study_provenance_path=output / PROVENANCE_FILENAME,
        promotion_manifest_path=output / PROMOTION_FILENAME,
        candidate_count=len(labels),
    )


def _require_stable_observation_read(bundle_dir: Path, *, expected_manifest_sha256: str) -> None:
    try:
        verified = verify_response_window_observations(bundle_dir)
    except ResponseWindowObservationArtifactError as exc:
        raise ResponseWindowLabelPromotionError(f"observation bundle drift detected during read: {exc}") from exc
    if verified.manifest_sha256 != expected_manifest_sha256:
        raise ResponseWindowLabelPromotionError("observation bundle drift detected during read.")


def _require_new_confined_output(output: Path, *, root: Path) -> None:
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ResponseWindowLabelPromotionError("label output must remain within the dataset root.") from exc
    if output.exists() and not output.is_dir():
        raise ResponseWindowLabelPromotionError(f"label output is not a directory: {output}")
    if output.exists():
        raise ResponseWindowLabelPromotionError(
            "label promotion already exists and is immutable; publish a new versioned directory "
            f"and update the campaign binding instead: {output}"
        )


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "ResponseWindowLabelPromotionError",
    "ResponseWindowLabelPromotionResult",
    "publish_response_window_labels",
]
