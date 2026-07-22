"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/publisher.py

Publish verified response-window observations through OPAL's label contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact import (
    RECORD_FILES,
    verify_response_window_observations,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
    read_json_object,
)

from .contracts import (
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
from .cumulative import extend_label_frame, load_prior_promotion, merge_candidate_exclusions
from .exclusions import derive_candidate_selection_exclusions, require_exclusion_candidates_in_records
from .lineage import (
    lineage_publication_lock,
    load_lineage_head,
    require_current_parent,
)
from .materialization import stage_and_publish
from .preflight import require_new_confined_output, require_stable_observation_read


def publish_response_window_labels(
    *,
    observation_bundle_dir: Path,
    dataset_root: Path,
    output_relative_directory: str = DEFAULT_OUTPUT_DIRECTORY,
    prior_promotion_manifest_path: Path | None = None,
) -> ResponseWindowLabelPromotionResult:
    """Publish one create-only label bundle and advance the study lineage head."""

    root = Path(dataset_root).expanduser().resolve()
    relative_dir = confined_relative_directory(output_relative_directory)
    output = (root / relative_dir).resolve()
    with lineage_publication_lock(root):
        require_new_confined_output(output, root=root)
        return _publish_locked(
            observation_bundle_dir=observation_bundle_dir,
            root=root,
            relative_dir=relative_dir,
            output=output,
            prior_promotion_manifest_path=prior_promotion_manifest_path,
        )


def _publish_locked(
    *,
    observation_bundle_dir: Path,
    root: Path,
    relative_dir: PurePosixPath,
    output: Path,
    prior_promotion_manifest_path: Path | None,
) -> ResponseWindowLabelPromotionResult:
    verified = verify_response_window_observations(observation_bundle_dir)
    observation_manifest = read_json_object(verified.manifest_json, label="response-window observation manifest")
    contract = require_observation_contract(observation_manifest.get("observation_contract"))
    observations = pd.read_parquet(verified.observations_parquet)
    contributions = pd.read_parquet(verified.manifest_json.parent / RECORD_FILES["contributions"])
    incoming_exclusions = derive_candidate_selection_exclusions(observations, contributions)
    require_stable_observation_read(observation_bundle_dir, expected_manifest_sha256=verified.manifest_sha256)

    records_path = root / "records.parquet"
    if not records_path.is_file():
        raise ResponseWindowLabelPromotionError(f"OPAL candidate records not found: {records_path}")
    candidate_sha256 = file_sha256(records_path)
    records = pd.read_parquet(records_path, columns=["id", "sequence"])
    if file_sha256(records_path) != candidate_sha256:
        raise ResponseWindowLabelPromotionError("OPAL candidate records changed while labels were being prepared.")
    verify_candidate_identity(observations, records=records)

    observed_round = int(contract["observed_round"])
    batch_id = str(contract["batch_id"])
    incoming = build_label_frame(observations, observed_round=observed_round, batch_id=batch_id)
    prior = load_prior_promotion(
        prior_promotion_manifest_path,
        dataset_root=root,
        expected_width=len(VALUE_COLUMNS),
    )
    require_current_parent(
        head=load_lineage_head(root),
        prior_reference=None if prior is None else prior.reference,
        incoming_round=observed_round,
    )
    labels = extend_label_frame(prior, incoming)
    exclusions = merge_candidate_exclusions(
        [] if prior is None else prior.candidate_exclusions,
        incoming_exclusions,
        cumulative_labels=labels,
        incoming_labels=incoming,
    )
    require_exclusion_candidates_in_records(exclusions, records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_and_publish(
        verified_manifest_path=verified.manifest_json,
        verified_manifest_sha256=verified.manifest_sha256,
        observation_manifest=observation_manifest,
        records_path=records_path,
        candidate_sha256=candidate_sha256,
        records=records,
        labels=labels,
        incoming=incoming,
        observed_round=observed_round,
        batch_id=batch_id,
        exclusions=exclusions,
        prior_reference=None if prior is None else prior.reference,
        root=root,
        relative_dir=relative_dir,
        output=output,
    )
    return ResponseWindowLabelPromotionResult(
        output_directory=output,
        label_path=output / LABEL_FILENAME,
        study_provenance_path=output / PROVENANCE_FILENAME,
        promotion_manifest_path=output / PROMOTION_FILENAME,
        label_event_count=len(labels),
        unique_candidate_count=int(labels["id"].astype(str).nunique()),
    )


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "ResponseWindowLabelPromotionError",
    "ResponseWindowLabelPromotionResult",
    "publish_response_window_labels",
]
