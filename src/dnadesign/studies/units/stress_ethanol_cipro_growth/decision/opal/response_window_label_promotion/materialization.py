"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/materialization.py

Stage, verify, atomically publish, and finalize one response-window label promotion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
)

from .contracts import (
    LABEL_FILENAME,
    PROMOTION_FILENAME,
    PROVENANCE_FILENAME,
    SOURCE_OBSERVATION_MANIFEST_FILENAME,
    ResponseWindowLabelPromotionError,
)
from .lineage import update_lineage_head
from .provenance import build_study_provenance
from .publication import build_promotion_manifest, publish_new_directory, verify_label_bundle


def stage_and_publish(
    *,
    verified_manifest_path: Path,
    verified_manifest_sha256: str,
    observation_manifest: dict[str, object],
    records_path: Path,
    candidate_sha256: str,
    records: pd.DataFrame,
    labels: pd.DataFrame,
    incoming: pd.DataFrame,
    observed_round: int,
    batch_id: str,
    exclusions: list[dict[str, str]],
    prior_reference: dict[str, object] | None,
    root: Path,
    relative_dir: PurePosixPath,
    output: Path,
) -> None:
    with TemporaryDirectory(prefix=".response-window-labels-staging-", dir=output.parent) as temporary:
        staging_root = Path(temporary)
        staged = staging_root / relative_dir
        staged.mkdir(parents=True, exist_ok=False)
        label_path = staged / LABEL_FILENAME
        provenance_path = staged / PROVENANCE_FILENAME
        promotion_path = staged / PROMOTION_FILENAME
        source_manifest_path = staged / SOURCE_OBSERVATION_MANIFEST_FILENAME
        labels.to_parquet(label_path, index=False)
        source_bytes = verified_manifest_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != verified_manifest_sha256:
            raise ResponseWindowLabelPromotionError("source observation manifest drifted before it could be copied.")
        source_manifest_path.write_bytes(source_bytes)
        provenance = build_study_provenance(
            observation_manifest=observation_manifest,
            source_observation_manifest_path=relative_dir / SOURCE_OBSERVATION_MANIFEST_FILENAME,
            source_observation_manifest_sha256=file_sha256(source_manifest_path),
            candidate_records_sha256=candidate_sha256,
            candidate_record_count=len(records),
            label_frame=labels,
            appended_label_event_count=len(incoming),
            observed_round=observed_round,
            batch_id=batch_id,
            candidate_exclusion_entries=exclusions,
            prior_promotion=prior_reference,
        )
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        promotion = build_promotion_manifest(
            relative_dir=relative_dir,
            label_path=label_path,
            provenance_path=provenance_path,
            candidate_path=records_path,
            label_event_count=len(labels),
            candidate_exclusion_entries=exclusions,
        )
        promotion_path.write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_label_bundle(
            staging_root,
            relative_dir=relative_dir,
            expected_width=len(VALUE_COLUMNS),
            candidate_root=root,
        )
        if file_sha256(records_path) != candidate_sha256:
            raise ResponseWindowLabelPromotionError("OPAL candidate records changed before label publication.")
        try:
            publish_new_directory(staged_dir=staged, output_dir=output)
        except OSError as exc:
            raise ResponseWindowLabelPromotionError(f"could not publish complete label promotion: {exc}") from exc
    _finalize_publication(root=root, relative_dir=relative_dir, output=output, labels=labels)


def _finalize_publication(*, root: Path, relative_dir: PurePosixPath, output: Path, labels: pd.DataFrame) -> None:
    try:
        verify_label_bundle(root, relative_dir=relative_dir, expected_width=len(VALUE_COLUMNS))
        update_lineage_head(
            root,
            manifest_path=output / PROMOTION_FILENAME,
            label_event_count=len(labels),
            unique_candidate_count=int(labels["id"].astype(str).nunique()),
            max_observed_round=int(labels["observed_round"].astype(int).max()),
        )
    except Exception as exc:
        try:
            shutil.rmtree(output)
        except OSError as cleanup_exc:
            raise ResponseWindowLabelPromotionError(
                f"published label verification failed and cleanup also failed: {cleanup_exc}"
            ) from exc
        if isinstance(exc, ResponseWindowLabelPromotionError):
            raise
        raise ResponseWindowLabelPromotionError(f"published label verification failed: {exc}") from exc


__all__ = ["stage_and_publish"]
