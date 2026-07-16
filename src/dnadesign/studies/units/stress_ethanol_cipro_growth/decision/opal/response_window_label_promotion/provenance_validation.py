"""Deep verification for response-window label study provenance."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path, PurePosixPath

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
    read_json_object,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_manifest import (
    validate_manifest_identity,
)

from .contracts import (
    PROVENANCE_SCHEMA_ID,
    SOURCE_OBSERVATION_MANIFEST_FILENAME,
    STUDY_ID,
    Y_SPACE,
    ResponseWindowLabelPromotionError,
    validate_label_frame,
)
from .exclusions import validate_candidate_selection_exclusion_provenance

_PROVENANCE_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "created_at",
    "observation_bundle",
    "candidate_table",
    "candidate_selection_exclusions",
    "label_contract",
    "prior_promotion",
}
_OBSERVATION_FIELDS = {"schema_id", "manifest_path", "manifest_sha256", "policy", "source_manifests"}
_CANDIDATE_FIELDS = {"path", "records_sha256", "record_count"}
_LABEL_FIELDS = {
    "y_space",
    "value_order",
    "observed_round",
    "batch_id",
    "label_event_count",
    "unique_candidate_count",
    "appended_label_event_count",
    "appended_unique_candidate_count",
    "observed_rounds",
    "batch_ids",
}
_PRIOR_FIELDS = {
    "label_path",
    "label_sha256",
    "manifest_path",
    "manifest_sha256",
    "label_event_count",
    "unique_candidate_count",
    "max_observed_round",
}


def validate_study_provenance(
    provenance: dict[str, object],
    *,
    bundle_root: Path,
    candidate_root: Path,
    label_path: Path,
    candidate_sha256: str,
    candidate_row_count: int,
) -> list[dict[str, str]]:
    """Cross-check every derived claim against the artifacts it describes."""

    if set(provenance) != _PROVENANCE_FIELDS:
        raise ResponseWindowLabelPromotionError("published study provenance fields disagree.")
    if (
        provenance["schema_id"] != PROVENANCE_SCHEMA_ID
        or provenance["schema_version"] != "4"
        or provenance["study_id"] != STUDY_ID
    ):
        raise ResponseWindowLabelPromotionError("published study provenance identity disagrees.")
    _timestamp(provenance["created_at"])
    labels = pd.read_parquet(label_path)
    validate_label_frame(labels, context="published promotion")
    source_manifest = _verify_observation_claims(provenance["observation_bundle"], bundle_root=bundle_root)
    _verify_candidate_claims(
        provenance["candidate_table"],
        candidate_sha256=candidate_sha256,
        candidate_row_count=candidate_row_count,
    )
    _verify_label_claims(provenance["label_contract"], labels=labels, source_manifest=source_manifest)
    _verify_prior_promotion_reference(
        provenance["prior_promotion"],
        candidate_root=candidate_root,
        labels=labels,
    )
    return validate_candidate_selection_exclusion_provenance(provenance["candidate_selection_exclusions"])


def _verify_observation_claims(value: object, *, bundle_root: Path) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _OBSERVATION_FIELDS:
        raise ResponseWindowLabelPromotionError("published observation-bundle provenance is malformed.")
    path = _confined_path(value["manifest_path"], root=bundle_root, expected_name=SOURCE_OBSERVATION_MANIFEST_FILENAME)
    digest = _sha256(value["manifest_sha256"], label="source observation manifest")
    if not path.is_file() or file_sha256(path) != digest:
        raise ResponseWindowLabelPromotionError("copied source observation manifest digest disagrees.")
    try:
        manifest = read_json_object(path, label="copied source observation manifest")
        validate_manifest_identity(manifest)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResponseWindowLabelPromotionError(f"copied source observation manifest is invalid: {exc}") from exc
    if (
        value["schema_id"] != manifest["schema_id"]
        or value["policy"] != manifest["policy"]
        or value["source_manifests"] != manifest["source_manifests"]
    ):
        raise ResponseWindowLabelPromotionError("observation-bundle provenance disagrees with its copied manifest.")
    return manifest


def _verify_candidate_claims(value: object, *, candidate_sha256: str, candidate_row_count: int) -> None:
    if not isinstance(value, dict) or set(value) != _CANDIDATE_FIELDS or value["path"] != "records.parquet":
        raise ResponseWindowLabelPromotionError("published candidate-table provenance is malformed.")
    if value["records_sha256"] != candidate_sha256 or value["record_count"] != candidate_row_count:
        raise ResponseWindowLabelPromotionError("candidate-table provenance disagrees with the verified artifact.")


def _verify_label_claims(value: object, *, labels: pd.DataFrame, source_manifest: dict[str, object]) -> None:
    if not isinstance(value, dict) or set(value) != _LABEL_FIELDS:
        raise ResponseWindowLabelPromotionError("published label-contract provenance is malformed.")
    source = source_manifest["observation_contract"]
    observed_round = int(source["observed_round"])
    batch_id = str(source["batch_id"])
    appended = labels.loc[
        labels["observed_round"].astype(int).eq(observed_round) & labels["batch_id"].astype(str).eq(batch_id)
    ]
    expected = {
        "y_space": Y_SPACE,
        "value_order": list(VALUE_COLUMNS),
        "observed_round": observed_round,
        "batch_id": batch_id,
        "label_event_count": len(labels),
        "unique_candidate_count": int(labels["id"].astype(str).nunique()),
        "appended_label_event_count": len(appended),
        "appended_unique_candidate_count": int(appended["id"].astype(str).nunique()),
        "observed_rounds": sorted(labels["observed_round"].astype(int).unique().tolist()),
        "batch_ids": list(dict.fromkeys(labels["batch_id"].astype(str).tolist())),
    }
    if value != expected or len(appended) != int(source["candidate_count"]):
        raise ResponseWindowLabelPromotionError("label-contract provenance disagrees with verified label evidence.")


def _verify_prior_promotion_reference(value: object, *, candidate_root: Path, labels: pd.DataFrame) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _PRIOR_FIELDS:
        raise ResponseWindowLabelPromotionError("published prior-promotion provenance is malformed.")
    manifest_path = _confined_path(value["manifest_path"], root=candidate_root, expected_name="promotion.manifest.json")
    label_path = _confined_path(value["label_path"], root=candidate_root, expected_name="observed_labels.parquet")
    for path, field in ((manifest_path, "manifest_sha256"), (label_path, "label_sha256")):
        if not path.is_file() or file_sha256(path) != _sha256(value[field], label=f"prior {field}"):
            raise ResponseWindowLabelPromotionError(f"published prior-promotion {field} digest disagrees.")
    prior = pd.read_parquet(label_path)
    validate_label_frame(prior, context="published prior promotion")
    if (
        value["label_event_count"] != len(prior)
        or value["unique_candidate_count"] != int(prior["id"].astype(str).nunique())
        or value["max_observed_round"] != int(prior["observed_round"].astype(int).max())
        or not prior.equals(labels.iloc[: len(prior)].reset_index(drop=True))
    ):
        raise ResponseWindowLabelPromotionError("published prior-promotion inventory disagrees with cumulative labels.")


def _confined_path(value: object, *, root: Path, expected_name: str) -> Path:
    if not isinstance(value, str):
        raise ResponseWindowLabelPromotionError("published provenance path must be a string.")
    relative = PurePosixPath(value)
    if not value or "\\" in value or relative.is_absolute() or ".." in relative.parts or relative.name != expected_name:
        raise ResponseWindowLabelPromotionError("published provenance path is not dataset-confined.")
    path = (Path(root).resolve() / Path(*relative.parts)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ResponseWindowLabelPromotionError("published provenance path escapes its artifact root.")
    return path


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ResponseWindowLabelPromotionError(f"published {label} digest is malformed.")
    return value


def _timestamp(value: object) -> None:
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ResponseWindowLabelPromotionError("published study provenance timestamp is invalid.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ResponseWindowLabelPromotionError("published study provenance timestamp must be timezone-aware.")


__all__ = ["validate_study_provenance"]
