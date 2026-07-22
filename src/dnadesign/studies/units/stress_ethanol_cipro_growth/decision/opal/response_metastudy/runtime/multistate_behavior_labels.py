"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_labels.py

Verified promoted labels and feature rows for grouped behavior validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from dnadesign.opal import (
    ObservedLabelPromotionBinding,
    candidate_exclusion_sets_from_config,
    load_config,
    verify_observed_label_snapshot,
)

from ..evaluation.multistate_behavior_cohort import behavior_component_columns
from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .multistate_behavior_json import load_strict_behavior_json
from .publication import sha256_file


@dataclass(frozen=True)
class VerifiedBehaviorValidationLabels:
    labels: pd.DataFrame
    x: np.ndarray
    source: dict[str, str]
    label_artifact_sha256: str
    central_label_equivalence_sha256: str
    promoted_label_event_count: int
    promoted_candidate_count: int


def load_verified_behavior_validation_labels(
    *,
    campaign_config_path: Path,
    current_measurements: pd.DataFrame,
    source_observation_bundle_root: Path,
    protocol: MultistateBehaviorShadowProtocol,
) -> VerifiedBehaviorValidationLabels:
    """Bind exact study observations to OPAL's verified immutable label source."""

    config = load_config(Path(campaign_config_path).resolve())
    location = config.data.location
    label_source = config.labels.source
    if getattr(location, "kind", None) != "usr" or getattr(label_source, "kind", None) != "usr_sidecar":
        raise ValueError("behavior grouped validation requires a manifest-pinned USR label source.")
    dataset_root = (Path(location.path) / str(label_source.dataset)).resolve()
    manifest_path = str(getattr(label_source, "manifest_path", ""))
    snapshot = verify_observed_label_snapshot(
        ObservedLabelPromotionBinding(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            label_path=str(label_source.path),
            campaign_slug=str(config.campaign.slug),
            study_id=str(config.ownership.study_id),
            y_space=str(config.labels.y_space),
            candidate_id_column=str(config.labels.id_column),
            candidate_x_column=str(config.data.x_column_name),
            candidate_exclusion_sets=candidate_exclusion_sets_from_config(config),
        ),
        expected_y_width=len(behavior_component_columns(protocol)),
    )
    latest = snapshot.labels.sort_values(["id", "r"], kind="mergesort").groupby("id", sort=False).tail(1)
    source_observation_sha, source_observations = _verify_exact_source_observation(
        dataset_root=dataset_root,
        study_provenance_path=snapshot.promotion.study_provenance_path,
        source_observation_bundle_root=source_observation_bundle_root,
        protocol=protocol,
    )
    observations = _validated_observations(source_observations, protocol=protocol)
    if set(latest["id"].astype(str)) != set(observations["candidate_id"].astype(str)):
        raise ValueError("verified promoted labels disagree with the approved study observation candidates.")
    latest_by_id = latest.set_index("id")
    ordered = observations.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    components = behavior_component_columns(protocol)
    promoted_y = np.vstack([latest_by_id.loc[candidate_id, "y"] for candidate_id in ordered["candidate_id"]]).astype(
        float
    )
    observed_y = ordered.loc[:, list(components)].to_numpy(dtype=float)
    if not np.allclose(promoted_y, observed_y, rtol=0.0, atol=0.0):
        raise ValueError("verified promoted label vectors disagree with approved exact study observations.")
    current = _current_source_rows(
        current_measurements,
        source_observations=ordered,
        protocol=protocol,
    )
    current_y = current.loc[:, list(components)].to_numpy(dtype=float)
    if not np.array_equal(current_y, observed_y):
        raise ValueError("corrected Reader central vectors disagree with the immutable promoted observation source.")
    label_rows = current.loc[
        :, ["candidate_id", "display_label", "label_source_reader_experiment_id", *components]
    ].copy()

    candidate_path = snapshot.promotion.candidate_path
    x_column = str(config.data.x_column_name)
    candidate_ids = label_rows["candidate_id"].astype(str).tolist()
    feature_rows = pd.read_parquet(
        candidate_path,
        columns=["id", x_column],
        filters=[("id", "in", candidate_ids)],
    )
    if feature_rows["id"].astype(str).duplicated().any() or set(feature_rows["id"].astype(str)) != set(candidate_ids):
        raise ValueError("candidate feature table does not cover every promoted validation label exactly once.")
    feature_by_id = feature_rows.set_index(feature_rows["id"].astype(str))[x_column]
    x = np.vstack([feature_by_id.loc[candidate_id] for candidate_id in candidate_ids]).astype(float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("promoted behavior validation features must form one finite two-dimensional matrix.")

    return VerifiedBehaviorValidationLabels(
        labels=label_rows.reset_index(drop=True),
        x=x,
        source={
            "promotion_manifest_sha256": f"sha256:{snapshot.promotion.manifest_sha256}",
            "candidate_records_sha256": f"sha256:{snapshot.promotion.candidate_sha256}",
            "source_observation_manifest_sha256": f"sha256:{source_observation_sha}",
            "x_column_name": x_column,
        },
        label_artifact_sha256=f"sha256:{snapshot.promotion.label_sha256}",
        central_label_equivalence_sha256=_central_equivalence_sha256(
            current,
            components=components,
        ),
        promoted_label_event_count=len(snapshot.labels),
        promoted_candidate_count=len(latest),
    )


def _validated_observations(
    frame: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    components = behavior_component_columns(protocol)
    required = {"candidate_id", "display_label", "label_source_reader_experiment_id", *components}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"approved behavior observations lack fields: {missing}")
    rows = frame.loc[:, ["candidate_id", "display_label", "label_source_reader_experiment_id", *components]].copy()
    if rows.empty or rows["candidate_id"].astype(str).duplicated().any():
        raise ValueError("approved behavior observations must contain unique candidate identities.")
    for field in ("candidate_id", "display_label", "label_source_reader_experiment_id"):
        values = rows[field].astype(str)
        if values.eq("").any() or values.str.strip().ne(values).any():
            raise ValueError(f"approved behavior observation field {field!r} must be exact and nonempty.")
        rows[field] = values
    if not np.isfinite(rows.loc[:, list(components)].to_numpy(dtype=float)).all():
        raise ValueError("approved behavior observations must contain finite exact point estimates.")
    return rows


def _verify_exact_source_observation(
    *,
    dataset_root: Path,
    study_provenance_path: Path,
    source_observation_bundle_root: Path,
    protocol: MultistateBehaviorShadowProtocol,
) -> tuple[str, pd.DataFrame]:
    provenance = load_strict_behavior_json(study_provenance_path)
    observation = provenance.get("observation_bundle")
    if not isinstance(observation, dict):
        raise ValueError("label promotion study provenance lacks its source observation bundle.")
    relative = observation.get("manifest_path")
    expected_sha = observation.get("manifest_sha256")
    if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise ValueError("source observation manifest path must remain within the USR dataset.")
    provenance_copy_path = (dataset_root / relative).resolve()
    if not provenance_copy_path.is_relative_to(dataset_root) or not provenance_copy_path.is_file():
        raise ValueError("source observation manifest is missing or escapes the USR dataset.")
    observed_sha = sha256_file(provenance_copy_path)
    if expected_sha != observed_sha:
        raise ValueError("source observation manifest digest disagrees with study provenance.")
    source_root = Path(source_observation_bundle_root).resolve()
    source_manifest_path = (source_root / "manifest.json").resolve()
    if not source_manifest_path.is_relative_to(source_root) or not source_manifest_path.is_file():
        raise ValueError("protocol-declared source observation manifest is missing or escapes its bundle.")
    if sha256_file(source_manifest_path) != observed_sha:
        raise ValueError("protocol-declared source observation manifest disagrees with immutable label provenance.")
    manifest = load_strict_behavior_json(source_manifest_path)
    contract = manifest.get("observation_contract")
    if not isinstance(contract, dict):
        raise ValueError("source observation manifest lacks its observation contract.")
    expected = {
        "primary_reduction_id": protocol.primary_reduction_id,
        "primary_value_requirement": "exact",
        "nonexact_label_action": "exclude_candidate",
        "y_space": "reader_response_window_vector_v1",
    }
    if any(contract.get(field) != value for field, value in expected.items()):
        raise ValueError("source observation manifest does not prove exact-only promoted labels.")
    records = manifest.get("records")
    if not isinstance(records, dict) or not isinstance(records.get("observations"), dict):
        raise ValueError("source observation manifest lacks its observations record.")
    record = records["observations"]
    relative_record = record.get("path")
    expected_record_sha = record.get("sha256")
    if (
        not isinstance(relative_record, str)
        or PurePosixPath(relative_record).is_absolute()
        or ".." in PurePosixPath(relative_record).parts
    ):
        raise ValueError("source observation record path must remain within its bundle.")
    record_path = (source_root / relative_record).resolve()
    if not record_path.is_relative_to(source_root) or not record_path.is_file():
        raise ValueError("source observation record is missing or escapes its bundle.")
    if expected_record_sha != sha256_file(record_path):
        raise ValueError("source observation record digest disagrees with its manifest.")
    return observed_sha, pd.read_parquet(record_path)


def _current_source_rows(
    measurements: pd.DataFrame,
    *,
    source_observations: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    """Select the same candidate and experiment identities from corrected Reader evidence."""

    components = behavior_component_columns(protocol)
    required = {
        "candidate_id",
        "reader_experiment_id",
        "display_label",
        "reduction_id",
        *components,
        *(f"{component}_bound_kind" for component in components),
    }
    if missing := sorted(required - set(measurements.columns)):
        raise ValueError(f"corrected Reader measurements lack central-equivalence fields: {missing}")
    primary = measurements.loc[measurements["reduction_id"].astype(str).eq(protocol.primary_reduction_id)].copy()
    identities = source_observations.loc[:, ["candidate_id", "label_source_reader_experiment_id"]].copy()
    rows = identities.merge(
        primary,
        left_on=["candidate_id", "label_source_reader_experiment_id"],
        right_on=["candidate_id", "reader_experiment_id"],
        how="left",
        validate="one_to_one",
    )
    if rows[list(components)].isna().any().any():
        raise ValueError("corrected Reader evidence lacks a promoted candidate/source-experiment row.")
    bound_columns = [f"{component}_bound_kind" for component in components]
    if not rows.loc[:, bound_columns].astype(str).eq("exact").all().all():
        raise ValueError("corrected Reader central-equivalence rows must remain exact and uncensored.")
    rows["display_label"] = rows["display_label"].astype(str)
    return rows


def _central_equivalence_sha256(frame: pd.DataFrame, *, components: tuple[str, ...]) -> str:
    """Digest exact candidate, source-experiment, and central-vector equivalence."""

    records = []
    for row in frame.sort_values("candidate_id", kind="mergesort").itertuples(index=False):
        records.append(
            (
                str(row.candidate_id),
                str(row.label_source_reader_experiment_id),
                tuple(float(getattr(row, component)).hex() for component in components),
            )
        )
    payload = repr(records).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = ["VerifiedBehaviorValidationLabels", "load_verified_behavior_validation_labels"]
