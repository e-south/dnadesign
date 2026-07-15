"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact.py

Publish and verify candidate-level response-window observation evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .artifact_contract import (
    RECORD_FILES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    ResponseWindowObservationArtifactError,
    ResponseWindowObservationVerification,
    ResponseWindowObservationWriteResult,
)
from .artifact_io import confined_path, file_sha256, publish_new_directory, read_json_object
from .artifact_manifest import build_manifest, is_sha256, validate_manifest_identity
from .artifact_validation import validate_frames
from .censoring import bounded_label_blockers
from .contracts import ResponseWindowObservationPreview
from .source_integrity import verify_source_bundle_records
from .sources import ResponseWindowObservationEvidence


def materialize_response_window_observations(
    evidence: ResponseWindowObservationEvidence,
    *,
    out_dir: Path,
    allowed_output_root: Path,
) -> ResponseWindowObservationWriteResult:
    """Atomically publish an approved, blocker-free study observation bundle."""

    blockers = tuple(sorted({*evidence.preview.blockers, *bounded_label_blockers(evidence.preview.contributions)}))
    if blockers:
        raise ResponseWindowObservationArtifactError(
            f"response-window observation publication is blocked: {list(blockers)}"
        )
    if evidence.policy.approval_status != "approved":
        raise ResponseWindowObservationArtifactError("response-window observation policy is not approved.")
    _verify_live_sources(evidence)
    validate_frames(evidence.preview, bootstrap_samples=evidence.policy.aggregation.bootstrap_samples)
    output = _output_path(out_dir, allowed_output_root=allowed_output_root)
    frames = _record_frames(evidence.preview)
    try:
        with TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temporary:
            staged = Path(temporary)
            for record_id, filename in RECORD_FILES.items():
                frames[record_id].to_parquet(staged / filename, index=False)
            manifest = build_manifest(evidence, staged=staged, frames=frames)
            (staged / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            verify_response_window_observations(staged, allowed_root=staged)
            _verify_live_sources(evidence)
            publish_new_directory(staged_dir=staged, output_dir=output)
    except ResponseWindowObservationArtifactError:
        raise
    except (OSError, ValueError) as exc:
        raise ResponseWindowObservationArtifactError(f"could not publish observation bundle: {exc}") from exc
    return ResponseWindowObservationWriteResult(
        manifest_json=output / "manifest.json",
        observations_parquet=output / RECORD_FILES["observations"],
        candidate_count=len(evidence.preview.observations),
    )


def verify_response_window_observations(
    bundle_dir: Path,
    *,
    allowed_root: Path | None = None,
) -> ResponseWindowObservationVerification:
    """Verify manifest identity, every record digest, and cross-record invariants."""

    bundle = _bundle_path(bundle_dir, allowed_root=allowed_root)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ResponseWindowObservationArtifactError(f"observation manifest not found: {manifest_path}")
    try:
        payload = read_json_object(manifest_path, label="observation manifest")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResponseWindowObservationArtifactError(f"could not parse observation manifest: {exc}") from exc
    validate_manifest_identity(payload)
    frames = _read_record_frames(bundle, records=payload["records"])
    preview = ResponseWindowObservationPreview(
        observations=frames["observations"],
        contributions=frames["contributions"],
        bootstrap_draws=frames["hierarchical_bootstrap_draws"],
        uncertainty=frames["uncertainty"],
        repeat_diagnostics=frames["repeat_diagnostics"],
        reduction_sensitivity=frames["reduction_sensitivity"],
        event_time_sensitivity=frames["event_time_sensitivity"],
        blockers=(),
    )
    contract = payload["observation_contract"]
    if int(contract["candidate_count"]) != len(preview.observations):
        raise ResponseWindowObservationArtifactError("observation candidate count disagrees with records.")
    validate_frames(preview, bootstrap_samples=int(contract["hierarchical_bootstrap_samples"]))
    return ResponseWindowObservationVerification(
        manifest_json=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        observations_parquet=bundle / RECORD_FILES["observations"],
        candidate_count=len(preview.observations),
        policy_id=str(payload["policy"]["policy_id"]),
        y_space=str(contract["y_space"]),
    )


def _read_record_frames(bundle: Path, *, records: object) -> dict[str, pd.DataFrame]:
    if not isinstance(records, dict):
        raise ResponseWindowObservationArtifactError("observation record inventory is malformed.")
    frames: dict[str, pd.DataFrame] = {}
    for record_id, filename in RECORD_FILES.items():
        raw = records.get(record_id)
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "row_count", "columns"}:
            raise ResponseWindowObservationArtifactError(f"observation record {record_id!r} is malformed.")
        if raw["path"] != filename or not is_sha256(raw["sha256"]):
            raise ResponseWindowObservationArtifactError(f"observation record {record_id!r} identity is invalid.")
        path = bundle / filename
        if not path.is_file():
            raise ResponseWindowObservationArtifactError(f"observation record is missing: {path}")
        actual_digest = file_sha256(path)
        if actual_digest != raw["sha256"]:
            raise ResponseWindowObservationArtifactError(
                f"observation record {record_id!r} digest mismatch: expected={raw['sha256']} actual={actual_digest}"
            )
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise ResponseWindowObservationArtifactError(
                f"could not read observation record {record_id!r}: {exc}"
            ) from exc
        if len(frame) != raw["row_count"] or frame.columns.tolist() != raw["columns"]:
            raise ResponseWindowObservationArtifactError(f"observation record {record_id!r} shape contract disagrees.")
        frames[record_id] = frame
    return frames


def _record_frames(preview: ResponseWindowObservationPreview) -> dict[str, pd.DataFrame]:
    return {
        "observations": preview.observations,
        "contributions": preview.contributions,
        "hierarchical_bootstrap_draws": preview.bootstrap_draws,
        "uncertainty": preview.uncertainty,
        "repeat_diagnostics": preview.repeat_diagnostics,
        "reduction_sensitivity": preview.reduction_sensitivity,
        "event_time_sensitivity": preview.event_time_sensitivity,
    }


def _verify_live_sources(evidence: ResponseWindowObservationEvidence) -> None:
    if file_sha256(evidence.reader_manifest_path) != evidence.reader_manifest_sha256:
        raise ResponseWindowObservationArtifactError("Reader manifest drift detected after observation preview.")
    if file_sha256(evidence.candidate_bindings_manifest_path) != evidence.candidate_bindings_manifest_sha256:
        raise ResponseWindowObservationArtifactError(
            "candidate-binding manifest drift detected after observation preview."
        )
    if file_sha256(evidence.policy.config_path) != evidence.policy.config_sha256:
        raise ResponseWindowObservationArtifactError("observation policy drift detected after preview.")
    if evidence.reader_manifest_sha256 != evidence.policy.reader_bundle_sha256:
        raise ResponseWindowObservationArtifactError("Reader manifest is not pinned by the observation policy.")
    if evidence.candidate_bindings_manifest_sha256 != evidence.policy.candidate_bindings_sha256:
        raise ResponseWindowObservationArtifactError(
            "candidate-binding manifest is not pinned by the observation policy."
        )
    verify_source_bundle_records(
        reader_manifest_path=evidence.reader_manifest_path,
        candidate_bindings_manifest_path=evidence.candidate_bindings_manifest_path,
        candidate_bindings_path=evidence.candidate_bindings_path,
    )
    if (
        file_sha256(evidence.reader_manifest_path) != evidence.reader_manifest_sha256
        or file_sha256(evidence.candidate_bindings_manifest_path) != evidence.candidate_bindings_manifest_sha256
    ):
        raise ResponseWindowObservationArtifactError("source manifest drift detected during record revalidation.")


def _output_path(out_dir: Path, *, allowed_output_root: Path) -> Path:
    root = Path(allowed_output_root).expanduser().resolve()
    try:
        output = confined_path(out_dir, root=root, label="observation output directory")
    except ValueError as exc:
        raise ResponseWindowObservationArtifactError(str(exc)) from exc
    if output.exists() and not output.is_dir():
        raise ResponseWindowObservationArtifactError(f"observation output is not a directory: {output}")
    if output.exists():
        raise ResponseWindowObservationArtifactError(
            f"observation bundle already exists and is immutable; publish a new named directory instead: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _bundle_path(bundle_dir: Path, *, allowed_root: Path | None) -> Path:
    bundle = Path(bundle_dir).expanduser().resolve()
    if allowed_root is None:
        return bundle
    try:
        return confined_path(bundle, root=Path(allowed_root).expanduser().resolve(), label="observation bundle")
    except ValueError as exc:
        raise ResponseWindowObservationArtifactError(str(exc)) from exc


__all__ = [
    "RECORD_FILES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ResponseWindowObservationArtifactError",
    "ResponseWindowObservationVerification",
    "ResponseWindowObservationWriteResult",
    "materialize_response_window_observations",
    "verify_response_window_observations",
]
