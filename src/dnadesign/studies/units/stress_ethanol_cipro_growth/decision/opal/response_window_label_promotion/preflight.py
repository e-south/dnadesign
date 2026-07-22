"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/preflight.py

Fail-closed input and output checks for response-window label publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact import (
    ResponseWindowObservationArtifactError,
    verify_response_window_observations,
)

from .contracts import ResponseWindowLabelPromotionError


def require_stable_observation_read(bundle_dir: Path, *, expected_manifest_sha256: str) -> None:
    try:
        verified = verify_response_window_observations(bundle_dir)
    except ResponseWindowObservationArtifactError as exc:
        raise ResponseWindowLabelPromotionError(f"observation bundle drift detected during read: {exc}") from exc
    if verified.manifest_sha256 != expected_manifest_sha256:
        raise ResponseWindowLabelPromotionError("observation bundle drift detected during read.")


def require_new_confined_output(output: Path, *, root: Path) -> None:
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


__all__ = ["require_new_confined_output", "require_stable_observation_read"]
