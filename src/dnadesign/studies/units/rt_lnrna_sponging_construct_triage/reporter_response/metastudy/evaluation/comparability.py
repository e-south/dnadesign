"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation/comparability.py

Selection comparability across raw and reference-normalized profile variants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

from ...measurement_profile import ReporterMeasurementProfile


def profiles_are_selection_comparable(rows: Iterable[object]) -> bool:
    """Require one raw estimand while comparing stricter projections separately."""

    profiles = tuple(row.profile for row in rows)
    base_keys = {(profile.observation_policy.digest, profile.reduction, profile.dose_grid_uM) for profile in profiles}
    normalized_keys = {
        profile.comparability_key
        for profile in profiles
        if not (isinstance(profile, ReporterMeasurementProfile) or getattr(profile, "reference_normalization", None))
    }
    return len(profiles) >= 2 and len(base_keys) == 1 and len(normalized_keys) <= 1


__all__ = ["profiles_are_selection_comparable"]
