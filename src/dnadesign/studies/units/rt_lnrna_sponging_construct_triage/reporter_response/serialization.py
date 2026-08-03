"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/serialization.py

Canonical serialization for descriptive reporter profiles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._contract_values import ReporterResponseContractError
from ._contract_values import json_value as _json_value
from .measurement_profile import ReporterMeasurementProfile
from .profile.normalized import ReporterResponseProfile


def profile_to_dict(profile: ReporterResponseProfile | ReporterMeasurementProfile) -> dict[str, object]:
    """Serialize one already-validated profile without private closure fields."""

    if not isinstance(profile, (ReporterResponseProfile, ReporterMeasurementProfile)):
        raise ReporterResponseContractError("profile must be a typed reporter profile")
    payload = _json_value(profile)
    assert isinstance(payload, dict)
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("_bound_subject_id", None)
    provenance.pop("_source_closed", None)
    provenance.pop("_declared_biological_replicate_scopes", None)
    return payload


__all__ = ["profile_to_dict"]
