"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/conditions.py

Condition ontology for the RT-lnRNA Reader SPOP composite.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

BASELINE_ROLE = "baseline"
POSITIVE_CONTROL_ROLE = "positive_control"
IPTG_DOSE_ROLE = "iptg_dose"

BASELINE_CONDITION_KEY = "0 nm aTc; 0 uM IPTG"

_ROLE_ORDER = {
    BASELINE_ROLE: 0,
    POSITIVE_CONTROL_ROLE: 1,
    IPTG_DOSE_ROLE: 2,
}


def condition_key_for_positive_control(atc_nM: float) -> str:
    dose = float(atc_nM)
    if dose <= 0:
        raise ValueError("positive-control aTc dose must be greater than zero.")
    return f"{dose:g} nm aTc; 0 uM IPTG"


def condition_key_for_iptg_dose(iptg_uM: float) -> str:
    dose = float(iptg_uM)
    if dose <= 0:
        raise ValueError("IPTG dose condition must be greater than zero.")
    return f"0 nm aTc; {dose:g} uM IPTG"


def condition_sort_key(
    *,
    condition_role: str,
    atc_nM: float,
    iptg_uM: float,
    condition_key: str,
) -> tuple[int, float, float, str]:
    return (
        _ROLE_ORDER.get(condition_role, 99),
        float(atc_nM) if condition_role == POSITIVE_CONTROL_ROLE else 0.0,
        float(iptg_uM),
        condition_key,
    )


def short_condition_label(condition_key: str) -> str:
    return condition_key.replace("; ", "\n")
