"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/specs.py

Design-class declarations for Eco1 RT fixed-backbone expansion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    BASELINE_CLASS_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    DesignClassSpec,
)

BASELINE_SPEC = DesignClassSpec(
    design_class_id=BASELINE_CLASS_ID,
    path_id="clade9_p25_5a",
    role="baseline_existing",
    conservation_profile_id="ec86_clade9_conservation_v1",
    conservation_threshold=0.25,
    contact_threshold_angstrom=5.0,
    batch_id="eco1_rt_p25_5a_n96_20260624",
    premise=(
        "Current conservative class: clade 9 25% WT-plurality conservation plus retained DNA/RNA contact within 5 A"
    ),
    rationale=(
        "This is the existing 96-candidate class. It is retained as the baseline and is not regenerated "
        "by the expansion materializer."
    ),
)

GENERATED_SPECS: tuple[DesignClassSpec, ...] = (
    DesignClassSpec(
        design_class_id="eco1_rt_clade9_plurality25_contact6a_v1",
        path_id="clade9_p25_6a",
        role="generated_request",
        conservation_profile_id="ec86_clade9_conservation_v1",
        conservation_threshold=0.25,
        contact_threshold_angstrom=6.0,
        batch_id="eco1_rt_clade9_p25_6a_n96_20260701",
        premise="A 6 A contact shell tests a modestly stricter substrate-protection rule",
        rationale=(
            "This class keeps the clade 9 25% WT-plurality rule and expands only the retained DNA/RNA "
            "distance cutoff from 5 A to 6 A."
        ),
    ),
    DesignClassSpec(
        design_class_id="eco1_rt_clade9_plurality25_contact8a_v1",
        path_id="clade9_p25_8a",
        role="generated_request",
        conservation_profile_id="ec86_clade9_conservation_v1",
        conservation_threshold=0.25,
        contact_threshold_angstrom=8.0,
        batch_id="eco1_rt_clade9_p25_8a_n96_20260701",
        premise="An 8 A contact shell tests a stronger retained-substrate protection rule",
        rationale=(
            "This class keeps the broad clade 9 conservation rule and asks whether a more protected "
            "substrate shell still leaves enough scaffold surface for useful ProteinMPNN sampling."
        ),
    ),
    DesignClassSpec(
        design_class_id="eco1_rt_clade9_plurality25_contact10a_v1",
        path_id="clade9_p25_10a",
        role="generated_request",
        conservation_profile_id="ec86_clade9_conservation_v1",
        conservation_threshold=0.25,
        contact_threshold_angstrom=10.0,
        batch_id="eco1_rt_clade9_p25_10a_n96_20260701",
        premise="A 10 A contact shell is a conservative sentinel class with a small mutable surface",
        rationale=(
            "This class is expected to be close to WT because most mapped residues become protected. "
            "It is useful as a sensitivity class, not as a replacement for the 5 A baseline."
        ),
    ),
    DesignClassSpec(
        design_class_id="eco1_rt_clade9_plurality50_contact5a_v1",
        path_id="clade9_p50_5a",
        role="generated_request",
        conservation_profile_id="ec86_clade9_conservation_v1",
        conservation_threshold=0.50,
        contact_threshold_angstrom=5.0,
        batch_id="eco1_rt_clade9_p50_5a_n96_20260701",
        premise="A 50% clade 9 plurality threshold tests a less restrictive conservation mask",
        rationale=(
            "This class keeps the 5 A contact rule but protects only positions where Eco1 is the clade 9 "
            "plurality residue at or above 50% non-gap frequency."
        ),
    ),
    DesignClassSpec(
        design_class_id="eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1",
        path_id="iia3_42_1_p50_5a",
        role="generated_request",
        conservation_profile_id="ec86_iia3_cluster42_1_conservation_v1",
        conservation_threshold=0.50,
        contact_threshold_angstrom=5.0,
        batch_id="eco1_rt_iia3_42_1_p50_5a_n96_20260701",
        premise="A subtype-family 50% plurality rule tests the closer Eco1-family conservation denominator",
        rationale=(
            "This class uses the II-A3 cluster 42_1 panel rather than the full clade 9 panel, while keeping "
            "the 5 A retained DNA/RNA contact rule."
        ),
    ),
)

ALL_SPECS: tuple[DesignClassSpec, ...] = (BASELINE_SPEC, *GENERATED_SPECS)


def select_specs(class_ids: Iterable[str] | None = None) -> tuple[DesignClassSpec, ...]:
    """Return generated design specs selected by id, excluding the existing baseline."""

    if class_ids is None:
        return GENERATED_SPECS
    selected = list(class_ids)
    by_id = {spec.design_class_id: spec for spec in GENERATED_SPECS}
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown design class id(s): {', '.join(unknown)}")
    return tuple(by_id[class_id] for class_id in selected)
