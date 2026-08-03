"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/protocol.py

Predeclared reporter-response meta-study protocol contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ._values import MetastudyContractError, _digest, canonical_digest

PROTOCOL_ID = "rt_lnrna_reporter_response_metastudy.v3"

Window = tuple[float, float]
CANONICAL_CONDITION_ONTOLOGY_DIGEST = "sha256:d40f953d8415a515b79565078f59e47a3eb726d5f6569bdf84d039a017b34b28"
CANONICAL_OBSERVATION_POLICY_DIGEST = "sha256:5d1ebf8a2e4c0cac751fcd80e378ca32d0986799db7ed53b5c6913298d293dec"


@dataclass(frozen=True, slots=True)
class MetastudyProtocol:
    """Predeclared study policy; endpoints and alternate widths are sensitivity only."""

    protocol_id: str
    primary_dose_uM: float
    sensitivity_doses_uM: tuple[float, ...]
    candidate_windows_h: tuple[Window, ...]
    endpoint_sensitivity_h: tuple[float, ...]
    centered_window_sensitivity_widths_h: tuple[float, ...]
    time_summary_statistic: Literal["median"]
    within_acquisition_observation_reduction: Literal["median"]
    ratio_reduction_order: Literal["ratio_then_reduce"]
    window_boundary: Literal["inclusive"]
    channel_time_alignment: Literal["exact"]
    expected_sampling_interval_h: float
    minimum_aligned_timepoints_per_4h_window: int
    minimum_within_acquisition_observations_per_stratum: int
    growth_phase_slope_window_h: float
    growth_phase_scale_quantile: float
    growth_phase_minimum_slope_points: int
    growth_phase_start_minimum: float
    growth_phase_end_minimum: float
    growth_phase_end_maximum: float
    within_acquisition_range_method: Literal["within_acquisition_observation_range"]
    within_acquisition_range_reference: Literal["endpoint_10h"]
    minimum_kinetic_experiments: int
    planned_kinetic_experiments: int
    planned_kinetic_experiment_ids: tuple[str, ...]
    excluded_snapshot_experiment_ids: tuple[str, ...]
    anchor_subject_order: tuple[str, ...]
    planned_anchor_experiment_ids: tuple[str, ...]
    reference_panel_target_ordered_acquisitions: int
    planned_anchor_acquisitions: int
    loo_same_or_adjacent_target_fraction: float
    clipping_or_capping: Literal["forbidden"]
    selection_order: tuple[str, ...]
    condition_ontology_digest: str
    observation_policy_digest: str

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise MetastudyContractError(f"protocol_id must equal {PROTOCOL_ID!r}")
        if self.primary_dose_uM != 500.0 or self.sensitivity_doses_uM != (5.0, 50.0):
            raise MetastudyContractError(
                "dose cohorts must remain the predeclared 500 uM primary and 5/50 uM sensitivity"
            )
        if self.candidate_windows_h != (
            (4.0, 8.0),
            (6.0, 10.0),
            (8.0, 12.0),
            (10.0, 14.0),
            (12.0, 16.0),
        ):
            raise MetastudyContractError("candidate windows must remain the five predeclared equal-width windows")
        if self.endpoint_sensitivity_h != (8.0, 10.0, 12.0, 14.0, 16.0):
            raise MetastudyContractError("endpoint sensitivity set changed")
        if self.centered_window_sensitivity_widths_h != (2.0, 6.0):
            raise MetastudyContractError("centered-window sensitivity widths changed")
        if (
            self.time_summary_statistic != "median"
            or self.within_acquisition_observation_reduction != "median"
            or self.ratio_reduction_order != "ratio_then_reduce"
        ):
            raise MetastudyContractError("reduction semantics changed")
        if self.window_boundary != "inclusive" or self.channel_time_alignment != "exact":
            raise MetastudyContractError("window boundaries and channel-time alignment changed")
        if self.expected_sampling_interval_h != 1.0 / 6.0:
            raise MetastudyContractError("expected sampling interval must remain ten minutes")
        if self.minimum_aligned_timepoints_per_4h_window != 25:
            raise MetastudyContractError("four-hour windows require 25 aligned inclusive timepoints")
        if self.minimum_within_acquisition_observations_per_stratum != 3:
            raise MetastudyContractError("condition strata require at least three within-acquisition observations")
        if self.growth_phase_slope_window_h != 1.0:
            raise MetastudyContractError("growth-phase slopes must use one-hour log-normalizer windows")
        if self.growth_phase_scale_quantile != 0.9:
            raise MetastudyContractError("growth-phase slopes must use the positive-slope 90th percentile scale")
        if self.growth_phase_minimum_slope_points != 4:
            raise MetastudyContractError("growth-phase slopes require at least four observations")
        if (
            self.growth_phase_start_minimum,
            self.growth_phase_end_minimum,
            self.growth_phase_end_maximum,
        ) != (0.5, 0.1, 0.6):
            raise MetastudyContractError("growth-phase thresholds changed")
        if (
            self.within_acquisition_range_method != "within_acquisition_observation_range"
            or self.within_acquisition_range_reference != "endpoint_10h"
        ):
            raise MetastudyContractError("within-acquisition range method or reference changed")
        if (self.minimum_kinetic_experiments, self.planned_kinetic_experiments) != (7, 8):
            raise MetastudyContractError("kinetic experiment gate must remain at least 7 of 8")
        if self.planned_kinetic_experiment_ids != (
            "20250622_retron_Eco1_26_43_benchmark",
            "20250707_retron_Eco1_26_43_45_46_benchmark",
            "20250718_retron_Eco1_26_45_47_48_benchmark",
            "20260418_retron_Eco1_26_43_170_171_benchmark",
            "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
            "20260529_retron_Eco1_26_43_177_186_benchmark",
            "20260705_retron_Eco1_26_195_196_180_199_200_197_198_benchmark",
            "20260720_retron_Eco1_26_180_201_202_203_204_benchmark",
        ):
            raise MetastudyContractError("planned kinetic experiment identities changed")
        if self.excluded_snapshot_experiment_ids != ("20251105_retron_Eco1_RT_variants",):
            raise MetastudyContractError("excluded snapshot experiment identity changed")
        if self.anchor_subject_order != (
            "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO",
            "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
        ):
            raise MetastudyContractError("anchor subject ordering must remain failed-anchor to working-anchor")
        if self.planned_anchor_experiment_ids != (
            "20250622_retron_Eco1_26_43_benchmark",
            "20250707_retron_Eco1_26_43_45_46_benchmark",
            "20260418_retron_Eco1_26_43_170_171_benchmark",
            "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
            "20260529_retron_Eco1_26_43_177_186_benchmark",
        ):
            raise MetastudyContractError("planned anchor co-measurement experiment identities changed")
        if not set(self.planned_anchor_experiment_ids) <= set(self.planned_kinetic_experiment_ids):
            raise MetastudyContractError("planned anchor experiments must be planned kinetic experiments")
        if (self.reference_panel_target_ordered_acquisitions, self.planned_anchor_acquisitions) != (4, 5):
            raise MetastudyContractError("reference-panel support target must remain 4 of 5 acquisitions")
        if self.loo_same_or_adjacent_target_fraction != 0.75:
            raise MetastudyContractError("leave-one-out stability target must remain 0.75")
        if self.clipping_or_capping != "forbidden":
            raise MetastudyContractError("clipping and capping are forbidden")
        if self.selection_order != (
            "require_active_to_decelerating_growth_phase",
            "maximize_worst_experiment_control_separation",
            "minimize_repeated_anchor_drift",
            "minimize_within_acquisition_observation_range",
            "earlier_end_tie_break",
        ):
            raise MetastudyContractError("selection must use the predeclared lexicographic order")
        _digest(self.condition_ontology_digest, label="condition_ontology_digest")
        if self.observation_policy_digest != CANONICAL_OBSERVATION_POLICY_DIGEST:
            raise MetastudyContractError("observation policy digest changed")


DEFAULT_PROTOCOL = MetastudyProtocol(
    protocol_id=PROTOCOL_ID,
    primary_dose_uM=500.0,
    sensitivity_doses_uM=(5.0, 50.0),
    candidate_windows_h=((4.0, 8.0), (6.0, 10.0), (8.0, 12.0), (10.0, 14.0), (12.0, 16.0)),
    endpoint_sensitivity_h=(8.0, 10.0, 12.0, 14.0, 16.0),
    centered_window_sensitivity_widths_h=(2.0, 6.0),
    time_summary_statistic="median",
    within_acquisition_observation_reduction="median",
    ratio_reduction_order="ratio_then_reduce",
    window_boundary="inclusive",
    channel_time_alignment="exact",
    expected_sampling_interval_h=1.0 / 6.0,
    minimum_aligned_timepoints_per_4h_window=25,
    minimum_within_acquisition_observations_per_stratum=3,
    growth_phase_slope_window_h=1.0,
    growth_phase_scale_quantile=0.9,
    growth_phase_minimum_slope_points=4,
    growth_phase_start_minimum=0.5,
    growth_phase_end_minimum=0.1,
    growth_phase_end_maximum=0.6,
    within_acquisition_range_method="within_acquisition_observation_range",
    within_acquisition_range_reference="endpoint_10h",
    minimum_kinetic_experiments=7,
    planned_kinetic_experiments=8,
    planned_kinetic_experiment_ids=(
        "20250622_retron_Eco1_26_43_benchmark",
        "20250707_retron_Eco1_26_43_45_46_benchmark",
        "20250718_retron_Eco1_26_45_47_48_benchmark",
        "20260418_retron_Eco1_26_43_170_171_benchmark",
        "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
        "20260529_retron_Eco1_26_43_177_186_benchmark",
        "20260705_retron_Eco1_26_195_196_180_199_200_197_198_benchmark",
        "20260720_retron_Eco1_26_180_201_202_203_204_benchmark",
    ),
    excluded_snapshot_experiment_ids=("20251105_retron_Eco1_RT_variants",),
    anchor_subject_order=(
        "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO",
        "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
    ),
    planned_anchor_experiment_ids=(
        "20250622_retron_Eco1_26_43_benchmark",
        "20250707_retron_Eco1_26_43_45_46_benchmark",
        "20260418_retron_Eco1_26_43_170_171_benchmark",
        "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
        "20260529_retron_Eco1_26_43_177_186_benchmark",
    ),
    reference_panel_target_ordered_acquisitions=4,
    planned_anchor_acquisitions=5,
    loo_same_or_adjacent_target_fraction=0.75,
    clipping_or_capping="forbidden",
    selection_order=(
        "require_active_to_decelerating_growth_phase",
        "maximize_worst_experiment_control_separation",
        "minimize_repeated_anchor_drift",
        "minimize_within_acquisition_observation_range",
        "earlier_end_tie_break",
    ),
    condition_ontology_digest=CANONICAL_CONDITION_ONTOLOGY_DIGEST,
    observation_policy_digest=CANONICAL_OBSERVATION_POLICY_DIGEST,
)


def protocol_digest(protocol: MetastudyProtocol = DEFAULT_PROTOCOL) -> str:
    """Return the canonical digest of the complete protocol policy."""

    return canonical_digest(asdict(protocol))


__all__ = [
    "CANONICAL_CONDITION_ONTOLOGY_DIGEST",
    "CANONICAL_OBSERVATION_POLICY_DIGEST",
    "DEFAULT_PROTOCOL",
    "PROTOCOL_ID",
    "MetastudyProtocol",
    "Window",
    "protocol_digest",
]
