"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_vocabulary.py

Closed publication vocabulary for response-metastudy plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from ..core.policies import audit_policy_specs

TARGET_VIEW_ORDER = ("ethanol", "ciprofloxacin", "and", "or")
TARGET_VIEW_LABELS = {
    "ethanol": "Ethanol",
    "ciprofloxacin": "Ciprofloxacin",
    "and": "AND",
    "or": "OR (prospective)",
}
POLICY_LABELS = {policy.id: policy.label for policy in audit_policy_specs()}
POLICY_COMPACT_LABELS = {
    "sfxi_beta1_gamma1": "Canonical SFXI",
    "logic_first_beta4_gamma05": "Logic-weighted",
    "logic_gate055_effect": "Logic gate 0.55",
    "lexicographic_logic_effect": "Lexicographic",
    "off_state_logic_eta2_beta2_gamma05": "OFF-state penalty",
}
PANEL_ROLE_LABELS = {
    "canonical_sfxi_high_effect": "Canonical SFXI high effect",
    "comparison_shape_effect": "Shape/effect comparison",
    "high_logic_lower_effect": "High logic, lower effect",
    "off_state_logic_penalized": "OFF-state logic penalty",
    "canonical_sfxi_shared_overlap": "Canonical SFXI shared overlap",
    "target_view_specific_comparison": "Target-view-specific comparison",
}
MODEL_METRIC_LABELS = {
    "v00": "Logic\nNo stress (v00)",
    "v10": "Logic\nEthanol (v10)",
    "v01": "Logic\nCiprofloxacin (v01)",
    "v11": "Logic\nBoth stresses (v11)",
    "y00_star": "Fluorescence\nNo stress (y*00)",
    "y10_star": "Fluorescence\nEthanol (y*10)",
    "y01_star": "Fluorescence\nCiprofloxacin (y*01)",
    "y11_star": "Fluorescence\nBoth stresses (y*11)",
    "ethanol": "Ethanol objective",
    "ciprofloxacin": "Ciprofloxacin objective",
    "and": "AND objective",
}
REPRESENTATION_ORDER = (
    "event_logmean_6_12h_post",
    "event_logmean_6_12h_post__factorial_contrast7",
    "event_logmean_4_8h_post",
    "event_logmean_8_12h_post",
    "event_linear_auc_6_12h_post",
    "event_logmean_6_12h_delta_pre1h",
)
REDUCTION_ORDER = (
    "event_logmean_6_12h_post",
    "event_logmean_4_8h_post",
    "event_logmean_8_12h_post",
    "event_linear_auc_6_12h_post",
    "event_logmean_6_12h_delta_pre1h",
)
REPRESENTATION_LABELS = {
    "event_logmean_6_12h_post": "6-12 h\nlog mean",
    "event_logmean_6_12h_post__factorial_contrast7": "6-12 h\nfactorial\ncontrasts",
    "event_logmean_4_8h_post": "4-8 h\nlog mean",
    "event_logmean_8_12h_post": "8-12 h\nlog mean",
    "event_linear_auc_6_12h_post": "6-12 h\nlinear AUC",
    "event_logmean_6_12h_delta_pre1h": "6-12 h\nlog mean -\npre-stress",
}
REPRESENTATION_ROLES = {
    "event_logmean_6_12h_post": "Candidate",
    "event_logmean_6_12h_post__factorial_contrast7": "Candidate",
    "event_logmean_4_8h_post": "Sensitivity",
    "event_logmean_8_12h_post": "Sensitivity",
    "event_linear_auc_6_12h_post": "Sensitivity",
    "event_logmean_6_12h_delta_pre1h": "Sensitivity",
}
PREDICTION_COMPONENT_LABELS = {
    "median_logic_fidelity": "Median logic fidelity",
    "median_effect_scaled": "Median scaled effect",
}
SELECTION_LAYER_LABELS = {
    "candidate sample": "Candidate sample",
    "selected top-6": "Selected top 6",
}
STATE_TICK_LABELS = {
    "00": "No stress",
    "10": "Ethanol",
    "01": "Ciprofloxacin",
    "11": "Both stresses",
}
READER_EXPERIMENT_LABELS = {
    "20260117_sfxi_ref-pDual10": "2026-01-17 | pDual-10 reference",
    "20260121_sfxi_ref-pDual10": "2026-01-21 | pDual-10 reference",
    "20260619_sfxi_sensor-panel-m9-glu-1-10": "2026-06-19 | sensor panel 1-10",
    "20260620_sfxi_sensor-panel-m9-glu-12-19": "2026-06-20 | sensor panel 12-19",
    "20260621_sfxi_sensors-opal-20-28": "2026-06-21 | OPAL sensor panel 20-28",
    "20260622_sfxi_sensor-panel-m9-glu-29-30-sulAp-spyp": "2026-06-22 | panel 29-30 + sulAp/SpyP",
    "20260706_sfxi_sensor-panel-m9-glu-secg": "2026-07-06 | SECG sensor panel",
    "20260707_sfxi_sensor-panel-m9-glu-secg": "2026-07-07 | SECG sensor panel",
}


def target_view_label(value: object) -> str:
    return _closed_label(TARGET_VIEW_LABELS, value, kind="target view")


def policy_label(value: object) -> str:
    return _closed_label(POLICY_LABELS, value, kind="policy")


def compact_policy_label(value: object) -> str:
    return _closed_label(POLICY_COMPACT_LABELS, value, kind="compact policy")


def panel_role_label(value: object) -> str:
    return _closed_label(PANEL_ROLE_LABELS, value, kind="panel role")


def model_metric_label(value: object) -> str:
    return _closed_label(MODEL_METRIC_LABELS, value, kind="model metric")


def representation_label(value: object) -> str:
    return _closed_label(REPRESENTATION_LABELS, value, kind="label representation")


def representation_role(value: object) -> str:
    return _closed_label(REPRESENTATION_ROLES, value, kind="label representation role")


def prediction_component_label(value: object) -> str:
    return _closed_label(PREDICTION_COMPONENT_LABELS, value, kind="prediction component")


def selection_layer_label(value: object) -> str:
    return _closed_label(SELECTION_LAYER_LABELS, value, kind="selection layer")


def reader_experiment_label(value: object) -> str:
    return _closed_label(READER_EXPERIMENT_LABELS, value, kind="Reader experiment")


def _closed_label(mapping: Mapping[str, str], value: object, *, kind: str) -> str:
    key = str(value)
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"{kind} {key!r} has no publication label.") from exc


__all__ = [
    "TARGET_VIEW_LABELS",
    "TARGET_VIEW_ORDER",
    "REDUCTION_ORDER",
    "REPRESENTATION_ORDER",
    "STATE_TICK_LABELS",
    "compact_policy_label",
    "model_metric_label",
    "panel_role_label",
    "policy_label",
    "prediction_component_label",
    "reader_experiment_label",
    "representation_label",
    "representation_role",
    "selection_layer_label",
    "target_view_label",
]
