"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_narrative.py

Reviewer-facing rationale and claim boundaries for metastudy plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

PLOT_RATIONALES: dict[str, str] = {
    "measured_response_examples": (
        "One measured assay summary should remain fixed while the declared target determines its ON and OFF partition."
    ),
    "rmf_cardinality_pressure": (
        "The generic K-state contract needs an explicit test for the order-statistic bias introduced by hard minima "
        "and maxima."
    ),
    "policy_guardrail_matrix": "A promotion decision needs one auditable view of every declared gate.",
    "policy_decision_frontier": (
        "A policy should retain useful predicted fluorescence while improving target-shape fidelity."
    ),
    "score_component_dominance": "Component dominance explains why different setpoints can produce similar ranks.",
    "selected_setpoint_residuals": (
        "State-level residuals reveal which parts of each biological specification are missed."
    ),
    "logic_gate_feasibility": "A shape requirement is operational only when enough candidates satisfy it.",
    "logic_effect_topk_scatter": (
        "A scalar rank can hide the tradeoff between target fidelity and predicted fluorescence."
    ),
    "score_correlation_matrix": "Target-view-specific objectives should not remain nearly rank-equivalent.",
    "selected_vec8_profiles": "Predicted profiles provide a direct check on whether selections differ by setpoint.",
    "sfxi_score_contours": "The score surface makes exponent and clipping behavior inspectable before tuning.",
    "target_view_pareto_fronts": "Pareto structure shows whether the candidate surface contains genuine tradeoffs.",
    "denominator_sensitivity": "A within-round scale constant should not silently determine candidate identity.",
    "policy_comparison_panel_roles": "The exported diagnostic panel must expose why each row was included.",
    "model_validation": "Selection is not actionable when the predictor cannot order held-out observations.",
    "candidate_logic_support": "No downstream selector can recover response shapes absent from model predictions.",
    "logic_effect_tradeoff_overlap": (
        "Overlap is useful as a sensitivity diagnostic after objective support is established."
    ),
    "logic_effect_tradeoff_fidelity": (
        "Tradeoff tuning is relevant only when it improves the weakest target view's shape fidelity."
    ),
    "policy_overlap_summary": (
        "A compact appendix view records candidate reuse without making uniqueness the objective."
    ),
    "topk_overlap_curve": "Overlap at several K values distinguishes a top-six artifact from global score coupling.",
    "reader_event_intervals": "Event-relative labels require a visible bound on intervention-time uncertainty.",
    "response_separation_stability": (
        "Reduction choice should not reverse the component ordering that drives selection."
    ),
    "response_constraint_coverage": (
        "Target-specific support must be visible before one selection posture is applied across target views."
    ),
    "response_uncertainty_sources": "Noise-standardized constraints need scales tied to measured assay variation.",
    "label_model_screen": (
        "The response metric is not actionable unless sequence features preserve its ordering, and sensitivity "
        "reductions must remain visible without becoming post hoc promotion candidates."
    ),
    "retrospective_enrichment": "A retrospective proxy can reject a model before a prospective round is spent.",
    "greedy_support_evidence": (
        "Grouped evidence should be visible before assay capacity is assigned to model-directed choices."
    ),
    "repeated_design_agreement": "Repeated measurements expose assay-source variation hidden by one-row labels.",
}

PLOT_NON_CLAIM_BOUNDARIES: dict[str, str] = {
    "measured_response_examples": (
        "SpyP and sulAp are measured interpretation examples, not complete or optimal biological archetypes."
    ),
    "rmf_cardinality_pressure": (
        "This deterministic screen assumes independent Gaussian noise and is not an assay calibration or a reason "
        "to tune the four-state objective."
    ),
    "policy_guardrail_matrix": "Guardrail passage would authorize review, not establish biological responsiveness.",
    "policy_decision_frontier": "Predicted policy tradeoffs are not measured promoter performance.",
    "score_component_dominance": "Correlation diagnoses scalarization behavior and does not identify mechanism.",
    "selected_setpoint_residuals": "Residuals use predicted vec8 values and are not assay validation.",
    "logic_gate_feasibility": "The 0.45 line is a provisional review threshold, not a biological cutoff.",
    "logic_effect_topk_scatter": "Top-k placement is model-derived and does not establish a responsive promoter.",
    "score_correlation_matrix": "Low correlation alone is not evidence that target-view selections are correct.",
    "selected_vec8_profiles": "Mean predicted profiles can hide candidate-level errors and assay variability.",
    "sfxi_score_contours": "The contour is metric algebra, not an empirical response surface.",
    "target_view_pareto_fronts": "The frontier is predicted and depends on the fitted model and X representation.",
    "denominator_sensitivity": "Within-round sensitivity does not make SFXI scores comparable across assay rounds.",
    "policy_comparison_panel_roles": "Panel membership is for calibration review, not synthesis authorization.",
    "model_validation": "Held-out association measures ranking support and does not prove causal biology.",
    "candidate_logic_support": "Support is computed from model predictions, not measured candidate responses.",
    "logic_effect_tradeoff_overlap": "Reduced overlap is not itself a selection-quality criterion.",
    "logic_effect_tradeoff_fidelity": "Improved predicted fidelity does not replace prospective assay validation.",
    "policy_overlap_summary": "Candidate uniqueness is not evidence of objective correctness.",
    "topk_overlap_curve": "Overlap is diagnostic and should not be optimized independently of fidelity and support.",
    "reader_event_intervals": "A narrow interval does not recover the exact physical stress-addition timestamp.",
    "response_separation_stability": (
        "Agreement among reductions does not establish which interval is biologically optimal."
    ),
    "response_constraint_coverage": "Negative margins do not prove that active learning will find a feasible promoter.",
    "response_uncertainty_sources": (
        "Bootstrap scales quantify observed assay variation, not biological effect thresholds."
    ),
    "label_model_screen": "This fixed screen is not a hyperparameter promotion or prospective model validation.",
    "retrospective_enrichment": "Held-out enrichment is not a measured active-learning hill climb.",
    "greedy_support_evidence": (
        "Retrospective enrichment neither assigns synthesis slots nor proves prospective improvement."
    ),
    "repeated_design_agreement": (
        "Cross-experiment range does not identify the correct aggregation rule or a biological response."
    ),
}

PLOT_DATA_TABLES: dict[str, str] = {
    "measured_response_examples": "tables/measured_response_examples.csv",
    "rmf_cardinality_pressure": "tables/rmf_cardinality_pressure.csv",
    "policy_guardrail_matrix": "tables/policy_summary.csv",
    "policy_decision_frontier": "tables/policy_summary.csv",
    "score_component_dominance": "tables/score_correlations.csv",
    "selected_setpoint_residuals": "tables/top_candidates.csv",
    "logic_gate_feasibility": "tables/policy_summary.csv",
    "logic_effect_topk_scatter": "tables/top_candidates.csv",
    "score_correlation_matrix": "tables/score_correlations.csv",
    "selected_vec8_profiles": "tables/top_candidates.csv",
    "sfxi_score_contours": "tables/policy_summary.csv",
    "target_view_pareto_fronts": "tables/top_candidates.csv",
    "denominator_sensitivity": "tables/denominator_sensitivity.csv",
    "policy_comparison_panel_roles": "tables/policy_comparison_panel.csv",
    "model_validation": "tables/model_validation.csv",
    "candidate_logic_support": "tables/setpoint_support.csv",
    "logic_effect_tradeoff_overlap": "tables/policy_summary.csv",
    "logic_effect_tradeoff_fidelity": "tables/policy_summary.csv",
    "policy_overlap_summary": "tables/policy_summary.csv",
    "topk_overlap_curve": "tables/overlap_by_k.csv",
    "reader_event_intervals": "tables/reader_event_intervals.csv",
    "response_separation_stability": "tables/response_separation_stability.csv",
    "response_constraint_coverage": "tables/response_separation_stability.csv",
    "response_uncertainty_sources": "tables/response_separation_uncertainty.csv",
    "label_model_screen": "tables/label_model_screen.csv",
    "retrospective_enrichment": "tables/retrospective_enrichment_summary.csv",
    "greedy_support_evidence": "tables/campaign_greedy_support.csv",
    "repeated_design_agreement": "tables/repeated_design_agreement.csv",
}
