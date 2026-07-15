"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_definitions.py

Declarative plot definitions for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .plot_contracts import PlotSpec

PLOT_SPECS: tuple[PlotSpec, ...] = (
    PlotSpec(
        plot_id="measured_response_examples",
        filename="measured_response_examples.png",
        tier="primary_decision",
        review_step=1,
        visual_type="measured-vector mask and component decomposition",
        premise="The target mask changes which fixed Reader states define each RMF requirement.",
        decision_value=(
            "Shows how the same measured SpyP and sulAp summaries produce different response and fluorescence "
            "components under the ethanol, ciprofloxacin, and AND masks."
        ),
        alt_text=(
            "Three target-view columns show the fixed four-state Reader response and pDual-10-relative fluorescence "
            "summaries for SpyP and sulAp, the ON/OFF mask over those states, and the three resulting RMF component "
            "bars before target-view standardization. Higher bars move toward feasibility and zero is the provisional "
            "boundary. The measured values stay fixed while the ethanol, ciprofloxacin, and AND masks change."
        ),
    ),
    PlotSpec(
        plot_id="rmf_cardinality_pressure",
        filename="rmf_cardinality_pressure.png",
        tier="metric_diagnostic",
        visual_type="synthetic state-cardinality sensitivity",
        premise="Hard extrema become more noise-sensitive as the state panel grows.",
        decision_value=(
            "Shows when RMF's worst-state components need replicate-aware uncertainty or a prespecified robust "
            "extremum before the objective is extended beyond the four-state assay."
        ),
        alt_text=(
            "Three square line panels show synthetic noise bias in response separation, minimum target-ON "
            "magnitude, and maximum target-OFF magnitude for two, four, eight, and sixteen states. Lines compare "
            "one-ON, balanced, and one-OFF target masks. Bias moves away from zero as the relevant extrema include "
            "more states."
        ),
    ),
    PlotSpec(
        plot_id="policy_guardrail_matrix",
        filename="policy_guardrail_matrix.png",
        tier="screen_appendix",
        visual_type="guardrail heatmap",
        premise="Policy promotion requires both metric guardrails and held-out predictor support.",
        decision_value=(
            "Shows which policies fail because of top-k feasibility, logic fidelity, overlap, score coupling, "
            "or held-out model support."
        ),
        alt_text=(
            "Heatmap where rows are policies and columns are guardrails for full top-k, eligible pool, "
            "logic fidelity, overlap, score coupling, and held-out model support; one means the policy passes."
        ),
    ),
    PlotSpec(
        plot_id="policy_decision_frontier",
        filename="policy_decision_frontier.png",
        tier="screen_appendix",
        visual_type="tradeoff scatter",
        premise="The sweep should be judged by target-shape fidelity and retained effect, not overlap alone.",
        decision_value="Shows whether any policy reaches the logic guardrail while retaining useful effect.",
        alt_text=(
            "Scatter plot of policy-level weakest-target-view median logic fidelity against mean selected effect; "
            "color encodes all-target-view overlap and marker style marks whether every target view produced a full "
            "eligible top-k."
        ),
    ),
    PlotSpec(
        plot_id="score_component_dominance",
        filename="score_component_dominance.png",
        tier="metric_diagnostic",
        visual_type="correlation bars",
        premise="A setpoint-directed score should not be dominated by the effect term.",
        decision_value="Shows whether score tracks logic fidelity or scaled effect in each target view.",
        alt_text=(
            "Grouped bar plot comparing within-target-view Pearson correlation of policy score with logic fidelity "
            "and with scaled effect for canonical SFXI and the declared shape-ceiling comparison."
        ),
    ),
    PlotSpec(
        plot_id="selected_setpoint_residuals",
        filename="selected_setpoint_residuals.png",
        tier="screen_appendix",
        visual_type="residual heatmap",
        premise="Selected predicted logic profiles should move toward each target-view setpoint.",
        decision_value="Shows which SFXI states remain over- or under-predicted after selection.",
        alt_text=(
            "Heatmap of mean selected predicted logic minus the target-view setpoint for states 00, 10, 01, and 11 "
            "under canonical SFXI and the declared shape-ceiling comparison."
        ),
    ),
    PlotSpec(
        plot_id="logic_gate_feasibility",
        filename="logic_gate_feasibility.png",
        tier="metric_diagnostic",
        visual_type="gate sweep scatter",
        premise="Logic gates are only useful if each target view still has enough eligible candidates.",
        decision_value="Shows where stricter logic gates stop producing a full top-k.",
        alt_text=(
            "Scatter plot of logic gate threshold against minimum effective top-k across target views; "
            "color indicates weakest-target-view median selected logic fidelity."
        ),
    ),
    PlotSpec(
        plot_id="logic_effect_topk_scatter",
        filename="logic_effect_topk_scatter.png",
        tier="metric_diagnostic",
        visual_type="component scatter",
        premise="Top-k candidates should be visible as both target-fidelity and effect tradeoffs.",
        decision_value="Separates high-effect candidates from candidates with stronger setpoint fidelity.",
        alt_text=(
            "Scatter plot of top-k candidates by logic fidelity and scaled effect, "
            "faceted by target view and colored by policy."
        ),
    ),
    PlotSpec(
        plot_id="score_correlation_matrix",
        filename="score_correlation_matrix.png",
        tier="metric_diagnostic",
        visual_type="correlation heatmap",
        premise="Setpoint-specific policies should reduce cross-target-view score coupling.",
        decision_value="Shows whether policy changes actually separate target-view score surfaces.",
        alt_text=(
            "Heatmap of pairwise score correlations between ethanol, ciprofloxacin, "
            "and AND target views for focus policies."
        ),
    ),
    PlotSpec(
        plot_id="selected_vec8_profiles",
        filename="selected_vec8_profiles.png",
        tier="metric_diagnostic",
        visual_type="profile heatmap",
        premise="Selected candidates should show visibly different predicted logic profiles across setpoints.",
        decision_value="Shows mean selected vec8 logic profiles by policy and target view.",
        alt_text=(
            "Heatmap of mean predicted logic-state values for selected candidates under focus policies and "
            "target views."
        ),
    ),
    PlotSpec(
        plot_id="sfxi_score_contours",
        filename="sfxi_score_contours.png",
        tier="metric_diagnostic",
        visual_type="score-surface contour",
        premise="SFXI policy changes should visibly alter how logic fidelity and effect trade off.",
        decision_value="Shows why effect can dominate the score unless logic is weighted or gated.",
        alt_text=(
            "Contour plots of SFXI score over logic fidelity and scaled effect for canonical SFXI "
            "and the declared normalized scalar tradeoff."
        ),
    ),
    PlotSpec(
        plot_id="target_view_pareto_fronts",
        filename="target_view_pareto_fronts.png",
        tier="metric_diagnostic",
        visual_type="Pareto scatter",
        premise=(
            "A target view should expose candidates that trade off logic fidelity and effect, not one scalar alone."
        ),
        decision_value="Shows selected candidates against the predicted candidate cloud for each stress setpoint.",
        alt_text=(
            "Scatter plots of sampled candidates by logic fidelity and scaled effect, faceted by target view, "
            "with selected canonical SFXI and shape-ceiling comparison candidates overlaid."
        ),
    ),
    PlotSpec(
        plot_id="denominator_sensitivity",
        filename="denominator_sensitivity.png",
        tier="metric_diagnostic",
        visual_type="sensitivity line plot",
        premise="Intensity scaling should not be the hidden driver of setpoint-directed selection.",
        decision_value="Shows whether changing the SFXI denominator changes top-k logic or effect summaries.",
        alt_text=(
            "Line plot of median top-k logic fidelity and scaled effect as the denominator is scaled down "
            "or up for focus policies."
        ),
    ),
    PlotSpec(
        plot_id="policy_comparison_panel_roles",
        filename="policy_comparison_panel_roles.png",
        tier="screen_appendix",
        visual_type="panel composition bars",
        premise="Policy review should compare declared diagnostic strata instead of presenting a winner list.",
        decision_value="Shows which metric-behavior strata are represented in policy_comparison_panel.csv.",
        alt_text=(
            "Bar plot counting policy-comparison rows by role and target view, including canonical SFXI high-effect, "
            "shape/effect, logic-first, OFF-state-logic-penalized, shared-overlap, and target-view-specific rows."
        ),
    ),
    PlotSpec(
        plot_id="model_validation",
        filename="model_validation.png",
        tier="metric_diagnostic",
        visual_type="held-out performance summary",
        premise=(
            "A metric rerank is not actionable unless held-out vec8 predictions preserve observed response ordering."
        ),
        decision_value="Shows target-level and selection-view validation across repeated retraining seeds.",
        alt_text=(
            "Point and interval plot of held-out Spearman correlations for the eight vec8 targets and the "
            "ethanol, ciprofloxacin, and AND SFXI scores, separated by shuffled and leave-experiment-out "
            "validation; values near zero indicate weak out-of-sample ordering support."
        ),
    ),
    PlotSpec(
        plot_id="candidate_logic_support",
        filename="candidate_logic_support.png",
        tier="metric_diagnostic",
        visual_type="threshold support curve",
        premise="A scalarizer cannot select response shapes absent from the predicted candidate surface.",
        decision_value="Shows how many candidates remain as the required setpoint fidelity increases.",
        alt_text=(
            "Line plot of candidate count versus minimum logic fidelity for ethanol, ciprofloxacin, and AND; "
            "a vertical line marks the provisional review guardrail and a horizontal line marks six candidates."
        ),
    ),
    PlotSpec(
        plot_id="logic_effect_tradeoff_overlap",
        filename="logic_effect_tradeoff_overlap.png",
        tier="screen_appendix",
        visual_type="tradeoff line plot",
        premise="The normalized logic-effect tradeoff changes target-view overlap.",
        decision_value="Shows which identifiable tradeoff weights increase unique top-k selections.",
        alt_text=(
            "Line plot of unique candidate IDs across the three target-view selections as the normalized logic "
            "tradeoff weight increases from effect-only to logic-only."
        ),
    ),
    PlotSpec(
        plot_id="logic_effect_tradeoff_fidelity",
        filename="logic_effect_tradeoff_fidelity.png",
        tier="screen_appendix",
        visual_type="tradeoff line plot",
        premise="A logic-effect tradeoff should improve target-shape fidelity, not only candidate uniqueness.",
        decision_value="Shows the weakest target view's selected logic fidelity across identifiable tradeoff weights.",
        alt_text=(
            "Line plot of weakest-target-view median top-k logic fidelity as normalized logic weight increases; "
            "higher values indicate closer predicted response shapes."
        ),
    ),
    PlotSpec(
        plot_id="policy_overlap_summary",
        filename="policy_overlap_summary.png",
        tier="screen_appendix",
        visual_type="overlap bar plot",
        premise="Primary policy families differ in how much they reuse candidates across target views.",
        decision_value="Summarizes top-k uniqueness and overlap for canonical SFXI and candidate policy families.",
        alt_text="Bar plot of unique top-k sequences with overlap annotations for primary policy families.",
    ),
    PlotSpec(
        plot_id="topk_overlap_curve",
        filename="topk_overlap_curve.png",
        tier="screen_appendix",
        visual_type="overlap curve",
        premise="Target-view collapse should be checked beyond top-6.",
        decision_value="Shows whether shared candidates remain high as K increases.",
        alt_text="Line plot of observed all-three selected-candidate overlap across K for the focus policies.",
    ),
    PlotSpec(
        plot_id="reader_event_intervals",
        filename="reader_event_intervals.png",
        tier="metric_diagnostic",
        visual_type="event interval bars",
        premise="Recorded stress addition is bounded tightly enough to support an event-relative sensitivity screen.",
        decision_value="Shows the unresolved transition interval and available post-stress coverage for each source.",
        alt_text=(
            "Horizontal bars show the gap between the last pre-stress and first post-stress Reader acquisition "
            "for eight experiments; annotations report the available post-event coverage."
        ),
    ),
    PlotSpec(
        plot_id="response_separation_stability",
        filename="response_separation_stability.png",
        tier="primary_decision",
        review_step=2,
        visual_type="component stability heatmap",
        premise=(
            "The primary reduction should preserve response and anchored fluorescence ordering across nearby summaries."
        ),
        decision_value="Shows the weakest component agreement with the primary 6-12 hour post-stress log mean.",
        alt_text=(
            "Heatmap of the minimum Spearman correlation across response separation, ON fluorescence, and OFF "
            "fluorescence for each target mask and prespecified time reduction relative to the 6-12 hour "
            "post-stress log mean. Nearby reductions retain high component ordering, with the weakest active-"
            "target-view agreement at approximately 0.92."
        ),
    ),
    PlotSpec(
        plot_id="response_constraint_coverage",
        filename="response_constraint_coverage.png",
        tier="metric_diagnostic",
        visual_type="constraint support bars",
        premise="The same 35 observations provide uneven support for the campaign masks and the OR pressure test.",
        decision_value="Separates response ordering from all three response and fluorescence requirements by target.",
        alt_text=(
            "Grouped bars compare the number of observed designs where every target-ON YFP-to-CFP response exceeds "
            "every target-OFF response with the number that also meet the provisional pDual-10-relative ON and OFF "
            "fluorescence boundaries for the three campaign masks and the OR pressure-test mask."
        ),
    ),
    PlotSpec(
        plot_id="response_uncertainty_sources",
        filename="response_uncertainty_sources.png",
        tier="metric_diagnostic",
        visual_type="uncertainty source bars",
        premise="Metric scales should reflect the dominant measured uncertainty source.",
        decision_value="Compares replicate-bootstrap variation with event-time interval sensitivity by component.",
        alt_text=(
            "Grouped bars show median replicate-bootstrap standard deviation and maximum event-bound deviation for "
            "response separation, ON fluorescence, and OFF fluorescence under the three campaign masks and the "
            "OR pressure-test mask."
        ),
    ),
    PlotSpec(
        plot_id="label_model_screen",
        filename="label_model_screen.png",
        tier="primary_decision",
        review_step=3,
        visual_type="grouped validation heatmap",
        premise=(
            "A label representation is actionable only if sequence features preserve response-separation and "
            "feasibility ordering."
        ),
        decision_value=(
            "Compares fixed RF, PLS, and PCA-ridge challengers while holding out complete Reader experiments."
        ),
        alt_text=(
            "Heatmap of the weakest median within-experiment Spearman correlation across response separation, "
            "feasibility, ethanol, ciprofloxacin, and AND for each label representation and fixed model challenger; "
            "adjacent-window, AUC, and pre-window-delta columns are marked as sensitivity analyses. The configured "
            "campaign model, baseline, and fixed challengers remain separate, and the screen does not promote a model."
        ),
    ),
    PlotSpec(
        plot_id="retrospective_enrichment",
        filename="retrospective_enrichment.png",
        tier="metric_diagnostic",
        visual_type="held-out enrichment heatmap",
        premise="A plausible active-learning direction should enrich held-out choices above random ordering.",
        decision_value="Shows the true within-experiment percentile selected by each representation's best challenger.",
        alt_text=(
            "Heatmap of median true feasibility percentile for the predicted best design in each held-out Reader "
            "experiment, split by active target view and label representation; 0.5 is median rather than proof of "
            "a hill climb."
        ),
    ),
    PlotSpec(
        plot_id="greedy_support_evidence",
        filename="greedy_support_evidence.png",
        tier="primary_decision",
        review_step=4,
        visual_type="grouped evidence interval plot",
        premise="Pure greedy selection requires grouped evidence that predicted leaders enrich held-out measurements.",
        decision_value=("Shows configured-campaign-model retrospective enrichment and its finite-sample uncertainty."),
        alt_text=(
            "Points and exact 95 percent binomial intervals show the fraction of held-out Reader experiments where "
            "the configured campaign random forest's predicted best design beat that experiment's median true score "
            "for each target view. The intervals are retrospective risk evidence, not selection authority."
        ),
    ),
    PlotSpec(
        plot_id="repeated_design_agreement",
        filename="repeated_design_agreement.png",
        tier="metric_diagnostic",
        visual_type="cross-experiment range heatmap",
        premise="Repeated Reader experiments must agree well enough to justify one candidate-level label.",
        decision_value=(
            "Shows which response or anchored-fluorescence channels depend strongly on the selected experiment."
        ),
        alt_text=(
            "Heatmap of cross-experiment range for repeated designs. Columns are grouped into four YFP-to-CFP "
            "response fields r00, r10, r01, and r11 and four pDual-10-relative YFP-to-OD600 fluorescence fields "
            "b00, b10, b01, and b11, with each condition named; larger values indicate stronger source sensitivity."
        ),
    ),
)
