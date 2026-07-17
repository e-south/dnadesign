"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/multistate_behavior_report.py

Plain scientific report for the multistate behavior shadow decision.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def render_multistate_behavior_report(
    decision: Mapping[str, object],
    *,
    grouped_validation: pd.DataFrame,
    allocation_comparison: pd.DataFrame,
    hard_behavior_summary: pd.DataFrame,
    observed_control_face_validity: pd.DataFrame,
    family_cardinality_pressure: pd.DataFrame,
) -> str:
    """Render the split verdict and its claim boundaries without promotional prose."""

    predictive = _mapping(decision, "predictive_support")
    robustness = _mapping(decision, "normalization_robustness")
    technical = _mapping(decision, "technical_readiness")
    implementation = _mapping(decision, "shadow_implementation")
    audit = _mapping(decision, "independent_adversarial_implementation_audit")
    lines = [
        "# Multistate Response Behavior shadow decision",
        "",
        "## Decision",
        "",
        f"Overall promotion decision: **{decision['promotion_decision']}**.",
        "",
        (
            f"Semantic-fit verdict: **{_mapping(decision, 'semantic_fit')['verdict']}**. "
            f"Shadow-implementation verdict: **{implementation['verdict']}**. Grouped retrospective support is "
            "insufficient and prospective hill-climb efficacy is unproven. Campaign activation and synthesis therefore "
            "remain prohibited."
        ),
        "",
        "## Biological behavior",
        "",
        (
            "The score rewards three families: every intended ON response above every intended OFF response, strong "
            "intended-ON signal relative to same-state pDual-10, and suppressed intended-OFF signal relative to "
            "same-state pDual-10. In real arithmetic, every desirable coordinate change strictly improves the score. "
            "At finite-precision extremes the implementation guarantees nondecrease. The score does not claim absolute "
            "absence of expression."
        ),
        "",
        "## Observed biological face validity",
        "",
        _face_validity_table(observed_control_face_validity),
        "",
        (
            "SpyP and sulAp are existing assay north stars, not score thresholds and not DenseGen-architecture "
            "matches. The study claims no equivalent positive AND control."
        ),
        "",
        "## Family-cardinality pressure",
        "",
        _cardinality_table(family_cardinality_pressure),
        "",
        (
            "Family balancing prevents masks with more coordinates from silently receiving more prior weight, but it "
            "does not make the selector noncompensatory. One weak coordinate receives less bottleneck weight as its "
            "family grows. The conditional semantic GO applies only to within-view ranking under the same ordered "
            "states, "
            "target mask, normalization record, and protocol. Scores are not comparable across views or state spaces. "
            "The behavior score is not a conformance test; the hard bottleneck is observable but does not constrain "
            "selection. In this study, ethanol/ciprofloxacin response coordinates have prior 1/12 "
            "(log(12)=2.485 maximum gap), while AND response/OFF coordinates have prior 1/9 (log(9)=2.197). "
            "A materially negative coordinate can coexist with a positive score."
        ),
        "",
        "## Normalization robustness",
        "",
        (
            "Protocol-frozen quantile and leave-one-source-experiment-out scenarios: "
            f"{robustness['scenario_count']}. "
            "Minimum rank correlation with the q90 primary convention: "
            f"{float(robustness['minimum_rank_spearman_vs_primary']):.3f}. "
            f"Minimum raw Top-6 overlap: {int(robustness['minimum_raw_top_k_overlap'])}/6."
        ),
        "",
        "These values characterize scale dependence. They are not tuned acceptance thresholds.",
        "",
        "## Same-prediction objective comparison",
        "",
        _objective_comparison_table(hard_behavior_summary),
        "",
        (
            "This comparison scores one fixed raw-Y prediction matrix with both objectives. It measures "
            "objective-induced ranking changes; it is not prediction-to-truth validation or prospective efficacy "
            "evidence."
        ),
        "",
        "## Grouped prediction-to-truth support",
        "",
        (
            f"Verified exact promoted candidates: {int(predictive['promoted_candidate_count'])}; "
            f"label-source experiments: {int(predictive['label_source_experiment_count'])}; "
            "minimum rank-defined held-out groups: "
            f"{int(predictive['minimum_rank_defined_group_count'])}; seeds: {int(predictive['seed_count'])}. "
            "Weakest median within-heldout-group Spearman: "
            f"{float(predictive['weakest_median_within_group_spearman']):.3f}; "
            f"weakest pooled out-of-fold Spearman: {float(predictive['weakest_pooled_oof_spearman']):.3f}."
        ),
        "",
        _grouped_table(grouped_validation),
        "",
        (
            "The median excludes held-out groups whose within-group rank correlation is undefined, including singleton "
            "groups. This is prediction-to-truth evidence. It is distinct from comparing two objective functions on "
            "the "
            "same prediction matrix, and it cannot establish prospective hill-climb efficacy. Pooled out-of-fold "
            "Spearman is secondary and descriptive because candidates from different folds use different "
            "train-fold-only scales."
        ),
        "",
        "## Allocation comparison",
        "",
        _allocation_table(allocation_comparison),
        "",
        (
            "Both allocations use OPAL's public round-robin next-best-unallocated runtime with sequence deduplication. "
            "This is a read-only preview, not a campaign mutation."
        ),
        "",
        "## Corrected Reader source equivalence",
        "",
        (
            f"Technical verdict: **{technical['verdict']}**. The corrected Reader bundle proves pDual-10 "
            "self-reference central values, bootstrap standard deviations, and all joint-bootstrap draws are exactly "
            "zero. Its selected central candidate vectors are exactly equal to the immutable promoted labels, so no "
            "new observation version is required for point-label reuse. The shadow evidence is a new source-bound "
            "publication "
            "because its uncertainty evidence and digests changed."
        ),
        "",
        "## Independent adversarial implementation audit",
        "",
        f"Status: **{audit['status']}**. Method: {audit['method']}.",
        "",
        "Scope: " + "; ".join(str(item) for item in audit["scope"]),
        "",
        "Findings:",
        "",
        *(f"- {item}" for item in audit["findings"]),
        "",
        "Blockers:",
        "",
        *((f"- {item}" for item in audit["blockers"]) if audit["blockers"] else ("- None.",)),
        "",
        "## Claim boundary",
        "",
        (
            "The evidence supports semantic improvement and a verified shadow implementation. "
            if implementation["verdict"] == "go"
            else "The evidence supports semantic improvement, but implementation audit blockers remain. "
        )
        + (
            "It does not yet support the claim that this objective will hill-climb more effectively than RMF. That "
            "comparison requires predictions frozen before new measurements over prospective rounds."
        ),
        "",
        (
            "Growth, viability, and expression burden remain separate assay-QC evidence. They are not encoded in this "
            "signal and response objective and must not be inferred from its score."
        ),
        "",
    ]
    return "\n".join(lines)


def _grouped_table(frame: pd.DataFrame) -> str:
    summary = frame.drop_duplicates(["seed", "selection_view_id", "objective_name"])
    rows = (
        summary.groupby(["objective_name", "selection_view_id"], sort=True)[
            ["median_within_group_spearman", "pooled_oof_spearman"]
        ]
        .median()
        .reset_index()
    )
    output = ["| Objective | View | Median within-group rho | Pooled OOF rho |", "| --- | --- | ---: | ---: |"]
    for row in rows.itertuples(index=False):
        output.append(
            f"| {row.objective_name} | {row.selection_view_id} | "
            f"{float(row.median_within_group_spearman):.3f} | {float(row.pooled_oof_spearman):.3f} |"
        )
    return "\n".join(output)


def _allocation_table(frame: pd.DataFrame) -> str:
    objectives = tuple(sorted(frame["objective_name"].astype(str).unique()))
    selected = {name: set(frame.loc[frame["objective_name"].eq(name), "id"].astype(str)) for name in objectives}
    overlap = len(selected[objectives[0]] & selected[objectives[1]])
    output = ["| Objective | Unique sequences | View quotas |", "| --- | ---: | --- |"]
    for name in objectives:
        rows = frame.loc[frame["objective_name"].eq(name)]
        quotas = ", ".join(f"{view}: {count}" for view, count in rows.groupby("selection_view_id").size().items())
        output.append(f"| {name} | {rows['sequence_sha256'].nunique()} | {quotas} |")
    output.append(f"\nCross-objective candidate overlap: {overlap}/18.")
    return "\n".join(output)


def _objective_comparison_table(frame: pd.DataFrame) -> str:
    output = [
        "| View | Rank rho | Raw Top-6 overlap | Median absolute rank shift | Maximum absolute rank shift |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.sort_values("selection_view_id", kind="mergesort").itertuples(index=False):
        output.append(
            f"| {row.selection_view_id} | {float(row.hard_behavior_spearman):.3f} | "
            f"{int(row.raw_top_k_overlap)}/{int(row.raw_top_k)} | {float(row.median_absolute_rank_shift):.1f} | "
            f"{int(row.maximum_absolute_rank_shift)} |"
        )
    return "\n".join(output)


def _face_validity_table(frame: pd.DataFrame) -> str:
    output = [
        (
            "| Control | View | Reader experiment | Candidate-experiment rank | Behavior score | "
            "Hard bottleneck | Limiting coordinate |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in frame.sort_values(["selection_view_id", "reader_experiment_id"], kind="mergesort").itertuples(
        index=False
    ):
        output.append(
            f"| {row.display_label} | {row.selection_view_id} | {row.reader_experiment_id} | "
            f"{int(row.observed_unit_rank)}/{int(row.observed_unit_count)} | {float(row.behavior_score):.3f} | "
            f"{float(row.hard_bottleneck_clearance):.3f} | {row.limiting_coordinate} |"
        )
    return "\n".join(output)


def _cardinality_table(frame: pd.DataFrame) -> str:
    output = [
        (
            "| States | Response pairs | Global max gap | Weak-coordinate bound | Realizable score | "
            "Hard minimum | Weak-coordinate weight |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.sort_values("state_count", kind="mergesort").itertuples(index=False):
        output.append(
            f"| {int(row.state_count)} | {int(row.response_pair_count)} | "
            f"{float(row.analytic_global_maximum_soft_vs_hard_gap):.2f} | "
            f"{float(row.weak_coordinate_analytic_soft_vs_hard_gap):.2f} | "
            f"{float(row.realizable_behavior_score):.3f} | "
            f"{float(row.realizable_hard_bottleneck):.1f} | {float(row.weak_coordinate_bottleneck_weight):.3f} |"
        )
    return "\n".join(output)


def _mapping(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"behavior report decision field {key!r} is malformed.")
    return value


__all__ = ["render_multistate_behavior_report"]
