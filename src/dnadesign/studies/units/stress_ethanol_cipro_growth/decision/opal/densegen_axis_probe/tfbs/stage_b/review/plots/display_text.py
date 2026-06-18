"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/review/plots/display_text.py

Manuscript-facing text contract for Stage B realized-label review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ....label_text import (
    SEQUENCE_MATCHED_METADATA_LABEL,
    tfbs_control_display_label,
    tfbs_label_compact_title,
    tfbs_label_expression,
    tfbs_label_title,
)

NO_ENRICHMENT_BASELINE_LABEL = "Pool average"
NO_ENRICHMENT_BASELINE_CONTRACT = "No enrichment: selected mean equals the same label-table pool mean"
SAME_BATCH_TOP_K_REFERENCE_LABEL = "Best possible batch"
SAME_BATCH_TOP_K_REFERENCE_CONTRACT = (
    "Best possible single batch: mean of the top selection_k label values in the same label table divided by the "
    "same label-table pool mean. This is a full-pool reference, not an observed campaign and not the multi-round "
    "same-budget known-label ranking."
)
TRAJECTORY_X_AXIS_LABEL = "Round"
INITIAL_BATCH_TICK_LABEL = "Initial seed\nbatch"
TRAILING_TRAJECTORY_NOTE = (
    "Faint lines = seed runs; bold line = mean selected batch; diamond = initial seed batch; "
    "dashed = pool average; dotted = best possible batch."
)
REALIZED_REVIEW_TEXT_CONTRACT = {
    "baseline": NO_ENRICHMENT_BASELINE_CONTRACT,
    "count_fraction_label": (
        "count_fraction label = target TFBS count / 3 per sequence; plotted values are enrichment ratios, "
        "not raw counts"
    ),
    "initial_batch": "diamond markers are the same initial seed-batch IDs scored by each label table before round 0",
    "interval": "mean plus/minus sample SD across seed runs; n is recorded; not an inferential CI",
    "legend_layout": "legend below the plot; wrap when needed to avoid clipping",
    "pairing": (
        "sequence-matched metadata and control campaigns share initial selected IDs; only the label table differs"
    ),
    "role_labels": "sequence-matched metadata versus profile-appropriate matched control",
    "selected_label_values": (
        "selected_true_* artifact columns are selected values from that campaign's label table; for shuffled "
        "controls this is a control-label value, not post hoc sequence-matched metadata truth"
    ),
    "trajectory_semantics": (
        "line points are per-round top-k selected batches; round 0 is the first acquired batch after the "
        "initial seed-batch IDs, not the initial seed batch itself"
    ),
    "same_batch_top_k_reference": SAME_BATCH_TOP_K_REFERENCE_CONTRACT,
    "subtitle_layout": "centered single-line subtitle",
    "title_alignment": "title centered over the axes frame; title may wrap, subtitle must not wrap",
    "type_scale": "axis labels, tick labels, subtitle, and legend use the same review body size",
}


def role_display_label(role: object, *, label_name: object, control_role: object | None = None) -> str:
    """Return a reviewer-facing label source name for the campaign role."""

    role_text = str(role)
    if role_text == "positive":
        return SEQUENCE_MATCHED_METADATA_LABEL
    if role_text == "matched_null":
        default_role = "" if _is_slot_label(label_name) else "matched_label_permutation_negative_control"
        return tfbs_control_display_label(control_role or default_role, label_name=label_name).capitalize()
    return role_text.replace("_", " ")


def trajectory_plot_title(
    label_name: object,
    *,
    replicate_count: int,
    control_role: object | None = None,
) -> str:
    """Return the visible title for the round trajectory plot."""

    del replicate_count
    if _is_slot_label(label_name):
        return f"Active selection enriches {_trajectory_title_label(label_name)} over slot-shuffled control"
    return f"Active selection enriches {_trajectory_title_label(label_name)} over row-shuffled control"


def trajectory_plot_subtitle(
    label_name: object,
    *,
    replicate_count: int,
    control_role: object | None = None,
) -> str:
    """Return the visible subtitle carrying pairing and interval semantics."""

    del label_name, replicate_count, control_role
    return ""


def label_definition(label_name: object) -> str:
    """Return the visible definition of the plotted label value."""

    expression = tfbs_label_expression(label_name)
    if expression:
        return f"{tfbs_label_compact_title(label_name)} = {expression} per sequence"
    return f"{tfbs_label_compact_title(label_name)} label value"


def trajectory_y_axis_label(label_name: object) -> str:
    """Return a label-specific y-axis title that distinguishes enrichment from raw label values."""

    return r"Enrichment vs pool ($\bar{y}_{sel}/\bar{y}_{pool}$)"


def enrichment_formula_text(label_name: object) -> str:
    """Return visible wording for the plotted enrichment ratio."""

    if tfbs_label_expression(label_name):
        return "y = selected mean fraction / same label-table pool mean fraction"
    return "y = selected mean label value / same label-table pool mean"


def trajectory_encoding_note(*, replicate_count: int) -> str:
    """Return the bottom encoding note for the trajectory plot."""

    if replicate_count > 1:
        return TRAILING_TRAJECTORY_NOTE
    return (
        "Line = selected batch trajectory; diamond = initial seed-batch IDs before round 0; "
        "dashed = pool average; dotted = best possible batch."
    )


def seed_run_sample_sd_label(*, replicate_count: int) -> str:
    """Return visible wording for a seed-run sample-SD interval."""

    return f"Mean +/- SD (n={int(replicate_count)})"


def seed_pair_sample_sd_label(*, replicate_count: int) -> str:
    """Return visible wording for a paired-seed sample-SD interval."""

    return f"Mean +/- SD (n={int(replicate_count)})"


def positive_null_summary_title(
    label_name: object,
    *,
    replicate_count: int,
    control_role: object | None = None,
) -> str:
    """Return the visible title for the sequence-matched-minus-control endpoint summary."""

    del replicate_count
    if _is_slot_label(label_name):
        return f"Sequence-matched metadata beats slot-shuffled control for {_summary_title_label(label_name)}"
    return f"Sequence-matched metadata beats row-shuffled control for {_summary_title_label(label_name)}"


def positive_null_summary_subtitle(*, replicate_count: int) -> str:
    """Return the visible subtitle for the endpoint summary."""

    del replicate_count
    return ""


def plot_manifest_title(
    kind: str,
    *,
    label_name: object,
    replicate_count: int,
    control_role: object | None = None,
) -> str:
    """Return the manifest title that mirrors the visible plot title."""

    if kind == "realized_label_lift_trajectory":
        return trajectory_plot_title(label_name, replicate_count=replicate_count, control_role=control_role)
    if kind == "positive_null_lift_summary":
        return positive_null_summary_title(label_name, replicate_count=replicate_count, control_role=control_role)
    return tfbs_label_title(label_name)


def plot_manifest_alt_text(
    kind: str,
    *,
    label_name: object,
    replicate_count: int,
    control_role: object | None = None,
) -> str:
    """Return plot alt text with the same design semantics as the rendered figure."""

    label_title = tfbs_label_compact_title(label_name)
    control = tfbs_control_display_label(control_role, label_name=label_name)
    if kind == "realized_label_lift_trajectory":
        interval = "band=SD" if replicate_count > 1 else "single seed run"
        return (
            f"{label_title} selected-batch enrichment vs pool. "
            f"Round 0 follows the initial seed batch. "
            f"Lines show sequence-matched metadata and {control}; dotted = best possible single batch; "
            f"{interval}, not CI."
        )
    if kind == "positive_null_lift_summary":
        interval = (
            f"mean bars with sample-SD whiskers across {int(replicate_count)} seed pairs"
            if replicate_count > 1
            else "single seed pair"
        )
        return (
            f"{label_title} sequence-matched-minus-control enrichment for final round and trajectory AUC; {interval}."
        )
    return f"Review plot for {label_title}."


def _paired_seed_run_phrase(replicate_count: int) -> str:
    count = int(replicate_count)
    noun = "start" if count == 1 else "starts"
    return f"{count} paired {noun}"


def _trajectory_title_label(label_name: object) -> str:
    text = str(label_name or "")
    if text == "lexA_in_slot0":
        return "LexA in the leftmost slot"
    if text == "cpxR_or_baeR_in_slot2":
        return "CpxR/BaeR in the rightmost slot"
    if text == "baeR_in_slot1":
        return "BaeR in the middle slot"
    return tfbs_label_compact_title(label_name)


def _summary_title_label(label_name: object) -> str:
    text = str(label_name or "")
    if text == "cpxR_or_baeR_in_slot2":
        return "CpxR/BaeR rightmost placement"
    if text == "lexA_in_slot0":
        return "LexA leftmost placement"
    if text == "baeR_in_slot1":
        return "BaeR middle placement"
    return tfbs_label_compact_title(label_name)


def _is_slot_label(label_name: object) -> bool:
    return "_in_slot" in str(label_name or "")
