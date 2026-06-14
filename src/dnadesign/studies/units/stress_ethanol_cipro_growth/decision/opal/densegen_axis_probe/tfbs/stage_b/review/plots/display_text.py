"""Manuscript-facing text contract for Stage B realized-label review plots."""

from __future__ import annotations

from ....label_text import (
    tfbs_control_display_label,
    tfbs_control_pair_label,
    tfbs_label_compact_title,
    tfbs_label_expression,
    tfbs_label_title,
)

NO_ENRICHMENT_BASELINE_LABEL = "Baseline"
NO_ENRICHMENT_BASELINE_CONTRACT = "No enrichment: selected mean = full candidate-pool mean"
TRAJECTORY_X_AXIS_LABEL = "Round"
INITIAL_BATCH_TICK_LABEL = "Shared\nstart"
TRAILING_TRAJECTORY_NOTE = (
    "Faint lines = individual seeds; bold line = mean selected batch; squares = shared start; dashed line = no "
    "enrichment."
)
REALIZED_REVIEW_TEXT_CONTRACT = {
    "baseline": NO_ENRICHMENT_BASELINE_CONTRACT,
    "count_fraction_label": (
        "count_fraction label = target TFBS count / 3 per sequence; plotted values are enrichment ratios, "
        "not raw counts"
    ),
    "initial_batch": "square markers are the same initial IDs scored by each label source before round 0",
    "interval": "mean plus/minus sample SD across seed runs; n is recorded; not an inferential CI",
    "legend_layout": "single row below the plot",
    "pairing": ("DenseGen-label and control campaigns share initial selected IDs; only the label table differs"),
    "role_labels": "DenseGen label versus profile-appropriate matched control",
    "selected_label_values": (
        "selected_true_* artifact columns are selected values from that campaign's label table; for shuffled "
        "controls this is a control-label value, not post hoc DenseGen truth"
    ),
    "trajectory_semantics": (
        "line points are per-round top-k acquired selected batches; round 0 is the first acquired batch after the "
        "shared initial IDs, not the initial seed batch"
    ),
    "subtitle_layout": "centered single-line subtitle",
    "title_alignment": "centered title; title may wrap, subtitle must not wrap",
    "type_scale": "axis labels, tick labels, subtitle, and legend use the same review body size",
}


def role_display_label(role: object, *, label_name: object, control_role: object | None = None) -> str:
    """Return a reviewer-facing label source name for the campaign role."""

    role_text = str(role)
    if role_text == "positive":
        return "DenseGen label"
    if role_text == "matched_null":
        default_role = "" if _is_slot_label(label_name) else "matched_label_permutation_negative_control"
        return tfbs_control_display_label(control_role or default_role, label_name=label_name).capitalize()
    return role_text.replace("_", " ")


def trajectory_plot_title(label_name: object, *, replicate_count: int) -> str:
    """Return the visible title for the round trajectory plot."""

    return f"{_trajectory_title_label(label_name)} enrichment from promoter embeddings"


def trajectory_plot_subtitle(
    label_name: object,
    *,
    replicate_count: int,
    control_role: object | None = None,
) -> str:
    """Return the visible subtitle carrying pairing and interval semantics."""

    seed_phrase = _paired_seed_run_phrase(replicate_count)
    if _is_slot_label(label_name):
        control = tfbs_control_display_label(control_role, label_name=label_name)
        return f"DenseGen slot label vs {control} across {seed_phrase}"
    control_pair = tfbs_control_pair_label(
        control_role or "matched_label_permutation_negative_control",
        label_name=label_name,
    )
    return f"{control_pair} across {seed_phrase}"


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
        return "y = selected mean fraction / pool mean fraction"
    return "y = selected mean label value / pool mean label value"


def trajectory_encoding_note(*, replicate_count: int) -> str:
    """Return the bottom encoding note for the trajectory plot."""

    if replicate_count > 1:
        return TRAILING_TRAJECTORY_NOTE
    return "Line = selected batch trajectory; square = shared initial IDs before round 0; dashed line = no enrichment."


def seed_run_sample_sd_label(*, replicate_count: int) -> str:
    """Return visible wording for a seed-run sample-SD interval."""

    return f"Mean +/- SD (n={int(replicate_count)})"


def seed_pair_sample_sd_label(*, replicate_count: int) -> str:
    """Return visible wording for a paired-seed sample-SD interval."""

    return f"Mean +/- SD (n={int(replicate_count)})"


def positive_null_summary_title(label_name: object, *, replicate_count: int) -> str:
    """Return the visible title for the DenseGen-minus-control endpoint summary."""

    label = tfbs_label_compact_title(label_name)
    return f"{label} enrichment over control"


def positive_null_summary_subtitle(*, replicate_count: int) -> str:
    """Return the visible subtitle for the endpoint summary."""

    if replicate_count > 1:
        interval = f"Bars = mean; whiskers = SD across {int(replicate_count)} paired seed runs"
    else:
        interval = "Single paired seed run; no seed-pair spread"
    return interval


def plot_manifest_title(kind: str, *, label_name: object, replicate_count: int) -> str:
    """Return the manifest title that mirrors the visible plot title."""

    if kind == "realized_label_lift_trajectory":
        return trajectory_plot_title(label_name, replicate_count=replicate_count)
    if kind == "positive_null_lift_summary":
        return positive_null_summary_title(label_name, replicate_count=replicate_count)
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
            f"{label_title} selected-batch enrichment versus the candidate pool. "
            f"Round 0 is the first acquired batch after the shared start. "
            f"Lines show DenseGen label and {control}; {interval}, not a confidence interval."
        )
    if kind == "positive_null_lift_summary":
        interval = (
            f"mean bars with sample-SD whiskers across {int(replicate_count)} seed pairs"
            if replicate_count > 1
            else "single seed pair"
        )
        return f"{label_title} DenseGen-minus-control enrichment for final round and trajectory AUC; {interval}."
    return f"Review plot for {label_title}."


def _paired_seed_run_phrase(replicate_count: int) -> str:
    count = int(replicate_count)
    noun = "run" if count == 1 else "runs"
    return f"{count} paired seed {noun}"


def _trajectory_title_label(label_name: object) -> str:
    label = tfbs_label_compact_title(label_name)
    if tfbs_label_expression(label_name):
        return label.replace(" count-fraction", " motif-count")
    return label


def _is_slot_label(label_name: object) -> bool:
    return "_in_slot" in str(label_name or "")
