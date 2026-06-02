"""Registry-backed visual vocabulary for TFBS Stage B review surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

TidySource = Literal["trajectory", "pair_summary", "count_distribution"]
_ALLOWED_TEMPLATE_FIELDS = {"label_name", "label_slug"}


@dataclass(frozen=True)
class StageBNotebookVisualSpec:
    """Contract for one study-owned visual entry in an OPAL collection notebook."""

    kind: str
    visual_id_template: str
    label: str
    group_key: str
    metric_name: str
    metric_label: str
    metric_expression: str
    summary_name: str
    tidy_source: TidySource
    caption: str
    plot_filename_template: str
    plot_title_template: str
    alt_text: str

    def visual_id(self, *, label_name: str | None = None) -> str:
        return _format_label_template(
            self.visual_id_template,
            label_name=label_name,
            kind=self.kind,
            field="visual_id_template",
        )

    def plot_filename(self, *, label_name: str | None = None) -> str:
        return _format_label_template(
            self.plot_filename_template,
            label_name=label_name,
            kind=self.kind,
            field="plot_filename_template",
        )

    def plot_title(self, *, label_name: str | None = None) -> str:
        return _format_label_template(
            self.plot_title_template,
            label_name=label_name,
            kind=self.kind,
            field="plot_title_template",
        )

    def tidy_csv_path(
        self,
        *,
        trajectory_csv_path: Path,
        pair_summary_csv_path: Path,
        count_distribution_csv_path: Path | None = None,
    ) -> Path:
        if self.tidy_source == "trajectory":
            return trajectory_csv_path
        if self.tidy_source == "pair_summary":
            return pair_summary_csv_path
        if self.tidy_source == "count_distribution":
            if count_distribution_csv_path is None:
                raise ValueError(f"Visual kind {self.kind!r} requires a count distribution CSV")
            return count_distribution_csv_path
        raise ValueError(f"Unsupported Stage B visual tidy source: {self.tidy_source!r}")


def realized_visual_spec(kind: str) -> StageBNotebookVisualSpec:
    """Return the registered realized-label review visual spec for ``kind``."""

    return _lookup_visual_spec(
        REALIZED_REVIEW_VISUAL_SPECS,
        kind,
        surface="Stage B realized review",
    )


def slot_visual_spec(kind: str) -> StageBNotebookVisualSpec:
    """Return the registered slot-count diagnostic visual spec for ``kind``."""

    return _lookup_visual_spec(
        SLOT_DIAGNOSTIC_VISUAL_SPECS,
        kind,
        surface="Stage B slot diagnostic",
    )


def build_visual_spec_registry(
    specs: Iterable[StageBNotebookVisualSpec],
    *,
    surface: str,
) -> Mapping[str, StageBNotebookVisualSpec]:
    """Build a fail-fast visual registry keyed by plot ``kind``."""

    registry: dict[str, StageBNotebookVisualSpec] = {}
    filenames: set[str] = set()
    for spec in specs:
        kind = spec.kind.strip()
        if not kind:
            raise ValueError(f"{surface} visual spec has an empty kind")
        if kind in registry:
            raise ValueError(f"Duplicate {surface} visual spec kind: {kind}")
        _validate_visual_spec(spec, surface=surface)
        filename = spec.plot_filename_template.strip()
        if filename in filenames:
            raise ValueError(f"Duplicate {surface} visual spec plot filename template: {filename}")
        registry[kind] = spec
        filenames.add(filename)
    if not registry:
        raise ValueError(f"{surface} visual registry must contain at least one spec")
    return MappingProxyType(registry)


def _lookup_visual_spec(
    registry: Mapping[str, StageBNotebookVisualSpec],
    kind: str,
    *,
    surface: str,
) -> StageBNotebookVisualSpec:
    token = str(kind).strip()
    try:
        return registry[token]
    except KeyError as exc:
        raise ValueError(f"Unsupported {surface} plot kind: {token!r}") from exc


def slug_token(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "label"


def _format_label_template(
    template: str,
    *,
    label_name: str | None,
    kind: str,
    field: str,
) -> str:
    _validate_template_fields(template, field=field, surface=f"visual kind {kind!r}")
    fields = _template_fields(template)
    if fields and (label_name is None or not str(label_name).strip()):
        raise ValueError(f"Visual kind {kind!r} requires a nonempty label_name for {field}")
    label_value = str(label_name or "")
    return template.format(label_name=label_value, label_slug=slug_token(label_value))


def _validate_visual_spec(spec: StageBNotebookVisualSpec, *, surface: str) -> None:
    for field in (
        "visual_id_template",
        "label",
        "group_key",
        "metric_name",
        "metric_label",
        "metric_expression",
        "summary_name",
        "caption",
        "plot_filename_template",
        "plot_title_template",
        "alt_text",
    ):
        if not str(getattr(spec, field)).strip():
            raise ValueError(f"{surface} visual spec {spec.kind!r} has an empty {field}")
    _validate_template_fields(spec.visual_id_template, field="visual_id_template", surface=surface)
    _validate_template_fields(spec.plot_filename_template, field="plot_filename_template", surface=surface)
    _validate_template_fields(spec.plot_title_template, field="plot_title_template", surface=surface)
    if not spec.plot_filename_template.endswith(".png"):
        raise ValueError(f"{surface} visual spec {spec.kind!r} must write a .png filename")


def _validate_template_fields(template: str, *, field: str, surface: str) -> None:
    unknown = sorted(_template_fields(template) - _ALLOWED_TEMPLATE_FIELDS)
    if unknown:
        raise ValueError(f"{surface} {field} contains unsupported template field(s): {unknown}")


def _template_fields(template: str) -> set[str]:
    return {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}


REALIZED_REVIEW_VISUAL_SPECS = build_visual_spec_registry(
    (
        StageBNotebookVisualSpec(
            kind="realized_label_lift_trajectory",
            visual_id_template="tfbs_stage_b_{label_slug}_realized_label_lift_trajectory",
            label="Realized selected true-label lift trajectory",
            group_key="label_oracle_kind",
            metric_name="selected_true_lift_ratio",
            metric_label="Selected true-label lift ratio",
            metric_expression="selected_true_mean / pool_baseline",
            summary_name="per_round",
            tidy_source="trajectory",
            caption=(
                "Realized selected-label lift by round, computed by joining selected row IDs to the positive or "
                "null/control oracle label table. The square marker is the initial labeled seed batch; round 0 is "
                "the first model-selected acquisition batch after those labels are ingested. This is the "
                "learnability evidence surface; predicted score remains an acquisition trace."
            ),
            plot_filename_template="{label_slug}__selected_true_lift_trajectory.png",
            plot_title_template="{label_name}: selected true-label lift over rounds",
            alt_text=(
                "Line plot of selected true-label lift over active-learning rounds for each sentinel label, with the "
                "initial seed batch shown as a square marker before round zero and positive and matched-null roles "
                "drawn separately."
            ),
        ),
        StageBNotebookVisualSpec(
            kind="positive_null_lift_summary",
            visual_id_template="tfbs_stage_b_{label_slug}_positive_null_lift_summary",
            label="Realized positive-minus-null lift summary",
            group_key="peer_review_claim_status",
            metric_name="positive_minus_null_lift_ratio",
            metric_label="Positive-minus-null lift ratio",
            metric_expression="positive_lift_ratio - null_or_control_lift_ratio",
            summary_name="final_and_normalized_auc",
            tidy_source="pair_summary",
            caption=(
                "Final and normalized trajectory positive-minus-null/control lift for each sentinel TFBS label. "
                "Rows marked as confound controls should not be interpreted as clean negative-control separation."
            ),
            plot_filename_template="{label_slug}__positive_minus_null_lift_summary.png",
            plot_title_template="{label_name}: positive-minus-null realized lift",
            alt_text=(
                "Bar plot comparing final and normalized trajectory AUC positive-minus-null lift for each sentinel "
                "label, using realized oracle labels from selected rows."
            ),
        ),
    ),
    surface="Stage B realized review",
)

SLOT_DIAGNOSTIC_VISUAL_SPECS = build_visual_spec_registry(
    (
        StageBNotebookVisualSpec(
            kind="slot_target_count_mean_trajectory",
            visual_id_template="tfbs_stage_b_slot_target_count_mean_trajectory",
            label="Slot selected target-family count trajectory",
            group_key="label_oracle_kind",
            metric_name="selected_target_count_mean",
            metric_label="Selected target-family count mean",
            metric_expression="mean(selected target-family count)",
            summary_name="per_round",
            tidy_source="trajectory",
            caption=(
                "Selected target-family count by round for slot-label campaigns. A null/control can look strong "
                "when OPAL selects rows with high target-family count rather than learning slot position."
            ),
            plot_filename_template="slot_target_count_mean_trajectory.png",
            plot_title_template="Selected target-family count over rounds",
            alt_text=(
                "Line plot of selected target-family count over active-learning rounds for slot-label campaigns, "
                "with positive and matched-null or control roles shown separately."
            ),
        ),
        StageBNotebookVisualSpec(
            kind="slot_count_stratified_lift_trajectory",
            visual_id_template="tfbs_stage_b_slot_count_stratified_lift_trajectory",
            label="Slot count-stratified lift trajectory",
            group_key="label_oracle_kind",
            metric_name="count_stratified_lift_ratio",
            metric_label="Count-stratified slot-label lift ratio",
            metric_expression="selected_nondeterministic_true_mean / selected_count_stratum_baseline",
            summary_name="per_round",
            tidy_source="trajectory",
            caption=(
                "Slot-label lift after excluding deterministic count strata and comparing selected rows with the "
                "baseline for their own target-family count strata."
            ),
            plot_filename_template="slot_count_stratified_lift_trajectory.png",
            plot_title_template="Count-stratified slot-label lift over rounds",
            alt_text=(
                "Line plot of slot-label lift over rounds after deterministic count strata are excluded and selected "
                "rows are compared to count-stratum baselines. X markers at y equals zero indicate rounds with no "
                "nondeterministic selected rows rather than missing plotted data."
            ),
        ),
        StageBNotebookVisualSpec(
            kind="slot_count_stratified_lift_summary",
            visual_id_template="tfbs_stage_b_slot_count_stratified_lift_summary",
            label="Slot count-stratified positive-minus-null summary",
            group_key="slot_diagnostic_status",
            metric_name="positive_minus_null_count_stratified_lift_ratio",
            metric_label="Positive-minus-null count-stratified lift ratio",
            metric_expression="positive_count_stratified_lift_ratio - null_or_control_count_stratified_lift_ratio",
            summary_name="final_and_normalized_auc",
            tidy_source="pair_summary",
            caption=(
                "Final and normalized trajectory positive-minus-null/control lift for slot labels after "
                "controlling for selected target-family count composition."
            ),
            plot_filename_template="slot_count_stratified_lift_summary.png",
            plot_title_template="Count-stratified positive-minus-null slot lift",
            alt_text=(
                "Bar plot comparing final and normalized trajectory AUC positive-minus-null slot lift after "
                "controlling for target-family count."
            ),
        ),
    ),
    surface="Stage B slot diagnostic",
)
