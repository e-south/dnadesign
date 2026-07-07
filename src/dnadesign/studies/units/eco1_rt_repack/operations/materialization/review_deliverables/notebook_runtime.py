"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_runtime.py

Runtime helpers for the Eco1 review-deliverables marimo notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_ESMC_FEATURE_REVIEW,
    SECTION_FEASIBILITY_AND_HANDOFF,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
)

from .notebook_selection_panel import render_selection_panel_table
from .notebook_selection_summary import (
    render_handoff_readiness,
    render_selection_funnel_summary,
)
from .notebook_sequences import handoff_sequence_list_html
from .notebook_visuals import render_image

_NOTEBOOK_HIDDEN_DELIVERABLE_IDS = {
    "foldcheck_review_structure_overlay_panel",
    "foldcheck_review_structure_overlay_skipped",
    "mask_structure_context_png",
}
_NOTEBOOK_LANE_ROLES = {
    "main_review": {"manuscript_facing", "interactive_review"},
    "audit_supplement": {"review_only", "operator_review", "optional_heavy"},
}
_NOTEBOOK_LANE_LABELS = {
    "main_review": "Core evidence",
    "audit_supplement": "Model and method checks",
}
_NOTEBOOK_EVIDENCE_ARTIFACT_KINDS = {
    "selection_funnel_summary",
    "selection_panel_table",
    "candidate_handoff_sequence_csv",
    "handoff_readiness",
}
_SECTION_DELIVERABLE_ORDER = {
    SECTION_FEASIBILITY_AND_HANDOFF: (
        *CURRENT_SELECTION_PLOT_IDS,
        "selected_panel_structure_browser_manifest",
        "selection_funnel_summary",
        "selection_panel_table",
        "selection_handoff_sequences",
        "selection_handoff_readiness",
    )
}


def load_review_manifest(notebook_file: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    """Load the manifest adjacent to the generated notebook."""

    manifest_path = Path(notebook_file).resolve().parents[1] / "review_deliverable_manifest.yaml"
    manifest_root = manifest_path.parent
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return manifest, list(manifest["deliverables"]), manifest_path, manifest_root


def render_intro(mo: Any) -> Any:
    """Render the study premise without code-self-referential copy."""

    intro_lead = (
        "Eco1/Ec86 is a retron reverse transcriptase with a cryoEM-supported RNA/DNA-bound scaffold. "
        "The study asks whether fold-preserved RT sequence variants can support downstream assays for "
        "processivity and strand displacement. The computational deliverable is a bounded set of protein "
        "sequence candidates for review, not an activity claim."
    )
    intro_flow = (
        "Evidence order: cryoEM scaffold, Tao-style mask evidence, ProteinMPNN sequence proposals, "
        "ColabFold fold checks, localized triage, panel selection, and sequence export. "
        "The active design classes vary homolog-conservation "
        "and retained-DNA/RNA proximity thresholds while always protecting catalytic anchors and Wang/Ec86 "
        "direct-contact priors. The active mask uses catalytic anchors, Wang/Ec86 direct-contact priors, "
        "retained-substrate proximity, and Mestre-derived clade 9 plurality at the 25% threshold. "
        "ProteinMPNN proposes variants at unprotected residues to repack the remaining designable residues, "
        "ColabFold removes poor "
        "fold predictions, and the remaining pool is stratified by MSA support, local mutation geography, "
        "chemistry near retained DNA/RNA or thumb-track, and selected model checks. WT ESMC masked-marginal scores "
        "and Biohub ESMC "
        "SAE features annotate the review set; they are not mask inputs or acceptance gates. The selected protein "
        "sequences are exported as a flat CSV for RT-only handoff planning."
    )
    paragraph_style = (
        "margin:0; width:100%; max-width:none; color:inherit; opacity:0.86; "
        "font-size:1.02rem; line-height:1.5; white-space:normal;"
    )
    return mo.Html(
        f"""
        <section style="width:100%; border-bottom:1px solid #d8dee4;
                        padding:0 0 0.8rem 0; margin-bottom:0.5rem;">
          <h1 style="margin:0 0 0.42rem 0; font-size:2.15rem; line-height:1.12;
                     font-family:ui-serif, Georgia, 'Times New Roman', serif;
                     font-weight:650; letter-spacing:0;">
            Repacking Eco1 reverse transcriptase for structured-template assays
          </h1>
          <p style="{paragraph_style}">{intro_lead}</p>
          <p style="{paragraph_style}; margin-top:0.5rem;">{intro_flow}</p>
        </section>
        """
    )


def resolve_manifest_path(manifest_root: Path, value: str) -> Path:
    """Resolve a manifest-relative artifact path."""

    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else manifest_root / candidate


def review_lane_lookup(deliverables: list[dict[str, Any]]) -> dict[str, str]:
    """Return available notebook lanes, preserving the intended default order."""

    observed_roles = {str(row.get("role") or "manuscript_facing") for row in deliverables}
    lanes: dict[str, str] = {}
    for lane_id, roles in _NOTEBOOK_LANE_ROLES.items():
        if roles & observed_roles:
            lanes[_NOTEBOOK_LANE_LABELS[lane_id]] = lane_id
    return lanes


def visual_deliverables(
    deliverables: list[dict[str, Any]],
    *,
    selected_lane: str = "main_review",
) -> list[dict[str, Any]]:
    """Return rendered visual and interactive-review rows for notebook selection."""

    allowed_roles = _NOTEBOOK_LANE_ROLES.get(selected_lane)
    if allowed_roles is None:
        raise ValueError(f"unknown review deliverable lane: {selected_lane}")
    return [
        row
        for row in deliverables
        if str(row.get("role") or "manuscript_facing") in allowed_roles and _is_publication_visual(row)
    ]


def evidence_deliverables(
    deliverables: list[dict[str, Any]],
    *,
    selected_lane: str = "main_review",
) -> list[dict[str, Any]]:
    """Return evidence/export rows for notebook sections outside figure selectors."""

    allowed_roles = _NOTEBOOK_LANE_ROLES.get(selected_lane)
    if allowed_roles is None:
        raise ValueError(f"unknown review deliverable lane: {selected_lane}")
    return [
        row
        for row in deliverables
        if str(row.get("role") or "manuscript_facing") in allowed_roles
        and str(row.get("artifact_kind") or "") in _NOTEBOOK_EVIDENCE_ARTIFACT_KINDS
    ]


def section_label_lookup(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map display section labels to section ids, preserving manifest order."""

    sections: list[str] = []
    seen_sections: set[str] = set()
    for row in rows:
        section = str(row["section"])
        if section not in seen_sections:
            seen_sections.add(section)
            sections.append(section)
    return {format_section_label(section): section for section in sections}


def section_deliverables(rows: list[dict[str, Any]], selected_section: str) -> list[dict[str, Any]]:
    """Filter visual deliverables to the selected section."""

    section_rows = [row for row in rows if str(row.get("section") or "") == selected_section]
    order = _SECTION_DELIVERABLE_ORDER.get(selected_section)
    if order is None:
        return section_rows
    order_lookup = {deliverable_id: index for index, deliverable_id in enumerate(order)}
    return sorted(
        section_rows,
        key=lambda row: (order_lookup.get(str(row.get("deliverable_id") or ""), len(order_lookup)),),
    )


def deliverable_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map display deliverable labels to manifest rows."""

    return {format_deliverable_label(row): row for row in rows}


def selected_deliverable(
    *,
    selected_label: str,
    lookup: dict[str, dict[str, Any]],
    options: list[str],
) -> dict[str, Any] | None:
    """Resolve a selected deliverable row, defaulting to the first visible option."""

    if selected_label not in lookup and options:
        selected_label = str(options[0])
    return lookup.get(selected_label) if selected_label else None


def render_deliverable_artifact(row: dict[str, Any], *, mo: Any, manifest_root: Path) -> Any:
    """Render one visual artifact at notebook-column width."""

    media_path = resolve_manifest_path(manifest_root, str(row["path"]))
    artifact_kind = str(row.get("artifact_kind") or "")
    suffix = media_path.suffix.lower()
    if artifact_kind == "selection_funnel_summary":
        return render_selection_funnel_summary(row, mo=mo, manifest_path=media_path)
    if artifact_kind == "selection_panel_table":
        return render_selection_panel_table(row, mo=mo, table_path=media_path)
    if artifact_kind == "candidate_handoff_sequence_csv":
        return _render_handoff_sequence_csv(row, mo=mo, table_path=media_path)
    if artifact_kind == "proteinmpnn_residue_frequency_bundle":
        return render_residue_frequency_bundle(row, mo=mo, manifest_root=manifest_root)
    if artifact_kind == "handoff_readiness":
        return render_handoff_readiness(row, mo=mo, manifest_path=media_path)
    if media_path.exists() and suffix in {".svg", ".png"}:
        return render_image(row, mo=mo, media_path=media_path)
    artifact_path = str(row.get("path") or "")
    if media_path.exists():
        return mo.md(f"Artifact file: `{artifact_path}`")
    skip_reason = str(row.get("skip_reason") or "artifact path does not exist")
    return mo.md(
        f"Artifact unavailable: `{artifact_path}`\n\nArtifact state: `{row.get('status')}`. Reason: {skip_reason}"
    )


def render_deliverable_panel(row: dict[str, Any], *, mo: Any, manifest_root: Path) -> Any:
    """Render one visual plus collapsible method/evidence details."""

    body = render_deliverable_artifact(row, mo=mo, manifest_root=manifest_root)
    return mo.vstack([body, render_deliverable_details(row, mo=mo)], gap=0.35)


def render_deliverable_details(row: dict[str, Any], *, mo: Any) -> Any:
    """Render collapsible method/evidence details for one manifest row."""

    evidence_rows = [
        {"field": "title", "value": str(row.get("title") or "")},
        {"field": "path", "value": str(row.get("path") or "")},
        {"field": "selection role", "value": str(row.get("selection_role") or "")},
        {"field": "funnel stage", "value": str(row.get("funnel_stage_id") or "")},
        {"field": "not a selector", "value": str(row.get("not_a_selector_reason") or "")},
        {"field": "input sources", "value": ", ".join(row.get("source_tables", []))},
        {"field": "alt_text", "value": str(row.get("alt_text") or "")},
        {"field": "skip_reason", "value": str(row.get("skip_reason") or "")},
    ]
    detail_panels = {
        "Premise": mo.md(str(row.get("description") or "")),
        "Interpretation limit": mo.md(str(row.get("interpretation_limit") or "")),
        "Sources": mo.ui.table(evidence_rows, page_size=8),
    }
    method_summary = str(row.get("method_summary") or "")
    evidence_summary = row.get("evidence_summary") or {}
    if method_summary or evidence_summary:
        method_text = method_summary or "LLR = log P(alternate) - log P(WT)."
        method_rows = [{"field": str(key), "value": str(value)} for key, value in dict(evidence_summary).items()]
        detail_panels["Method and row counts"] = mo.vstack(
            [mo.md(method_text), mo.ui.table(method_rows, page_size=8)],
            gap=0.25,
        )
    return mo.accordion(detail_panels, multiple=True, lazy=True)


def is_interactive_structure_deliverable(row: dict[str, Any] | None) -> bool:
    """Return whether a manifest row should render as an interactive structure view."""

    if row is None:
        return False
    artifact_kind = str(row.get("artifact_kind") or "")
    status = str(row.get("status") or "")
    return artifact_kind == "structure_browser_manifest" and status == "rendered"


def is_residue_frequency_bundle_deliverable(row: dict[str, Any] | None) -> bool:
    """Return whether a manifest row owns a ProteinMPNN residue-frequency bundle."""

    if row is None:
        return False
    return str(row.get("artifact_kind") or "") == "proteinmpnn_residue_frequency_bundle"


def residue_frequency_view_lookup(row: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map residue-frequency view labels to design-class view rows."""

    if not is_residue_frequency_bundle_deliverable(row):
        return {}
    evidence_summary = row.get("evidence_summary") or {}
    view_rows = list(dict(evidence_summary).get("design_class_views") or [])
    return {str(view.get("label") or ""): dict(view) for view in view_rows if view.get("label")}


def select_residue_frequency_view(
    *,
    selected_label: str,
    lookup: dict[str, dict[str, Any]],
    options: list[str],
) -> dict[str, Any] | None:
    """Resolve the selected ProteinMPNN design-class view."""

    if selected_label not in lookup and options:
        selected_label = str(options[0])
    return lookup.get(selected_label) if selected_label else None


def format_section_label(section: str) -> str:
    labels = {
        SECTION_CONSTRAINT_EVIDENCE: "Mask basis",
        SECTION_DESIGNS_AND_FOLD_TRIAGE: "Sequence proposals and fold checks",
        SECTION_ESMC_FEATURE_REVIEW: "ESMC and SAE checks",
        SECTION_FEASIBILITY_AND_HANDOFF: "Panel selection",
    }
    return labels.get(str(section), str(section).replace("_", " ").title())


def format_deliverable_label(row: dict[str, Any] | str) -> str:
    deliverable_id = str(row.get("deliverable_id") if isinstance(row, dict) else row)
    row_title = str(row.get("title") or "") if isinstance(row, dict) else ""
    if isinstance(row, dict) and row.get("notebook_group_label"):
        return f"{row['notebook_group_label']} - {row_title or deliverable_id.replace('_', ' ').title()}"
    if row_title:
        return row_title
    return deliverable_id.replace("_", " ").title()


def _is_publication_visual(row: dict[str, Any]) -> bool:
    if str(row.get("deliverable_id") or "") in _NOTEBOOK_HIDDEN_DELIVERABLE_IDS:
        return False
    if str(row.get("artifact_kind") or "") in {
        "selection_funnel_summary",
        "selection_panel_table",
        "candidate_handoff_sequence_csv",
        "handoff_readiness",
    }:
        return False
    if str(row.get("artifact_kind") or "") == "sae_feature_heatmap_manifest":
        return str(row.get("status") or "") == "rendered"
    if is_interactive_structure_deliverable(row):
        return True
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    return suffix in {".svg", ".png"} and str(row.get("status") or "") in {
        "rendered",
        "linked_existing",
        "reused_existing_optional_render",
    }


def _render_handoff_sequence_csv(row: dict[str, Any], *, mo: Any, table_path: Path) -> Any:
    if not table_path.exists():
        return mo.md(f"Selected sequence CSV unavailable: `{row.get('path')}`")
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return mo.vstack(
        [
            mo.Html("<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>Selected RT protein sequences</h3>"),
            mo.Html(handoff_sequence_list_html(rows)),
            mo.ui.table(rows, page_size=10),
        ],
        gap=0.25,
    )


def render_residue_frequency_bundle(
    row: dict[str, Any],
    *,
    mo: Any,
    manifest_root: Path,
    selected_view: dict[str, Any] | None = None,
    design_class_ui: Any | None = None,
) -> Any:
    """Render a residue-frequency design-class view with notebook-owned controls."""

    view_lookup = residue_frequency_view_lookup(row)
    if not view_lookup:
        media_path = resolve_manifest_path(manifest_root, str(row["path"]))
        return render_image(row, mo=mo, media_path=media_path)
    options = list(view_lookup)
    selected_view = selected_view or select_residue_frequency_view(
        selected_label=options[0] if options else "",
        lookup=view_lookup,
        options=options,
    )
    selected_view = selected_view or {}
    selected_label = str(selected_view.get("label") or "")
    selected_path = resolve_manifest_path(manifest_root, str(selected_view.get("path") or row["path"]))
    selected_row = dict(row)
    selected_row["path"] = str(selected_path)
    selected_row["title"] = f"{row.get('title')} | {selected_label}" if selected_label else str(row.get("title") or "")
    selected_row["alt_text"] = (
        f"{row.get('alt_text')} Selected design class: {selected_label}."
        if selected_label
        else str(row.get("alt_text") or "")
    )
    rendered_items = [render_image(selected_row, mo=mo, media_path=selected_path)]
    if design_class_ui is not None:
        rendered_items.insert(0, design_class_ui)
    return mo.vstack(
        rendered_items,
        gap=0.25,
    )
