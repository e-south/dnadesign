"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_runtime.py

Runtime helpers for the Eco1 review-deliverables marimo notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import yaml

_NOTEBOOK_HIDDEN_DELIVERABLE_IDS = {
    "foldcheck_review_structure_overlay_panel",
    "foldcheck_review_structure_overlay_skipped",
    "mask_structure_context_png",
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
        "Eco1/Ec86 is a retron reverse transcriptase with a cryoEM-supported fold that "
        "wraps the retron RNA/DNA substrate. This study tests whether Tao-style fixed-backbone "
        "redesign can repack residues outside catalytic and substrate-contact constraints "
        "while preserving the RT scaffold."
    )
    intro_flow = (
        "The evidence stack is sequential: Mestre-derived clade 9 alignments and ESMC "
        "masked-marginal scores flag constrained WT residues, the cryoEM structure defines "
        "substrate-proximal positions, ProteinMPNN proposes sequences only in the unprotected "
        "canvas, and ColabFold checks whether those full-length variants retain the fold. "
        "Activity, strand displacement, and structured-template readthrough remain assay questions."
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
            Repacking Eco1 reverse transcriptase while preserving the RT scaffold
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


def visual_deliverables(deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rendered visual and interactive-review rows for notebook selection."""

    return [row for row in deliverables if _is_publication_visual(row)]


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

    return [row for row in rows if str(row.get("section") or "") == selected_section]


def deliverable_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map display deliverable labels to manifest rows."""

    return {format_deliverable_label(row): row for row in rows}


def selected_deliverable(
    *,
    selected_label: str,
    lookup: dict[str, dict[str, Any]],
    options: list[str],
) -> dict[str, Any] | None:
    """Resolve a selected deliverable row with a stable fallback."""

    if selected_label not in lookup and options:
        selected_label = str(options[0])
    return lookup.get(selected_label) if selected_label else None


def render_deliverable_artifact(row: dict[str, Any], *, mo: Any, manifest_root: Path) -> Any:
    """Render one visual artifact at notebook-column width."""

    media_path = resolve_manifest_path(manifest_root, str(row["path"]))
    suffix = media_path.suffix.lower()
    if media_path.exists() and suffix in {".svg", ".png"}:
        return _render_image(row, mo=mo, media_path=media_path)
    if media_path.exists():
        return mo.md(f"Generated non-image artifact: `{media_path}`")
    skip_reason = str(row.get("skip_reason") or "artifact path does not exist")
    return mo.md(
        f"Artifact unavailable: `{media_path}`\n\nArtifact state: `{row.get('status')}`. Reason: {skip_reason}"
    )


def render_deliverable_panel(row: dict[str, Any], *, mo: Any, manifest_root: Path) -> Any:
    """Render one visual plus collapsible method/evidence details."""

    body = render_deliverable_artifact(row, mo=mo, manifest_root=manifest_root)
    return mo.vstack([body, render_deliverable_details(row, mo=mo)], gap=0.35)


def render_interpretation_note(row: dict[str, Any], *, mo: Any) -> Any:
    """Render the claim boundary where reviewers can see it."""

    limit = str(row.get("interpretation_limit") or "").strip()
    if not limit:
        return mo.md("")
    safe_limit = html.escape(limit)
    return mo.Html(
        f"""
        <div style="border-left:3px solid #8c959f; padding:0.38rem 0.55rem;
                    margin:0.35rem 0 0 0; color:#57606a; background:#f6f8fa;
                    font-size:0.92rem; line-height:1.35;">
          <strong>Interpretation limit:</strong> {safe_limit}
        </div>
        """
    )


def render_deliverable_details(row: dict[str, Any], *, mo: Any) -> Any:
    """Render collapsible method/evidence details for one manifest row."""

    evidence_rows = [
        {"field": "title", "value": str(row.get("title") or "")},
        {"field": "status", "value": str(row.get("status") or "")},
        {"field": "path", "value": str(row.get("path") or "")},
        {"field": "role", "value": str(row.get("role") or "")},
        {"field": "sources", "value": ", ".join(row.get("source_tables", []))},
        {"field": "alt_text", "value": str(row.get("alt_text") or "")},
        {"field": "skip_reason", "value": str(row.get("skip_reason") or "")},
    ]
    detail_panels = {
        "What this visual shows": mo.md(str(row.get("description") or "")),
        "Interpretation limit": mo.md(str(row.get("interpretation_limit") or "")),
        "Evidence": mo.ui.table(evidence_rows, page_size=8),
    }
    method_summary = str(row.get("method_summary") or "")
    evidence_summary = row.get("evidence_summary") or {}
    if method_summary or evidence_summary:
        method_text = method_summary or "For this section, LLR = log P(alternate) - log P(WT)."
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


def format_section_label(section: str) -> str:
    labels = {
        "scaffold_and_mask": "Reference sequence, alignment, and mask",
        "proteinmpnn": "ProteinMPNN sequence proposals",
        "fold_review": "ColabFold structure triage",
        "wt_model_constraint_audit": "WT ESMC substitution constraint",
        "biohub_esmc_sae_interpretation": "Biohub ESMC SAE interpretation",
    }
    return labels.get(str(section), str(section).replace("_", " ").title())


def format_deliverable_label(row: dict[str, Any] | str) -> str:
    deliverable_id = str(row.get("deliverable_id") if isinstance(row, dict) else row)
    row_title = str(row.get("title") or "") if isinstance(row, dict) else ""
    if row_title:
        return row_title
    return deliverable_id.replace("_", " ").title()


def _is_publication_visual(row: dict[str, Any]) -> bool:
    if str(row.get("deliverable_id") or "") in _NOTEBOOK_HIDDEN_DELIVERABLE_IDS:
        return False
    if is_interactive_structure_deliverable(row):
        return True
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    return suffix in {".svg", ".png"} and str(row.get("status") or "") in {"rendered", "linked_existing"}


def _render_image(row: dict[str, Any], *, mo: Any, media_path: Path) -> Any:
    mime_type = "image/svg+xml" if media_path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
    alt_text = html.escape(str(row["alt_text"]), quote=True)
    caption = html.escape(format_deliverable_label(row), quote=True)
    limit = html.escape(str(row.get("interpretation_limit") or ""), quote=False)
    return mo.Html(
        f"""
        <figure style="margin:0;">
          <div style="overflow:hidden; width:100%; border:1px solid #d8dee4;
                      border-radius:6px; background:#ffffff; padding:0.5rem;">
            <img src="data:{mime_type};base64,{encoded}" alt="{alt_text}"
                 style="display:block; width:100%; max-width:100%; max-height:min(72vh, 780px);
                        height:auto; object-fit:contain;" />
          </div>
          <figcaption style="font-size:0.92rem; color:#57606a; margin-top:0.35rem;">{caption}</figcaption>
          <div style="border-left:3px solid #8c959f; padding:0.38rem 0.55rem;
                      margin:0.35rem 0 0 0; color:#57606a; background:#f6f8fa;
                      font-size:0.92rem; line-height:1.35;">
            <strong>Interpretation limit:</strong> {limit}
          </div>
        </figure>
        """
    )
