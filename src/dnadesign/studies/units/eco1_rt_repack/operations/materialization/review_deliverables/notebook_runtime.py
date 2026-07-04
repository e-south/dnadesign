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
import hashlib
import html
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_ESMC_FEATURE_REVIEW,
    SECTION_FEASIBILITY_AND_HANDOFF,
)

from .notebook_selection_panel import render_selection_panel_table
from .notebook_selection_summary import (
    render_handoff_readiness,
    render_selection_funnel_summary,
)

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
        "The current Tao-style fixed-backbone workflow builds a conservative redesign set for downstream "
        "structured-template assays: protect residues supported by motifs, "
        "substrate contacts, and homolog conservation, then repack the remaining designable residues."
    )
    intro_flow = (
        "Evidence order: scaffold, mask evidence, sequence proposals, fold checks, panel selection, and model checks. "
        "The active mask uses catalytic anchors, Wang/Ec86 direct-contact priors, retained-substrate "
        "proximity, and Mestre-derived clade 9 plurality at the 25% threshold. WT ESMC masked-marginal "
        "scoring appears beside those inputs as a model check, not as a mask input. "
        "ProteinMPNN proposes variants at unprotected residues, ColabFold checks fold-model compatibility, and "
        "Biohub ESMC SAE features annotate WT and candidate sequences. Activity, strand displacement, "
        "and structured-template readthrough remain assay questions."
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
    if artifact_kind == "handoff_readiness":
        return render_handoff_readiness(row, mo=mo, manifest_path=media_path)
    if artifact_kind == "handoff_boundary":
        return _render_handoff_boundary(row, mo=mo)
    if media_path.exists() and suffix in {".svg", ".png"}:
        return _render_image(row, mo=mo, media_path=media_path)
    if media_path.exists():
        return mo.md(f"Artifact file: `{media_path}`")
    skip_reason = str(row.get("skip_reason") or "artifact path does not exist")
    return mo.md(
        f"Artifact unavailable: `{media_path}`\n\nArtifact state: `{row.get('status')}`. Reason: {skip_reason}"
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
    if row_title:
        return row_title
    return deliverable_id.replace("_", " ").title()


def _is_publication_visual(row: dict[str, Any]) -> bool:
    if str(row.get("deliverable_id") or "") in _NOTEBOOK_HIDDEN_DELIVERABLE_IDS:
        return False
    if str(row.get("artifact_kind") or "") in {
        "selection_funnel_summary",
        "selection_panel_table",
        "handoff_readiness",
    }:
        return str(row.get("status") or "") == "linked_existing"
    if str(row.get("artifact_kind") or "") == "handoff_boundary":
        return str(row.get("status") or "") == "linked_existing"
    if str(row.get("artifact_kind") or "") == "sae_feature_heatmap_manifest":
        return str(row.get("status") or "") == "rendered"
    if is_interactive_structure_deliverable(row):
        return True
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    return suffix in {".svg", ".png"} and str(row.get("status") or "") in {"rendered", "linked_existing"}


def _render_image(row: dict[str, Any], *, mo: Any, media_path: Path) -> Any:
    mime_type = "image/svg+xml" if media_path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
    alt_text = html.escape(str(row["alt_text"]), quote=True)
    caption = html.escape(format_deliverable_label(row), quote=True)
    is_wide = is_wide_visual(row)
    render_mode = html.escape(str(row.get("render_mode") or "standard_visual"), quote=True)
    container_id = f"zoomable-visual-{hashlib.sha256(str(media_path).encode()).hexdigest()[:12]}"
    image_id = f"{container_id}-image"
    container_max_height = "86vh" if is_wide else "82vh"
    image_style = "display:block; width:100%; max-width:none; max-height:none; height:auto; transform-origin:top left;"
    zoom_controls = render_visual_zoom_controls(container_id)
    zoom_script = visual_zoom_script(container_id=container_id, image_id=image_id)
    frame_html = zoom_frame_html(
        body_html=f"""
          {zoom_controls}
          <div id="{container_id}" data-render-mode="{render_mode}"
               style="overflow:auto; width:100%; height:calc(100vh - 2.6rem);
                      border:1px solid #d8dee4; border-radius:6px; background:#ffffff; padding:0.25rem;
                      box-sizing:border-box;">
            <img id="{image_id}" src="data:{mime_type};base64,{encoded}" alt="{alt_text}"
                 style="{image_style}" />
          </div>
          {zoom_script}
        """
    )
    caption_html = mo.Html(
        f"""
        <figcaption style="font-size:0.92rem; color:#57606a; margin-top:0.2rem;">{caption}</figcaption>
        """
    )
    return mo.vstack(
        [
            render_zoom_frame(
                mo=mo,
                frame_html=frame_html,
                title=f"Zoomable visual: {caption}",
                height_css=container_max_height,
            ),
            caption_html,
        ],
        gap=0.15,
    )


def _render_handoff_boundary(row: dict[str, Any], *, mo: Any) -> Any:
    title = html.escape(str(row.get("title") or "Panel status"))
    description = html.escape(str(row.get("description") or ""))
    return mo.Html(
        f"""
        <section style="border:1px solid #d8dee4; border-radius:6px; padding:0.8rem 0.9rem;
                        background:#ffffff;">
          <h3 style="margin:0 0 0.35rem 0; font-size:1.06rem;">{title}</h3>
          <p style="margin:0; line-height:1.45; color:#57606a;">{description}</p>
        </section>
        """
    )


def render_zoom_frame(*, mo: Any, frame_html: str, title: str, height_css: str) -> Any:
    safe_srcdoc = html.escape(frame_html, quote=True)
    safe_title = html.escape(title, quote=True)
    safe_height = html.escape(height_css, quote=True)
    return mo.Html(
        f"""
        <iframe title="{safe_title}" srcdoc="{safe_srcdoc}"
                style="display:block; width:100%; height:{safe_height}; min-height:520px;
                       border:0; margin:0; padding:0; background:#ffffff;"
                loading="lazy"></iframe>
        """
    )


def zoom_frame_html(*, body_html: str) -> str:
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #ffffff;
            color: #24292f;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          * {{ box-sizing: border-box; }}
        </style>
      </head>
      <body>
        {body_html}
      </body>
    </html>
    """


def render_visual_zoom_controls(container_id: str) -> str:
    safe_container_id = html.escape(container_id, quote=True)
    return f"""
          <div style="display:flex; gap:0.35rem; align-items:center; justify-content:flex-end;
                      margin:0 0 0.25rem 0;">
            <button type="button" data-zoom-target="{safe_container_id}" data-zoom-action="out"
                    aria-label="Zoom out"
                    style="border:1px solid #d8dee4; background:#ffffff; border-radius:4px;
                           padding:0.12rem 0.45rem; line-height:1.25; cursor:pointer;">-</button>
            <button type="button" data-zoom-target="{safe_container_id}" data-zoom-action="fit"
                    aria-label="Fit image to panel"
                    style="border:1px solid #d8dee4; background:#ffffff; border-radius:4px;
                           padding:0.12rem 0.45rem; line-height:1.25; cursor:pointer;">Fit</button>
            <button type="button" data-zoom-target="{safe_container_id}" data-zoom-action="in"
                    aria-label="Zoom in"
                    style="border:1px solid #d8dee4; background:#ffffff; border-radius:4px;
                           padding:0.12rem 0.45rem; line-height:1.25; cursor:pointer;">+</button>
          </div>
    """


def visual_zoom_script(*, container_id: str, image_id: str) -> str:
    safe_container_id = html.escape(container_id, quote=True)
    safe_image_id = html.escape(image_id, quote=True)
    return f"""
          <script>
          (function() {{
            const container = document.getElementById("{safe_container_id}");
            const image = document.getElementById("{safe_image_id}");
            if (!container || !image || container.dataset.zoomReady === "true") {{
              return;
            }}
            container.dataset.zoomReady = "true";
            const minScale = 0.2;
            const maxScale = 24.0;
            let scale = 1.0;
            let baseWidth = 0;
            const setBaseWidth = function() {{
              const rect = image.getBoundingClientRect();
              const computedStyle = window.getComputedStyle(container);
              const horizontalPadding = parseFloat(computedStyle.paddingLeft || "0") +
                parseFloat(computedStyle.paddingRight || "0");
              baseWidth = Math.max(320, container.clientWidth - horizontalPadding, rect.width || 0);
              image.style.width = baseWidth + "px";
            }};
            const applyScale = function(nextScale) {{
              if (!baseWidth) {{
                setBaseWidth();
              }}
              scale = Math.max(minScale, Math.min(maxScale, nextScale));
              image.dataset.zoomScale = String(scale);
              image.style.width = (baseWidth * scale) + "px";
            }};
            const fitImage = function() {{
              scale = 1.0;
              setBaseWidth();
              image.dataset.zoomScale = String(scale);
              container.scrollLeft = 0;
              container.scrollTop = 0;
            }};
            window.requestAnimationFrame(fitImage);
            container.addEventListener("wheel", function(event) {{
              if (!event.ctrlKey && !event.metaKey) {{
                return;
              }}
              event.preventDefault();
              const factor = event.deltaY < 0 ? 1.14 : 0.88;
              applyScale(scale * factor);
            }}, {{ passive: false }});
            document.querySelectorAll('[data-zoom-target="{safe_container_id}"]').forEach(function(button) {{
              button.addEventListener("click", function() {{
                const action = button.getAttribute("data-zoom-action");
                if (action === "fit") {{
                  fitImage();
                }} else if (action === "in") {{
                  applyScale(scale * 1.25);
                }} else if (action === "out") {{
                  applyScale(scale * 0.8);
                }}
              }});
            }});
            window.addEventListener("resize", function() {{
              if (scale === 1.0) {{
                fitImage();
              }}
            }}, {{ passive: true }});
          }})();
          </script>
    """


def is_wide_visual(row: dict[str, Any]) -> bool:
    """Return whether a visual should preserve horizontal detail instead of squeezing to fit."""

    return str(row.get("render_mode") or "") == "wide_visual"
