"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook.py

Marimo notebook writer for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_review_deliverables_notebook(path: Path) -> None:
    """Write a manifest-backed marimo notebook for Eco1 review deliverables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '''import marimo

__generated_with = "dnadesign.eco1_rt_repack.review_deliverables"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import html
    import marimo as mo
    import re
    from pathlib import Path
    import yaml
    return Path, base64, html, mo, re, yaml


@app.cell
def _(Path, yaml):
    manifest_path = Path(__file__).resolve().parents[1] / "review_deliverable_manifest.yaml"
    manifest_root = manifest_path.parent
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    deliverables = manifest["deliverables"]
    return deliverables, manifest, manifest_path, manifest_root


@app.cell
def _(Path, manifest_root):
    def resolve_manifest_path(value):
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else manifest_root / candidate

    return resolve_manifest_path,


@app.cell
def _(mo):
    mo.Html(
        """
        <section style="border-bottom:1px solid #d8dee4; padding:0 0 0.8rem 0; margin-bottom:0.5rem;">
          <h1 style="margin:0 0 0.35rem 0; font-size:2rem; line-height:1.12;">
            Repacking the Eco1 reverse transcriptase
          </h1>
          <p style="margin:0; max-width:70ch; color:inherit; opacity:0.82; font-size:1rem; line-height:1.45;">
            This review surface follows the current Eco1 RT repack study from scaffold evidence to
            ProteinMPNN sequence proposals and ColabFold-based structure review. It displays
            pre-rendered figures only; it does not rerun ProteinMPNN, ColabFold, Biohub, Atlas,
            ChimeraX, or selection logic.
          </p>
          <p style="margin:0.45rem 0 0 0; max-width:70ch; color:inherit;
                    opacity:0.82; font-size:1rem; line-height:1.45;">
            ProteinMPNN proposes sequence variants on the Ec86 scaffold. The figures show what was
            fixed, what was allowed to change, and how the current candidates look under structural review.
          </p>
        </section>
        """
    )


@app.cell
def _(Path, deliverables):
    def format_section_label(section):
        labels = {
            "scaffold_and_mask": "Scaffold and mask",
            "proteinmpnn": "ProteinMPNN proposals",
            "fold_review": "Fold review",
        }
        return labels.get(str(section), str(section).replace("_", " ").title())

    def format_deliverable_label(deliverable_id):
        labels = {
            "msa_plurality_mask_panel": "Clade 9 alignment and mask context",
            "linear_mask_tracks": "Linear mask tracks",
            "mask_structure_context_png": "3D mask-context render",
            "proteinmpnn_score_mutation_burden": "ProteinMPNN score and sequence identity",
            "proteinmpnn_mutation_density": "Mutation density across Ec86",
            "foldcheck_review_review_class_counts": "Fold-review class counts",
            "foldcheck_review_fold_metric_scatter": "ColabFold confidence and RMSD",
            "foldcheck_review_cryoem_vs_runtime_rmsd": "Runtime RMSD versus cryoEM RMSD",
            "foldcheck_review_biohub_esmc_sae_coverage": "Biohub ESMC SAE coverage",
        }
        return labels.get(str(deliverable_id), str(deliverable_id).replace("_", " ").title())

    def is_publication_visual(_row):
        suffix = Path(str(_row.get("path") or "")).suffix.lower()
        status = str(_row.get("status") or "")
        return suffix in {".svg", ".png"} and status in {"rendered", "linked_existing"}

    visual_deliverables = [_row for _row in deliverables if is_publication_visual(_row)]
    sections = []
    seen_sections = set()
    for _row in visual_deliverables:
        section = str(_row["section"])
        if section not in seen_sections:
            seen_sections.add(section)
            sections.append(section)
    section_label_lookup = {format_section_label(section): section for section in sections}
    return format_deliverable_label, format_section_label, section_label_lookup, sections, visual_deliverables


@app.cell
def _(mo, section_label_lookup):
    section_options = list(section_label_lookup)
    deliverable_section_ui = mo.ui.dropdown(
        section_options,
        value=section_options[0] if section_options else None,
        label="Review section",
        full_width=True,
    )
    return deliverable_section_ui


@app.cell
def _(deliverable_section_ui, section_label_lookup):
    selected_section = ""
    if deliverable_section_ui is not None and deliverable_section_ui.value is not None:
        selected_section = str(section_label_lookup.get(str(deliverable_section_ui.value), ""))
    return selected_section


@app.cell
def _(format_deliverable_label, selected_section, visual_deliverables):
    section_deliverables = [
        _row for _row in visual_deliverables if str(_row.get("section") or "") == selected_section
    ]
    deliverable_options = [format_deliverable_label(str(_row["deliverable_id"])) for _row in section_deliverables]
    deliverable_lookup = {
        format_deliverable_label(str(_row["deliverable_id"])): _row for _row in section_deliverables
    }
    return deliverable_lookup, deliverable_options


@app.cell
def _(deliverable_options, mo):
    deliverable_id_ui = mo.ui.dropdown(
        deliverable_options,
        value=deliverable_options[0] if deliverable_options else None,
        label="Visual",
        full_width=True,
    )
    return deliverable_id_ui


@app.cell
def _(deliverable_id_ui, deliverable_lookup):
    selected_deliverable = None
    if deliverable_id_ui is not None and deliverable_id_ui.value is not None:
        selected_deliverable = deliverable_lookup.get(str(deliverable_id_ui.value))
    return selected_deliverable


@app.cell
def _(base64, format_deliverable_label, html, mo, re, resolve_manifest_path):
    def image_aspect_ratio(media_path):
        if media_path.suffix.lower() != ".svg":
            return None
        text = media_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'viewBox="[^"]*?([0-9.]+)\\s+([0-9.]+)"', text)
        if not match:
            return None
        width = float(match.group(1))
        height = float(match.group(2))
        return width / height if height else None

    def render_deliverable_artifact(_row, *, max_height="min(72vh, 780px)", include_description=False):
        media_path = resolve_manifest_path(_row["path"])
        suffix = media_path.suffix.lower()
        if media_path.exists() and suffix in {".svg", ".png"}:
            mime_type = "image/svg+xml" if suffix == ".svg" else "image/png"
            encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
            aspect_ratio = image_aspect_ratio(media_path) or 2.0
            target_height = 170 if include_description else 360
            display_width_px = int(min(9000, max(1200, aspect_ratio * target_height)))
            alt_text = html.escape(str(_row["alt_text"]), quote=True)
            caption = html.escape(format_deliverable_label(str(_row.get("deliverable_id") or "")), quote=True)
            max_height_style = html.escape(max_height, quote=True)
            rendered = mo.Html(
                f"""
                <figure style="margin:0;">
                  <div style="overflow-x:auto; overflow-y:hidden; width:100%;
                              border:1px solid #d8dee4; border-radius:6px;
                              background:#ffffff; padding:0.5rem;">
                    <img
                      src="data:{mime_type};base64,{encoded}"
                      alt="{alt_text}"
                      style="display:block; width:min({display_width_px}px, 100%);
                             max-width:100%; max-height:{max_height_style};
                             height:auto; object-fit:contain;"
                    />
                  </div>
                  <figcaption style="font-size:0.86rem; color:#57606a; margin-top:0.35rem;">
                    {caption}
                  </figcaption>
                </figure>
                """
            )
        elif media_path.exists():
            rendered = mo.md(f"Generated non-image artifact: `{media_path}`")
        else:
            skip_reason = str(_row.get("skip_reason") or "artifact path does not exist")
            rendered = mo.md(
                f"Artifact unavailable: `{media_path}`\\n\\n"
                f"Artifact state: `{_row.get('status')}`. Reason: {skip_reason}"
            )
        if include_description:
            rendered = mo.vstack(
                [
                    rendered,
                    mo.md(
                        f"**{_row.get('deliverable_id')}**  \\n"
                        f"`{_row.get('status')}`  \\n"
                        f"{_row.get('description')}"
                    ),
                ],
                gap=0.2,
            )
        return rendered

    return render_deliverable_artifact,


@app.cell
def _(
    deliverable_id_ui,
    deliverable_section_ui,
    mo,
    render_deliverable_artifact,
    selected_deliverable,
):
    if selected_deliverable is None:
        panel = mo.md("No deliverable is available for the selected section.")
    else:
        body = render_deliverable_artifact(selected_deliverable)
        evidence_rows = [
            {"field": "status", "value": str(selected_deliverable.get("status") or "")},
            {"field": "path", "value": str(selected_deliverable.get("path") or "")},
            {"field": "role", "value": str(selected_deliverable.get("role") or "")},
            {"field": "sources", "value": ", ".join(selected_deliverable.get("source_tables", []))},
            {"field": "alt_text", "value": str(selected_deliverable.get("alt_text") or "")},
            {"field": "skip_reason", "value": str(selected_deliverable.get("skip_reason") or "")},
        ]
        details = mo.accordion(
            {
                "What this visual shows": mo.md(str(selected_deliverable.get("description") or "")),
                "Interpretation limit": mo.md(str(selected_deliverable.get("interpretation_limit") or "")),
                "Evidence": mo.ui.table(evidence_rows, page_size=8),
            },
            multiple=True,
            lazy=True,
        )
        panel = mo.vstack(
            [
                mo.hstack([deliverable_section_ui, deliverable_id_ui], justify="start", gap=1.0),
                mo.md("## Selected visual"),
                body,
                details,
            ],
            gap=0.45,
        )
    panel


if __name__ == "__main__":
    app.run()
''',
        encoding="utf-8",
    )
