"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/notebook.py

Marimo notebook writer for Eco1 fold-check review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_review_notebook(path: Path) -> None:
    """Write a scoped marimo notebook that reads the visual manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '''import marimo

__generated_with = "dnadesign.eco1_rt_repack.foldcheck_review"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import yaml
    return Path, mo, yaml


@app.cell
def _(Path, yaml):
    manifest_path = Path(__file__).resolve().parents[1] / "review_visual_manifest.yaml"
    manifest_root = manifest_path.parent
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    plots = manifest["plots"]
    return manifest, manifest_path, manifest_root, plots


@app.cell
def _(Path, manifest_root):
    def resolve_manifest_path(value):
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else manifest_root / candidate

    return resolve_manifest_path,


@app.cell
def _(manifest, mo):
    mo.md(
        f"""
        # Eco1 fold-check review

        This notebook reads a scoped visual manifest and displays generated plots
        with their alt text and interpretation limits. The plots summarize
        fold-review and Biohub ESMC/SAE artifacts; they do not accept candidates.

        **Plot count:** {manifest["plot_count"]}
        """
    )


@app.cell
def _(plots):
    plot_labels = [str(plot["title"]) for plot in plots]
    plot_lookup = {str(plot["title"]): plot for plot in plots}
    plot_inventory_rows = [
        {
            "plot_id": str(plot.get("plot_id") or ""),
            "title": str(plot.get("title") or ""),
            "path": str(plot.get("path") or ""),
            "sources": ", ".join(str(source) for source in plot.get("data_sources", [])),
        }
        for plot in plots
    ]
    return plot_inventory_rows, plot_labels, plot_lookup


@app.cell
def _(mo, plot_labels):
    if plot_labels:
        visual_surface_ui = mo.ui.dropdown(
            plot_labels,
            value=plot_labels[0],
            label="Review surface",
            full_width=True,
        )
    else:
        visual_surface_ui = None
    return visual_surface_ui


@app.cell
def _(plot_lookup, visual_surface_ui):
    selected_plot = None
    if visual_surface_ui is not None:
        selected_plot = plot_lookup.get(str(visual_surface_ui.value))
    return selected_plot


@app.cell
def _(mo, plot_inventory_rows, resolve_manifest_path, selected_plot, visual_surface_ui):
    if selected_plot is None:
        plot_panel = mo.md("No review plots are available in the visual manifest.")
    else:
        media_path = resolve_manifest_path(selected_plot["path"])
        if media_path.exists():
            visual = mo.image(
                media_path.read_bytes(),
                alt=str(selected_plot["alt_text"]),
                caption=str(selected_plot.get("title") or ""),
                rounded=True,
                style={
                    "width": "auto",
                    "max-height": "min(70vh, 760px)",
                    "max-width": "100%",
                    "height": "auto",
                    "object-fit": "contain",
                    "margin": "0 auto",
                    "display": "block",
                    "background": "white",
                },
            )
        else:
            visual = mo.md(f"Plot media missing: `{media_path}`")

        evidence_rows = [
            {"field": "plot_id", "value": str(selected_plot.get("plot_id") or "")},
            {"field": "path", "value": str(selected_plot.get("path") or "")},
            {
                "field": "data_sources",
                "value": ", ".join(str(source) for source in selected_plot.get("data_sources", [])),
            },
            {"field": "alt_text", "value": str(selected_plot.get("alt_text") or "")},
        ]
        details = mo.accordion(
            {
                "What this visual shows": mo.md(str(selected_plot.get("description") or "")),
                "Interpretation limit": mo.md(str(selected_plot.get("interpretation_limit") or "")),
                "Evidence": mo.ui.table(evidence_rows, page_size=8),
                "Plot inventory": mo.ui.table(plot_inventory_rows, page_size=8),
            },
            multiple=True,
            lazy=True,
        )
        plot_panel = mo.vstack([visual_surface_ui, visual, details], gap=0.45)
    plot_panel


if __name__ == "__main__":
    app.run()
''',
        encoding="utf-8",
    )
