"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_components.py

Reusable generated-cell components for OPAL marimo notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from textwrap import dedent


def render_plot_gallery_cells() -> str:
    """Render generated cells for manifest-backed plot selection."""

    return dedent(
        """
        @app.cell
        def _(Path, notebook_view_model, plot_entries):
            plots_dir = Path(notebook_view_model["campaign"]["workdir"]) / "outputs" / "plots"
            manifest_rows = [
                row
                for row in notebook_view_model.get("plot_manifests", [])
                if row.get("status") == "written"
            ]
            active_by_name = {str(row.get("name")): row for row in manifest_rows}
            plot_choices = []
            missing_outputs = []
            for plot_entry_choice in plot_entries:
                _manifest = active_by_name.get(str(plot_entry_choice["name"]))
                if _manifest is None:
                    missing_outputs.append(plot_entry_choice["name"])
                    continue
                media_outputs = [
                    output
                    for output in _manifest.get("outputs", [])
                    if output.get("role") == "media" and output.get("exists")
                ]
                if not media_outputs:
                    missing_outputs.append(plot_entry_choice["name"])
                    continue
                path = Path(media_outputs[0]["path"])
                label = f"{plot_entry_choice['name']} ({path.name})"
                plot_choices.append(
                    {
                        "label": label,
                        "path": path,
                        "entry": plot_entry_choice,
                        "manifest": _manifest,
                    }
                )
            stale_plot_artifacts = notebook_view_model.get("stale_artifacts", [])
            return plots_dir, plot_choices, missing_outputs, stale_plot_artifacts


        @app.cell
        def _(mo, plot_cfg_error, plot_choices, plots_dir, missing_outputs, stale_plot_artifacts):
            plot_ui = None
            gallery_scope = "All configured plots with written manifests."
            if plot_cfg_error:
                plot_gallery_note = (
                    "### Plot artifacts (`outputs/plots`)\\n\\n"
                    f"Plot config unavailable: `{plot_cfg_error}`"
                )
            elif not plot_choices:
                _lines = [
                    "### Plot artifacts (`outputs/plots`)",
                    "",
                    f"No manifest-backed plot outputs found in `{plots_dir}`.",
                    "Run `uv run opal plot -c <campaign.yaml>` to generate plots.",
                    gallery_scope,
                ]
                if missing_outputs:
                    _lines.append(
                        f"Configured plots without outputs: {', '.join(missing_outputs)}"
                    )
                if stale_plot_artifacts:
                    _lines.append(f"Stale artifact warnings: `{len(stale_plot_artifacts)}`")
                plot_gallery_note = "\\n".join(_lines)
            else:
                labels = [plot_choice["label"] for plot_choice in plot_choices]
                plot_ui = mo.ui.dropdown(labels, value=labels[0], label="Plot")
                plot_gallery_note = "### Plot artifacts (`outputs/plots`)\\n\\n" + gallery_scope
                if stale_plot_artifacts:
                    plot_gallery_note += f"\\n\\nStale artifact warnings: `{len(stale_plot_artifacts)}`"
            return plot_ui, plot_gallery_note


        @app.cell
        def _(mo, plot_choices, plot_gallery_note, plot_ui):
            if plot_ui is None:
                plot_panel = mo.md(plot_gallery_note)
            else:
                selected = str(plot_ui.value)
                choice = next(
                    (
                        plot_choice
                        for plot_choice in plot_choices
                        if plot_choice["label"] == selected
                    ),
                    None,
                )
                if choice is None:
                    raise ValueError(f"Plot selection not found: {selected}")
                plot_entry_selected = choice["entry"]
                _manifest = choice["manifest"]
                _plot_tags_str = (
                    ", ".join(plot_entry_selected["tags"])
                    if plot_entry_selected["tags"]
                    else "none"
                )
                tidy_csv = _manifest.get("tidy_csv") or "none"
                params = _manifest.get("params") or {}
                details = [
                    plot_gallery_note,
                    "",
                    f"**Plot**: `{plot_entry_selected['name']}`",
                    f"Kind: `{plot_entry_selected['kind']}`",
                    f"Tags: `{_plot_tags_str}`",
                    f"Status: `{_manifest.get('status')}`",
                    f"Generated: `{_manifest.get('generated_at')}`",
                    f"Run ID: `{_manifest.get('run_id')}`",
                    f"Rounds: `{_manifest.get('rounds')}`",
                    f"File: `{choice['path']}`",
                    f"Tidy CSV: `{tidy_csv}`",
                    f"Params: `{params}`",
                ]
                plot_panel = mo.vstack(
                    [
                        mo.md("\\n".join(details)),
                        plot_ui,
                        mo.image(choice["path"].read_bytes()),
                    ]
                )
            return plot_panel
        """
    ).strip("\n")
