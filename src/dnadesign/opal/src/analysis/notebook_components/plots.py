from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Mapping

from ._support import first_media_output, mapping, plot_entries_from_manifests, sequence


def build_notebook_plot_gallery_model(
    view_model: Mapping[str, Any],
    *,
    plot_entries: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a manifest-authoritative plot gallery model for marimo templates."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir") or ""
    plots_dir = str(Path(str(workdir)) / "outputs" / "plots") if workdir else "outputs/plots"
    manifest_rows = [
        manifest
        for manifest in sequence(view_model.get("plot_manifests"))
        if isinstance(manifest, Mapping) and manifest.get("status") == "written"
    ]
    active_by_name = {str(row.get("name")): row for row in manifest_rows}
    configured_entries = plot_entries_from_manifests(manifest_rows) if plot_entries is None else list(plot_entries)

    choices: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for entry in configured_entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        manifest = active_by_name.get(name)
        if manifest is None:
            missing_outputs.append(name)
            continue
        media_output = first_media_output(manifest)
        if media_output is None:
            missing_outputs.append(name)
            continue
        path = str(media_output.get("path"))
        choices.append(
            {
                "label": f"{name} ({Path(path).name})",
                "path": path,
                "entry": dict(entry),
                "manifest": dict(manifest),
            }
        )
    return {
        "plots_dir": plots_dir,
        "choices": choices,
        "missing_outputs": missing_outputs,
        "stale_artifacts": list(sequence(view_model.get("stale_artifacts"))),
    }


def build_notebook_plot_card_lines(choice: Mapping[str, Any]) -> list[str]:
    """Build manifest-backed plot-card detail lines for generated notebooks."""

    entry = mapping(choice.get("entry"))
    manifest = mapping(choice.get("manifest"))
    freshness = mapping(manifest.get("freshness"))
    inputs = [
        item
        for item in sequence(manifest.get("inputs"))
        if isinstance(item, Mapping) and (item.get("path") or item.get("role"))
    ]
    source_data = "; ".join(f"{item.get('role') or 'input'}={item.get('path') or 'unrecorded'}" for item in inputs[:5])
    warnings = sequence(manifest.get("warnings"))
    tags = ", ".join(str(tag) for tag in sequence(entry.get("tags"))) or "none"
    return [
        "### Plot deliverables",
        "",
        f"**Plot**: `{entry.get('name') or manifest.get('name')}`",
        f"Kind: `{entry.get('kind') or manifest.get('kind')}`",
        f"Tags: `{tags}`",
        f"Status: `{manifest.get('status')}`",
        f"Freshness: `{freshness.get('status') or manifest.get('stale_state') or 'unknown'}`",
        f"Generated: `{manifest.get('generated_at')}`",
        f"Run ID: `{manifest.get('run_id')}`",
        f"Rounds: `{manifest.get('rounds')}`",
        f"Media: `{choice.get('path')}`",
        f"Tidy CSV: `{manifest.get('tidy_csv') or 'none'}`",
        f"Source data: `{source_data or 'not recorded'}`",
        f"Params: `{manifest.get('params') or {}}`",
        f"Warnings: `{len(warnings)}`",
    ]


def render_plot_gallery_cells() -> str:
    """Render generated cells for manifest-backed plot selection."""

    return dedent(
        """
        @app.cell
        def _(Path, build_notebook_plot_gallery_model, notebook_view_model, plot_entries):
            plot_gallery_model = build_notebook_plot_gallery_model(
                notebook_view_model,
                plot_entries=plot_entries,
            )
            plots_dir = Path(plot_gallery_model["plots_dir"])
            plot_choices = plot_gallery_model["choices"]
            missing_outputs = plot_gallery_model["missing_outputs"]
            stale_plot_artifacts = plot_gallery_model["stale_artifacts"]
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
        def _(Path, build_notebook_plot_card_lines, mo, plot_choices, plot_gallery_note, plot_ui):
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
                details = [plot_gallery_note, "", *build_notebook_plot_card_lines(choice)]
                plot_panel = mo.vstack(
                    [
                        mo.md("\\n".join(details)),
                        plot_ui,
                        mo.image(Path(choice["path"]).read_bytes()),
                    ]
                )
            return plot_panel
        """
    ).strip("\n")
