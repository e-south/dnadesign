"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_set_template.py

Renders marimo notebook templates for OPAL campaign sets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def render_campaign_set_notebook(config_paths: list[Path], *, round_selector: str) -> str:
    """Render a marimo notebook template for a campaign set."""

    try:
        import marimo as _marimo
    except Exception:
        _marimo = None
    marimo_version = "unknown" if _marimo is None else getattr(_marimo, "__version__", "unknown")
    path_literals = repr([str(Path(path)) for path in config_paths])
    template = dedent(
        """
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="full")


        @app.cell
        def _():
            generated_with = "__GENERATED_WITH__"
            from pathlib import Path

            import marimo as mo
            import polars as pl

            from dnadesign.opal.notebooks.api import (
                build_campaign_set_notebook_view_model,
                build_notebook_artifact_garden_lines,
                build_notebook_artifact_garden_rows,
                build_notebook_at_a_glance_rows,
                build_notebook_campaign_summary_row,
                build_notebook_change_lines,
                build_notebook_change_rows,
                build_notebook_evidence_rows,
                build_notebook_metric_definition_rows,
                build_notebook_plot_card_rows,
                build_notebook_plot_method_sections,
                build_notebook_visual_surface_model,
                build_notebook_validity_lines,
            )
            return (
                Path,
                build_campaign_set_notebook_view_model,
                build_notebook_artifact_garden_lines,
                build_notebook_artifact_garden_rows,
                build_notebook_at_a_glance_rows,
                build_notebook_campaign_summary_row,
                build_notebook_change_lines,
                build_notebook_change_rows,
                build_notebook_evidence_rows,
                build_notebook_metric_definition_rows,
                build_notebook_plot_card_rows,
                build_notebook_plot_method_sections,
                build_notebook_visual_surface_model,
                build_notebook_validity_lines,
                generated_with,
                mo,
                pl,
            )


        @app.cell
        def _(Path, build_campaign_set_notebook_view_model):
            config_paths = [Path(path) for path in __CONFIG_PATHS__]
            campaign_set_view_model = build_campaign_set_notebook_view_model(
                config_paths,
                round_selector=__DEFAULT_ROUND__,
            )
            campaigns = campaign_set_view_model["campaigns"]
            return campaign_set_view_model, campaigns, config_paths


        @app.cell
        def _(build_notebook_campaign_summary_row, campaign_set_view_model, campaigns, mo, pl):
            _rows = [build_notebook_campaign_summary_row(campaign_model) for campaign_model in campaigns]
            _labels = [row["label"] for row in _rows]
            campaign_ui = mo.ui.dropdown(_labels, value=_labels[0], label="Campaign")
            campaign_summary_df = pl.DataFrame(_rows)
            header_md = mo.md(
                "# Campaigns\\n\\n"
                f"`{campaign_set_view_model['campaign_count']}` campaigns. "
                f"Round selector: `{campaign_set_view_model['round_selector']}`."
            )
            return campaign_summary_df, campaign_ui, header_md


        @app.cell
        def _(campaign_ui, campaigns):
            selected_label = str(campaign_ui.value)
            selected_campaign_model = next(
                campaign_model
                for campaign_model in campaigns
                if f"{campaign_model['campaign']['slug']} | "
                f"{campaign_model.get('status', {}).get('progress_status') or 'unknown'}"
                == selected_label
            )
            return selected_campaign_model, selected_label


        @app.cell
        def _(build_notebook_at_a_glance_rows, build_notebook_validity_lines, mo, pl, selected_campaign_model):
            selected_overview_panel = mo.ui.table(
                pl.DataFrame(build_notebook_at_a_glance_rows(selected_campaign_model)),
                page_size=14,
            )
            selected_validity_md = mo.md(
                "\\n".join(build_notebook_validity_lines(selected_campaign_model))
            )
            return selected_overview_panel, selected_validity_md


        @app.cell
        def _(build_notebook_visual_surface_model, selected_campaign_model):
            visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
            plot_choices = visual_surface_model["choices"]
            return plot_choices


        @app.cell
        def _(mo, plot_choices):
            if plot_choices:
                _labels = [choice["label"] for choice in plot_choices]
                plot_ui = mo.ui.dropdown(_labels, value=_labels[0], label="Visual")
            else:
                plot_ui = None
            return plot_ui


        @app.cell
        def _(
            Path,
            build_notebook_plot_card_rows,
            build_notebook_plot_method_sections,
            mo,
            pl,
            plot_choices,
            plot_ui,
        ):
            if plot_ui is None:
                plot_panel = mo.md(
                    "No written manifest-backed plot media are available for this campaign."
                )
            else:
                _selected = str(plot_ui.value)
                _choice = next(choice for choice in plot_choices if choice["label"] == _selected)
                def _plot_image(plot_choice):
                    _path = Path(plot_choice["path"])
                    if not _path.exists():
                        return mo.md(f"Plot media missing: `{plot_choice['path_label']}`")
                    return mo.image(
                        _path.read_bytes(),
                        alt=str(plot_choice.get("alt_text") or plot_choice["title"]),
                        caption=str(plot_choice.get("caption") or "") or None,
                        rounded=True,
                        style={
                            "width": "100%",
                            "max-width": "100%",
                            "height": "auto",
                            "object-fit": "contain",
                            "margin": "0 auto",
                            "display": "block",
                            "background": "white",
                        },
                    )
                _controls = mo.hstack([plot_ui], justify="start", align="end", wrap=True, gap=0.35)
                _method_sections = build_notebook_plot_method_sections(_choice)
                plot_panel = mo.vstack(
                    [
                        _controls,
                        _plot_image(_choice),
                        mo.accordion(
                            {
                                **{label: mo.md(text) for label, text in _method_sections.items()},
                                "Evidence": mo.ui.table(
                                    pl.DataFrame(build_notebook_plot_card_rows(_choice)),
                                    page_size=12,
                                ),
                            },
                            multiple=True,
                        ),
                    ],
                    gap=0.45,
                )
            return plot_panel


        @app.cell
        def _(build_notebook_evidence_rows, mo, pl, selected_campaign_model):
            _rows = build_notebook_evidence_rows(selected_campaign_model)
            if _rows:
                evidence_panel = mo.ui.table(pl.DataFrame(_rows), page_size=10)
            else:
                evidence_panel = mo.md("No warnings or stale artifacts reported for this campaign.")
            return evidence_panel


        @app.cell
        def _(
            build_notebook_artifact_garden_lines,
            build_notebook_artifact_garden_rows,
            build_notebook_change_lines,
            build_notebook_change_rows,
            build_notebook_metric_definition_rows,
            mo,
            pl,
            selected_campaign_model,
        ):
            _metric_rows = build_notebook_metric_definition_rows(selected_campaign_model)
            if _metric_rows:
                metric_definitions_panel = mo.ui.table(pl.DataFrame(_metric_rows), page_size=10)
            else:
                metric_definitions_panel = mo.md("No manifest-backed plot metric definitions are available.")

            _change_rows = build_notebook_change_rows(selected_campaign_model)
            if _change_rows:
                changes_table = mo.ui.table(pl.DataFrame(_change_rows), page_size=10)
            else:
                changes_table = mo.md("No round changes are available yet.")
            changes_panel = mo.vstack(
                [
                    mo.md("\\n".join(build_notebook_change_lines(selected_campaign_model))),
                    changes_table,
                ]
            )

            _artifact_lines = build_notebook_artifact_garden_lines(selected_campaign_model)
            _artifact_rows = build_notebook_artifact_garden_rows(selected_campaign_model)
            artifact_rows_panel = (
                mo.ui.table(pl.DataFrame(_artifact_rows), page_size=10)
                if _artifact_rows
                else mo.md("No artifact garden rows are available.")
            )
            artifact_garden_panel = mo.vstack(
                [
                    mo.md("\\n".join(_artifact_lines)),
                    artifact_rows_panel,
                ]
            )
            return artifact_garden_panel, changes_panel, metric_definitions_panel


        @app.cell
        def _(
            artifact_garden_panel,
            campaign_summary_df,
            campaign_ui,
            changes_panel,
            evidence_panel,
            header_md,
            metric_definitions_panel,
            mo,
            plot_panel,
            selected_overview_panel,
            selected_validity_md,
        ):
            mo.vstack(
                [
                    header_md,
                    campaign_ui,
                    plot_panel,
                    mo.accordion(
                        {
                            "Campaigns at a glance": mo.ui.table(campaign_summary_df, page_size=12),
                            "Selected campaign": selected_overview_panel,
                            "Validity": selected_validity_md,
                            "Changes": changes_panel,
                            "Metric definitions": metric_definitions_panel,
                            "Artifacts": artifact_garden_panel,
                            "Warnings and stale artifacts": evidence_panel,
                        },
                        multiple=True,
                        lazy=True,
                    ),
                ]
            )
            return


        if __name__ == "__main__":
            app.run()
        """
    ).strip("\n")
    return (
        template.replace("__CONFIG_PATHS__", path_literals)
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )
