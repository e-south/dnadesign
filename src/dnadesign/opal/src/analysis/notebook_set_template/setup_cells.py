from __future__ import annotations

from ._support import block


def render_setup_cells() -> str:
    """Render campaign-set imports and view-model setup cells."""

    return "\n\n".join((_preamble(), _import_cell(), _view_model_cell()))


def _preamble() -> str:
    return block(
        """
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="full")
        """
    )


def _import_cell() -> str:
    return block(
        """
        @app.cell
        def _():
            generated_with = "__GENERATED_WITH__"
            from pathlib import Path

            import marimo as mo
            import polars as pl

            from dnadesign.opal.notebooks.api.generated import (
                build_campaign_set_notebook_view_model,
                build_notebook_artifact_garden_rows,
                build_notebook_artifact_garden_summary_rows,
                build_notebook_at_a_glance_rows,
                build_notebook_campaign_header_lines,
                build_notebook_campaign_summary_row,
                build_notebook_change_rows,
                build_notebook_change_summary_rows,
                build_notebook_evidence_rows,
                build_notebook_metric_definition_rows,
                build_notebook_plot_card_rows,
                build_notebook_plot_method_sections,
                build_notebook_visual_surface_model,
                build_notebook_validity_rows,
            )
            return (
                Path,
                build_campaign_set_notebook_view_model,
                build_notebook_artifact_garden_rows,
                build_notebook_artifact_garden_summary_rows,
                build_notebook_at_a_glance_rows,
                build_notebook_campaign_header_lines,
                build_notebook_campaign_summary_row,
                build_notebook_change_rows,
                build_notebook_change_summary_rows,
                build_notebook_evidence_rows,
                build_notebook_metric_definition_rows,
                build_notebook_plot_card_rows,
                build_notebook_plot_method_sections,
                build_notebook_visual_surface_model,
                build_notebook_validity_rows,
                generated_with,
                mo,
                pl,
            )
        """
    )


def _view_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(Path, build_campaign_set_notebook_view_model):
            config_paths = [Path(path) for path in __CONFIG_PATHS__]
            campaign_set_view_model = build_campaign_set_notebook_view_model(
                config_paths,
                round_selector=__DEFAULT_ROUND__,
            )
            campaigns = campaign_set_view_model["campaigns"]
            return campaign_set_view_model, campaigns, config_paths
        """
    )


__all__ = ["render_setup_cells"]
