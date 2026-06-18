"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_template/plot_cells.py

Notebook template builders for plot cells OPAL analysis notebook template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from textwrap import dedent

PLOT_CONFIG_CELLS = dedent(
    """
    @app.cell
    def _(campaign, config_path, load_plot_config):
        plot_cfg = None
        plot_cfg_error = None
        try:
            plot_cfg = load_plot_config(
                campaign_cfg=campaign.read_config_dict(),
                campaign_yaml=config_path,
                campaign_dir=campaign.workspace.workdir,
                plot_config_opt=None,
            )
        except Exception as exc:
            plot_cfg_error = str(exc)
        return plot_cfg, plot_cfg_error


    @app.cell
    def _(list_configured_plot_specs, plot_cfg):
        plot_entries = []
        if plot_cfg is not None:
            for spec in list_configured_plot_specs(
                plots_cfg=plot_cfg.plots,
                plot_presets=plot_cfg.plot_presets,
            ):
                if not spec["enabled"]:
                    continue
                plot_entries.append(
                    {
                        "name": spec["name"],
                        "kind": spec["kind"],
                        "tags": spec.get("tags") or [],
                        "round_selector": spec.get("round_selector"),
                    }
                )
        return plot_entries
    """
).strip("\n")
