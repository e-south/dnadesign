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
    def _(parse_enabled, parse_tags, plot_cfg):
        plot_entries = []
        if plot_cfg is not None:
            for plot_entry_item in plot_cfg.plots:
                if not isinstance(plot_entry_item, dict):
                    raise ValueError(
                        "Plot entry must be a mapping (got "
                        f"{type(plot_entry_item).__name__})."
                    )
                name = plot_entry_item.get("name")
                if not name:
                    raise ValueError("Plot entry missing name.")
                preset_name = plot_entry_item.get("preset")
                preset = plot_cfg.plot_presets.get(preset_name) if preset_name else {}
                kind = plot_entry_item.get("kind") or preset.get("kind")
                if not kind:
                    raise ValueError(f"Plot '{name}' missing kind.")
                enabled = parse_enabled(
                    plot_entry_item.get("enabled")
                    if "enabled" in plot_entry_item
                    else preset.get("enabled"),
                    ctx=name,
                )
                if not enabled:
                    continue
                _plot_tags_list = []
                if preset_name:
                    _plot_tags_list += parse_tags(
                        preset.get("tags"),
                        ctx=f"plot_presets.{preset_name}",
                    )
                _plot_tags_list += parse_tags(
                    plot_entry_item.get("tags"),
                    ctx=f"plot {name}",
                )
                plot_entries.append(
                    {"name": name, "kind": kind, "tags": _plot_tags_list}
                )
        return plot_entries
    """
).strip("\n")
