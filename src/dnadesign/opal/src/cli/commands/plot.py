"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/plot.py

CLI plotting command for OPAL campaign outputs and ledgers. Resolves plot
configs, rounds, and output locations for charts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ...analysis.facade import CampaignAnalysis, parse_round_selector, round_suffix
from ...core.utils import OpalError, print_stdout
from ...plots.config import list_configured_plots, load_plot_config
from ...plots.runner import PlotRequest, resolve_run_round, run_plots
from ...registries.plots import get_plot_meta, list_plots
from ..formatting import bullet_list
from ..registry import cli_command
from ._common import print_config_context


@cli_command("plot", help="Generate plots from plot_config (preferred) or inline 'plots:'.")
def cmd_plot(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to campaign.yaml or campaign directory."),
    plot_config: Optional[Path] = typer.Option(
        None,
        "--plot-config",
        help="Path to plots YAML (overrides campaign.plot_config).",
    ),
    list_registry: bool = typer.Option(
        False,
        "--list",
        help="List registered plot kinds and exit.",
    ),
    list_config: bool = typer.Option(
        False,
        "--list-config",
        help="List plots configured in YAML and exit (requires --config).",
    ),
    describe: Optional[str] = typer.Option(
        None,
        "--describe",
        help="Describe a plot kind (params + required fields) and exit.",
    ),
    round: Optional[str] = typer.Option(
        None,
        "--round",
        "-r",
        help="Round selector: latest | all | 3 | 1,3,7 | 2-5 (omitted = unspecified).",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Explicit run_id to disambiguate ledger predictions (required if multiple runs per round).",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Run a single plot by its 'name' in the YAML."),
    tag: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        help="Run plots with the given tag (repeatable).",
    ),
) -> None:
    """
    Runs all plots by default (or a single plot via --name).
    Overwrites output files by default.
    Continues on error, printing full tracebacks.
    Exit code 1 if any plot failed.
    """
    if describe:
        try:
            meta = get_plot_meta(describe)
        except KeyError as e:
            raise ValueError(str(e)) from e
        print_stdout(f"Plot: {describe}")
        if meta is None:
            print_stdout("No metadata available for this plot.")
        else:
            print_stdout(f"Summary: {meta.summary}")
            if meta.requires:
                print_stdout(bullet_list("Required fields", meta.requires))
            if meta.params:
                rows = [f"{k}: {v}" for k, v in meta.params.items()]
                print_stdout(bullet_list("Params", rows))
            if meta.notes:
                print_stdout(bullet_list("Notes", meta.notes))
        return

    if list_registry and not list_config:
        rows = []
        for name in list_plots():
            meta = get_plot_meta(name)
            rows.append(f"{name} - {meta.summary}" if meta and meta.summary else name)
        print_stdout(bullet_list("Registered plots", rows))
        return

    analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
    cfg_path = analysis.config_path
    campaign_yaml = cfg_path
    campaign_dir = analysis.workspace.workdir
    campaign_cfg = analysis.read_config_dict()
    cfg = analysis.config
    store = analysis.records_store()
    ws = analysis.workspace
    print_config_context(campaign_yaml, cfg=cfg, records_path=store.records_path)

    try:
        plot_cfg = load_plot_config(
            campaign_cfg=campaign_cfg,
            campaign_yaml=campaign_yaml,
            campaign_dir=campaign_dir,
            plot_config_opt=plot_config,
        )
    except ValueError as e:
        msg = str(e)
        if "No plots found" in msg:
            raise ValueError("[plot] No plots found. Add plots.yaml or define plots in campaign.yaml.") from e
        raise

    if list_registry or list_config:
        if list_registry:
            rows = []
            for name in list_plots():
                meta = get_plot_meta(name)
                rows.append(f"{name} - {meta.summary}" if meta and meta.summary else name)
            print_stdout(bullet_list("Registered plots", rows))
        if list_config:
            rows = list_configured_plots(
                plots_cfg=plot_cfg.plots,
                plot_presets=plot_cfg.plot_presets,
            )
            print_stdout(bullet_list("Configured plots", rows))
        return

    tag_filters = [str(t) for t in (tag or [])]

    try:
        rounds_sel = parse_round_selector(round)
    except OpalError as e:
        raise typer.BadParameter(str(e), param_hint="--round") from e
    if run_id:
        try:
            runs_df = analysis.read_runs()
        except OpalError as e:
            raise ValueError(f"[plot] {e}") from e
        run_round = resolve_run_round(runs_df, run_id)
        if rounds_sel == "all":
            raise ValueError("[plot] Do not combine --run-id with --round all; run_id is single-round.")
        if rounds_sel in ("unspecified", "latest"):
            rounds_sel = [run_round]
        elif isinstance(rounds_sel, list):
            if run_round not in rounds_sel:
                raise ValueError(
                    f"[plot] run_id {run_id!r} belongs to as_of_round={run_round}, but --round={round!r} excludes it."
                )
        else:
            if int(rounds_sel) != int(run_round):
                raise ValueError(
                    f"[plot] run_id {run_id!r} belongs to as_of_round={run_round}, "
                    f"but --round={round!r} selects a different round."
                )
    suffix = round_suffix(rounds_sel)

    req = PlotRequest(
        plots_cfg=plot_cfg.plots,
        plot_defaults=plot_cfg.plot_defaults,
        plot_presets=plot_cfg.plot_presets,
        plot_cfg_dir=plot_cfg.source_dir,
        campaign_dir=campaign_dir,
        workspace=ws,
        store=store,
        rounds_sel=rounds_sel,
        run_id=run_id,
        round_suffix=suffix,
        name_filter=name,
        tag_filters=tag_filters,
    )

    any_fail = run_plots(req)
    raise typer.Exit(code=1 if any_fail else 0)
