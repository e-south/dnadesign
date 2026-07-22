"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/plot.py

CLI plotting command for OPAL campaign outputs and ledgers. Resolves plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ...analysis.campaign import CampaignAnalysis
from ...analysis.ledger import parse_round_selector, round_suffix
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...plots.config import list_configured_plot_specs, list_configured_plots, load_plot_config
from ...plots.runner import PlotRequest, resolve_run_round, run_plots
from ...registries.objectives import get_objective_family
from ...registries.plots import describe_plot_kind, get_plot_meta, list_plots
from ..formatting import bullet_list
from ..registry import cli_command
from ._common import json_error, json_out, opal_error, print_config_context


@cli_command("plot", help="Generate plots from the campaign plot_config or an explicit plot-config path.")
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
    view: Optional[str] = typer.Option(
        None,
        "--view",
        help="Selection view ID; required for multi-view campaigns.",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Run a single plot by its 'name' in the YAML."),
    tag: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        help="Run plots with the given tag (repeatable).",
    ),
    json_output: bool = typer.Option(False, "--json/--text", help="Output JSON for list/describe surfaces."),
) -> None:
    """
    Runs all plots by default (or a single plot via --name).
    Overwrites output files by default.
    Collects per-plot render failures while surfacing CLI/config contract
    failures as OPAL errors.
    Exit code 1 if any plot failed.
    """
    try:
        _run_plot_command(
            config=config,
            plot_config=plot_config,
            list_registry=list_registry,
            list_config=list_config,
            describe=describe,
            round=round,
            run_id=run_id,
            view=view,
            name=name,
            tag=tag,
            json_output=json_output,
        )
    except OpalError as e:
        ctx = "plot list-config" if list_config and not describe else "plot"
        if json_output:
            json_error(ctx, e)
        else:
            opal_error(ctx, e)
        raise typer.Exit(code=e.exit_code) from e


def _run_plot_command(
    *,
    config: Optional[Path],
    plot_config: Optional[Path],
    list_registry: bool,
    list_config: bool,
    describe: Optional[str],
    round: Optional[str],
    run_id: Optional[str],
    view: Optional[str],
    name: Optional[str],
    tag: Optional[List[str]],
    json_output: bool,
) -> None:
    if describe:
        try:
            meta = get_plot_meta(describe)
        except KeyError as e:
            err = OpalError(f"Unknown plot kind: {describe!r}", ExitCodes.BAD_ARGS)
            if json_output:
                json_error("plot describe", err)
                raise typer.Exit(code=err.exit_code) from e
            raise typer.BadParameter(str(err), param_hint="--describe") from e
        if json_output:
            json_out(
                {
                    "schema_version": "opal.plot_description.v1",
                    "plot": describe_plot_kind(describe),
                }
            )
            return
        print_stdout(f"Plot: {describe}")
        if meta is None:
            print_stdout("No metadata available for this plot.")
        else:
            print_stdout(f"Summary: {meta.summary}")
            if meta.requires:
                print_stdout(bullet_list("Required fields", meta.requires))
            if meta.data_shape:
                print_stdout(f"Data shape: {meta.data_shape}")
            if meta.tidy_schema:
                print_stdout(bullet_list("Tidy CSV schema", meta.tidy_schema))
            if meta.failure_modes:
                print_stdout(bullet_list("Failure modes", meta.failure_modes))
            if meta.params:
                rows = [f"{k}: {v}" for k, v in meta.params.items()]
                print_stdout(bullet_list("Params", rows))
            if meta.notes:
                print_stdout(bullet_list("Notes", meta.notes))
        return

    if list_registry and not list_config:
        if json_output:
            json_out(
                {
                    "schema_version": "opal.plot_registry.v1",
                    "plots": [describe_plot_kind(name) for name in list_plots()],
                }
            )
            return
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
    if not json_output:
        print_config_context(campaign_yaml, cfg=cfg, records_path=store.records_path)

    try:
        plot_cfg = load_plot_config(
            campaign_cfg=campaign_cfg,
            campaign_yaml=campaign_yaml,
            plot_config_opt=plot_config,
        )
    except ValueError as e:
        msg = str(e)
        if "No plots found" in msg:
            err = OpalError("[plot] No plots found. Set campaign.plot_config or pass --plot-config.")
            if json_output:
                json_error("plot list-config" if list_config else "plot", err)
                raise typer.Exit(code=err.exit_code) from e
            raise typer.BadParameter(str(err), param_hint="--plot-config") from e
        raise

    if list_registry or list_config:
        if json_output:
            if list_config and not list_registry:
                json_out(
                    {
                        "schema_version": "opal.plot_config.v1",
                        "config_path": str(campaign_yaml),
                        "plots": list_configured_plot_specs(
                            plots_cfg=plot_cfg.plots,
                            plot_presets=plot_cfg.plot_presets,
                        ),
                    }
                )
                return
            payload: dict[str, object] = {
                "schema_version": "opal.plot_cli_list.v1",
                "config_path": str(campaign_yaml),
            }
            if list_registry:
                payload["registered_plots"] = [describe_plot_kind(name) for name in list_plots()]
            if list_config:
                payload["configured_plots"] = list_configured_plot_specs(
                    plots_cfg=plot_cfg.plots,
                    plot_presets=plot_cfg.plot_presets,
                )
            json_out(payload)
            return
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
    configured_views = [selection_view.id for selection_view in cfg.selection_views]
    if view is None:
        if len(configured_views) != 1:
            raise OpalError(f"[plot] --view is required. Available: {configured_views}", ExitCodes.BAD_ARGS)
        view = configured_views[0]
    elif view not in configured_views:
        raise OpalError(f"[plot] Unknown --view {view!r}. Available: {configured_views}", ExitCodes.BAD_ARGS)
    selected_view = next(selection_view for selection_view in cfg.selection_views if selection_view.id == view)
    objective_name = selected_view.objective.name
    objective_family = get_objective_family(objective_name)

    try:
        rounds_sel = parse_round_selector(round)
    except OpalError as e:
        raise typer.BadParameter(str(e), param_hint="--round") from e
    if run_id:
        try:
            runs_df = analysis.read_runs()
        except OpalError as e:
            raise OpalError(f"[plot] {e}", e.exit_code) from e
        try:
            run_round = resolve_run_round(runs_df, run_id)
        except ValueError as e:
            raise OpalError(str(e), ExitCodes.BAD_ARGS) from e
        if rounds_sel == "all":
            raise OpalError(
                "[plot] Do not combine --run-id with --round all; run_id is single-round.",
                ExitCodes.BAD_ARGS,
            )
        if rounds_sel in ("unspecified", "latest"):
            rounds_sel = [run_round]
        elif isinstance(rounds_sel, list):
            if run_round not in rounds_sel:
                raise OpalError(
                    f"[plot] run_id {run_id!r} belongs to as_of_round={run_round}, but --round={round!r} excludes it.",
                    ExitCodes.BAD_ARGS,
                )
        else:
            if int(rounds_sel) != int(run_round):
                raise OpalError(
                    f"[plot] run_id {run_id!r} belongs to as_of_round={run_round}, "
                    f"but --round={round!r} selects a different round.",
                    ExitCodes.BAD_ARGS,
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
        selection_view_id=view,
        objective_name=objective_name,
        objective_family=objective_family,
        multi_view_campaign=len(configured_views) > 1,
        round_suffix=suffix,
        name_filter=name,
        tag_filters=tag_filters,
    )

    any_fail = run_plots(req)
    raise typer.Exit(code=1 if any_fail else 0)
