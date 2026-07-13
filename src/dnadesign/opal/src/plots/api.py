"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/api.py

Public helpers for generating configured OPAL plot artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..analysis.campaign import CampaignAnalysis
from ..analysis.ledger import parse_round_selector, round_suffix
from ..core.utils import OpalError
from .config import load_plot_config
from .runner import PlotRequest, resolve_run_round, run_plots


def run_campaign_plots(
    config_path: str | Path | None,
    *,
    plot_config_path: str | Path | None = None,
    round_selector: str | None = None,
    run_id: str | None = None,
    selection_view_id: str | None = None,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Generate configured plots for one campaign in the current Python process."""
    analysis = CampaignAnalysis.from_config_path(Path(config_path) if config_path is not None else None, allow_dir=True)
    cfg_path = analysis.config_path
    campaign_dir = analysis.workspace.workdir
    campaign_cfg = analysis.read_config_dict()
    store = analysis.records_store()
    workspace = analysis.workspace
    configured_views = [view.id for view in analysis.config.selection_views]
    if selection_view_id is None:
        if len(configured_views) != 1:
            raise OpalError(f"selection_view_id is required. Available: {configured_views}")
        selection_view_id = configured_views[0]
    elif selection_view_id not in configured_views:
        raise OpalError(f"Unknown selection_view_id {selection_view_id!r}. Available: {configured_views}")
    plot_cfg = load_plot_config(
        campaign_cfg=campaign_cfg,
        campaign_yaml=cfg_path,
        plot_config_opt=Path(plot_config_path) if plot_config_path is not None else None,
    )

    try:
        rounds_sel = parse_round_selector(round_selector)
    except OpalError:
        raise
    if run_id:
        runs_df = analysis.read_runs()
        run_round = resolve_run_round(runs_df, str(run_id))
        if rounds_sel == "all":
            raise ValueError("[plot] Do not combine --run-id with --round all; run_id is single-round.")
        if rounds_sel in ("unspecified", "latest"):
            rounds_sel = [run_round]
        elif isinstance(rounds_sel, list):
            if run_round not in rounds_sel:
                raise ValueError(
                    f"[plot] run_id {run_id!r} belongs to as_of_round={run_round}, "
                    f"but --round={round_selector!r} excludes it."
                )
        elif int(rounds_sel) != int(run_round):
            raise ValueError(
                f"[plot] run_id {run_id!r} belongs to as_of_round={run_round}, "
                f"but --round={round_selector!r} selects a different round."
            )

    suffix = round_suffix(rounds_sel)
    request = PlotRequest(
        plots_cfg=plot_cfg.plots,
        plot_defaults=plot_cfg.plot_defaults,
        plot_presets=plot_cfg.plot_presets,
        plot_cfg_dir=plot_cfg.source_dir,
        campaign_dir=campaign_dir,
        workspace=workspace,
        store=store,
        rounds_sel=rounds_sel,
        run_id=run_id,
        selection_view_id=selection_view_id,
        multi_view_campaign=len(configured_views) > 1,
        round_suffix=suffix,
        name_filter=name,
        tag_filters=[str(tag) for tag in (tags or [])],
        emit_status=not quiet,
    )
    any_fail = run_plots(request)
    return {
        "schema_version": "opal.plot_run.v1",
        "config_path": str(cfg_path),
        "campaign_dir": str(campaign_dir),
        "round_selector": round_selector or "unspecified",
        "round_suffix": suffix,
        "run_id": run_id,
        "selection_view_id": selection_view_id,
        "any_fail": bool(any_fail),
        "plot_manifest_path": str(
            campaign_dir
            / "outputs"
            / "plots"
            / (Path("selection_views") / selection_view_id if len(configured_views) > 1 else Path())
            / "plot_manifest.json"
        ),
    }
