"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/runner.py

Executes configured plots for OPAL campaigns using PlotContext. Owns plot
dispatch, output resolution, and error handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import typer

from ..analysis.facade import RoundSelector
from ..plots._context import PlotContext
from ..plots._mpl_utils import ensure_mpl_config_dir
from ..registries.plots import get_plot
from ..storage.data_access import RecordsStore
from ..storage.workspace import CampaignWorkspace
from .config import apply_data_entries, parse_enabled, parse_tags, validate_plot_entry


@dataclass(frozen=True)
class PlotRequest:
    plots_cfg: List[Dict[str, Any]]
    plot_defaults: Dict[str, Any]
    plot_presets: Dict[str, Dict[str, Any]]
    plot_cfg_dir: Path
    campaign_dir: Path
    workspace: CampaignWorkspace
    store: RecordsStore
    rounds_sel: RoundSelector
    run_id: Optional[str]
    round_suffix: str
    name_filter: Optional[str]
    tag_filters: List[str]


def resolve_run_round(runs_df: pl.DataFrame, run_id: str) -> int:
    if runs_df.is_empty():
        raise ValueError("[plot] outputs/ledger/runs.parquet is empty; cannot resolve run_id.")
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise ValueError("[plot] outputs/ledger/runs.parquet missing required columns (run_id, as_of_round).")
    df = runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("as_of_round").drop_nulls().unique())
    if df.is_empty():
        raise ValueError(f"[plot] run_id not found in outputs/ledger/runs.parquet: {run_id!r}.")
    rounds = sorted({int(x) for x in df.to_series().to_list()})
    if len(rounds) > 1:
        raise ValueError(f"[plot] run_id {run_id!r} appears in multiple rounds {rounds}.")
    return rounds[0]


def _resolve_output_dir(
    out_cfg: dict,
    *,
    campaign_dir: Path,
    workspace: CampaignWorkspace,
    plot_name: str,
    plot_kind: str,
    round_suffix: str,
) -> Path:
    out_dir_tpl = out_cfg.get("dir")
    if out_dir_tpl:
        out_dir_str = str(out_dir_tpl).format(
            campaign=str(campaign_dir),
            workdir=str(workspace.workdir),
            name=plot_name,
            kind=plot_kind,
            round_suffix=round_suffix,
        )
        out_dir = Path(out_dir_str)
        if not out_dir.is_absolute():
            out_dir = (campaign_dir / out_dir).resolve()
        else:
            out_dir = out_dir.resolve()
    else:
        out_dir = (campaign_dir / "outputs" / "plots").resolve()
    return out_dir


def run_plots(req: PlotRequest) -> bool:
    ensure_mpl_config_dir(workdir=req.workspace.workdir)

    builtins = {
        "records": Path(req.store.records_path),
        "outputs": req.workspace.outputs_dir,
        "ledger_predictions_dir": req.workspace.ledger_predictions_dir,
        "ledger_runs_parquet": req.workspace.ledger_runs_path,
        "ledger_labels_parquet": req.workspace.ledger_labels_path,
    }
    builtin_resolved = {k: p for k, p in builtins.items() if p.exists()}

    any_fail = False

    for entry in req.plots_cfg:
        if not isinstance(entry, dict):
            raise ValueError(f"[plot] Each plot entry must be a mapping (got {type(entry).__name__}).")
        validate_plot_entry(entry, ctx="plot entry")

        preset: Dict[str, Any] = {}
        preset_name = entry.get("preset")
        if preset_name is not None:
            if not isinstance(preset_name, str):
                raise ValueError(f"[plot] plot preset name must be a string (got {type(preset_name).__name__}).")
            if preset_name not in req.plot_presets:
                raise ValueError(f"[plot] Unknown plot preset: {preset_name!r}")
            preset = req.plot_presets.get(preset_name) or {}

        pname = entry.get("name")
        if not pname or not isinstance(pname, str):
            raise ValueError("[plot] Each plot requires a string 'name'.")

        if req.name_filter and pname != req.name_filter:
            continue

        pkind = entry.get("kind") or preset.get("kind")
        if not pkind or not isinstance(pkind, str):
            raise ValueError(f"[plot] Plot '{pname}' is missing 'kind' (or preset kind).")

        enabled = parse_enabled(
            entry.get("enabled") if "enabled" in entry else preset.get("enabled"),
            ctx=pname,
        )
        if not enabled:
            if req.name_filter:
                raise ValueError(f"[plot] Plot '{pname}' is disabled (enabled: false).")
            typer.echo(f"[plot] Skipping disabled plot: {pname}")
            continue

        tags = parse_tags(preset.get("tags"), ctx=f"preset:{preset_name}") + parse_tags(
            entry.get("tags"), ctx=f"plot:{pname}"
        )
        if req.tag_filters:
            if not set(tags).intersection(req.tag_filters):
                if req.name_filter:
                    raise ValueError(f"[plot] Plot '{pname}' does not match tags: {req.tag_filters}")
                continue

        data_paths = dict(builtin_resolved)
        apply_data_entries(
            data_paths,
            req.plot_defaults.get("data"),
            base_dir=req.plot_cfg_dir,
            ctx="plot_defaults.data",
        )
        if preset:
            apply_data_entries(
                data_paths,
                preset.get("data"),
                base_dir=req.plot_cfg_dir,
                ctx=f"plot_presets.{preset_name}.data",
            )
        apply_data_entries(
            data_paths,
            entry.get("data"),
            base_dir=req.plot_cfg_dir,
            ctx=f"plot '{pname}'.data",
        )

        preset_out = preset.get("output") or {}
        if preset and not isinstance(preset_out, dict):
            raise ValueError(f"[plot] plot_presets.{preset_name}.output must be a mapping.")
        entry_out = entry.get("output")
        if entry_out is None:
            entry_out = {}
        if not isinstance(entry_out, dict):
            raise ValueError(f"[plot] plot '{pname}' output must be a mapping.")
        out_cfg = {
            **(req.plot_defaults.get("output") or {}),
            **preset_out,
            **entry_out,
        }
        out_dir = _resolve_output_dir(
            out_cfg,
            campaign_dir=req.campaign_dir,
            workspace=req.workspace,
            plot_name=pname,
            plot_kind=pkind,
            round_suffix=req.round_suffix,
        )
        fmt = (out_cfg.get("format") or "png").lower()
        dpi = int(out_cfg.get("dpi", 600))
        fname = (out_cfg.get("filename") or "{name}{round_suffix}.png").format(
            name=pname,
            round_suffix=req.round_suffix,
        )
        if not fname.lower().endswith(f".{fmt}"):
            base = fname.rsplit(".", 1)[0] if "." in fname else fname
            fname = f"{base}.{fmt}"
        save_data = bool(out_cfg.get("save_data", False))

        raw_params = entry.get("params", None)
        if raw_params is None:
            if "params" in entry:
                raise ValueError(f"[plot] plot '{pname}' has an empty 'params:' block. Use {{}} or remove it.")
            entry_params = {}
        elif not isinstance(raw_params, dict):
            raise ValueError(f"[plot] plot '{pname}' has a non-mapping 'params' (type={type(raw_params).__name__}).")
        else:
            entry_params = dict(raw_params)

        preset_params = preset.get("params") or {}
        if preset and not isinstance(preset_params, dict):
            raise ValueError(f"[plot] plot_presets.{preset_name}.params must be a mapping.")

        params = {
            **(req.plot_defaults.get("params") or {}),
            **preset_params,
            **entry_params,
        }

        import logging

        logger = logging.getLogger(f"opal.plot.{pname}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            try:
                from rich.logging import RichHandler

                h = RichHandler(rich_tracebacks=False, markup=True, show_path=False, show_time=False)
            except Exception:
                h = logging.StreamHandler()
            h.setLevel(logging.INFO)
            if not isinstance(h, logging.StreamHandler):
                h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(h)
            logger.propagate = False

        ctx = PlotContext(
            campaign_dir=req.campaign_dir,
            workspace=req.workspace,
            rounds=req.rounds_sel,
            run_id=req.run_id,
            data_paths=data_paths,
            output_dir=Path(out_dir),
            filename=fname,
            dpi=dpi,
            format=fmt,
            logger=logger,
            save_data=save_data,
        )

        try:
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            debug = str(os.getenv("OPAL_DEBUG", "")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if debug:
                if isinstance(entry.get("params"), dict):
                    params_preview = {k: entry["params"].get(k) for k in (entry.get("params") or {}).keys()}
                else:
                    params_preview = "(not a dict)"
                typer.secho(
                    f"[plot] entry '{pname}': keys={sorted(entry.keys())} "
                    f"params_type={type(entry.get('params')).__name__} "
                    f"params_preview={params_preview}",
                    fg=typer.colors.BLUE,
                )

            get_plot(pkind)(ctx, params)
            typer.secho(
                f"[ok] {pname} ({pkind}) → {ctx.output_dir / ctx.filename}",
                fg=typer.colors.GREEN,
            )
        except Exception:
            any_fail = True
            typer.secho(f"[fail] {pname} ({pkind})", fg=typer.colors.RED)
            traceback.print_exc()

    return any_fail
