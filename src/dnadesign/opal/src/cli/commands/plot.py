"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/plot.py

OPAL CLI — plot

A lean driver that:
- resolves campaign.yaml (or campaign directory),
- builds a minimal PlotContext,
- auto-injects built-in data sources if present (records, outputs, ledger sinks),
- dispatches to plot plugins via the registry,
- writes outputs (overwrite by default),
- continues on errors and prints full tracebacks,
- exits with code 1 if any plot failed.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional, Union

import typer
import yaml

from ...plots._context import PlotContext
from ...registries.plot import get_plot
from ...workspace import CampaignWorkspace
from ..registry import cli_command
from ._common import load_cli_config, resolve_config_path, store_from_cfg


def _parse_round_selector(sel: Optional[str]) -> Union[str, list[int]]:
    if not sel:
        return "unspecified"
    sel = sel.strip().lower()
    if sel in {"latest", "all"}:
        return sel
    if "-" in sel:
        a, b = sel.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in sel:
        return [int(x) for x in sel.split(",") if x]
    return [int(sel)]


def _round_suffix(rounds: Union[str, list[int]]) -> str:
    if rounds == "unspecified":
        return ""
    if rounds == "latest":
        return "_rlatest"
    if rounds == "all":
        return "_rall"
    if isinstance(rounds, list) and len(rounds) == 1:
        return f"_r{rounds[0]}"
    if isinstance(rounds, list):
        return f"_r{','.join(map(str, rounds))}"
    return ""


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


def _load_campaign_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Campaign YAML did not parse to a mapping: {path}")
    return cfg


def _resolve_campaign_dir(config_path: Path) -> tuple[Path, dict, Path]:
    """
    Accept a YAML path or a directory. If directory, find a campaign YAML inside.
    """
    p = config_path
    if p.is_dir():
        for cand in ("campaign.yaml", "campaign.yml", "opal.yaml", "opal.yml"):
            c = p / cand
            if c.exists():
                return p.resolve(), _load_campaign_yaml(c), c.resolve()
        raise ValueError(f"No campaign YAML found in directory: {p}")
    # YAML file
    return p.parent.resolve(), _load_campaign_yaml(p), p.resolve()


@cli_command("plot", help="Generate plots defined under the campaign's top-level 'plots:' block.")
def cmd_plot(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to campaign.yaml or campaign directory."),
    round: Optional[str] = typer.Option(
        None,
        "--round",
        "-r",
        help="Round selector: latest | all | 3 | 1,3,7 | 2-5 (omitted = unspecified).",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Run a single plot by its 'name' in the YAML."),
) -> None:
    """
    Runs all plots by default (or a single plot via --name).
    Overwrites output files by default.
    Continues on error, printing full tracebacks.
    Exit code 1 if any plot failed.
    """
    # Resolve campaign.yaml
    cfg_path = resolve_config_path(config)
    campaign_dir, campaign_cfg, campaign_yaml = _resolve_campaign_dir(cfg_path)
    cfg = load_cli_config(campaign_yaml)
    store = store_from_cfg(cfg)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    typer.secho(f"[plot] Using config: {cfg_path}", fg=typer.colors.CYAN)

    plots_cfg = campaign_cfg.get("plots") or []
    if not isinstance(plots_cfg, list):
        raise typer.Exit(code=1)

    if name:
        filtered = [p for p in plots_cfg if p.get("name") == name]
        if not filtered:
            typer.echo(f"[plot] No plot with name '{name}' in campaign YAML.")
            raise typer.Exit(code=1)
        plots_cfg = filtered

    rounds_sel = _parse_round_selector(round)
    suffix = _round_suffix(rounds_sel)

    # Built-in data sources (auto-injected if present)
    builtins = {
        # Records path resolved from campaign data location
        "records": Path(store.records_path),
        # Workspace outputs and ledger sinks (always under outputs/)
        "outputs": ws.outputs_dir,
        "ledger_predictions_dir": ws.ledger_predictions_dir,
        "ledger_runs_parquet": ws.ledger_runs_path,
        "ledger_labels_parquet": ws.ledger_labels_path,
    }
    builtin_resolved = {k: p for k, p in builtins.items() if p.exists()}

    any_fail = False

    for entry in plots_cfg:
        try:
            pname = entry["name"]
            pkind = entry["kind"]
        except Exception:
            typer.echo("[plot] Each plot requires 'name' and 'kind'. Skipping invalid entry.")
            any_fail = True
            continue

        # Merge data paths: built-ins first, then YAML overrides
        data_paths = dict(builtin_resolved)
        for d in entry.get("data") or []:
            n = d.get("name")
            p = d.get("path")
            if not n or not p:
                raise ValueError(f"Invalid data entry in plot '{pname}': expected name+path")
            pp = Path(p)
            if not pp.is_absolute():
                pp = (campaign_dir / pp).resolve()
            data_paths[n] = pp

        out_cfg = entry.get("output") or {}
        # Resolve output directory (default: outputs/plots)
        out_dir = _resolve_output_dir(
            out_cfg,
            campaign_dir=campaign_dir,
            workspace=ws,
            plot_name=pname,
            plot_kind=pkind,
            round_suffix=suffix,
        )
        fmt = (out_cfg.get("format") or "png").lower()
        dpi = int(out_cfg.get("dpi", 600))
        fname = (out_cfg.get("filename") or "{name}{round_suffix}.png").format(name=pname, round_suffix=suffix)
        # Normalize extension to chosen format
        if not fname.lower().endswith(f".{fmt}"):
            base = fname.rsplit(".", 1)[0] if "." in fname else fname
            fname = f"{base}.{fmt}"
        save_data = bool(out_cfg.get("save_data", False))

        # Accept only mappings for params; rescue common mistakes.
        raw_params = entry.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            typer.echo(
                f"[plot] WARN: plot '{pname}' has a non-mapping 'params' "
                f"(type={type(raw_params).__name__}). Coercing to empty dict."
            )
            raw_params = {}
        params = dict(raw_params)

        # Optional strictness: refuse empty/non-mapping params when a 'params' block exists.
        import os

        STRICT = str(os.getenv("OPAL_PLOT_STRICT", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if STRICT and "params" in entry and not params:
            raise ValueError(f"[plot] plot '{pname}' has an empty or non-mapping 'params:' block in YAML.")

        # Convenience: if users put plotting keys at top-level, lift them into params.
        TOPLEVEL = {
            "hue",
            "hue_field",
            "color",
            "color_by",
            "colour_by",
            "size",
            "size_by",
            "size_field",
            "point_size_by",
            "alpha",
            "rank_mode",
            "score_field",
            "threshold",
            "mode",
            "delta",
            "cmap",
            "rasterize_at",
            "figsize_in",
            "size_min",
            "size_max",
            # swarm/violin & geometry knobs used by built-ins
            "swarm",
            "swarm_max_points",
            "swarm_jitter",
            "swarm_alpha",
            "violin",
            "violin_alpha",
            "panel_size_in",
        }
        lifted = {k: entry[k] for k in TOPLEVEL if k in entry}
        for k, v in lifted.items():
            params.setdefault(k, v)
        if lifted:
            typer.echo(
                f"[plot] Note: moved top-level keys {sorted(lifted)} into 'params' "
                f"for plot '{pname}'. (Place them under 'params:' to silence this.)"
            )
            if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                raise ValueError(f"Plot '{pname}' has plotting keys at the top level. Move them under 'params:'.")

        # Build context
        import logging

        logger = logging.getLogger(f"opal.plot.{pname}")
        logger.setLevel(logging.INFO)

        # Ensure a handler so plugin logger.info lines are visible
        if not logger.handlers:
            try:
                from rich.logging import RichHandler

                h = RichHandler(rich_tracebacks=False, markup=True, show_path=False, show_time=False)
            except Exception:
                h = logging.StreamHandler()
            h.setLevel(logging.INFO)
            if not isinstance(h, logging.StreamHandler):  # i.e., RichHandler
                h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(h)
            logger.propagate = False

        ctx = PlotContext(
            campaign_dir=campaign_dir,
            workspace=ws,
            rounds=rounds_sel,
            data_paths=data_paths,
            output_dir=Path(out_dir),
            filename=fname,
            dpi=dpi,
            format=fmt,
            logger=logger,
            save_data=save_data,
        )

        # Run plugin (overwrite outputs by default)
        try:
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            typer.secho(
                f"[plot] entry '{pname}': keys={sorted(entry.keys())} "
                f"params_type={type(entry.get('params')).__name__} "
                f"params_preview={ {k: entry['params'].get(k) for k in (entry.get('params') or {}).keys()} if isinstance(entry.get('params'), dict) else '(not a dict)' }",  # noqa
                fg=typer.colors.BLUE,
            )

            get_plot(pkind)(ctx, params)
            typer.secho(
                f"[ok] {pname} ({pkind}) → {ctx.output_dir / ctx.filename}",
                fg=typer.colors.GREEN,
            )
        except Exception:  # full traceback always
            any_fail = True
            typer.secho(f"[fail] {pname} ({pkind})", fg=typer.colors.RED)
            traceback.print_exc()

    raise typer.Exit(code=1 if any_fail else 0)
