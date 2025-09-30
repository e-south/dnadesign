"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/plot.py

OPAL CLI — plot

A lean driver that:
- resolves campaign.yaml (or campaign directory),
- builds a minimal PlotContext,
- auto-injects built-in data sources if present (events, records, artifacts),
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
from ..registry import cli_command
from ._common import resolve_config_path


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


def _load_campaign_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Campaign YAML did not parse to a mapping: {path}")
    return cfg


def _resolve_campaign_dir(config_path: Path) -> tuple[Path, dict]:
    """
    Accept a YAML path or a directory. If directory, find a campaign YAML inside.
    """
    p = config_path
    if p.is_dir():
        for cand in ("campaign.yaml", "campaign.yml", "opal.yaml", "opal.yml"):
            c = p / cand
            if c.exists():
                return p.resolve(), _load_campaign_yaml(c)
        raise ValueError(f"No campaign YAML found in directory: {p}")
    # YAML file
    return p.parent.resolve(), _load_campaign_yaml(p)


@cli_command(
    "plot", help="Generate plots defined under the campaign's top-level 'plots:' block."
)
def cmd_plot(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to campaign.yaml or campaign directory."
    ),
    round: Optional[str] = typer.Option(
        None,
        "--round",
        "-r",
        help="Round selector: latest | all | 3 | 1,3,7 | 2-5 (omitted = unspecified).",
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Run a single plot by its 'name' in the YAML."
    ),
) -> None:
    """
    Runs all plots by default (or a single plot via --name).
    Overwrites output files by default.
    Continues on error, printing full tracebacks.
    Exit code 1 if any plot failed.
    """
    # Resolve campaign.yaml
    cfg_path = resolve_config_path(config)
    campaign_dir, campaign_cfg = _resolve_campaign_dir(cfg_path)

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
        # Canonical ledger thin index (SoT for plots):
        "events": campaign_dir / "outputs" / "ledger.index.parquet",
        # Optional convenience if a local records cache is colocated with campaign:
        "records": campaign_dir / "records.parquet",
        "artifacts": campaign_dir / "artifacts",
    }
    builtin_resolved = {k: p for k, p in builtins.items() if p.exists()}

    any_fail = False

    for entry in plots_cfg:
        try:
            pname = entry["name"]
            pkind = entry["kind"]
        except Exception:
            typer.echo(
                "[plot] Each plot requires 'name' and 'kind'. Skipping invalid entry."
            )
            any_fail = True
            continue

        # Merge data paths: built-ins first, then YAML overrides
        data_paths = dict(builtin_resolved)
        for d in entry.get("data") or []:
            n = d.get("name")
            p = d.get("path")
            if not n or not p:
                raise ValueError(
                    f"Invalid data entry in plot '{pname}': expected name+path"
                )
            pp = Path(p)
            if not pp.is_absolute():
                pp = (campaign_dir / pp).resolve()
            data_paths[n] = pp

        out_cfg = entry.get("output") or {}
        # Force a flat output directory for all plots
        out_dir = (campaign_dir / "outputs" / "plots").resolve()
        fmt = (out_cfg.get("format") or "png").lower()
        dpi = int(out_cfg.get("dpi", 600))
        fname = (out_cfg.get("filename") or "{name}{round_suffix}.png").format(
            name=pname, round_suffix=suffix
        )
        # Normalize extension to chosen format
        if not fname.lower().endswith(f".{fmt}"):
            base = fname.rsplit(".", 1)[0] if "." in fname else fname
            fname = f"{base}.{fmt}"
        save_data = bool(out_cfg.get("save_data", False))

        params = entry.get("params") or {}

        # Build context
        import logging

        logger = logging.getLogger(f"opal.plot.{pname}")
        logger.setLevel(logging.INFO)

        ctx = PlotContext(
            campaign_dir=campaign_dir,
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
            get_plot(pkind)(ctx, params)
            typer.echo(f"[ok] {pname} ({pkind}) → {ctx.output_dir / ctx.filename}")
        except Exception:  # full traceback always
            any_fail = True
            typer.echo(f"[fail] {pname} ({pkind})")
            traceback.print_exc()

    raise typer.Exit(code=1 if any_fail else 0)
