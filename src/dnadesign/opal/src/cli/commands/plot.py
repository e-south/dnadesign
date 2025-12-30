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

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
import yaml

from ...config.loader import resolve_path_like
from ...plots._context import PlotContext
from ...registries.plot import get_plot
from ...workspace import CampaignWorkspace
from ..registry import cli_command
from ._common import load_cli_config, print_config_context, resolve_config_path, store_from_cfg


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


ALLOWED_PLOT_KEYS = {
    "name",
    "kind",
    "params",
    "output",
    "data",
    "enabled",
    "tags",
    "preset",
}
ALLOWED_PRESET_KEYS = {
    "kind",
    "params",
    "output",
    "data",
    "enabled",
    "tags",
}
ALLOWED_PLOT_DEFAULT_KEYS = {"output", "data", "params"}
ALLOWED_PLOT_CONFIG_KEYS = {"plots", "plot_defaults", "plot_presets"}


def _ensure_mapping(val: Any, *, ctx: str) -> Dict[str, Any]:
    if not isinstance(val, dict):
        raise ValueError(f"[plot] {ctx} must be a mapping (got {type(val).__name__}).")
    return val


def _ensure_list(val: Any, *, ctx: str) -> List[Any]:
    if not isinstance(val, list):
        raise ValueError(f"[plot] {ctx} must be a list (got {type(val).__name__}).")
    return val


def _validate_keys(obj: Dict[str, Any], *, allowed: set[str], ctx: str) -> None:
    extra = sorted(set(obj.keys()) - allowed)
    if extra:
        raise ValueError(f"[plot] Unknown keys in {ctx}: {extra}. Allowed: {sorted(allowed)}")


def _parse_enabled(val: Any, *, ctx: str) -> bool:
    if val is None:
        return True
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "yes", "on", "1"}:
            return True
        if v in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"[plot] Invalid 'enabled' value for {ctx}: {val!r} (expected boolean).")


def _parse_tags(val: Any, *, ctx: str) -> List[str]:
    if val is None:
        return []
    if not isinstance(val, list):
        raise ValueError(f"[plot] {ctx}.tags must be a list of strings.")
    tags: List[str] = []
    for t in val:
        if not isinstance(t, str):
            raise ValueError(f"[plot] {ctx}.tags must contain only strings (got {type(t).__name__}).")
        tags.append(t)
    return tags


def _apply_data_entries(
    data_paths: Dict[str, Path],
    entries: Any,
    *,
    base_dir: Path,
    ctx: str,
) -> None:
    if entries is None:
        return
    _ensure_list(entries, ctx=ctx)
    for d in entries:
        if not isinstance(d, dict):
            raise ValueError(f"[plot] {ctx} entries must be mappings.")
        n = d.get("name")
        p = d.get("path")
        if not n or not p:
            raise ValueError(f"[plot] Invalid data entry in {ctx}: expected name+path.")
        pp = Path(p)
        if not pp.is_absolute():
            pp = (base_dir / pp).resolve()
        data_paths[str(n)] = pp


def _parse_plot_defaults(raw: Any, *, ctx: str) -> Dict[str, Any]:
    if raw is None:
        return {"output": {}, "data": [], "params": {}}
    d = _ensure_mapping(raw, ctx=f"{ctx}.plot_defaults")
    _validate_keys(d, allowed=ALLOWED_PLOT_DEFAULT_KEYS, ctx=f"{ctx}.plot_defaults")
    out = d.get("output") or {}
    if not isinstance(out, dict):
        raise ValueError(f"[plot] {ctx}.plot_defaults.output must be a mapping.")
    data = d.get("data") or []
    if not isinstance(data, list):
        raise ValueError(f"[plot] {ctx}.plot_defaults.data must be a list.")
    params = d.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"[plot] {ctx}.plot_defaults.params must be a mapping.")
    return {"output": dict(out), "data": list(data), "params": dict(params)}


def _parse_plot_presets(raw: Any, *, ctx: str) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    presets = _ensure_mapping(raw, ctx=f"{ctx}.plot_presets")
    for name, preset in presets.items():
        if not isinstance(preset, dict):
            raise ValueError(f"[plot] {ctx}.plot_presets.{name} must be a mapping.")
        if "name" in preset:
            raise ValueError(f"[plot] plot preset '{name}' must not define 'name'.")
        _validate_keys(preset, allowed=ALLOWED_PRESET_KEYS, ctx=f"{ctx}.plot_presets.{name}")
    return presets  # type: ignore[return-value]


def _resolve_plot_config_source(
    *,
    campaign_cfg: dict,
    campaign_yaml: Path,
    campaign_dir: Path,
    plot_config_opt: Optional[Path],
) -> tuple[dict, Path, Path, str]:
    inline_plots = campaign_cfg.get("plots", None)
    inline_defaults = campaign_cfg.get("plot_defaults", None)
    inline_presets = campaign_cfg.get("plot_presets", None)
    inline_present = inline_plots is not None or inline_defaults is not None or inline_presets is not None
    cfg_plot_path = campaign_cfg.get("plot_config", None)

    opt_path: Optional[Path] = None
    if plot_config_opt is not None:
        opt_path = Path(plot_config_opt).expanduser()
        opt_path = opt_path if opt_path.is_absolute() else (Path.cwd() / opt_path)
        opt_path = opt_path.resolve()

    if opt_path is not None and cfg_plot_path:
        cfg_path = resolve_path_like(campaign_yaml, cfg_plot_path)
        if cfg_path.resolve() != opt_path.resolve():
            raise ValueError(
                f"[plot] Both --plot-config and campaign.plot_config are set and differ: {opt_path} vs {cfg_path}."
            )

    if opt_path is not None:
        if inline_present:
            raise ValueError("[plot] Remove inline plots/plot_defaults/plot_presets when using --plot-config.")
        if not opt_path.exists():
            raise ValueError(f"[plot] plot_config not found: {opt_path}")
        plot_cfg = _load_campaign_yaml(opt_path)
        extra = sorted(set(plot_cfg.keys()) - ALLOWED_PLOT_CONFIG_KEYS)
        if extra:
            raise ValueError(f"[plot] Unknown top-level keys in plot config {opt_path}: {extra}")
        return plot_cfg, opt_path.resolve(), opt_path.parent.resolve(), "--plot-config"

    if cfg_plot_path:
        if inline_present:
            raise ValueError("[plot] Remove inline plots/plot_defaults/plot_presets when using campaign.plot_config.")
        plot_cfg_path = resolve_path_like(campaign_yaml, cfg_plot_path)
        if not plot_cfg_path.exists():
            raise ValueError(f"[plot] plot_config not found: {plot_cfg_path}")
        plot_cfg = _load_campaign_yaml(plot_cfg_path)
        extra = sorted(set(plot_cfg.keys()) - ALLOWED_PLOT_CONFIG_KEYS)
        if extra:
            raise ValueError(f"[plot] Unknown top-level keys in plot config {plot_cfg_path}: {extra}")
        return plot_cfg, plot_cfg_path.resolve(), plot_cfg_path.parent.resolve(), "campaign.plot_config"

    if inline_plots is None:
        raise ValueError("[plot] No plots found. Provide a plots list or set plot_config.")

    plot_cfg = {
        "plots": inline_plots,
        "plot_defaults": inline_defaults,
        "plot_presets": inline_presets,
    }
    return plot_cfg, campaign_yaml.resolve(), campaign_dir.resolve(), "campaign.yaml"


@cli_command("plot", help="Generate plots from plot_config (preferred) or inline 'plots:'.")
def cmd_plot(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to campaign.yaml or campaign directory."),
    plot_config: Optional[Path] = typer.Option(
        None,
        "--plot-config",
        help="Path to plots YAML (overrides campaign.plot_config).",
    ),
    round: Optional[str] = typer.Option(
        None,
        "--round",
        "-r",
        help="Round selector: latest | all | 3 | 1,3,7 | 2-5 (omitted = unspecified).",
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
    # Resolve campaign.yaml
    cfg_path = resolve_config_path(config, allow_dir=True)
    campaign_dir, campaign_cfg, campaign_yaml = _resolve_campaign_dir(cfg_path)
    cfg = load_cli_config(campaign_yaml)
    store = store_from_cfg(cfg)
    ws = CampaignWorkspace.from_config(cfg, campaign_yaml)
    print_config_context(campaign_yaml, cfg=cfg, records_path=store.records_path)

    plot_cfg, plot_cfg_path, plot_cfg_dir, plot_src = _resolve_plot_config_source(
        campaign_cfg=campaign_cfg,
        campaign_yaml=campaign_yaml,
        campaign_dir=campaign_dir,
        plot_config_opt=plot_config,
    )
    plots_cfg = plot_cfg.get("plots")
    if not isinstance(plots_cfg, list):
        raise ValueError(f"[plot] plots must be a list in {plot_cfg_path}.")
    plot_defaults = _parse_plot_defaults(plot_cfg.get("plot_defaults"), ctx=str(plot_cfg_path))
    plot_presets = _parse_plot_presets(plot_cfg.get("plot_presets"), ctx=str(plot_cfg_path))

    if name:
        filtered = [p for p in plots_cfg if isinstance(p, dict) and p.get("name") == name]
        if not filtered:
            raise ValueError(f"[plot] No plot with name '{name}' in {plot_src}.")
        plots_cfg = filtered

    tag_filters = [str(t) for t in (tag or [])]

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
        if not isinstance(entry, dict):
            raise ValueError(f"[plot] Each plot entry must be a mapping (got {type(entry).__name__}).")
        _validate_keys(entry, allowed=ALLOWED_PLOT_KEYS, ctx="plot entry")

        preset: Dict[str, Any] = {}
        preset_name = entry.get("preset")
        if preset_name is not None:
            if not isinstance(preset_name, str):
                raise ValueError(f"[plot] plot preset name must be a string (got {type(preset_name).__name__}).")
            if preset_name not in plot_presets:
                raise ValueError(f"[plot] Unknown plot preset: {preset_name!r}")
            preset = plot_presets.get(preset_name) or {}

        pname = entry.get("name")
        if not pname or not isinstance(pname, str):
            raise ValueError("[plot] Each plot requires a string 'name'.")

        pkind = entry.get("kind") or preset.get("kind")
        if not pkind or not isinstance(pkind, str):
            raise ValueError(f"[plot] Plot '{pname}' is missing 'kind' (or preset kind).")

        enabled = _parse_enabled(entry.get("enabled") if "enabled" in entry else preset.get("enabled"), ctx=pname)
        if not enabled:
            if name:
                raise ValueError(f"[plot] Plot '{pname}' is disabled (enabled: false).")
            typer.echo(f"[plot] Skipping disabled plot: {pname}")
            continue

        tags = _parse_tags(preset.get("tags"), ctx=f"preset:{preset_name}") + _parse_tags(
            entry.get("tags"), ctx=f"plot:{pname}"
        )
        if tag_filters:
            if not set(tags).intersection(tag_filters):
                if name:
                    raise ValueError(f"[plot] Plot '{pname}' does not match tags: {tag_filters}")
                continue

        # Merge data paths: built-ins first, then defaults/preset/entry overrides
        data_paths = dict(builtin_resolved)
        _apply_data_entries(data_paths, plot_defaults.get("data"), base_dir=plot_cfg_dir, ctx="plot_defaults.data")
        if preset:
            _apply_data_entries(
                data_paths, preset.get("data"), base_dir=plot_cfg_dir, ctx=f"plot_presets.{preset_name}.data"
            )
        _apply_data_entries(data_paths, entry.get("data"), base_dir=plot_cfg_dir, ctx=f"plot '{pname}'.data")

        # Merge output defaults → preset → entry
        preset_out = preset.get("output") or {}
        if preset and not isinstance(preset_out, dict):
            raise ValueError(f"[plot] plot_presets.{preset_name}.output must be a mapping.")
        entry_out = entry.get("output")
        if entry_out is None:
            entry_out = {}
        if not isinstance(entry_out, dict):
            raise ValueError(f"[plot] plot '{pname}' output must be a mapping.")
        out_cfg = {
            **(plot_defaults.get("output") or {}),
            **preset_out,
            **entry_out,
        }
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

        # Accept only mappings for params; fail fast on ambiguous configs.
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
            **(plot_defaults.get("params") or {}),
            **preset_params,
            **entry_params,
        }

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
            debug = str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")
            if debug:
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
