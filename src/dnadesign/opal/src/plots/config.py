"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/config.py

Parses plot configuration for OPAL campaigns and validates plot entries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config.loader import resolve_path_like

ALLOWED_PLOT_KEYS = {
    "name",
    "kind",
    "params",
    "output",
    "data",
    "enabled",
    "tags",
    "preset",
    "round_selector",
    "round_variants",
}
ALLOWED_PRESET_KEYS = {
    "kind",
    "params",
    "output",
    "data",
    "enabled",
    "tags",
    "round_selector",
    "round_variants",
}
ALLOWED_PLOT_DEFAULT_KEYS = {"output", "data", "params"}
ALLOWED_PLOT_CONFIG_KEYS = {"plots", "plot_defaults", "plot_presets"}


@dataclass(frozen=True)
class PlotConfig:
    plots: List[Dict[str, Any]]
    plot_defaults: Dict[str, Any]
    plot_presets: Dict[str, Dict[str, Any]]
    source_path: Path
    source_dir: Path
    source_label: str


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Campaign YAML did not parse to a mapping: {path}")
    return cfg


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
    plot_config_opt: Optional[Path],
) -> tuple[dict, Path, Path, str]:
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
        if not opt_path.exists():
            raise ValueError(f"[plot] plot_config not found: {opt_path}")
        plot_cfg = _load_yaml(opt_path)
        extra = sorted(set(plot_cfg.keys()) - ALLOWED_PLOT_CONFIG_KEYS)
        if extra:
            raise ValueError(f"[plot] Unknown top-level keys in plot config {opt_path}: {extra}")
        return plot_cfg, opt_path.resolve(), opt_path.parent.resolve(), "--plot-config"

    if cfg_plot_path:
        plot_cfg_path = resolve_path_like(campaign_yaml, cfg_plot_path)
        if not plot_cfg_path.exists():
            raise ValueError(f"[plot] plot_config not found: {plot_cfg_path}")
        plot_cfg = _load_yaml(plot_cfg_path)
        extra = sorted(set(plot_cfg.keys()) - ALLOWED_PLOT_CONFIG_KEYS)
        if extra:
            raise ValueError(f"[plot] Unknown top-level keys in plot config {plot_cfg_path}: {extra}")
        return (
            plot_cfg,
            plot_cfg_path.resolve(),
            plot_cfg_path.parent.resolve(),
            "campaign.plot_config",
        )

    raise ValueError("[plot] No plots found. Set campaign.plot_config or pass --plot-config.")


def load_plot_config(
    *,
    campaign_cfg: dict,
    campaign_yaml: Path,
    plot_config_opt: Optional[Path],
) -> PlotConfig:
    plot_cfg, plot_cfg_path, plot_cfg_dir, plot_src = _resolve_plot_config_source(
        campaign_cfg=campaign_cfg,
        campaign_yaml=campaign_yaml,
        plot_config_opt=plot_config_opt,
    )
    plots_cfg = plot_cfg.get("plots")
    if not isinstance(plots_cfg, list):
        raise ValueError(f"[plot] plots must be a list in {plot_cfg_path}.")
    plot_defaults = _parse_plot_defaults(plot_cfg.get("plot_defaults"), ctx=str(plot_cfg_path))
    plot_presets = _parse_plot_presets(plot_cfg.get("plot_presets"), ctx=str(plot_cfg_path))
    return PlotConfig(
        plots=plots_cfg,
        plot_defaults=plot_defaults,
        plot_presets=plot_presets,
        source_path=plot_cfg_path,
        source_dir=plot_cfg_dir,
        source_label=plot_src,
    )


def list_configured_plots(
    *,
    plots_cfg: List[Dict[str, Any]],
    plot_presets: Dict[str, Dict[str, Any]],
) -> List[str]:
    rows: List[str] = []
    for spec in list_configured_plot_specs(plots_cfg=plots_cfg, plot_presets=plot_presets):
        status = "disabled" if not spec["enabled"] else "enabled"
        tags = spec.get("tags") or []
        tag_str = f" tags={tags}" if tags else ""
        round_str = f" round_selector={spec['round_selector']}" if spec.get("round_selector") else ""
        variant_str = f" round_variants={spec['round_variants']}" if spec.get("round_variants") else ""
        rows.append(f"{spec['name']}: {spec['kind']} ({status}){tag_str}{round_str}{variant_str}")
    return rows


def list_configured_plot_specs(
    *,
    plots_cfg: List[Dict[str, Any]],
    plot_presets: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
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

        enabled = _parse_enabled(
            entry.get("enabled") if "enabled" in entry else preset.get("enabled"),
            ctx=pname,
        )
        tags = _parse_tags(preset.get("tags"), ctx=f"preset:{preset_name}") + _parse_tags(
            entry.get("tags"), ctx=f"plot:{pname}"
        )
        spec: Dict[str, Any] = {
            "name": pname,
            "kind": pkind,
            "enabled": enabled,
            "tags": tags,
        }
        if preset_name is not None:
            spec["preset"] = preset_name
        entry_round_selector = entry.get("round_selector", preset.get("round_selector"))
        if entry_round_selector is not None:
            spec["round_selector"] = str(entry_round_selector)
        entry_round_variants = entry.get("round_variants", preset.get("round_variants"))
        if entry_round_variants is not None:
            spec["round_variants"] = entry_round_variants
        specs.append(spec)
    return specs


def apply_data_entries(
    data_paths: Dict[str, Path],
    entries: Any,
    *,
    base_dir: Path,
    ctx: str,
) -> None:
    _apply_data_entries(data_paths, entries, base_dir=base_dir, ctx=ctx)


def parse_enabled(val: Any, *, ctx: str) -> bool:
    return _parse_enabled(val, ctx=ctx)


def parse_tags(val: Any, *, ctx: str) -> List[str]:
    return _parse_tags(val, ctx=ctx)


def validate_plot_entry(entry: Dict[str, Any], *, ctx: str) -> None:
    _validate_keys(entry, allowed=ALLOWED_PLOT_KEYS, ctx=ctx)
