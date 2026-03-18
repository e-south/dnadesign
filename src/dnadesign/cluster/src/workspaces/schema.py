"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/workspaces/schema.py

Workspace schema loading and validation for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import WorkspaceConfigError

SUPPORTED_WORKSPACE_SCHEMA_VERSION = 1
_RELATIVE_PATH_KEYS = frozenset({"file", "usr_root", "highlight", "out", "out_dir"})
_TOP_LEVEL_KEYS = frozenset({"schema_version", "input", "fit", "umap", "analyze"})
_INPUT_KEYS = frozenset({"dataset", "file", "usr_root", "key_col", "x_col", "x_cols"})
_FIT_KEYS = frozenset(
    {
        "name",
        "key_col",
        "x_col",
        "x_cols",
        "method",
        "preset",
        "method_params",
        "silhouette",
        "full_silhouette",
        "dedupe_policy",
        "reuse",
        "force",
        "write",
        "allow_overwrite",
        "inplace",
        "out",
        "plot",
    }
)
_UMAP_KEYS = frozenset(
    {
        "name",
        "key_col",
        "x_col",
        "x_cols",
        "neighbors",
        "min_dist",
        "metric",
        "random_state",
        "preset",
        "color_by",
        "highlight",
        "highlight_topn",
        "highlight_topn_col",
        "highlight_topn_asc",
        "highlight_hue_col",
        "alpha",
        "size",
        "dims",
        "font_scale",
        "opal_campaign",
        "opal_run",
        "opal_as_of_round",
        "opal_fields",
        "derive_ratio",
        "attach_coords",
        "write",
        "allow_overwrite",
        "inplace",
        "out",
        "plot",
    }
)
_ANALYZE_KEYS = frozenset(
    {
        "cluster_col",
        "group_by",
        "preset",
        "out_dir",
        "composition",
        "diversity",
        "difffeat",
        "plots",
        "numeric",
        "numeric_plots",
        "font_scale",
        "opal_campaign",
        "opal_as_of_round",
        "opal_fields",
        "plot",
    }
)
_UMAP_PLOT_KEYS = frozenset({"enabled", "alpha", "size", "dims", "font_scale", "color_by", "legend", "highlight"})
_ANALYZE_PLOT_KEYS = frozenset({"font_scale"})
_UMAP_LEGEND_KEYS = frozenset({"ncol", "bbox", "max_items", "frameon"})
_UMAP_HIGHLIGHT_KEYS = frozenset(
    {
        "overlay",
        "size",
        "size_multiplier",
        "alpha",
        "facecolor",
        "edgecolor",
        "linewidth",
        "marker",
        "legend",
        "palette",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedWorkspacePayload:
    schema_version: int
    input: dict[str, Any]
    fit: dict[str, Any]
    umap: dict[str, Any]
    analyze: dict[str, Any]
    fit_plot: dict[str, Any]
    umap_plot: dict[str, Any]
    analyze_plot: dict[str, Any]


def _normalize_dict_keys(d: dict[str, Any], *, path: tuple[str, ...] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seen_src: dict[str, str] = {}
    for key, value in d.items():
        normalized_key = key.replace("-", "_") if isinstance(key, str) else key
        if isinstance(value, dict):
            value = _normalize_dict_keys(value, path=(*path, str(normalized_key)))
        elif isinstance(value, list):
            value = [
                _normalize_dict_keys(item, path=(*path, str(normalized_key))) if isinstance(item, dict) else item
                for item in value
            ]
        if normalized_key in out and seen_src.get(str(normalized_key)) != key:
            here = ".".join(path) if path else "<root>"
            raise WorkspaceConfigError(
                f"Duplicate parameters after key normalization at {here}: "
                f"'{seen_src[str(normalized_key)]}' and '{key}' both map to '{normalized_key}'."
            )
        out[normalized_key] = value
        seen_src[str(normalized_key)] = key
    return out


def _resolve_relative_params(params: dict[str, Any], *, origin: Path) -> dict[str, Any]:
    resolved = dict(params)
    for key in _RELATIVE_PATH_KEYS:
        value = resolved.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            resolved[key] = str(candidate)
            continue
        resolved[key] = str((origin.parent / candidate).resolve())
    return resolved


def _reject_unknown_keys(
    section_name: str,
    section: dict[str, Any],
    *,
    allowed: frozenset[str],
    config_path: Path,
) -> None:
    unknown = sorted(set(section).difference(allowed))
    if unknown:
        allowed_list = ", ".join(sorted(allowed))
        raise WorkspaceConfigError(
            f"Workspace section '{section_name}' in {config_path} has unsupported keys: {', '.join(unknown)}. "
            f"Allowed keys: {allowed_list}."
        )


def _validate_input_source(input_section: dict[str, Any], *, config_path: Path) -> None:
    if input_section.get("dataset") and input_section.get("file"):
        raise WorkspaceConfigError(
            f"Workspace input in {config_path} must set exactly one of 'dataset' or 'file', not both."
        )


def _validate_method_params(fit_section: dict[str, Any], *, config_path: Path) -> None:
    raw = fit_section.get("method_params")
    if raw is not None and not isinstance(raw, dict):
        raise WorkspaceConfigError(f"Workspace fit.method_params must be a mapping in {config_path}.")


def _validated_plot_mapping(name: str, plot_cfg: Any, *, config_path: Path, allowed: frozenset[str]) -> dict[str, Any]:
    if plot_cfg in (None, {}):
        return {}
    if not isinstance(plot_cfg, dict):
        raise WorkspaceConfigError(f"Workspace section '{name}' must be a mapping in {config_path}.")
    _reject_unknown_keys(name, plot_cfg, allowed=allowed, config_path=config_path)
    return dict(plot_cfg)


def _validate_umap_plot(plot_cfg: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    legend = plot_cfg.get("legend")
    if legend not in (None, {}):
        if not isinstance(legend, dict):
            raise WorkspaceConfigError(f"Workspace section 'umap.plot.legend' must be a mapping in {config_path}.")
        _reject_unknown_keys("umap.plot.legend", legend, allowed=_UMAP_LEGEND_KEYS, config_path=config_path)
    highlight = plot_cfg.get("highlight")
    if highlight not in (None, {}):
        if not isinstance(highlight, dict):
            raise WorkspaceConfigError(f"Workspace section 'umap.plot.highlight' must be a mapping in {config_path}.")
        _reject_unknown_keys("umap.plot.highlight", highlight, allowed=_UMAP_HIGHLIGHT_KEYS, config_path=config_path)
    return plot_cfg


def load_workspace_payload(config_path: Path) -> ParsedWorkspacePayload:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise WorkspaceConfigError(f"Workspace config must be a mapping: {config_path}")

    config = _normalize_dict_keys(raw)
    _reject_unknown_keys("<root>", config, allowed=_TOP_LEVEL_KEYS, config_path=config_path)
    schema_version = int(config.get("schema_version", SUPPORTED_WORKSPACE_SCHEMA_VERSION))
    if schema_version != SUPPORTED_WORKSPACE_SCHEMA_VERSION:
        raise WorkspaceConfigError(
            f"Unsupported cluster workspace schema_version '{schema_version}' in {config_path}. "
            f"Expected {SUPPORTED_WORKSPACE_SCHEMA_VERSION}."
        )

    input_section = config.get("input", {}) or {}
    fit_section = config.get("fit", {}) or {}
    umap_section = config.get("umap", {}) or {}
    analyze_section = config.get("analyze", {}) or {}
    for name, section in {
        "input": input_section,
        "fit": fit_section,
        "umap": umap_section,
        "analyze": analyze_section,
    }.items():
        if not isinstance(section, dict):
            raise WorkspaceConfigError(f"Workspace section '{name}' must be a mapping in {config_path}.")
    _reject_unknown_keys("input", input_section, allowed=_INPUT_KEYS, config_path=config_path)
    _reject_unknown_keys("fit", fit_section, allowed=_FIT_KEYS, config_path=config_path)
    _reject_unknown_keys("umap", umap_section, allowed=_UMAP_KEYS, config_path=config_path)
    _reject_unknown_keys("analyze", analyze_section, allowed=_ANALYZE_KEYS, config_path=config_path)
    _validate_input_source(input_section, config_path=config_path)
    _validate_method_params(fit_section, config_path=config_path)

    fit_plot = _validated_plot_mapping(
        "fit.plot",
        fit_section.pop("plot", {}) or {},
        config_path=config_path,
        allowed=frozenset(),
    )
    umap_plot = _validate_umap_plot(
        _validated_plot_mapping(
            "umap.plot",
            umap_section.pop("plot", {}) or {},
            config_path=config_path,
            allowed=_UMAP_PLOT_KEYS,
        ),
        config_path=config_path,
    )
    analyze_plot = _validated_plot_mapping(
        "analyze.plot",
        analyze_section.pop("plot", {}) or {},
        config_path=config_path,
        allowed=_ANALYZE_PLOT_KEYS,
    )

    return ParsedWorkspacePayload(
        schema_version=schema_version,
        input=_resolve_relative_params(input_section, origin=config_path),
        fit=_resolve_relative_params(fit_section, origin=config_path),
        umap=_resolve_relative_params(umap_section, origin=config_path),
        analyze=_resolve_relative_params(analyze_section, origin=config_path),
        fit_plot=fit_plot,
        umap_plot=umap_plot,
        analyze_plot=analyze_plot,
    )


__all__ = ["ParsedWorkspacePayload", "SUPPORTED_WORKSPACE_SCHEMA_VERSION", "load_workspace_payload"]
