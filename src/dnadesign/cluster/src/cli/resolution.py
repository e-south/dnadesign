"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/resolution.py

Cluster CLI boundary-resolution helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import typer
from rich.console import Console

from ..layout import ClusterLayoutError, explicit_results_root
from ..methods.params import parse_method_param_assignments
from ..presets.runtime import apply_preset
from ..workspaces import WorkspaceConfigError, load_workspace_config

LEGACY_FIT_METHOD_KEYS = frozenset({"neighbors", "resolution", "scale", "metric", "random_state", "backend"})
_MISSING = object()


@dataclass(frozen=True, slots=True)
class WorkspaceCommandContext:
    params: dict[str, Any]
    plot: dict[str, Any]
    workspace_id: str | None = None
    results_root: Path | None = None


def resolve_workspace_context(workspace: str | None, expected_section: str) -> WorkspaceCommandContext:
    if not workspace:
        return WorkspaceCommandContext(params={}, plot={})
    try:
        config = load_workspace_config(workspace)
    except (FileNotFoundError, WorkspaceConfigError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return WorkspaceCommandContext(
        params=config.section_params(expected_section),
        plot=config.section_plot(expected_section),
        workspace_id=config.workspace_id,
        results_root=config.results_root,
    )


def assert_no_method_overlap_with_preset(kind: str, job_params: Mapping[str, Any], preset_name: str | None) -> None:
    if not preset_name:
        return
    umap_keys = {"neighbors", "min_dist", "metric", "random_state"}
    banned = umap_keys if kind == "umap" else set()
    overlap = sorted(k for k in job_params.keys() if k in banned)
    if overlap:
        raise typer.BadParameter(
            f"Job provides {overlap} but also references a preset. "
            "Move method-specific knobs into the preset or pass them via CLI flags."
        )


def _job_method_params(job_params: Mapping[str, Any]) -> dict[str, Any]:
    legacy = sorted(k for k in job_params.keys() if k in LEGACY_FIT_METHOD_KEYS)
    if legacy:
        raise typer.BadParameter(
            "Legacy fit method keys are no longer accepted at the top level of workspace config params: "
            + ", ".join(legacy)
            + ". Move them under method_params or into the selected preset."
        )
    raw = job_params.get("method_params", {}) or {}
    if not isinstance(raw, dict):
        raise typer.BadParameter("Job 'method_params' must be a mapping.")
    return dict(raw)


def resolve_fit_method_params(
    job_params: Mapping[str, Any],
    cli_assignments: Sequence[str],
    *,
    preset_name: str | None = None,
) -> dict[str, Any]:
    params = _job_method_params(job_params)
    try:
        params.update(parse_method_param_assignments(list(cli_assignments)))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not preset_name or not params:
        return params
    preset_params = apply_preset("method", preset_name)
    overlap = sorted(set(params).intersection(preset_params))
    if overlap:
        raise typer.BadParameter(
            "Method params overlap with the selected preset: "
            + ", ".join(overlap)
            + ". Keep reusable method knobs in the preset or override them exclusively via --method-param."
        )
    return params


def resolve_cli_or_config_value(
    *,
    parameter_source: click.core.ParameterSource | None,
    cli_value: Any,
    config_value: Any,
) -> Any:
    if config_value is None:
        return cli_value
    if parameter_source in (None, click.core.ParameterSource.DEFAULT):
        return config_value
    return cli_value


def resolve_workspace_value(
    ctx: typer.Context,
    *,
    option_name: str,
    cli_value: Any,
    config_params: Mapping[str, Any],
    config_key: str | None = None,
    config_value: Any = _MISSING,
) -> Any:
    effective_config_value = config_params.get(config_key or option_name) if config_value is _MISSING else config_value
    return resolve_cli_or_config_value(
        parameter_source=ctx.get_parameter_source(option_name),
        cli_value=cli_value,
        config_value=effective_config_value,
    )


def runs_root_or_exit(
    *,
    console: Console,
    workspace_root: Path | None,
    results_root: str | None,
    materialize: bool = True,
) -> Path:
    from ..runs.store import runs_root

    if workspace_root is not None and results_root is not None:
        console.print("[red]Layout error:[/red] Pass either --workspace or --results-root, not both.")
        raise typer.Exit(code=2)
    root_spec: Path | str | None = workspace_root if workspace_root is not None else results_root
    try:
        return runs_root(root_spec) if materialize else explicit_results_root(root_spec)
    except ClusterLayoutError as exc:
        console.print(f"[red]Layout error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


__all__ = [
    "WorkspaceCommandContext",
    "assert_no_method_overlap_with_preset",
    "resolve_cli_or_config_value",
    "resolve_fit_method_params",
    "resolve_workspace_context",
    "resolve_workspace_value",
    "runs_root_or_exit",
]
