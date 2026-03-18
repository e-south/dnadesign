"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_umap.py

UMAP execution runtime for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from .execution_support import (
    CommandExecution,
    _log,
    _rule,
    append_command_record_or_warn,
    context_and_df,
    progress_scope,
)
from .io.read import extract_X
from .runs.contracts import EmbeddingRun, RunCounts, utc_now_iso
from .runs.recorder import CommandRecord, record_umap_run
from .runs.signatures import UmapSignature
from .runs.store import umap_run_dir
from .runtime_contracts import FeatureSpec, InputSource
from .umap.frame import prepare_umap_frame
from .umap.hues import resolve_hue as umap_resolve_hue
from .umap.overlays import write_umap_overlays
from .umap.requests import resolve_umap_request
from .util.checks import assert_no_duplicate_ids
from .util.slug import artifact_slug


def run_umap(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    name: str,
    key_col: str,
    x_col: str | None,
    x_cols: str | None,
    neighbors: int | None,
    min_dist: float | None,
    metric: str | None,
    random_state: int | None,
    preset: str | None,
    color_by: list[str],
    highlight: str | None,
    highlight_topn: int | None,
    highlight_topn_col: str | None,
    highlight_topn_asc: bool,
    highlight_hue_col: str | None,
    alpha: float | None,
    size: float | None,
    dims: str | None,
    font_scale: float | None,
    opal_campaign: str | None,
    opal_run: str | None,
    opal_as_of_round: int | None,
    opal_fields: str | None,
    derive_ratio: list[str],
    attach_coords: bool,
    write: bool,
    allow_overwrite: bool,
    inplace: bool,
    out: str | None,
    root: Path,
    workspace_id: str | None = None,
    workspace_params: dict[str, Any] | None = None,
    workspace_plot: dict[str, Any] | None = None,
    console: Console | None = None,
    render_plots: bool | None = None,
) -> CommandExecution:
    from .umap.compute import compute as umap_compute

    workspace_params = workspace_params or {}
    workspace_plot = workspace_plot or {}

    ictx, df = context_and_df(dataset, file, usr_root)
    _rule(console, "[bold]cluster umap[/]")
    df = assert_no_duplicate_ids(df, key_col=key_col, policy="error")
    attach_base_df = df.copy(deep=False)
    source = InputSource.from_context(ictx)
    feature_spec = FeatureSpec.from_inputs(x_col=x_col, x_cols=x_cols)
    if df.index.name != key_col:
        df = df.set_index(key_col, drop=False)

    request = resolve_umap_request(
        df=df,
        key_col=key_col,
        preset=preset,
        neighbors=neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        color_by=color_by,
        highlight=highlight,
        highlight_topn=highlight_topn,
        highlight_topn_col=highlight_topn_col,
        highlight_topn_asc=highlight_topn_asc,
        highlight_hue_col=highlight_hue_col,
        alpha=alpha,
        size=size,
        dims=dims,
        font_scale=font_scale,
        render_plots=render_plots,
        workspace_params=workspace_params,
        workspace_plot=workspace_plot,
        console=console,
    )
    df, derived_cols = prepare_umap_frame(
        df,
        name=name,
        key_col=key_col,
        color_by=request.color_by,
        highlight_payload=request.highlight_payload,
        opal_campaign=opal_campaign,
        opal_run=opal_run,
        opal_as_of_round=opal_as_of_round,
        opal_fields=opal_fields,
        derive_ratio=derive_ratio,
        resolve_hue_fn=umap_resolve_hue,
        console=console,
    )

    with progress_scope(console) as progress:
        build_task = progress.add_task("Preparing X...", total=None)
        X = extract_X(
            df,
            x_col=feature_spec.columns[0] if feature_spec.mode == "single_col" else None,
            x_cols=list(feature_spec.columns) if feature_spec.mode == "multi_col" else None,
        )
        progress.update(build_task, completed=1)
        umap_task = progress.add_task("Computing UMAP...", total=None)
        coords = umap_compute(
            X,
            neighbors=request.neighbors,
            min_dist=request.min_dist,
            metric=request.metric,
            seed=request.random_state,
        )
        progress.update(umap_task, completed=1)

    umap_params = {
        "neighbors": request.neighbors,
        "min_dist": request.min_dist,
        "metric": request.metric,
        "random_state": request.random_state,
    }
    umap_signature = UmapSignature(params=umap_params, libs={})
    created_utc = utc_now_iso()
    umap_slug = artifact_slug(name, created_utc=created_utc, fingerprint=umap_signature.hash())
    run_dir = root / name
    umap_dir = umap_run_dir(root, name, umap_slug)
    plot_root: Path | None = None
    if request.render_plots:
        from .umap.plot import scatter as umap_scatter

        out_path = umap_dir / f"{name}.png"
        umap_scatter(
            coords,
            df if df.index.name == "id" else df.set_index(key_col, drop=False),
            color_specs=list(request.color_by),
            name=name,
            highlight=request.highlight_payload,
            alpha=request.alpha,
            size=request.size,
            dims=request.dims,
            legend=request.legend,
            out_path=out_path,
            font_scale=request.font_scale,
            overlay_highlight=True,
            highlight_style=request.highlight_style,
        )
        plot_root = umap_dir
    coords_df = pd.DataFrame({"id": df.index.astype(str), "umap_x": coords[:, 0], "umap_y": coords[:, 1]})
    embedding_run = EmbeddingRun(
        alias=name,
        slug=umap_slug,
        created_utc=created_utc,
        source=source,
        feature=feature_spec,
        counts=RunCounts(n_rows=int(len(df))),
        params=umap_params,
        signature=umap_signature,
    )
    record_umap_run(root=root, artifact_dir=umap_dir, run=embedding_run, coords_df=coords_df, plot_root=plot_root)

    write_umap_overlays(
        ictx=ictx,
        attach_base_df=attach_base_df,
        df=df,
        name=name,
        key_col=key_col,
        coords=coords,
        derived_cols=derived_cols,
        attach_coords=attach_coords,
        write=write,
        allow_overwrite=allow_overwrite,
        inplace=inplace,
        out=out,
        console=console,
    )

    if request.render_plots:
        _log(console, "print", f"[green]Saved[/green] {len(request.color_by)} UMAP PNG(s) to {umap_dir}")
    else:
        _log(
            console,
            "print",
            f"[green]Saved[/green] UMAP coords to {umap_dir} [dim](plot rendering disabled)[/dim]",
        )
    append_command_record_or_warn(
        run_dir,
        CommandRecord(
            command="umap",
            subject=name,
            workspace=workspace_id,
            preset=preset or None,
            resolved={
                "name": name,
                "neighbors": neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
                "plot": {
                    "enabled": request.render_plots,
                    "alpha": request.alpha,
                    "size": request.size,
                    "dims": list(request.dims),
                    "font_scale": request.font_scale,
                    "legend": request.legend,
                    "color_by": list(request.color_by),
                    "highlight": request.highlight_style,
                },
            },
        ),
        console=console,
    )
    return CommandExecution(command="umap", subject=name, artifact_path=umap_dir, run_record_subject=name)


__all__ = ["run_umap"]
