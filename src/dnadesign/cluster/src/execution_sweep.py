"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_sweep.py

Resolution-sweep execution runtime for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .execution_support import (
    CommandExecution,
    _log,
    append_command_record_or_warn,
    context_and_df,
    resolve_scoped_out_dir,
)
from .io.read import extract_X
from .methods.registry import get_method
from .presets.runtime import apply_preset
from .runs.contracts import SweepRun, utc_now_iso
from .runs.recorder import CommandRecord, record_sweep_run
from .runs.signatures import MethodSignature
from .runs.store import sweep_run_dir
from .runtime_contracts import FeatureSpec, InputSource
from .util.checks import assert_no_duplicate_ids
from .util.slug import artifact_slug, slugify


def run_sweep(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    key_col: str,
    x_col: str | None,
    x_cols: str | None,
    method: str,
    preset: str | None,
    method_params: dict[str, Any],
    res_min: float,
    res_max: float,
    step: float,
    replicates: int,
    seeds: str,
    out_dir: str | None,
    root: Path,
    workspace_id: str | None = None,
    console: Console | None = None,
) -> CommandExecution:
    ictx, df = context_and_df(dataset, file, usr_root)
    df = assert_no_duplicate_ids(df, key_col=key_col, policy="error")
    feature_spec = FeatureSpec.from_inputs(x_col=x_col, x_cols=x_cols)
    X = extract_X(
        df,
        x_col=feature_spec.columns[0] if feature_spec.mode == "single_col" else None,
        x_cols=list(feature_spec.columns) if feature_spec.mode == "multi_col" else None,
    )
    method_spec = get_method(method)
    sweep_operation = method_spec.get_operation("resolution_sweep")
    if sweep_operation is None:
        raise typer.BadParameter(f"Method '{method}' does not expose a resolution sweep contract.")
    preset_params = apply_preset("method", preset)
    resolved_method_params = method_spec.resolve_fit_params(preset=preset_params, raw_params=method_params)
    sweep_alias = slugify(
        f"{method_spec.method_id}-sweep-r{res_min:g}-{res_max:g}-step{step:g}-"
        f"n{int(resolved_method_params.get('neighbors', 0))}"
    )
    created_utc = utc_now_iso()
    seed_values = (
        [int(x.strip()) for x in seeds.split(",")]
        if isinstance(seeds, str) and seeds.strip()
        else list(range(1, int(replicates) + 1))
    )
    sweep_slug = artifact_slug(
        sweep_alias,
        created_utc=created_utc,
        fingerprint=json.dumps(
            {
                "method": method_spec.method_id,
                "params": resolved_method_params,
                "res_min": float(res_min),
                "res_max": float(res_max),
                "step": float(step),
                "seeds": seed_values,
            },
            sort_keys=True,
        ),
    )
    default_out_dir = sweep_run_dir(root, sweep_alias, sweep_slug) if out_dir is None else None
    out_path = resolve_scoped_out_dir(requested=out_dir, root=root) if out_dir is not None else default_out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    sweep_operation(
        X,
        method_params=resolved_method_params,
        res_min=res_min,
        res_max=res_max,
        step=step,
        seeds=seed_values,
        out_dir=out_path,
    )
    sweep_run = SweepRun(
        alias=sweep_alias,
        slug=sweep_slug,
        created_utc=created_utc,
        source=InputSource.from_context(ictx),
        feature=feature_spec,
        method_signature=MethodSignature(method_id=method_spec.method_id, params=resolved_method_params, libs={}),
        res_min=float(res_min),
        res_max=float(res_max),
        step=float(step),
        seeds=tuple(int(seed) for seed in seed_values),
    )
    record_sweep_run(root=root, out_dir=out_path, run=sweep_run)
    append_command_record_or_warn(
        out_path,
        CommandRecord(
            command="sweep",
            subject=sweep_alias,
            workspace=workspace_id,
            preset=preset or None,
            resolved={
                "method": method_spec.method_id,
                "res_min": float(res_min),
                "res_max": float(res_max),
                "step": float(step),
                "seeds": list(seed_values),
                "out_dir": str(out_path),
                **resolved_method_params,
            },
        ),
        console=console,
    )
    _log(console, "print", f"[green]Saved sweep results[/green] to {out_path}")
    return CommandExecution(
        command="sweep",
        subject=sweep_alias,
        artifact_path=out_path,
        run_record_subject=sweep_alias,
    )


__all__ = ["run_sweep"]
