"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plotting.py

Plot runner and manifest writer for DenseGen diagnostics plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from rich.console import Console

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from ..core.artifacts.pool import POOL_MODE_TFBS, TFBSPoolArtifact, load_pool_artifact
from ..utils.rich_style import make_panel, make_table
from .plot_common import (  # noqa: F401
    _apply_style,
    _draw_tier_markers,
    _format_plot_path,
    _format_source_label,
    _palette,
)
from .plot_registry import PLOT_SPECS
from .plot_run import plot_run_health, plot_tfbs_usage
from .plot_stage_a import plot_stage_a_summary  # noqa: F401
from .plot_stage_a_strata import _build_stage_a_strata_overview_figure  # noqa: F401
from .plot_stage_a_yield import _build_stage_a_yield_bias_figure  # noqa: F401
from .plot_stage_b_placement import plot_placement_map

_console = Console()


def _plot_manifest_path(out_dir: Path) -> Path:
    return out_dir / "plot_manifest.json"


def _load_plot_manifest(out_dir: Path) -> dict:
    path = _plot_manifest_path(out_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_attempts(run_root: Path) -> pd.DataFrame:
    attempts_path = run_root / "outputs" / "tables" / "attempts.parquet"
    if not attempts_path.exists():
        raise ValueError(f"attempts.parquet not found: {attempts_path}")
    return pd.read_parquet(attempts_path)


def _load_events(run_root: Path) -> pd.DataFrame:
    events_path = run_root / "outputs" / "meta" / "events.jsonl"
    if not events_path.exists():
        raise ValueError(f"events.jsonl not found: {events_path}")
    rows = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _write_plot_manifest(
    out_dir: Path,
    *,
    entries: list[dict],
    run_root: Path,
    cfg_path: Path,
    source: str,
) -> None:
    existing = _load_plot_manifest(out_dir)
    merged: dict[str, dict] = {}
    for item in existing.get("plots", []):
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        if (out_dir / rel_path).exists():
            merged[rel_path] = item
    for item in entries:
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        merged[rel_path] = item
    payload = {
        "schema_version": "1.0",
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "plots": sorted(merged.values(), key=lambda x: (x.get("name", ""), x.get("path", ""))),
    }
    _plot_manifest_path(out_dir).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _ensure_out_dir(plots_cfg, cfg_path: Path, run_root: Path) -> Path:
    out_dir = plots_cfg.out_dir if plots_cfg else "outputs/plots"
    out = resolve_outputs_scoped_path(cfg_path, run_root, out_dir, label="plots.out_dir")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_stage_a_pools(run_root: Path) -> tuple[TFBSPoolArtifact, dict[str, pd.DataFrame]]:
    pools_dir = run_root / "outputs" / "pools"
    artifact = load_pool_artifact(pools_dir)
    pools: dict[str, pd.DataFrame] = {}
    for entry in artifact.inputs.values():
        if entry.pool_mode != POOL_MODE_TFBS:
            continue
        pool_path = pools_dir / entry.pool_path
        if not pool_path.exists():
            raise FileNotFoundError(f"Stage-A pool not found: {pool_path}")
        pools[entry.name] = pd.read_parquet(pool_path)
    if not pools:
        raise ValueError("No TFBS pools available for Stage-A plots.")
    return artifact, pools


# ---------------------- Plots ----------------------


def _maybe_load_stage_a_pools(run_root: Path) -> tuple[TFBSPoolArtifact | None, dict[str, pd.DataFrame] | None]:
    pools_dir = run_root / "outputs" / "pools"
    if not pools_dir.exists():
        return None, None
    return _load_stage_a_pools(run_root)


def _load_composition(run_root: Path) -> pd.DataFrame:
    path = run_root / "outputs" / "tables" / "composition.parquet"
    if not path.exists():
        raise ValueError(f"composition.parquet not found: {path}")
    return pd.read_parquet(path)


def _maybe_load_composition(run_root: Path) -> pd.DataFrame | None:
    path = run_root / "outputs" / "tables" / "composition.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_libraries(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    libs_dir = run_root / "outputs" / "libraries"
    builds_path = libs_dir / "library_builds.parquet"
    members_path = libs_dir / "library_members.parquet"
    if not builds_path.exists():
        raise ValueError(f"library_builds.parquet not found: {builds_path}")
    if not members_path.exists():
        raise ValueError(f"library_members.parquet not found: {members_path}")
    return pd.read_parquet(builds_path), pd.read_parquet(members_path)


def _maybe_load_libraries(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    libs_dir = run_root / "outputs" / "libraries"
    builds_path = libs_dir / "library_builds.parquet"
    members_path = libs_dir / "library_members.parquet"
    if not builds_path.exists() or not members_path.exists():
        return None
    return pd.read_parquet(builds_path), pd.read_parquet(members_path)


def _load_effective_config(run_root: Path) -> dict:
    path = run_root / "outputs" / "meta" / "effective_config.json"
    if not path.exists():
        raise ValueError(f"effective_config.json not found: {path}")
    return json.loads(path.read_text())


def _load_dense_arrays(run_root: Path) -> pd.DataFrame:
    path = run_root / "outputs" / "tables" / "dense_arrays.parquet"
    if not path.exists():
        raise ValueError(f"dense_arrays.parquet not found: {path}")
    return pd.read_parquet(path)


_PLOT_FNS = {
    "placement_map": plot_placement_map,
    "tfbs_usage": plot_tfbs_usage,
    "run_health": plot_run_health,
    "stage_a_summary": plot_stage_a_summary,
}

AVAILABLE_PLOTS: Dict[str, Dict[str, object]] = {}
for _name, _spec in PLOT_SPECS.items():
    _fn = _PLOT_FNS.get(_name)
    if _fn is None:
        raise RuntimeError(f"Plot function not registered for '{_name}'.")
    AVAILABLE_PLOTS[_name] = {
        "fn": _fn,
        "description": _spec.get("description", ""),
        "requires": _spec.get("requires"),
    }


# ---------------------- Runner with unknown-option filter ----------------------

# Options explicitly supported by each plot; unknown options raise errors (strict).
_ALLOWED_OPTIONS = {
    "placement_map": {"occupancy_alpha", "occupancy_max_categories", "tfbs_top_k_annotation"},
    "tfbs_usage": set(),
    "run_health": set(),
    "stage_a_summary": set(),
}


def _filter_kwargs(name: str, kwargs: dict) -> dict:
    allowed = _ALLOWED_OPTIONS.get(name)
    if allowed is None:
        raise ValueError(f"Unknown plot name: {name}")
    unknown = [
        k
        for k in list(kwargs.keys())
        if k not in allowed and k not in {"dims", "palette", "palette_no_repeat", "style"}
    ]
    if unknown:
        raise ValueError(f"Unknown options for plot '{name}': {unknown}")
    return kwargs


def _plot_required_sources(selected: Iterable[str]) -> set[str]:
    sources: set[str] = set()
    for name in selected:
        spec = AVAILABLE_PLOTS.get(name, {})
        requires = spec.get("requires")
        if requires:
            sources.update({str(item) for item in requires})
        else:
            sources.add("outputs")
    return sources


def _plot_required_columns(selected: Iterable[str], options: Dict[str, Dict[str, object]]) -> list[str]:
    return []


def run_plots_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    only: Optional[str] = None,
    source: str = "plot",
    absolute: bool = False,
) -> None:
    plots_cfg = root_cfg.plots
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_dir = _ensure_out_dir(plots_cfg, cfg_path, run_root)
    plot_format = plots_cfg.format if plots_cfg and getattr(plots_cfg, "format", None) else "pdf"
    default_list = plots_cfg.default if (plots_cfg and plots_cfg.default) else ["stage_a_summary", "placement_map"]
    selected = [p.strip() for p in (only.split(",") if only else default_list)]
    options = plots_cfg.options if plots_cfg else {}
    global_style = plots_cfg.style if plots_cfg else {}
    required_sources = _plot_required_sources(selected)
    cols: list[str] = []
    max_rows = plots_cfg.sample_rows if plots_cfg else None
    df = pd.DataFrame()
    src_label = "none"
    row_count = 0
    attempts_df: pd.DataFrame | None = None
    events_df: pd.DataFrame | None = None
    composition_df: pd.DataFrame | None = None
    dense_arrays_df: pd.DataFrame | None = None
    library_builds_df: pd.DataFrame | None = None
    library_members_df: pd.DataFrame | None = None
    cfg_effective: dict | None = None

    if "outputs" in required_sources:
        df, src_label = load_records_from_config(root_cfg, cfg_path, columns=cols, max_rows=max_rows)
        src_label = _format_source_label(src_label, run_root, absolute)
        row_count = len(df)
    if "composition" in required_sources:
        composition_df = _load_composition(run_root)
        if row_count == 0:
            row_count = len(composition_df)
            src_label = _format_source_label(
                f"composition:{run_root / 'outputs' / 'tables' / 'composition.parquet'}", run_root, absolute
            )
    if "libraries" in required_sources:
        library_builds_df, library_members_df = _load_libraries(run_root)
        if row_count == 0:
            row_count = len(library_members_df)
            src_label = _format_source_label(f"libraries:{run_root / 'outputs' / 'libraries'}", run_root, absolute)
    if "config" in required_sources:
        cfg_effective = _load_effective_config(run_root)
        if row_count == 0:
            row_count = 1
            src_label = _format_source_label(
                f"config:{run_root / 'outputs' / 'meta' / 'effective_config.json'}", run_root, absolute
            )
    if "attempts" in required_sources:
        attempts_df = _load_attempts(run_root)
        if row_count == 0:
            row_count = len(attempts_df)
            src_label = _format_source_label(
                f"attempts:{run_root / 'outputs' / 'tables' / 'attempts.parquet'}", run_root, absolute
            )
        events_path = run_root / "outputs" / "meta" / "events.jsonl"
        if events_path.exists():
            events_df = _load_events(run_root)
            if row_count == 0:
                row_count = len(events_df)
                src_label = _format_source_label(f"events:{events_path}", run_root, absolute)
    pools: dict[str, pd.DataFrame] | None = None
    pool_manifest: TFBSPoolArtifact | None = None
    if "pools" in required_sources:
        pool_manifest, pools = _load_stage_a_pools(run_root)
        if row_count == 0:
            row_count = sum(len(pool_df) for pool_df in pools.values())
            src_label = _format_source_label(f"pools:{run_root / 'outputs' / 'pools'}", run_root, absolute)
    if "tfbs_usage" in selected and pools is None:
        pool_manifest, pools = _maybe_load_stage_a_pools(run_root)
    if "tfbs_usage" in selected and library_members_df is None:
        libs = _maybe_load_libraries(run_root)
        if libs is not None:
            library_builds_df, library_members_df = libs
    if "dense_arrays" in required_sources:
        dense_arrays_df = _load_dense_arrays(run_root)
        if row_count == 0:
            row_count = len(dense_arrays_df)
            src_label = _format_source_label(
                f"dense_arrays:{run_root / 'outputs' / 'tables' / 'dense_arrays.parquet'}", run_root, absolute
            )

    out_label = _format_plot_path(out_dir, run_root, absolute)
    _console.print(
        make_panel(
            f"source: {src_label} | rows: {row_count:,}\nOutput: {out_label}",
            title="DenseGen plotting",
        )
    )
    summary = make_table("plot", "saved to", "status")
    errors: list[tuple[str, Exception]] = []
    manifest_entries: list[dict] = []

    for name in selected:
        if name not in AVAILABLE_PLOTS:
            raise ValueError(f"Unknown plot name requested: {name}")
        fn = AVAILABLE_PLOTS[name]["fn"]
        raw = (options.get(name, {}) or {}).copy()

        # absorb dims/palette into style
        dims = raw.pop("dims", None)
        style = {**global_style, **(raw.pop("style", {}) or {})}
        if dims:
            style["figsize"] = tuple(dims)
        pal_override = raw.pop("palette", None)
        if pal_override is not None:
            style["palette"] = pal_override
        if "palette_no_repeat" in raw:
            style["palette_no_repeat"] = bool(raw.pop("palette_no_repeat"))

        # drop unknown/retired options (e.g., promoter_scan_revcomp)
        kwargs = _filter_kwargs(name, raw)

        out_path = out_dir / f"{name}.{plot_format}"
        try:
            if name == "placement_map":
                result = fn(
                    df,
                    out_path,
                    style=style,
                    composition_df=composition_df,
                    dense_arrays_df=dense_arrays_df,
                    cfg=cfg_effective,
                    **kwargs,
                )
            elif name == "tfbs_usage":
                result = fn(
                    df,
                    out_path,
                    style=style,
                    composition_df=composition_df,
                    pools=pools,
                    library_members_df=library_members_df,
                    **kwargs,
                )
            elif name == "run_health":
                result = fn(df, out_path, style=style, attempts_df=attempts_df, events_df=events_df, **kwargs)
            elif name == "stage_a_summary":
                result = fn(df, out_path, style=style, pools=pools, pool_manifest=pool_manifest, **kwargs)
            else:
                result = fn(df, out_path, style=style, **kwargs)
            if result is None:
                paths = [out_path]
            elif isinstance(result, (list, tuple, set)):
                paths = [Path(p) for p in result]
            else:
                paths = [Path(result)]
            saved_label = _format_plot_path(paths[0], run_root, absolute)
            if len(paths) > 1:
                saved_label = f"{saved_label} (+{len(paths) - 1})"
            summary.add_row(name, saved_label, "[green]ok[/]")
            created_at = datetime.now(timezone.utc).isoformat()
            for path in paths:
                manifest_entries.append(
                    {
                        "name": name,
                        "path": str(path.relative_to(out_dir)),
                        "description": AVAILABLE_PLOTS[name]["description"],
                        "figsize": list(style.get("figsize", [])) if style.get("figsize") else None,
                        "generated_at": created_at,
                        "source": str(source),
                    }
                )
        except Exception as e:
            summary.add_row(name, "-", f"[red]failed[/] ({e})")
            errors.append((name, e))

    _console.print(summary)
    if errors:
        details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"{len(errors)} plot(s) failed: {details}")

    _write_plot_manifest(out_dir, entries=manifest_entries, run_root=run_root, cfg_path=cfg_path, source=source)
