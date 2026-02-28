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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from typing_extensions import Literal

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from ..core.artifacts.pool import POOL_MODE_TFBS, TFBSPoolArtifact, load_pool_artifact
from ..utils.rich_style import make_panel, make_table
from .dense_array_video import plot_dense_array_video_showcase
from .plot_common import (  # noqa: F401
    _apply_style,
    _draw_tier_markers,
    _format_plot_path,
    _format_source_label,
    _palette,
    plan_group_from_name,
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


def _is_supported_plot_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if not parts:
        return False
    if parts[0] == "stage_a":
        return len(parts) >= 2
    if parts[0] == "stage_b":
        return len(parts) >= 3
    if parts[0] == "run_health":
        return len(parts) >= 2
    return False


def _read_columns(columns: Iterable[str] | None) -> list[str] | None:
    if columns is None:
        return None
    cleaned = sorted({str(col).strip() for col in columns if str(col).strip()})
    return cleaned or None


def _resolve_composition_projection_columns(
    path: Path,
    columns: list[str] | None,
) -> tuple[list[str] | None, dict[str, str]]:
    if columns is None:
        return None, {}
    try:
        available = set(pq.read_schema(path).names)
    except Exception:
        return columns, {}
    read_cols: list[str] = []
    aliases: dict[str, str] = {}
    missing: list[str] = []
    for col in columns:
        if col == "tf":
            if "regulator" in available:
                read_cols.append("regulator")
                aliases["tf"] = "regulator"
            elif "tf" in available:
                read_cols.append("tf")
            else:
                missing.append("tf")
            continue
        if col == "tfbs":
            if "sequence" in available:
                read_cols.append("sequence")
                aliases["tfbs"] = "sequence"
            elif "tfbs" in available:
                read_cols.append("tfbs")
            else:
                missing.append("tfbs")
            continue
        if col in available:
            read_cols.append(col)
            continue
        missing.append(col)
    if missing:
        raise ValueError(
            "composition.parquet missing required columns: "
            f"{sorted(set(missing))}. Available columns: {sorted(available)}"
        )
    return sorted(set(read_cols)), aliases


def _read_composition_parquet(path: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    requested = _read_columns(columns)
    read_columns, aliases = _resolve_composition_projection_columns(path, requested)
    frame = pd.read_parquet(path, columns=read_columns)
    for dest, source in aliases.items():
        if dest not in frame.columns and source in frame.columns:
            frame[dest] = frame[source]
    return frame


def _load_attempts(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    attempts_path = run_root / "outputs" / "tables" / "attempts.parquet"
    if not attempts_path.exists():
        raise ValueError(f"attempts.parquet not found: {attempts_path}")
    return pd.read_parquet(attempts_path, columns=_read_columns(columns))


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
        if (out_dir / rel_path).exists() and _is_supported_plot_path(rel_path):
            merged[rel_path] = item
    for item in entries:
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        merged[rel_path] = item
    payload = {
        "schema_version": "2.0",
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


def _read_projected_parquet(path: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    requested = _read_columns(columns)
    if requested is None:
        return pd.read_parquet(path)
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(path).schema.names)
    projected = [name for name in requested if name in schema_names]
    if not projected:
        raise ValueError(f"Parquet file is missing all projected columns: {path}")
    return pd.read_parquet(path, columns=projected)


def _resolve_pool_path(pools_dir: Path, rel_path: Path, *, input_name: str) -> Path:
    candidate = pools_dir / rel_path
    root = pools_dir.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Pool path for input '{input_name}' escapes outputs/pools: {rel_path}") from exc
    return candidate


def _load_stage_a_pools(
    run_root: Path,
    *,
    columns: Iterable[str] | None = None,
) -> tuple[TFBSPoolArtifact, dict[str, pd.DataFrame]]:
    pools_dir = run_root / "outputs" / "pools"
    artifact = load_pool_artifact(pools_dir)
    pools: dict[str, pd.DataFrame] = {}
    for entry in artifact.inputs.values():
        if entry.pool_mode != POOL_MODE_TFBS:
            continue
        pool_path = _resolve_pool_path(pools_dir, entry.pool_path, input_name=entry.name)
        if not pool_path.exists():
            raise FileNotFoundError(f"Stage-A pool not found: {pool_path}")
        pools[entry.name] = _read_projected_parquet(pool_path, columns=columns)
    if not pools:
        raise ValueError("No TFBS pools available for Stage-A plots.")
    return artifact, pools


# ---------------------- Plots ----------------------


def _maybe_load_stage_a_pools(
    run_root: Path,
    *,
    columns: Iterable[str] | None = None,
) -> tuple[TFBSPoolArtifact | None, dict[str, pd.DataFrame] | None]:
    pools_dir = run_root / "outputs" / "pools"
    if not pools_dir.exists():
        return None, None
    return _load_stage_a_pools(run_root, columns=columns)


def _load_composition(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = run_root / "outputs" / "tables" / "composition.parquet"
    if not path.exists():
        raise ValueError(f"composition.parquet not found: {path}")
    return _read_composition_parquet(path, columns=columns)


def _maybe_load_composition(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame | None:
    path = run_root / "outputs" / "tables" / "composition.parquet"
    if not path.exists():
        return None
    return _read_composition_parquet(path, columns=columns)


def _load_libraries(
    run_root: Path,
    *,
    builds_columns: Iterable[str] | None = None,
    members_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    libs_dir = run_root / "outputs" / "libraries"
    builds_path = libs_dir / "library_builds.parquet"
    members_path = libs_dir / "library_members.parquet"
    if not builds_path.exists():
        raise ValueError(f"library_builds.parquet not found: {builds_path}")
    if not members_path.exists():
        raise ValueError(f"library_members.parquet not found: {members_path}")
    return (
        pd.read_parquet(builds_path, columns=_read_columns(builds_columns)),
        pd.read_parquet(members_path, columns=_read_columns(members_columns)),
    )


def _maybe_load_libraries(
    run_root: Path,
    *,
    builds_columns: Iterable[str] | None = None,
    members_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    libs_dir = run_root / "outputs" / "libraries"
    builds_path = libs_dir / "library_builds.parquet"
    members_path = libs_dir / "library_members.parquet"
    if not builds_path.exists() or not members_path.exists():
        return None
    return (
        pd.read_parquet(builds_path, columns=_read_columns(builds_columns)),
        pd.read_parquet(members_path, columns=_read_columns(members_columns)),
    )


def _load_effective_config(run_root: Path) -> dict:
    path = run_root / "outputs" / "meta" / "effective_config.json"
    if not path.exists():
        raise ValueError(f"effective_config.json not found: {path}")
    return json.loads(path.read_text())


def _load_dense_arrays(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = run_root / "outputs" / "tables" / "records.parquet"
    if path.exists():
        return pd.read_parquet(path, columns=_read_columns(columns))
    raise ValueError(f"records.parquet not found: {path}")


_PLOT_FNS = {
    "dense_array_video_showcase": plot_dense_array_video_showcase,
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
    "dense_array_video_showcase": set(),
    "placement_map": {"occupancy_alpha", "occupancy_max_categories", "scope", "max_plans", "drilldown_plans"},
    "tfbs_usage": {"scope", "max_plans", "drilldown_plans"},
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


@dataclass(frozen=True)
class StageBScopeOptions:
    scope: Literal["auto", "per_plan", "per_group"] = "auto"
    max_plans: int = 12
    drilldown_plans: int = 0


def _parse_stage_b_scope_options(raw: dict | None) -> StageBScopeOptions:
    payload = dict(raw or {})
    if not payload:
        return StageBScopeOptions()
    scope = str(payload.get("scope", "auto")).strip() or "auto"
    if scope not in {"auto", "per_plan", "per_group"}:
        raise ValueError(f"Invalid scope={scope!r}; expected auto|per_plan|per_group")
    try:
        max_plans = int(payload.get("max_plans", 12))
    except Exception as exc:
        raise ValueError("max_plans must be an integer > 0") from exc
    if max_plans <= 0:
        raise ValueError("max_plans must be > 0")
    try:
        drilldown_plans = int(payload.get("drilldown_plans", 0))
    except Exception as exc:
        raise ValueError("drilldown_plans must be an integer >= 0") from exc
    if drilldown_plans < 0:
        raise ValueError("drilldown_plans must be >= 0")
    return StageBScopeOptions(scope=scope, max_plans=max_plans, drilldown_plans=drilldown_plans)


def _clean_plot_subdir(out_dir: Path, subdir: str) -> None:
    target = out_dir / subdir
    if target.exists():
        shutil.rmtree(target)


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


_OUTPUT_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "dense_array_video_showcase": {"id", "sequence", "densegen__plan", "densegen__used_tfbs_detail"},
    "placement_map": {"id", "sequence", "densegen__input_name", "densegen__plan"},
    "run_health": {"densegen__compression_ratio", "densegen__plan"},
}
_COMPOSITION_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_map": {"solution_id", "input_name", "plan_name", "tf", "tfbs", "offset", "length", "end"},
    "tfbs_usage": {"input_name", "plan_name", "tf", "tfbs"},
    "run_health": {"input_name", "plan_name", "tf", "tfbs", "length"},
}
_ATTEMPT_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "run_health": {"status", "reason", "plan_name", "created_at", "detail_json"},
}
_LIBRARY_BUILDS_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_map": {"library_index", "library_hash", "input_name", "plan_name"},
    "tfbs_usage": {"library_index", "library_hash", "input_name", "plan_name"},
    "run_health": {"library_index", "library_hash", "input_name", "plan_name"},
}
_LIBRARY_MEMBERS_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_map": {"input_name", "plan_name", "tf", "tfbs"},
    "tfbs_usage": {"input_name", "plan_name", "tf", "tfbs"},
    "run_health": {"input_name", "plan_name", "tf", "tfbs"},
}
_DENSE_ARRAY_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_map": {"id", "sequence", "densegen__input_name", "densegen__plan"},
}
_POOL_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "stage_a_summary": {
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "tfbs_core",
        "sequence",
        "best_hit_score",
        "tier",
        "selection_score_norm",
        "nearest_selected_distance_norm",
        "selection_rank",
    }
}


def _required_columns_for_selected(
    selected: Iterable[str],
    *,
    mapping: Dict[str, set[str]],
) -> list[str]:
    columns: set[str] = set()
    for name in selected:
        columns.update(mapping.get(str(name), set()))
    return sorted(columns)


def _plot_required_columns(selected: Iterable[str], options: Dict[str, Dict[str, object]]) -> list[str]:
    del options
    return _required_columns_for_selected(selected, mapping=_OUTPUT_COLUMNS_BY_PLOT)


def _resolve_selected_plot_names(*, only: Optional[str], default_list: list[str]) -> list[str]:
    raw_selected = only.split(",") if only else list(default_list)
    selected = [str(name).strip() for name in raw_selected if str(name).strip()]
    if not selected:
        raise ValueError("No plot names selected. Use 'dense ls-plots' to list available plots.")
    unknown = [name for name in selected if name not in AVAILABLE_PLOTS]
    if unknown:
        valid = ", ".join(sorted(AVAILABLE_PLOTS))
        raise ValueError(
            f"Unknown plot name requested: {unknown[0]}. Valid plot names: {valid}. "
            "Use 'dense ls-plots' to list available plots."
        )
    return selected


def _cleanup_legacy_flat_outputs(out_dir: Path, selected: Iterable[str], plot_format: str) -> None:
    selected_set = {str(name) for name in selected}
    suffix = f".{plot_format.lstrip('.')}"
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        if path.name == "plot_manifest.json":
            continue
        if path.suffix != suffix:
            continue
        stem = path.stem
        remove = False
        if "stage_a_summary" in selected_set and stem.startswith("stage_a_summary__"):
            remove = True
        elif "placement_map" in selected_set and stem.startswith("placement_map__"):
            remove = True
        elif "tfbs_usage" in selected_set and stem.startswith("tfbs_usage__"):
            remove = True
        elif "run_health" in selected_set and stem == "run_health":
            remove = True
        if remove:
            path.unlink(missing_ok=True)


def _should_group_stage_b_plans(
    *,
    plan_names: list[str],
    scope_options: StageBScopeOptions,
) -> bool:
    if not plan_names:
        return False
    if scope_options.scope == "per_plan":
        return False
    grouped_names = [plan_group_from_name(name) for name in plan_names]
    grouped_unique = {name for name in grouped_names if name}
    if scope_options.scope == "per_group":
        return True
    if len(set(plan_names)) <= int(scope_options.max_plans):
        return False
    return len(grouped_unique) < len(set(plan_names))


def _map_stage_b_plan_group(
    *,
    dense_arrays_df: pd.DataFrame | None,
    composition_df: pd.DataFrame | None,
    library_members_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    def _normalize_pool_input(input_name: str, plan_name: str) -> str:
        raw_input = str(input_name or "").strip()
        grouped_plan = str(plan_name or "").strip()
        if not raw_input or not grouped_plan:
            return raw_input
        if raw_input.startswith("plan_pool__"):
            return f"plan_pool__{grouped_plan}"
        return raw_input

    dense_scoped = None
    if dense_arrays_df is not None:
        dense_scoped = dense_arrays_df.copy()
        if "densegen__plan" in dense_scoped.columns:
            grouped_plan = dense_scoped["densegen__plan"].astype(str).map(plan_group_from_name)
            dense_scoped["densegen__plan"] = grouped_plan
            if "densegen__input_name" in dense_scoped.columns:
                dense_scoped["densegen__input_name"] = [
                    _normalize_pool_input(input_name, plan_name)
                    for input_name, plan_name in zip(
                        dense_scoped["densegen__input_name"].astype(str),
                        grouped_plan,
                    )
                ]
    composition_scoped = None
    if composition_df is not None:
        composition_scoped = composition_df.copy()
        if "plan_name" in composition_scoped.columns:
            grouped_plan = composition_scoped["plan_name"].astype(str).map(plan_group_from_name)
            composition_scoped["plan_name"] = grouped_plan
            if "input_name" in composition_scoped.columns:
                composition_scoped["input_name"] = [
                    _normalize_pool_input(input_name, plan_name)
                    for input_name, plan_name in zip(
                        composition_scoped["input_name"].astype(str),
                        grouped_plan,
                    )
                ]
    library_members_scoped = None
    if library_members_df is not None:
        library_members_scoped = library_members_df.copy()
        if "plan_name" in library_members_scoped.columns:
            grouped_plan = library_members_scoped["plan_name"].astype(str).map(plan_group_from_name)
            library_members_scoped["plan_name"] = grouped_plan
            if "input_name" in library_members_scoped.columns:
                library_members_scoped["input_name"] = [
                    _normalize_pool_input(input_name, plan_name)
                    for input_name, plan_name in zip(
                        library_members_scoped["input_name"].astype(str),
                        grouped_plan,
                    )
                ]
    return dense_scoped, composition_scoped, library_members_scoped


def _top_stage_b_drilldown_plans(
    *,
    dense_arrays_df: pd.DataFrame | None,
    composition_df: pd.DataFrame | None,
    limit: int,
) -> list[str]:
    if int(limit) <= 0:
        return []
    counts_df: pd.DataFrame | None = None
    if dense_arrays_df is not None and "densegen__plan" in dense_arrays_df.columns:
        counts_df = (
            dense_arrays_df["densegen__plan"]
            .astype(str)
            .value_counts(dropna=True)
            .rename_axis("plan")
            .reset_index(name="count")
        )
    elif composition_df is not None and "plan_name" in composition_df.columns:
        counts_df = (
            composition_df["plan_name"]
            .astype(str)
            .value_counts(dropna=True)
            .rename_axis("plan")
            .reset_index(name="count")
        )
    if counts_df is None or counts_df.empty:
        return []
    ordered = counts_df.sort_values(by=["count", "plan"], ascending=[False, True]).reset_index(drop=True)
    return ordered["plan"].head(int(limit)).astype(str).tolist()


def _manifest_path_fields(name: str, rel_path: Path) -> dict:
    fields: dict[str, str] = {"plot_id": str(name)}
    parts = rel_path.parts
    stem = rel_path.stem
    if name == "dense_array_video_showcase":
        fields["group"] = "stage_b"
        fields["family"] = "showcase"
        if len(parts) >= 2:
            fields["plan_name"] = parts[1]
        fields["variant"] = stem
        return fields
    if name == "stage_a_summary":
        fields["group"] = "stage_a"
        fields["family"] = "stage_a"
        fields["plan_name"] = "stage_a"
        if stem == "background_logo":
            fields["variant"] = "background_logo"
            fields["input_name"] = "background"
        elif stem.endswith("__background_logo"):
            fields["variant"] = "background_logo"
            fields["input_name"] = stem[: -len("__background_logo")]
        else:
            fields["variant"] = stem
        return fields
    if name == "placement_map":
        fields["group"] = "stage_b"
        fields["family"] = "plan"
        if len(parts) >= 3:
            fields["plan_name"] = parts[1]
        if len(parts) >= 4:
            fields["input_name"] = parts[2]
        fields["variant"] = stem
        return fields
    if name == "tfbs_usage":
        fields["group"] = "stage_b"
        fields["family"] = "plan"
        if len(parts) >= 3:
            fields["plan_name"] = parts[1]
        if len(parts) >= 4:
            fields["input_name"] = parts[2]
        fields["variant"] = stem
        return fields
    if name == "run_health":
        fields["group"] = "run"
        fields["family"] = "run_health"
        fields["variant"] = stem
        return fields
    fields["variant"] = stem
    return fields


def run_plots_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    only: Optional[str] = None,
    source: str = "plot",
    absolute: bool = False,
    allow_truncated: bool = False,
) -> None:
    plots_cfg = root_cfg.plots
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_dir = _ensure_out_dir(plots_cfg, cfg_path, run_root)
    plot_format = plots_cfg.format if plots_cfg and getattr(plots_cfg, "format", None) else "pdf"
    default_list = plots_cfg.default if (plots_cfg and plots_cfg.default) else ["stage_a_summary", "placement_map"]
    if (
        only is None
        and plots_cfg is not None
        and bool(plots_cfg.video.enabled)
        and "dense_array_video_showcase" not in set(default_list)
    ):
        default_list = [*list(default_list), "dense_array_video_showcase"]
    selected = _resolve_selected_plot_names(only=only, default_list=list(default_list))
    options = plots_cfg.options if plots_cfg else {}
    global_style = plots_cfg.style if plots_cfg else {}
    _cleanup_legacy_flat_outputs(out_dir, selected, plot_format)
    if "placement_map" in selected or "tfbs_usage" in selected:
        _clean_plot_subdir(out_dir, "stage_b")
    required_sources = _plot_required_sources(selected)
    cols = _plot_required_columns(selected, options)
    composition_cols = _required_columns_for_selected(selected, mapping=_COMPOSITION_COLUMNS_BY_PLOT)
    attempt_cols = _required_columns_for_selected(selected, mapping=_ATTEMPT_COLUMNS_BY_PLOT)
    library_build_cols = _required_columns_for_selected(selected, mapping=_LIBRARY_BUILDS_COLUMNS_BY_PLOT)
    library_member_cols = _required_columns_for_selected(selected, mapping=_LIBRARY_MEMBERS_COLUMNS_BY_PLOT)
    dense_array_cols = _required_columns_for_selected(selected, mapping=_DENSE_ARRAY_COLUMNS_BY_PLOT)
    pool_cols = _required_columns_for_selected(selected, mapping=_POOL_COLUMNS_BY_PLOT)
    max_rows = plots_cfg.sample_rows if plots_cfg else None
    allow_truncated_records = bool(
        allow_truncated or (plots_cfg is not None and bool(getattr(plots_cfg, "allow_truncated", False)))
    )
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
        df, src_label = load_records_from_config(
            root_cfg,
            cfg_path,
            columns=cols,
            max_rows=max_rows,
            allow_truncated=allow_truncated_records,
        )
        src_label = _format_source_label(src_label, run_root, absolute)
        row_count = len(df)
    if "composition" in required_sources:
        composition_df = _load_composition(run_root, columns=composition_cols)
        if row_count == 0:
            row_count = len(composition_df)
            src_label = _format_source_label(
                f"composition:{run_root / 'outputs' / 'tables' / 'composition.parquet'}", run_root, absolute
            )
    if "libraries" in required_sources:
        library_builds_df, library_members_df = _load_libraries(
            run_root,
            builds_columns=library_build_cols,
            members_columns=library_member_cols,
        )
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
        attempts_df = _load_attempts(run_root, columns=attempt_cols)
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
        pool_manifest, pools = _load_stage_a_pools(run_root, columns=pool_cols)
        if row_count == 0:
            row_count = sum(
                int(entry.rows)
                for entry in pool_manifest.inputs.values()
                if str(entry.pool_mode or "") == POOL_MODE_TFBS
            )
            src_label = _format_source_label(f"pools:{run_root / 'outputs' / 'pools'}", run_root, absolute)
    if "tfbs_usage" in selected and library_members_df is None:
        libs = _maybe_load_libraries(
            run_root,
            builds_columns=library_build_cols,
            members_columns=library_member_cols,
        )
        if libs is not None:
            library_builds_df, library_members_df = libs
    if "dense_arrays" in required_sources:
        dense_arrays_df = _load_dense_arrays(run_root, columns=dense_array_cols)
        if row_count == 0:
            row_count = len(dense_arrays_df)
            src_label = _format_source_label(
                f"dense_arrays:{run_root / 'outputs' / 'tables' / 'records.parquet'}", run_root, absolute
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
        if name in {"placement_map", "tfbs_usage"}:
            scope_options = _parse_stage_b_scope_options(kwargs)
            kwargs.pop("scope", None)
            kwargs.pop("max_plans", None)
            kwargs.pop("drilldown_plans", None)
        else:
            scope_options = StageBScopeOptions(scope="per_plan", max_plans=1, drilldown_plans=0)

        out_path = out_dir / f"{name}.{plot_format}"
        try:
            if name == "dense_array_video_showcase":
                if plots_cfg is None:
                    raise ValueError("dense_array_video_showcase requires plots.video configuration.")
                result = fn(
                    df,
                    out_path,
                    video_cfg=plots_cfg.video,
                    **kwargs,
                )
            elif name == "placement_map":
                placement_dense_arrays_df = dense_arrays_df if dense_arrays_df is not None else df
                plan_names = (
                    placement_dense_arrays_df["densegen__plan"].astype(str).dropna().unique().tolist()
                    if placement_dense_arrays_df is not None and "densegen__plan" in placement_dense_arrays_df.columns
                    else []
                )
                use_group_scope = _should_group_stage_b_plans(plan_names=plan_names, scope_options=scope_options)
                if use_group_scope:
                    dense_grouped, composition_grouped, library_members_grouped = _map_stage_b_plan_group(
                        dense_arrays_df=placement_dense_arrays_df,
                        composition_df=composition_df,
                        library_members_df=library_members_df,
                    )
                    paths_grouped = fn(
                        df,
                        out_path,
                        style=style,
                        composition_df=composition_grouped,
                        dense_arrays_df=dense_grouped,
                        library_members_df=library_members_grouped,
                        cfg=cfg_effective,
                        **kwargs,
                    )
                    result_paths = list(paths_grouped or [])
                    drilldown_plan_names = _top_stage_b_drilldown_plans(
                        dense_arrays_df=placement_dense_arrays_df,
                        composition_df=composition_df,
                        limit=int(scope_options.drilldown_plans),
                    )
                    for drill_plan in drilldown_plan_names:
                        dense_drill = placement_dense_arrays_df[
                            placement_dense_arrays_df["densegen__plan"].astype(str) == str(drill_plan)
                        ].copy()
                        composition_drill = composition_df[
                            composition_df["plan_name"].astype(str) == str(drill_plan)
                        ].copy()
                        library_members_drill = (
                            library_members_df[library_members_df["plan_name"].astype(str) == str(drill_plan)].copy()
                            if library_members_df is not None and "plan_name" in library_members_df.columns
                            else library_members_df
                        )
                        if dense_drill.empty or composition_drill.empty:
                            continue
                        drill_paths = fn(
                            df,
                            out_path,
                            style=style,
                            composition_df=composition_drill,
                            dense_arrays_df=dense_drill,
                            library_members_df=library_members_drill,
                            cfg=cfg_effective,
                            **kwargs,
                        )
                        result_paths.extend(list(drill_paths or []))
                    result = result_paths
                else:
                    result = fn(
                        df,
                        out_path,
                        style=style,
                        composition_df=composition_df,
                        dense_arrays_df=placement_dense_arrays_df,
                        library_members_df=library_members_df,
                        cfg=cfg_effective,
                        **kwargs,
                    )
            elif name == "tfbs_usage":
                plan_names = (
                    composition_df["plan_name"].astype(str).dropna().unique().tolist()
                    if composition_df is not None and "plan_name" in composition_df.columns
                    else []
                )
                use_group_scope = _should_group_stage_b_plans(plan_names=plan_names, scope_options=scope_options)
                if use_group_scope:
                    _dense_unused, composition_grouped, library_members_grouped = _map_stage_b_plan_group(
                        dense_arrays_df=None,
                        composition_df=composition_df,
                        library_members_df=library_members_df,
                    )
                    paths_grouped = fn(
                        df,
                        out_path,
                        style=style,
                        composition_df=composition_grouped,
                        pools=pools,
                        library_members_df=library_members_grouped,
                        **kwargs,
                    )
                    result_paths = list(paths_grouped or [])
                    drilldown_plan_names = _top_stage_b_drilldown_plans(
                        dense_arrays_df=dense_arrays_df if dense_arrays_df is not None else df,
                        composition_df=composition_df,
                        limit=int(scope_options.drilldown_plans),
                    )
                    for drill_plan in drilldown_plan_names:
                        composition_drill = composition_df[
                            composition_df["plan_name"].astype(str) == str(drill_plan)
                        ].copy()
                        library_members_drill = (
                            library_members_df[library_members_df["plan_name"].astype(str) == str(drill_plan)].copy()
                            if library_members_df is not None and "plan_name" in library_members_df.columns
                            else library_members_df
                        )
                        if composition_drill.empty:
                            continue
                        drill_paths = fn(
                            df,
                            out_path,
                            style=style,
                            composition_df=composition_drill,
                            pools=pools,
                            library_members_df=library_members_drill,
                            **kwargs,
                        )
                        result_paths.extend(list(drill_paths or []))
                    result = result_paths
                else:
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
                result = fn(
                    df,
                    out_path,
                    style=style,
                    attempts_df=attempts_df,
                    composition_df=composition_df,
                    library_members_df=library_members_df,
                    events_df=events_df,
                    cfg=cfg_effective,
                    **kwargs,
                )
            elif name == "stage_a_summary":
                result = fn(df, out_path, style=style, pools=pools, pool_manifest=pool_manifest, **kwargs)
            else:
                result = fn(df, out_path, style=style, **kwargs)
            if result is None:
                paths = [out_path]
            elif isinstance(result, (list, tuple, set)):
                paths = [Path(p) for p in result if p is not None]
            else:
                paths = [Path(result)]
            if not paths:
                summary.add_row(name, "-", "[yellow]skipped[/] (not applicable for available artifacts)")
                continue
            saved_label = _format_plot_path(paths[0], run_root, absolute)
            if len(paths) > 1:
                saved_label = f"{saved_label} (+{len(paths) - 1})"
            summary.add_row(name, saved_label, "[green]ok[/]")
            created_at = datetime.now(timezone.utc).isoformat()
            for path in paths:
                rel_path = path.relative_to(out_dir)
                manifest_fields = _manifest_path_fields(name, rel_path)
                manifest_entries.append(
                    {
                        "name": name,
                        "path": str(rel_path),
                        "description": AVAILABLE_PLOTS[name]["description"],
                        "figsize": list(style.get("figsize", [])) if style.get("figsize") else None,
                        "generated_at": created_at,
                        "source": str(source),
                        **manifest_fields,
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
