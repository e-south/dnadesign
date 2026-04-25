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
from rich.console import Console
from typing_extensions import Literal

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_run_root
from ..core.artifacts.pool import POOL_MODE_TFBS, TFBSPoolArtifact
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
from .plot_data_loading import (
    _ensure_out_dir,
    _is_missing_composition_artifact_error,
    _load_attempts,
    _load_composition,
    _load_dense_arrays,
    _load_effective_config,
    _load_events,
    _load_libraries,
    _load_stage_a_pools,
    _maybe_load_libraries,
    _read_composition_parquet,
    _recover_composition_from_output_records,
    _root_config_to_dict,
)
from .plot_dataset import plot_source_cohort_concentration, plot_source_plan_input_heatmap
from .plot_inventory import (
    ANALYSIS_SURFACE_CONTRACT_VERSION,
    ARTIFACT_LEDGER_SCHEMA_VERSION,
    CURRENT_INVENTORY_SCHEMA_VERSION,
    LEGACY_PUBLIC_PLOT_IDS,
    base_plot_id,
    build_plot_text_contract,
    manifest_path_fields,
    resolve_plot_record,
)
from .plot_inventory import (
    artifact_ledger_path as resolve_artifact_ledger_path,
)
from .plot_inventory import (
    current_inventory_path as resolve_current_inventory_path,
)
from .plot_inventory import (
    plot_manifest_path as resolve_plot_manifest_path,
)
from .plot_registry import PLOT_SPECS
from .plot_run import (
    plot_attempt_outcome_timeline,
    plot_compression_ratio_by_plan,
    plot_solve_pressure_and_progress,
    plot_tfbs_concentration_profile,
)
from .plot_stage_a import (
    plot_background_sequence_logo,
    plot_stage_a_pool_diversity,
    plot_stage_a_pool_score_strata,
    plot_stage_a_sampling_yield,
)  # noqa: F401
from .plot_stage_a import plot_stage_a_summary as plot_stage_a_summary  # noqa: F401
from .plot_stage_a_strata import _build_stage_a_strata_overview_figure  # noqa: F401
from .plot_stage_a_yield import _build_stage_a_yield_bias_figure  # noqa: F401
from .plot_stage_b_placement import plot_placement_occupancy_map
from .plot_stage_b_summary import (
    plot_plan_regulator_deployment_heatmap,
    plot_retained_pool_coverage_by_regulator,
    plot_retained_vs_deployed_length_mix_by_regulator,
    plot_retained_vs_deployed_tier_mix_by_regulator,
    plot_score_strata_and_deployed_length_bridge,
    plot_upstream_motif_supply_and_pwm_strength,
)

_console = Console()
_read_composition_parquet = _read_composition_parquet


def _plot_manifest_path(out_dir: Path) -> Path:
    return resolve_plot_manifest_path(out_dir)


def _current_inventory_path(out_dir: Path) -> Path:
    return resolve_current_inventory_path(out_dir)


def _artifact_ledger_path(out_dir: Path) -> Path:
    return resolve_artifact_ledger_path(out_dir)


def _load_inventory_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_plot_manifest(out_dir: Path) -> dict:
    return _load_inventory_payload(_plot_manifest_path(out_dir))


def _load_artifact_ledger(out_dir: Path) -> dict:
    return _load_inventory_payload(_artifact_ledger_path(out_dir))


def _is_supported_plot_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if not parts:
        return False
    if parts[0] == "stage_a":
        return len(parts) >= 2
    if parts[0] == "stage_b":
        return len(parts) >= 3
    if parts[0] == "stage_b_summary":
        return len(parts) >= 2
    if parts[0] == "run_health":
        return len(parts) >= 2
    if parts[0] == "dataset":
        return len(parts) >= 2
    return False


def _is_legacy_public_plot_id(token: str) -> bool:
    raw = str(token or "").strip()
    if not raw:
        return False
    return raw in LEGACY_PUBLIC_PLOT_IDS or base_plot_id(raw) in LEGACY_PUBLIC_PLOT_IDS


def _write_plot_manifest(
    out_dir: Path,
    *,
    entries: list[dict],
    run_root: Path,
    cfg_path: Path,
    source: str,
) -> None:
    def _supported_entries(payload: dict, *, expected_schema_version: str | None = None) -> dict[str, dict]:
        if expected_schema_version is not None:
            schema_version = str(payload.get("schema_version") or "").strip()
            if schema_version != expected_schema_version:
                return {}
        supported: dict[str, dict] = {}
        for item in payload.get("plots", []):
            rel_path = str(item.get("path") or "")
            if not rel_path:
                continue
            plot_id = str(item.get("plot_id") or item.get("name") or "").strip()
            if _is_legacy_public_plot_id(plot_id):
                continue
            if (out_dir / rel_path).exists() and _is_supported_plot_path(rel_path):
                supported[rel_path] = item
        return supported

    current_entries: dict[str, dict] = {}
    for item in entries:
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        if (out_dir / rel_path).exists() and _is_supported_plot_path(rel_path):
            current_entries[rel_path] = item

    existing_current_entries = _supported_entries(
        _load_inventory_payload(_current_inventory_path(out_dir)),
        expected_schema_version=CURRENT_INVENTORY_SCHEMA_VERSION,
    )
    refreshed_plot_ids = {
        str(item.get("plot_id") or item.get("name") or "").strip()
        for item in current_entries.values()
        if str(item.get("plot_id") or item.get("name") or "").strip()
    }
    merged_current_entries = {
        rel_path: item
        for rel_path, item in existing_current_entries.items()
        if str(item.get("plot_id") or item.get("name") or "").strip() not in refreshed_plot_ids
    }
    merged_current_entries.update(current_entries)

    current_payload = {
        "schema_version": CURRENT_INVENTORY_SCHEMA_VERSION,
        "contract_version": ANALYSIS_SURFACE_CONTRACT_VERSION,
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "plots": sorted(merged_current_entries.values(), key=lambda x: (x.get("plot_id", ""), x.get("path", ""))),
    }
    serialized_current_payload = json.dumps(current_payload, indent=2, sort_keys=True)
    _current_inventory_path(out_dir).write_text(serialized_current_payload)

    ledger_entries = _supported_entries(
        _load_artifact_ledger(out_dir),
        expected_schema_version=ARTIFACT_LEDGER_SCHEMA_VERSION,
    )
    for rel_path, item in current_entries.items():
        ledger_entries[rel_path] = item
    artifact_ledger_payload = {
        "schema_version": ARTIFACT_LEDGER_SCHEMA_VERSION,
        "contract_version": ANALYSIS_SURFACE_CONTRACT_VERSION,
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "plots": sorted(ledger_entries.values(), key=lambda x: (x.get("name", ""), x.get("path", ""))),
    }
    serialized_ledger_payload = json.dumps(artifact_ledger_payload, indent=2, sort_keys=True)
    _artifact_ledger_path(out_dir).write_text(serialized_ledger_payload)
    _plot_manifest_path(out_dir).write_text(serialized_ledger_payload)


# ---------------------- Plots ----------------------


_PLOT_FNS = {
    "attempt_outcome_timeline": plot_attempt_outcome_timeline,
    "background_sequence_logo": plot_background_sequence_logo,
    "compression_ratio_by_plan": plot_compression_ratio_by_plan,
    "dense_array_showcase_video": plot_dense_array_video_showcase,
    "plan_regulator_deployment_heatmap": plot_plan_regulator_deployment_heatmap,
    "placement_occupancy_map": plot_placement_occupancy_map,
    "retained_pool_coverage_by_regulator": plot_retained_pool_coverage_by_regulator,
    "retained_vs_deployed_length_mix_by_regulator": plot_retained_vs_deployed_length_mix_by_regulator,
    "retained_vs_deployed_tier_mix_by_regulator": plot_retained_vs_deployed_tier_mix_by_regulator,
    "score_strata_and_deployed_length_bridge": plot_score_strata_and_deployed_length_bridge,
    "solve_pressure_and_progress": plot_solve_pressure_and_progress,
    "source_cohort_concentration": plot_source_cohort_concentration,
    "source_plan_input_heatmap": plot_source_plan_input_heatmap,
    "stage_a_pool_diversity": plot_stage_a_pool_diversity,
    "stage_a_pool_score_strata": plot_stage_a_pool_score_strata,
    "stage_a_sampling_yield": plot_stage_a_sampling_yield,
    "tfbs_concentration_profile": plot_tfbs_concentration_profile,
    "upstream_motif_supply_and_pwm_strength": plot_upstream_motif_supply_and_pwm_strength,
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
    "attempt_outcome_timeline": set(),
    "background_sequence_logo": set(),
    "compression_ratio_by_plan": set(),
    "dense_array_showcase_video": set(),
    "plan_regulator_deployment_heatmap": set(),
    "placement_occupancy_map": {"occupancy_alpha", "occupancy_max_categories", "scope", "max_plans", "drilldown_plans"},
    "retained_pool_coverage_by_regulator": set(),
    "retained_vs_deployed_length_mix_by_regulator": set(),
    "retained_vs_deployed_tier_mix_by_regulator": set(),
    "score_strata_and_deployed_length_bridge": set(),
    "solve_pressure_and_progress": set(),
    "source_cohort_concentration": {"max_sources"},
    "source_plan_input_heatmap": {"max_sources", "max_plans", "max_inputs"},
    "stage_a_pool_diversity": set(),
    "stage_a_pool_score_strata": set(),
    "stage_a_sampling_yield": set(),
    "tfbs_concentration_profile": {"scope", "max_plans", "drilldown_plans"},
    "upstream_motif_supply_and_pwm_strength": set(),
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


def _clean_selected_plot_files(out_dir: Path, *, subdir: str, plot_ids: Iterable[str]) -> None:
    target = out_dir / subdir
    if not target.exists():
        return
    selected = {str(plot_id).strip() for plot_id in plot_ids if str(plot_id).strip()}
    if not selected:
        return
    for path in target.iterdir():
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(out_dir))
        if path.stem not in selected or not _is_supported_plot_path(rel_path):
            continue
        path.unlink(missing_ok=True)
    if not any(target.iterdir()):
        target.rmdir()


def _clean_selected_stage_b_outputs(out_dir: Path, *, plot_ids: Iterable[str]) -> None:
    target = out_dir / "stage_b"
    if not target.exists():
        return
    selected = {str(plot_id).strip() for plot_id in plot_ids if str(plot_id).strip()}
    if not selected:
        return
    for path in sorted(target.rglob("*"), reverse=True):
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(out_dir))
        if not _is_supported_plot_path(rel_path):
            continue
        try:
            record = resolve_plot_record(plot_root=out_dir, plot_path=path)
        except Exception:
            continue
        plot_id = str(record.get("plot_id") or record.get("visual_plot_type") or "").strip()
        if plot_id not in selected:
            continue
        path.unlink(missing_ok=True)
    for directory in sorted(target.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if target.exists() and not any(target.iterdir()):
        target.rmdir()


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
    "compression_ratio_by_plan": {"densegen__compression_ratio", "densegen__plan"},
    "dense_array_showcase_video": {"id", "sequence", "densegen__plan", "densegen__used_tfbs_detail"},
    "plan_regulator_deployment_heatmap": {"densegen__plan", "densegen__used_tfbs_detail"},
    "placement_occupancy_map": {
        "id",
        "sequence",
        "densegen__input_name",
        "densegen__plan",
        "densegen__used_tfbs_detail",
    },
    "retained_pool_coverage_by_regulator": {"densegen__plan", "densegen__used_tfbs_detail"},
    "retained_vs_deployed_length_mix_by_regulator": {"densegen__plan", "densegen__used_tfbs_detail"},
    "retained_vs_deployed_tier_mix_by_regulator": {"densegen__plan", "densegen__used_tfbs_detail"},
    "score_strata_and_deployed_length_bridge": {"densegen__plan", "densegen__used_tfbs_detail"},
    "source_cohort_concentration": {"source", "densegen__plan", "densegen__input_name"},
    "source_plan_input_heatmap": {"source", "densegen__plan", "densegen__input_name"},
}
_COMPOSITION_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_occupancy_map": {"solution_id", "input_name", "plan_name", "tf", "tfbs", "offset", "length", "end"},
    "tfbs_concentration_profile": {"input_name", "plan_name", "tf", "tfbs"},
}
_ATTEMPT_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "attempt_outcome_timeline": {"status", "reason", "plan_name", "created_at", "detail_json"},
    "solve_pressure_and_progress": {"status", "reason", "plan_name", "created_at", "detail_json"},
}
_LIBRARY_BUILDS_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_occupancy_map": {"library_index", "library_hash", "input_name", "plan_name"},
    "tfbs_concentration_profile": {"library_index", "library_hash", "input_name", "plan_name"},
}
_LIBRARY_MEMBERS_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_occupancy_map": {"input_name", "plan_name", "tf", "tfbs"},
    "tfbs_concentration_profile": {"input_name", "plan_name", "tf", "tfbs"},
}
_DENSE_ARRAY_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_occupancy_map": {"id", "sequence", "densegen__input_name", "densegen__plan"},
}
_POOL_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "background_sequence_logo": {
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "sequence",
    },
    "retained_pool_coverage_by_regulator": {
        "input_name",
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "tier",
    },
    "retained_vs_deployed_length_mix_by_regulator": {
        "input_name",
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "tier",
    },
    "retained_vs_deployed_tier_mix_by_regulator": {
        "input_name",
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "tier",
    },
    "score_strata_and_deployed_length_bridge": {
        "input_name",
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "tfbs_core",
        "best_hit_score",
        "tier",
    },
    "stage_a_pool_diversity": {
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
    },
    "stage_a_pool_score_strata": {
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
    },
    "stage_a_sampling_yield": {
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
    },
    "upstream_motif_supply_and_pwm_strength": {
        "input_name",
        "tf",
        "regulator_id",
        "tfbs",
        "tfbs_sequence",
        "tier",
    },
}
_COMPOSITION_RECOVERY_OUTPUT_COLUMNS_BY_PLOT: Dict[str, set[str]] = {
    "placement_occupancy_map": {"id", "densegen__input_name", "densegen__plan", "densegen__used_tfbs_detail"},
    "tfbs_concentration_profile": {"id", "densegen__input_name", "densegen__plan", "densegen__used_tfbs_detail"},
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
        if {
            "background_sequence_logo",
            "stage_a_pool_diversity",
            "stage_a_pool_score_strata",
            "stage_a_sampling_yield",
        } & selected_set and stem.startswith("stage_a_"):
            remove = True
        elif "placement_occupancy_map" in selected_set and stem.startswith("placement_"):
            remove = True
        elif "tfbs_concentration_profile" in selected_set and stem.startswith("tfbs_"):
            remove = True
        elif {"attempt_outcome_timeline", "solve_pressure_and_progress", "compression_ratio_by_plan"} & selected_set:
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
    default_list = (
        plots_cfg.default
        if (plots_cfg and plots_cfg.default)
        else ["stage_a_sampling_yield", "placement_occupancy_map"]
    )
    selected = _resolve_selected_plot_names(only=only, default_list=list(default_list))
    options = plots_cfg.options if plots_cfg else {}
    global_style = plots_cfg.style if plots_cfg else {}
    _cleanup_legacy_flat_outputs(out_dir, selected, plot_format)
    stage_b_selected = [
        name
        for name in selected
        if name
        in {
            "dense_array_showcase_video",
            "placement_occupancy_map",
            "tfbs_concentration_profile",
        }
    ]
    if stage_b_selected:
        _clean_selected_stage_b_outputs(out_dir, plot_ids=stage_b_selected)
    dataset_plots = [name for name in selected if name in {"source_cohort_concentration", "source_plan_input_heatmap"}]
    if dataset_plots:
        _clean_selected_plot_files(out_dir, subdir="dataset", plot_ids=dataset_plots)
    required_sources = _plot_required_sources(selected)
    cols = _plot_required_columns(selected, options)
    composition_cols = _required_columns_for_selected(selected, mapping=_COMPOSITION_COLUMNS_BY_PLOT)
    attempt_cols = _required_columns_for_selected(selected, mapping=_ATTEMPT_COLUMNS_BY_PLOT)
    library_build_cols = _required_columns_for_selected(selected, mapping=_LIBRARY_BUILDS_COLUMNS_BY_PLOT)
    library_member_cols = _required_columns_for_selected(selected, mapping=_LIBRARY_MEMBERS_COLUMNS_BY_PLOT)
    dense_array_cols = _required_columns_for_selected(selected, mapping=_DENSE_ARRAY_COLUMNS_BY_PLOT)
    pool_cols = _required_columns_for_selected(selected, mapping=_POOL_COLUMNS_BY_PLOT)
    composition_recovery_output_cols = _required_columns_for_selected(
        selected,
        mapping=_COMPOSITION_RECOVERY_OUTPUT_COLUMNS_BY_PLOT,
    )
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
    source_errors: dict[str, Exception] = {}

    if "outputs" in required_sources:
        try:
            df, src_label = load_records_from_config(
                root_cfg,
                cfg_path,
                columns=cols,
                max_rows=max_rows,
                allow_truncated=allow_truncated_records,
                normalize_used_tfbs_detail=False,
            )
            src_label = _format_source_label(src_label, run_root, absolute)
            row_count = len(df)
        except Exception as exc:
            source_errors["outputs"] = exc
    if "composition" in required_sources:
        try:
            composition_df = _load_composition(run_root, columns=composition_cols)
        except Exception as exc:
            try:
                if not _is_missing_composition_artifact_error(exc) or not composition_recovery_output_cols:
                    raise
                recovery_cols = sorted(set(cols) | set(composition_recovery_output_cols))
                if df.empty or any(column not in df.columns for column in composition_recovery_output_cols):
                    df, src_label = load_records_from_config(
                        root_cfg,
                        cfg_path,
                        columns=recovery_cols,
                        max_rows=max_rows,
                        allow_truncated=allow_truncated_records,
                        normalize_used_tfbs_detail=False,
                    )
                    src_label = _format_source_label(src_label, run_root, absolute)
                    row_count = len(df)
                    source_errors.pop("outputs", None)
                composition_df = _recover_composition_from_output_records(df, columns=composition_cols)
            except Exception as recovery_exc:
                source_errors["composition"] = recovery_exc
        if composition_df is not None and row_count == 0:
            row_count = len(composition_df)
            src_label = _format_source_label(
                f"composition:{run_root / 'outputs' / 'tables' / 'composition.parquet'}", run_root, absolute
            )
    if "libraries" in required_sources:
        try:
            library_builds_df, library_members_df = _load_libraries(
                run_root,
                builds_columns=library_build_cols,
                members_columns=library_member_cols,
            )
            if row_count == 0:
                row_count = len(library_members_df)
                src_label = _format_source_label(f"libraries:{run_root / 'outputs' / 'libraries'}", run_root, absolute)
        except Exception:
            library_builds_df = None
            library_members_df = None
    if "config" in required_sources:
        try:
            cfg_effective = _load_effective_config(run_root)
            if row_count == 0:
                row_count = 1
                src_label = _format_source_label(
                    f"config:{run_root / 'outputs' / 'meta' / 'effective_config.json'}", run_root, absolute
                )
        except Exception as exc:
            if "effective_config.json not found:" in str(exc):
                cfg_effective = _root_config_to_dict(root_cfg)
                if row_count == 0:
                    row_count = 1
                    src_label = _format_source_label(f"config:{cfg_path}", run_root, absolute)
            else:
                source_errors["config"] = exc
    if "attempts" in required_sources:
        try:
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
        except Exception as exc:
            source_errors["attempts"] = exc
    pools: dict[str, pd.DataFrame] | None = None
    pool_manifest: TFBSPoolArtifact | None = None
    if "pools" in required_sources:
        try:
            pool_manifest, pools = _load_stage_a_pools(run_root, columns=pool_cols)
            if row_count == 0:
                row_count = sum(
                    int(entry.rows)
                    for entry in pool_manifest.inputs.values()
                    if str(entry.pool_mode or "") == POOL_MODE_TFBS
                )
                src_label = _format_source_label(f"pools:{run_root / 'outputs' / 'pools'}", run_root, absolute)
        except Exception as exc:
            source_errors["pools"] = exc
    if "tfbs_concentration_profile" in selected and library_members_df is None:
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
        plot_source_error = next(
            (
                source_errors[source_name]
                for source_name in _plot_required_sources([name])
                if source_name in source_errors
            ),
            None,
        )

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
        if name in {"placement_occupancy_map", "tfbs_concentration_profile"}:
            scope_options = _parse_stage_b_scope_options(kwargs)
            kwargs.pop("scope", None)
            kwargs.pop("max_plans", None)
            kwargs.pop("drilldown_plans", None)
        else:
            scope_options = StageBScopeOptions(scope="per_plan", max_plans=1, drilldown_plans=0)

        out_path = out_dir / f"{name}.{plot_format}"
        try:
            if plot_source_error is not None:
                raise plot_source_error
            if name == "dense_array_showcase_video":
                if plots_cfg is None:
                    raise ValueError("dense_array_showcase_video requires plots.video configuration.")
                if not bool(plots_cfg.video.enabled):
                    raise ValueError(
                        "dense_array_showcase_video requires plots.video.enabled: true when explicitly selected."
                    )
                result = fn(
                    df,
                    out_path,
                    video_cfg=plots_cfg.video,
                    workspace_name=cfg_path.parent.name,
                    **kwargs,
                )
            elif name == "placement_occupancy_map":
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
            elif name == "tfbs_concentration_profile":
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
            elif name in {"attempt_outcome_timeline", "solve_pressure_and_progress"}:
                result = fn(
                    df,
                    out_path,
                    style=style,
                    attempts_df=attempts_df,
                    events_df=events_df,
                    cfg=cfg_effective,
                    **kwargs,
                )
            elif name == "compression_ratio_by_plan":
                result = fn(
                    df,
                    out_path,
                    style=style,
                    **kwargs,
                )
            elif name in {
                "retained_pool_coverage_by_regulator",
                "retained_vs_deployed_length_mix_by_regulator",
                "retained_vs_deployed_tier_mix_by_regulator",
                "score_strata_and_deployed_length_bridge",
                "upstream_motif_supply_and_pwm_strength",
            }:
                result = fn(
                    df,
                    out_path,
                    style=style,
                    pools=pools,
                    pool_manifest=pool_manifest,
                    **kwargs,
                )
            elif name in {
                "background_sequence_logo",
                "stage_a_pool_diversity",
                "stage_a_pool_score_strata",
                "stage_a_sampling_yield",
            }:
                result = fn(
                    df,
                    out_path,
                    style=style,
                    pools=pools,
                    pool_manifest=pool_manifest,
                    **kwargs,
                )
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
                manifest_fields = manifest_path_fields(name, rel_path)
                text_contract = build_plot_text_contract(
                    name,
                    variant=str(manifest_fields.get("variant") or ""),
                    plan_name=str(manifest_fields.get("plan_name") or ""),
                    input_name=str(manifest_fields.get("input_name") or ""),
                )
                manifest_entries.append(
                    {
                        "name": name,
                        "path": str(rel_path),
                        "title": text_contract["title"],
                        "description": text_contract["description"] or AVAILABLE_PLOTS[name]["description"],
                        "caption": text_contract["caption"],
                        "alt_text": text_contract["alt_text"],
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
    _write_plot_manifest(out_dir, entries=manifest_entries, run_root=run_root, cfg_path=cfg_path, source=source)
    if errors:
        details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"{len(errors)} plot(s) failed: {details}")
