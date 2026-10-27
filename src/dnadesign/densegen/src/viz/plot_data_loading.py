"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_data_loading.py

Data-loading helpers for DenseGen plotting workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from ..config import RootConfig, resolve_outputs_scoped_path
from ..core.artifacts.pool import POOL_MODE_TFBS, TFBSPoolArtifact, load_pool_artifact
from ..core.pipeline.attempts import _load_attempts_snapshot
from ..core.record_metadata_recovery import recover_densegen_metadata_from_source
from ..core.record_values import require_list_of_dicts as _require_list_of_dicts


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


def _resolve_analysis_table_paths(
    tables_root: Path,
    *,
    final_name: str,
    part_glob: str,
) -> list[Path]:
    final_path = tables_root / final_name
    part_paths = sorted(tables_root.glob(part_glob))
    paths: list[Path] = []
    if final_path.exists():
        paths.append(final_path)
    paths.extend(part_paths)
    if not paths:
        raise ValueError(
            f"{final_name} not found: {final_path}. Expected finalized table or pending `{part_glob}` files."
        )
    return paths


def _load_attempts(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    tables_root = run_root / "outputs" / "tables"
    projected_columns = _read_columns(columns)
    return _load_attempts_snapshot(tables_root, columns=projected_columns)


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


def _ensure_out_dir(plots_cfg, cfg_path: Path, run_root: Path) -> Path:
    out_dir = plots_cfg.out_dir if plots_cfg else "outputs/plots"
    out = resolve_outputs_scoped_path(cfg_path, run_root, out_dir, label="plots.out_dir")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_projected_parquet(path: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    requested = _read_columns(columns)
    if requested is None:
        return pd.read_parquet(path)
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
    tables_root = run_root / "outputs" / "tables"
    composition_paths = _resolve_analysis_table_paths(
        tables_root,
        final_name="composition.parquet",
        part_glob="composition_part-*.parquet",
    )
    frames = [_read_composition_parquet(path, columns=columns) for path in composition_paths]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if {"solution_id", "placement_index"}.issubset(set(frame.columns)):
        frame = frame.drop_duplicates(subset=["solution_id", "placement_index"], keep="last", ignore_index=True)
    return frame


def _is_missing_composition_artifact_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return "composition.parquet not found:" in message


def _recover_composition_from_output_records(
    dense_arrays_df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    dense_arrays_df = recover_densegen_metadata_from_source(dense_arrays_df)
    required = {"id", "densegen__input_name", "densegen__plan", "densegen__used_tfbs_detail"}
    missing = required - set(dense_arrays_df.columns)
    if missing:
        raise ValueError(
            f"Output records are missing the columns required to recover composition data: {sorted(missing)}"
        )
    rows: list[dict[str, object]] = []
    for _, row in dense_arrays_df.iterrows():
        solution_id = str(row.get("id") or "").strip()
        input_name = str(row.get("densegen__input_name") or "").strip()
        plan_name = str(row.get("densegen__plan") or "").strip()
        if not solution_id or not input_name or not plan_name:
            raise ValueError(
                "Output records are missing id/input_name/plan metadata required for composition recovery."
            )
        library_hash = str(row.get("densegen__sampling_library_hash") or "").strip() or None
        library_index = row.get("densegen__sampling_library_index")
        for item in _require_list_of_dicts(row.get("densegen__used_tfbs_detail")):
            part_kind = str(item.get("part_kind") or "tfbs").strip().lower()
            if part_kind != "tfbs":
                continue
            tf = str(item.get("regulator") or "").strip()
            tfbs = str(item.get("sequence") or "").strip()
            if not tf or not tfbs:
                continue
            rows.append(
                {
                    "solution_id": solution_id,
                    "input_name": input_name,
                    "plan_name": plan_name,
                    "library_hash": library_hash,
                    "library_index": library_index,
                    "part_kind": "tfbs",
                    "tf": tf,
                    "tfbs": tfbs,
                    "regulator": tf,
                    "sequence": tfbs,
                    "offset": item.get("offset"),
                    "offset_raw": item.get("offset_raw"),
                    "length": item.get("length"),
                    "end": item.get("end"),
                    "orientation": item.get("orientation"),
                    "source": item.get("source"),
                    "motif_id": item.get("motif_id"),
                    "tfbs_id": item.get("tfbs_id"),
                    "site_id": item.get("site_id"),
                }
            )
    if not rows:
        raise ValueError("Output records include no TFBS placement annotations to recover composition data.")
    recovered = pd.DataFrame(rows)
    requested = _read_columns(columns)
    if requested is None:
        return recovered
    missing_requested = sorted(set(requested) - set(recovered.columns))
    if missing_requested:
        raise ValueError(f"Recovered composition data is missing required columns: {missing_requested}")
    return recovered[requested].copy()


def _maybe_load_composition(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame | None:
    tables_root = run_root / "outputs" / "tables"
    final_path = tables_root / "composition.parquet"
    part_paths = sorted(tables_root.glob("composition_part-*.parquet"))
    if not final_path.exists() and not part_paths:
        return None
    return _load_composition(run_root, columns=columns)


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


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump(mode="python", exclude_none=True))
    if hasattr(model, "dict"):
        return dict(model.dict(exclude_none=True))
    raise TypeError("DenseGen config model does not support dict export for plotting fallback.")


def _root_config_to_dict(root_cfg: RootConfig) -> dict:
    return {"config": _model_to_dict(root_cfg.densegen)}


def _load_dense_arrays(run_root: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = run_root / "outputs" / "tables" / "records.parquet"
    if path.exists():
        return pd.read_parquet(path, columns=_read_columns(columns))
    raise ValueError(f"records.parquet not found: {path}")
