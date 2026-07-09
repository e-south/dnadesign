"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_triptych_runtime.py

Runtime loading for Reader-backed SFXI triptych notebook plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

SFXI_TIDY_RECORD_ID = "promote_to_tidy_plus_map/df"


@dataclass(frozen=True)
class ReaderSFXITriptychRuntime:
    tidy: pd.DataFrame
    label_col: str
    time_col: str
    treatment_col: str
    treatment_case_sensitive: bool
    logic_channel: str
    common_times: tuple[float, ...]
    default_time_h: float
    induction_time_h: float | None
    sfxi_condition_order: tuple[str, ...]
    sfxi_condition_map: Mapping[str, str]


def load_reader_sfxi_triptych_runtime(row: Mapping[str, Any]) -> ReaderSFXITriptychRuntime:
    """Load Reader records and SFXI config for a flattened evidence row."""

    config_path = _reader_config_path(row)
    return _load_runtime_for_config(str(config_path.resolve()))


@lru_cache(maxsize=8)
def _load_runtime_for_config(config_path_text: str) -> ReaderSFXITriptychRuntime:
    config_path = Path(config_path_text)
    experiment_root = config_path.parent
    _ensure_reader_import_path(experiment_root)
    from reader.domains.logic.sfxi.api import load_sfxi_config  # noqa: PLC0415
    from reader.domains.logic.treatment_columns import (  # noqa: PLC0415
        choose_treatment_column,
        normalize_treatment_series,
    )
    from reader.runtime import builtin_runtime  # noqa: PLC0415
    from reader.workbench.notebooks.context import load_notebook_workbench_context  # noqa: PLC0415
    from reader.workbench.records import discover_dataframe_records  # noqa: PLC0415

    ctx = load_notebook_workbench_context(experiment_root)
    runtime = builtin_runtime()
    sfxi_cfg = _load_sfxi_config(ctx, runtime=runtime, load_sfxi_config=load_sfxi_config)
    record_info, _, record_note, _ = discover_dataframe_records(ctx.outputs_dir, allow_scan=False, runtime=runtime)
    tidy = pd.read_parquet(_record_path(record_info, record_id=SFXI_TIDY_RECORD_ID, record_note=record_note))
    logic_times, treatment_col = _channel_times(
        tidy,
        cfg=sfxi_cfg,
        channel=sfxi_cfg.response.logic_channel,
        choose_treatment_column=choose_treatment_column,
        normalize_treatment_series=normalize_treatment_series,
    )
    intensity_times, _ = _channel_times(
        tidy,
        cfg=sfxi_cfg,
        channel=sfxi_cfg.response.intensity_channel,
        choose_treatment_column=choose_treatment_column,
        normalize_treatment_series=normalize_treatment_series,
    )
    common_times = common_sfxi_times(logic_times, intensity_times)
    if not common_times:
        raise RuntimeError("No common SFXI time points exist for logic and intensity channels.")
    condition_order, condition_map = _condition_contract(sfxi_cfg.treatment_map, sfxi_cfg.treatment_case_sensitive)
    return ReaderSFXITriptychRuntime(
        tidy=tidy,
        label_col=str(sfxi_cfg.design_by[0]),
        time_col=str(sfxi_cfg.time_column),
        treatment_col=treatment_col,
        treatment_case_sensitive=bool(sfxi_cfg.treatment_case_sensitive),
        logic_channel=str(sfxi_cfg.response.logic_channel),
        common_times=common_times,
        default_time_h=default_sfxi_time(common_times, target=sfxi_cfg.target_time_h),
        induction_time_h=_induction_time(tidy, time_col=str(sfxi_cfg.time_column)),
        sfxi_condition_order=condition_order,
        sfxi_condition_map=condition_map,
    )


def _load_sfxi_config(ctx: Any, *, runtime: Any, load_sfxi_config: Any) -> Any:
    sfxi_step = None
    for step in ctx.workbench.pipeline:
        descriptor = runtime.plugins.resolve_descriptor(str(getattr(step, "plugin", "")))
        if descriptor.domain == "logic" and descriptor.family == "summary_transform" and "sfxi" in descriptor.tags:
            sfxi_step = step
    if sfxi_step is None:
        raise RuntimeError("No SFXI transform step is defined for the Reader experiment.")
    bound_protocol = runtime.bind_protocol(ctx.decl.experiment_semantics.protocol)
    payload = bound_protocol.effective_plugin_config(
        plugin_id=sfxi_step.plugin,
        step_with=dict(getattr(sfxi_step, "with_", {}) or {}),
    )
    logic_map_ref = payload.get("logic_map_ref")
    if isinstance(logic_map_ref, str):
        logic_spec = ctx.decl.experiment_semantics.annotations.resolve_logic_map(ref=logic_map_ref)
        payload = dict(payload)
        payload["treatment_map"] = dict(logic_spec.corners)
        payload["treatment_case_sensitive"] = bool(logic_spec.case_sensitive)
    return load_sfxi_config(payload)


def _channel_times(
    tidy: pd.DataFrame,
    *,
    cfg: Any,
    channel: str,
    choose_treatment_column: Any,
    normalize_treatment_series: Any,
) -> tuple[tuple[float, ...], str]:
    work = tidy[tidy["channel"].astype(str) == str(channel)].copy()
    if work.empty:
        raise RuntimeError(f"Reader tidy record has no rows for channel `{channel}`.")
    treatment_col = choose_treatment_column(
        work,
        dict(cfg.treatment_map),
        case_sensitive=bool(cfg.treatment_case_sensitive),
        preferred=cfg.treatment_column,
    )
    values = work[treatment_col].astype(str)
    if cfg.treatment_case_sensitive:
        mask = values.isin({str(value) for value in cfg.treatment_map.values()})
    else:
        wanted = {str(value).strip().casefold() for value in cfg.treatment_map.values()}
        mask = normalize_treatment_series(values).isin(wanted)
    times = pd.to_numeric(work.loc[mask, cfg.time_column], errors="coerce").dropna()
    return tuple(sorted({float(value) for value in times.tolist()})), treatment_col


def _reader_config_path(row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("reader_config_path") or "")).expanduser()
    if path.name == "config.yaml" and path.exists():
        return path
    artifact_path = Path(str(row.get("path") or "")).expanduser()
    for base in [artifact_path] + list(artifact_path.parents):
        candidate = base / "config.yaml"
        if candidate.exists():
            return candidate
    raise RuntimeError("Reader config.yaml could not be resolved for the selected evidence row.")


def _ensure_reader_import_path(experiment_root: Path) -> None:
    if importlib.util.find_spec("reader") is not None:
        return
    for base in [experiment_root] + list(experiment_root.parents):
        source_root = base / "src"
        if (source_root / "reader").is_dir():
            sys.path.insert(0, str(source_root))
            return
    raise RuntimeError("Reader Python source root is not importable from the selected experiment path.")


def _record_path(record_info: Mapping[str, Mapping[str, Any]], *, record_id: str, record_note: str) -> Path:
    for info in record_info.values():
        if str(info.get("record_id") or "") == record_id:
            path = Path(info["path"])
            if path.exists():
                return path
            raise RuntimeError(f"Reader dataframe record `{record_id}` points to a missing file: {path}")
    note = f" {record_note}" if record_note else ""
    raise RuntimeError(f"Reader dataframe record `{record_id}` is not available.{note}")


def common_sfxi_times(logic_times: tuple[float, ...], intensity_times: tuple[float, ...]) -> tuple[float, ...]:
    logic = {round(float(value), 12) for value in logic_times}
    intensity = {round(float(value), 12) for value in intensity_times}
    return tuple(sorted(logic & intensity))


def default_sfxi_time(times: tuple[float, ...], *, target: float | None) -> float:
    value = finite_float(target)
    if value is not None and times[0] <= value <= times[-1]:
        return value
    return times[-1]


def sfxi_time_step(times: tuple[float, ...]) -> float:
    diffs = [right - left for left, right in zip(times[:-1], times[1:], strict=False) if right > left]
    return min(diffs) if diffs else 0.25


def _condition_contract(
    treatment_map: Mapping[str, str],
    case_sensitive: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    order = tuple(f"{corner}: {treatment_map[corner]}" for corner in ("00", "10", "01", "11"))
    mapping = {
        _condition_key(treatment_map[corner], case_sensitive=case_sensitive): f"{corner}: {treatment_map[corner]}"
        for corner in ("00", "10", "01", "11")
    }
    return order, mapping


def _condition_key(value: object, *, case_sensitive: bool) -> str:
    text = str(value)
    return text if case_sensitive else text.strip().casefold()


def _induction_time(tidy: pd.DataFrame, *, time_col: str) -> float | None:
    for column in ("induction_time_h", "induction_time", "time_of_induction_h", "time_of_induction"):
        if column in tidy.columns:
            values = pd.to_numeric(tidy[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
    if "sheet_index" in tidy.columns:
        sheet_values = pd.to_numeric(tidy["sheet_index"], errors="coerce").dropna()
        if not sheet_values.empty:
            sheet_series = pd.to_numeric(tidy["sheet_index"], errors="coerce")
            times = pd.to_numeric(tidy.loc[sheet_series > float(sheet_values.min()), time_col], errors="coerce")
            times = times.dropna()
            if not times.empty:
                return float(times.min())
    return None


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "ReaderSFXITriptychRuntime",
    "finite_float",
    "load_reader_sfxi_triptych_runtime",
    "sfxi_time_step",
]
