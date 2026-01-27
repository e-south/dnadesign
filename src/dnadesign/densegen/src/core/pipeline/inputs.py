"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/inputs.py

Input metadata helpers for pipeline manifests and sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...config import resolve_relative_path

PWM_INPUT_TYPES = {
    "pwm_meme",
    "pwm_meme_set",
    "pwm_jaspar",
    "pwm_matrix_csv",
    "pwm_artifact",
    "pwm_artifact_set",
}


def _resolve_input_paths(source_cfg, cfg_path: Path) -> list[str]:
    paths: list[str] = []
    if hasattr(source_cfg, "path"):
        paths.append(str(resolve_relative_path(cfg_path, getattr(source_cfg, "path"))))
    if hasattr(source_cfg, "paths"):
        for path in getattr(source_cfg, "paths") or []:
            paths.append(str(resolve_relative_path(cfg_path, path)))
    return paths


def _sampling_attr(sampling, name: str, default=None):
    if sampling is None:
        return default
    if hasattr(sampling, name):
        return getattr(sampling, name)
    if isinstance(sampling, dict):
        return sampling.get(name, default)
    return default


def _mining_attr(mining, name: str, default=None):
    if mining is None:
        return default
    if hasattr(mining, name):
        return getattr(mining, name)
    if isinstance(mining, dict):
        return mining.get(name, default)
    return default


def _extract_pwm_sampling_config(source_cfg) -> dict | None:
    sampling = getattr(source_cfg, "sampling", None)
    if sampling is None:
        return None
    n_sites = _sampling_attr(sampling, "n_sites")
    oversample = _sampling_attr(sampling, "oversample_factor")
    requested = None
    generated = None
    if isinstance(n_sites, int) and isinstance(oversample, int):
        requested = int(n_sites) * int(oversample)
        generated = requested
    length_range = _sampling_attr(sampling, "length_range")
    if length_range is not None:
        length_range = list(length_range)
    mining = _sampling_attr(sampling, "mining")
    scoring_backend = _sampling_attr(sampling, "scoring_backend")
    mining_batch_size = _mining_attr(mining, "batch_size")
    mining_max_seconds = _mining_attr(mining, "max_seconds")
    mining_log_every_batches = _mining_attr(mining, "log_every_batches")
    return {
        "strategy": _sampling_attr(sampling, "strategy"),
        "scoring_backend": scoring_backend,
        "n_sites": _sampling_attr(sampling, "n_sites"),
        "oversample_factor": _sampling_attr(sampling, "oversample_factor"),
        "requested_candidates": requested,
        "generated_candidates": generated,
        "bgfile": _sampling_attr(sampling, "bgfile"),
        "keep_all_candidates_debug": _sampling_attr(sampling, "keep_all_candidates_debug"),
        "length_policy": _sampling_attr(sampling, "length_policy"),
        "length_range": length_range,
        "mining": {
            "batch_size": mining_batch_size,
            "max_seconds": mining_max_seconds,
            "log_every_batches": mining_log_every_batches,
        }
        if mining is not None
        else None,
    }


def _build_input_manifest_entry(
    *,
    source_cfg,
    cfg_path: Path,
    input_meta: dict,
    input_row_count: int,
    input_tf_count: int,
    input_tfbs_count: int,
    input_tf_tfbs_pair_count: int | None,
    meta_df: pd.DataFrame | None,
) -> dict:
    source_type = getattr(source_cfg, "type", "unknown")
    entry = {
        "name": getattr(source_cfg, "name", "unknown"),
        "type": source_type,
        "mode": input_meta.get("input_mode"),
        "resolved_paths": _resolve_input_paths(source_cfg, cfg_path),
        "resolved_root": str(resolve_relative_path(cfg_path, getattr(source_cfg, "root")))
        if hasattr(source_cfg, "root")
        else None,
        "dataset": getattr(source_cfg, "dataset", None),
        "counts": {
            "rows": int(input_row_count),
            "tf_count": int(input_tf_count),
            "tfbs_count": int(input_tfbs_count),
            "tf_tfbs_pair_count": int(input_tf_tfbs_pair_count) if input_tf_tfbs_pair_count is not None else None,
        },
    }
    if source_type in PWM_INPUT_TYPES:
        entry["pwm_ids_requested"] = []
        if source_type == "pwm_matrix_csv":
            motif_id = getattr(source_cfg, "motif_id", None)
            entry["pwm_ids_requested"] = [motif_id] if motif_id else []
        elif source_type in {"pwm_meme", "pwm_meme_set", "pwm_jaspar"}:
            entry["pwm_ids_requested"] = list(getattr(source_cfg, "motif_ids") or [])
        entry["pwm_sampling"] = _extract_pwm_sampling_config(source_cfg)
        entry["pwm_ids"] = list(input_meta.get("input_pwm_ids") or [])
        if meta_df is not None and "tf" in meta_df.columns:
            counts = meta_df["tf"].value_counts().to_dict()
            entry["pwm_sites_per_motif"] = {str(k): int(v) for k, v in counts.items()}
    return entry


def _input_metadata(source_cfg, cfg_path: Path) -> dict:
    source_type = getattr(source_cfg, "type", "unknown")
    source_name = getattr(source_cfg, "name", "unknown")
    meta = {"input_type": source_type, "input_name": source_name}
    if hasattr(source_cfg, "path"):
        meta["input_path"] = str(resolve_relative_path(cfg_path, getattr(source_cfg, "path")))
    if source_type == "usr_sequences":
        meta["input_dataset"] = getattr(source_cfg, "dataset", None)
        meta["input_root"] = str(resolve_relative_path(cfg_path, getattr(source_cfg, "root")))
        meta["input_mode"] = "sequence_library"
        meta["input_pwm_ids"] = []
    elif source_type == "sequence_library":
        meta["input_mode"] = "sequence_library"
        meta["input_pwm_ids"] = []
    elif source_type == "binding_sites":
        meta["input_mode"] = "binding_sites"
        meta["input_pwm_ids"] = []
    elif source_type in PWM_INPUT_TYPES:
        meta["input_mode"] = "pwm_sampled"
        if source_type == "pwm_matrix_csv":
            motif_id = getattr(source_cfg, "motif_id", None)
            meta["input_pwm_ids"] = [motif_id] if motif_id else []
        elif source_type in {"pwm_meme", "pwm_jaspar"}:
            meta["input_pwm_ids"] = list(getattr(source_cfg, "motif_ids") or [])
        else:
            meta["input_pwm_ids"] = []
        sampling = getattr(source_cfg, "sampling", None)
        if sampling is not None:
            meta["input_pwm_strategy"] = getattr(sampling, "strategy", None)
            meta["input_pwm_scoring_backend"] = getattr(sampling, "scoring_backend", None)
            mining_cfg = getattr(sampling, "mining", None)
            meta["input_pwm_mining_batch_size"] = _mining_attr(mining_cfg, "batch_size")
            meta["input_pwm_mining_max_seconds"] = _mining_attr(mining_cfg, "max_seconds")
            meta["input_pwm_mining_log_every_batches"] = _mining_attr(mining_cfg, "log_every_batches")
            meta["input_pwm_bgfile"] = getattr(sampling, "bgfile", None)
            meta["input_pwm_keep_all_candidates_debug"] = getattr(sampling, "keep_all_candidates_debug", None)
            meta["input_pwm_include_matched_sequence"] = getattr(sampling, "include_matched_sequence", None)
            meta["input_pwm_n_sites"] = getattr(sampling, "n_sites", None)
            meta["input_pwm_oversample_factor"] = getattr(sampling, "oversample_factor", None)
    else:
        meta["input_mode"] = source_type
        meta["input_pwm_ids"] = []
    return meta
