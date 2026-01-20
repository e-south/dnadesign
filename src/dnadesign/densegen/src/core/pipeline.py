"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/pipeline.py

DenseGen pipeline orchestration (CLI-agnostic).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import random
import time
import tomllib
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
from rich.console import Console

from ..adapters.optimizer import DenseArraysAdapter, OptimizerAdapter
from ..adapters.outputs import OutputRecord, SinkBase, build_sinks, load_records_from_config, resolve_bio_alphabet
from ..adapters.sources import data_source_factory
from ..adapters.sources.base import BaseDataSource
from ..config import (
    DenseGenConfig,
    LoadedConfig,
    ResolvedPlanItem,
    resolve_relative_path,
    resolve_run_root,
    schema_version_at_least,
)
from ..utils.logging_utils import install_native_stderr_filters
from .artifacts.ids import hash_tfbs_id
from .artifacts.library import write_library_artifact
from .artifacts.pool import POOL_MODE_SEQUENCE, POOL_MODE_TFBS, PoolData, build_pool_artifact
from .metadata import build_metadata
from .postprocess import random_fill
from .pvalue_bins import resolve_pvalue_bins
from .run_manifest import PlanManifest, RunManifest
from .run_paths import (
    ensure_run_meta_dir,
    inputs_manifest_path,
    run_manifest_path,
    run_outputs_root,
    run_state_path,
)
from .run_state import RunState, load_run_state
from .runtime_policy import RuntimePolicy
from .sampler import TFSampler
from .seeding import derive_seed_map

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    total_generated: int
    per_plan: dict[tuple[str, str], int]


def _write_run_state(
    path: Path,
    *,
    run_id: str,
    schema_version: str,
    config_sha256: str,
    run_root: str,
    counts: dict[tuple[str, str], int],
    created_at: str,
) -> None:
    state = RunState.from_counts(
        run_id=run_id,
        schema_version=schema_version,
        config_sha256=config_sha256,
        run_root=run_root,
        counts=counts,
        created_at=created_at,
    )
    state.write_json(path)


@dataclass(frozen=True)
class PipelineDeps:
    source_factory: Callable[[object, Path], BaseDataSource]
    sink_factory: Callable[[DenseGenConfig, Path], Iterable[SinkBase]]
    optimizer: OptimizerAdapter
    gap_fill: Callable[..., tuple[str, dict] | str]


def default_deps() -> PipelineDeps:
    return PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=build_sinks,
        optimizer=DenseArraysAdapter(),
        gap_fill=random_fill,
    )


def resolve_plan(loaded: LoadedConfig) -> List[ResolvedPlanItem]:
    return loaded.root.densegen.generation.resolve_plan()


def select_solver(
    preferred: str | None,
    optimizer: OptimizerAdapter,
    *,
    strategy: str,
    fallback_to_cbc: bool = False,
    test_length: int = 10,
) -> str | None:
    """
    Probe the requested solver once. If it fails, optionally fall back to CBC.
    """
    if strategy == "approximate":
        return preferred
    if not preferred:
        raise ValueError("solver.backend is required unless strategy=approximate")
    try:
        optimizer.probe_solver(preferred, test_length=test_length)
        return preferred
    except Exception as exc:
        if fallback_to_cbc and str(preferred).upper() != "CBC":
            log.warning(
                "Requested solver '%s' failed; falling back to CBC (solver.fallback_to_cbc=true).",
                preferred,
            )
            optimizer.probe_solver("CBC", test_length=test_length)
            return "CBC"
        raise RuntimeError(
            f"Requested solver '{preferred}' failed during probe: {exc}\n"
            "Please install/configure this solver or choose another in solver.backend."
        ) from exc


def _summarize_tf_counts(labels: List[str], max_items: int = 6) -> str:
    if not labels:
        return ""
    counts = Counter(labels)
    items = [f"{tf} x {n}" for tf, n in counts.most_common(max_items)]
    extra = len(counts) - min(len(counts), max_items)
    return ", ".join(items) + (f" (+{extra} TFs)" if extra > 0 else "")


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


def _resolve_pvalue_bins_meta(sampling) -> list[float] | None:
    if sampling is None:
        return None
    backend = str(_sampling_attr(sampling, "scoring_backend") or "densegen").lower()
    bins = _sampling_attr(sampling, "pvalue_bins")
    if backend == "fimo":
        return resolve_pvalue_bins(bins)
    if bins is None:
        return None
    return [float(v) for v in bins]


def _extract_pwm_sampling_config(source_cfg) -> dict | None:
    sampling = getattr(source_cfg, "sampling", None)
    if sampling is None:
        return None
    n_sites = _sampling_attr(sampling, "n_sites")
    oversample = _sampling_attr(sampling, "oversample_factor")
    max_candidates = _sampling_attr(sampling, "max_candidates")
    requested = None
    generated = None
    capped = False
    backend = str(_sampling_attr(sampling, "scoring_backend") or "densegen").lower()
    if isinstance(n_sites, int) and isinstance(oversample, int):
        requested = int(n_sites) * int(oversample)
        generated = requested
        if backend == "fimo":
            mining_cfg = _sampling_attr(sampling, "mining")
            mining_max_candidates = _mining_attr(mining_cfg, "max_candidates")
            if mining_max_candidates is not None:
                try:
                    cap_val = int(mining_max_candidates)
                except Exception:
                    cap_val = None
                if cap_val is not None:
                    generated = min(requested, cap_val)
                    capped = generated < requested
        else:
            if max_candidates is not None:
                try:
                    cap_val = int(max_candidates)
                except Exception:
                    cap_val = None
                if cap_val is not None:
                    generated = min(requested, cap_val)
                    capped = generated < requested
    length_range = _sampling_attr(sampling, "length_range")
    if length_range is not None:
        length_range = list(length_range)
    mining = _sampling_attr(sampling, "mining")
    mining_batch_size = _mining_attr(mining, "batch_size")
    mining_max_batches = _mining_attr(mining, "max_batches")
    mining_max_candidates = _mining_attr(mining, "max_candidates")
    mining_max_seconds = _mining_attr(mining, "max_seconds")
    mining_retain_bin_ids = _mining_attr(mining, "retain_bin_ids")
    mining_log_every_batches = _mining_attr(mining, "log_every_batches")
    return {
        "strategy": _sampling_attr(sampling, "strategy"),
        "scoring_backend": _sampling_attr(sampling, "scoring_backend"),
        "n_sites": _sampling_attr(sampling, "n_sites"),
        "oversample_factor": _sampling_attr(sampling, "oversample_factor"),
        "max_candidates": _sampling_attr(sampling, "max_candidates"),
        "max_seconds": _sampling_attr(sampling, "max_seconds"),
        "requested_candidates": requested,
        "generated_candidates": generated,
        "capped": capped,
        "score_threshold": _sampling_attr(sampling, "score_threshold"),
        "score_percentile": _sampling_attr(sampling, "score_percentile"),
        "pvalue_threshold": _sampling_attr(sampling, "pvalue_threshold"),
        "pvalue_bins": _resolve_pvalue_bins_meta(sampling),
        "selection_policy": _sampling_attr(sampling, "selection_policy"),
        "bgfile": _sampling_attr(sampling, "bgfile"),
        "keep_all_candidates_debug": _sampling_attr(sampling, "keep_all_candidates_debug"),
        "length_policy": _sampling_attr(sampling, "length_policy"),
        "length_range": length_range,
        "mining": {
            "batch_size": mining_batch_size,
            "max_batches": mining_max_batches,
            "max_candidates": mining_max_candidates,
            "max_seconds": mining_max_seconds,
            "retain_bin_ids": mining_retain_bin_ids,
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


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    while True:
        if (current / "uv.lock").exists() or (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _dense_arrays_version_from_uv_lock(root: Path) -> str | None:
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        return None
    try:
        data = tomllib.loads(lock_path.read_text())
    except Exception:
        return None
    packages = data.get("package", [])
    for pkg in packages:
        name = str(pkg.get("name", "")).lower()
        if name not in {"dense-arrays", "dense_arrays"}:
            continue
        version = pkg.get("version")
        if isinstance(version, str) and version:
            source = pkg.get("source") or {}
            if isinstance(source, dict):
                git_url = source.get("git")
                if isinstance(git_url, str) and "#" in git_url:
                    rev = git_url.split("#")[-1]
                    if rev:
                        return f"{version}+git.{rev[:7]}"
            return version
    return None


def _dense_arrays_version_from_pyproject(root: Path) -> str | None:
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        data = tomllib.loads(pyproject_path.read_text())
    except Exception:
        return None
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        dep_str = str(dep).strip()
        if dep_str.startswith("dense-arrays") or dep_str.startswith("dense_arrays"):
            return dep_str
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if isinstance(sources, dict):
        entry = sources.get("dense-arrays") or sources.get("dense_arrays")
        if isinstance(entry, dict):
            git_url = entry.get("git")
            if isinstance(git_url, str) and git_url:
                return f"git:{git_url}"
    return None


def _resolve_dense_arrays_version(cfg_path: Path) -> tuple[str | None, str]:
    try:
        import dense_arrays as da  # type: ignore

        version = getattr(da, "__version__", None)
        if isinstance(version, str) and version:
            return version, "installed"
    except Exception:
        pass
    for pkg_name in ("dense-arrays", "dense_arrays"):
        try:
            version = importlib.metadata.version(pkg_name)
            return version, "installed"
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            break
    root = _find_project_root(cfg_path)
    if root is None:
        root = _find_project_root(Path(__file__).resolve())
    if root is not None:
        version = _dense_arrays_version_from_uv_lock(root)
        if version:
            return version, "lock"
        version = _dense_arrays_version_from_pyproject(root)
        if version:
            return version, "pyproject"
    return None, "unknown"


def _min_count_by_regulator(labels: list[str] | None, min_count_per_tf: int) -> dict[str, int] | None:
    if not labels or min_count_per_tf <= 0:
        return None
    return {tf: int(min_count_per_tf) for tf in sorted(set(labels))}


def _merge_min_counts(base: dict[str, int] | None, override: dict[str, int] | None) -> dict[str, int] | None:
    if not base and not override:
        return None
    merged = dict(base or {})
    for key, val in (override or {}).items():
        merged[key] = max(int(val), merged.get(key, 0))
    return merged


def _extract_side_biases(fixed_elements) -> tuple[list[str], list[str]]:
    if fixed_elements is None:
        return [], []
    if hasattr(fixed_elements, "side_biases"):
        sb = getattr(fixed_elements, "side_biases")
    else:
        sb = (fixed_elements or {}).get("side_biases")
    if sb is None:
        return [], []
    if hasattr(sb, "left") or hasattr(sb, "right"):
        left = list(getattr(sb, "left", []) or [])
        right = list(getattr(sb, "right", []) or [])
    else:
        left = list((sb or {}).get("left") or [])
        right = list((sb or {}).get("right") or [])
    return left, right


def _fixed_elements_dump(fixed_elements) -> dict:
    if fixed_elements is None:
        return {
            "promoter_constraints": [],
            "side_biases": {"left": [], "right": []},
        }
    if hasattr(fixed_elements, "model_dump"):
        dump = fixed_elements.model_dump()
    else:
        dump = dict(fixed_elements or {})

    pcs_raw = dump.get("promoter_constraints") or []
    pcs = []
    keys = ("name", "upstream", "downstream", "spacer_length", "upstream_pos", "downstream_pos")
    for pc in pcs_raw:
        if hasattr(pc, "model_dump"):
            pc = pc.model_dump()
        if isinstance(pc, dict):
            pcs.append({k: pc.get(k) for k in keys})

    sb_raw = dump.get("side_biases") or {}
    if hasattr(sb_raw, "model_dump"):
        sb_raw = sb_raw.model_dump()
    left = list((sb_raw or {}).get("left") or [])
    right = list((sb_raw or {}).get("right") or [])

    return {"promoter_constraints": pcs, "side_biases": {"left": left, "right": right}}


def _max_fixed_element_len(fixed_elements_dump: dict) -> int:
    max_len = 0
    pcs = fixed_elements_dump.get("promoter_constraints") or []
    for pc in pcs:
        if not isinstance(pc, dict):
            continue
        for key in ("upstream", "downstream"):
            seq = pc.get(key)
            if isinstance(seq, str):
                seq = seq.strip().upper()
                if seq:
                    max_len = max(max_len, len(seq))
    return max_len


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
            meta["input_pwm_score_threshold"] = getattr(sampling, "score_threshold", None)
            meta["input_pwm_score_percentile"] = getattr(sampling, "score_percentile", None)
            meta["input_pwm_pvalue_threshold"] = getattr(sampling, "pvalue_threshold", None)
            meta["input_pwm_pvalue_bins"] = _resolve_pvalue_bins_meta(sampling)
            mining_cfg = getattr(sampling, "mining", None)
            retained_bins = _mining_attr(mining_cfg, "retain_bin_ids")
            meta["input_pwm_mining_batch_size"] = _mining_attr(mining_cfg, "batch_size")
            meta["input_pwm_mining_max_batches"] = _mining_attr(mining_cfg, "max_batches")
            meta["input_pwm_mining_max_candidates"] = _mining_attr(mining_cfg, "max_candidates")
            meta["input_pwm_mining_max_seconds"] = _mining_attr(mining_cfg, "max_seconds")
            meta["input_pwm_mining_retain_bin_ids"] = retained_bins
            meta["input_pwm_mining_log_every_batches"] = _mining_attr(mining_cfg, "log_every_batches")
            meta["input_pwm_selection_policy"] = getattr(sampling, "selection_policy", None)
            meta["input_pwm_bgfile"] = getattr(sampling, "bgfile", None)
            meta["input_pwm_keep_all_candidates_debug"] = getattr(sampling, "keep_all_candidates_debug", None)
            meta["input_pwm_include_matched_sequence"] = getattr(sampling, "include_matched_sequence", None)
            meta["input_pwm_n_sites"] = getattr(sampling, "n_sites", None)
            meta["input_pwm_oversample_factor"] = getattr(sampling, "oversample_factor", None)
    else:
        meta["input_mode"] = source_type
        meta["input_pwm_ids"] = []
    return meta


def _compute_used_tf_info(
    sol,
    library_for_opt,
    regulator_labels,
    fixed_elements,
    site_id_by_index,
    source_by_index,
    tfbs_id_by_index,
    motif_id_by_index,
):
    promoter_motifs = set()
    if fixed_elements is not None:
        if hasattr(fixed_elements, "promoter_constraints"):
            pcs = getattr(fixed_elements, "promoter_constraints") or []
        else:
            pcs = (fixed_elements or {}).get("promoter_constraints") or []
        for pc in pcs:
            if hasattr(pc, "upstream") or hasattr(pc, "downstream"):
                up = getattr(pc, "upstream", None)
                dn = getattr(pc, "downstream", None)
                for v in (up, dn):
                    if isinstance(v, str) and v.strip():
                        promoter_motifs.add(v.strip().upper())
            elif isinstance(pc, dict):
                for k in ("upstream", "downstream"):
                    v = pc.get(k)
                    if isinstance(v, str) and v.strip():
                        promoter_motifs.add(v.strip().upper())

    lib = getattr(sol, "library", [])
    orig_n = len(library_for_opt)
    used_simple: list[str] = []
    used_detail: list[dict] = []
    counts: dict[str, int] = {}
    used_tf_set: set[str] = set()

    for offset, idx in sol.offset_indices_in_order():
        base_idx = idx % len(lib)
        orientation = "fwd" if idx < len(lib) else "rev"
        motif = lib[base_idx]
        if motif in promoter_motifs or base_idx >= orig_n:
            continue
        tf_label = (
            regulator_labels[base_idx] if regulator_labels is not None and base_idx < len(regulator_labels) else ""
        )
        tfbs = motif
        used_simple.append(f"{tf_label}:{tfbs}" if tf_label else tfbs)
        entry = {
            "tf": tf_label,
            "tfbs": tfbs,
            "orientation": orientation,
            "offset": int(offset),
            "offset_raw": int(offset),
            "length": len(tfbs),
            "end": int(offset) + len(tfbs),
            "pad_left": 0,
        }
        if site_id_by_index is not None and base_idx < len(site_id_by_index):
            site_id = site_id_by_index[base_idx]
            if site_id is not None:
                entry["site_id"] = site_id
        if source_by_index is not None and base_idx < len(source_by_index):
            source = source_by_index[base_idx]
            if source is not None:
                entry["source"] = source
        if tfbs_id_by_index is not None and base_idx < len(tfbs_id_by_index):
            tfbs_id = tfbs_id_by_index[base_idx]
            if tfbs_id is not None:
                entry["tfbs_id"] = tfbs_id
        if motif_id_by_index is not None and base_idx < len(motif_id_by_index):
            motif_id = motif_id_by_index[base_idx]
            if motif_id is not None:
                entry["motif_id"] = motif_id
        used_detail.append(entry)
        if tf_label:
            counts[tf_label] = counts.get(tf_label, 0) + 1
            used_tf_set.add(tf_label)
    return used_simple, used_detail, counts, sorted(used_tf_set)


def _parse_used_tfbs_detail(val) -> list[dict]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                val = json.loads(s)
            except Exception as exc:
                raise ValueError(f"Failed to parse used_tfbs_detail JSON: {s[:120]}") from exc
    if isinstance(val, (list, tuple, np.ndarray)):
        out: list[dict] = []
        for item in list(val):
            if isinstance(item, dict):
                out.append(item)
        return out
    return []


def _update_usage_counts(
    usage_counts: dict[tuple[str, str], int],
    used_tfbs_detail: list[dict],
) -> None:
    for entry in used_tfbs_detail:
        tf = str(entry.get("tf") or "").strip()
        tfbs = str(entry.get("tfbs") or "").strip()
        if not tf or not tfbs:
            continue
        key = (tf, tfbs)
        usage_counts[key] = int(usage_counts.get(key, 0)) + 1


def _summarize_leaderboard(counts: dict, *, top: int = 5) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))
    items = items[: max(1, int(top))]
    parts = []
    for key, val in items:
        if isinstance(key, tuple):
            key_label = f"{key[0]}:{key[1]}"
        else:
            key_label = str(key)
        parts.append(f"{key_label}={int(val)}")
    return ", ".join(parts) if parts else "-"


def _leaderboard_items(counts: dict, *, top: int = 5) -> list[dict]:
    if not counts:
        return []
    items = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))
    items = items[: max(1, int(top))]
    out: list[dict] = []
    for key, val in items:
        if isinstance(key, tuple):
            tf = str(key[0])
            tfbs = str(key[1])
            out.append({"tf": tf, "tfbs": tfbs, "count": int(val)})
        else:
            out.append({"tf": str(key), "count": int(val)})
    return out


def _format_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[?]"
    filled = int(width * (current / max(1, total)))
    filled = min(width, max(0, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _short_seq(value: str, *, max_len: int = 16) -> str:
    if not value:
        return "-"
    if len(value) <= max_len:
        return value
    keep = max(1, max_len - 3)
    return value[:keep] + "..."


def _summarize_failure_totals(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
) -> str:
    total = 0
    unique = 0
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in reasons.values())
        if count > 0:
            total += count
            unique += 1
    if total <= 0:
        return "failed_sites=0 total_failures=0"
    return f"failed_sites={unique} total_failures={total}"


def _summarize_failure_leaderboard(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
    top: int = 5,
) -> str:
    if not failure_counts:
        return "-"
    totals: dict[tuple[str, str], int] = {}
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in reasons.values())
        if count <= 0:
            continue
        key = (str(tf), str(tfbs))
        totals[key] = totals.get(key, 0) + int(count)
    if not totals:
        return "-"
    items = sorted(totals.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    items = items[: max(1, int(top))]
    parts = []
    for (tf, tfbs), count in items:
        parts.append(f"{tf}:{_short_seq(tfbs)}={int(count)}")
    return ", ".join(parts) if parts else "-"


def _failure_leaderboard_items(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
    top: int = 5,
) -> list[dict]:
    if not failure_counts:
        return []
    totals: dict[tuple[str, str], int] = {}
    reasons_by_key: dict[tuple[str, str], dict[str, int]] = {}
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in (reasons or {}).values())
        if count <= 0:
            continue
        key = (str(tf), str(tfbs))
        totals[key] = totals.get(key, 0) + int(count)
        reason_counts = reasons_by_key.setdefault(key, {})
        for reason, n in (reasons or {}).items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(n)
    if not totals:
        return []
    items = sorted(totals.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    items = items[: max(1, int(top))]
    out: list[dict] = []
    for (tf, tfbs), count in items:
        reasons = reasons_by_key.get((tf, tfbs), {})
        top_reason = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else ""
        out.append({"tf": tf, "tfbs": tfbs, "failures": int(count), "top_reason": top_reason})
    return out


def _aggregate_failure_counts_for_sampling(
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
) -> dict[tuple[str, str], int]:
    if not failure_counts:
        return {}
    totals: dict[tuple[str, str], int] = {}
    for (inp, plan, tf, tfbs, _site_id), reasons in failure_counts.items():
        if inp != input_name or plan != plan_name:
            continue
        if not tfbs:
            continue
        count = sum(int(v) for v in (reasons or {}).values())
        if count <= 0:
            continue
        key = (str(tf), str(tfbs))
        totals[key] = totals.get(key, 0) + int(count)
    return totals


def _normalized_entropy(counts: dict) -> float | None:
    values = np.array(list(counts.values()), dtype=float)
    if values.size == 0:
        return None
    total = float(values.sum())
    if total <= 0:
        return None
    p = values / total
    ent = -np.sum(p * np.log(p))
    max_ent = math.log(len(values)) if len(values) > 1 else 0.0
    if max_ent <= 0:
        return 0.0
    return float(ent / max_ent)


def _summarize_diversity(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    *,
    library_tfs: list[str],
    library_tfbs: list[str],
) -> str:
    lib_tf_count = len(set(library_tfs)) if library_tfs else 0
    if library_tfs:
        lib_tfbs_count = len(set(zip(library_tfs, library_tfbs)))
    else:
        lib_tfbs_count = len(set(library_tfbs))
    used_tf_count = len(tf_usage_counts)
    used_tfbs_count = len(usage_counts)
    tf_cov = used_tf_count / max(1, lib_tf_count) if lib_tf_count else 0.0
    tfbs_cov = used_tfbs_count / max(1, lib_tfbs_count) if lib_tfbs_count else 0.0
    ent = _normalized_entropy(usage_counts)
    ent_label = f"{ent:.3f}" if ent is not None else "n/a"
    return (
        f"tf_coverage={tf_cov:.2f} ({used_tf_count}/{lib_tf_count}) | "
        f"tfbs_coverage={tfbs_cov:.2f} ({used_tfbs_count}/{lib_tfbs_count}) | "
        f"tfbs_entropy={ent_label}"
    )


def _diversity_snapshot(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    *,
    library_tfs: list[str],
    library_tfbs: list[str],
) -> dict[str, object]:
    lib_tf_count = len(set(library_tfs)) if library_tfs else 0
    if library_tfs:
        lib_tfbs_count = len(set(zip(library_tfs, library_tfbs)))
    else:
        lib_tfbs_count = len(set(library_tfbs))
    used_tf_count = len(tf_usage_counts)
    used_tfbs_count = len(usage_counts)
    tf_cov = used_tf_count / max(1, lib_tf_count) if lib_tf_count else 0.0
    tfbs_cov = used_tfbs_count / max(1, lib_tfbs_count) if lib_tfbs_count else 0.0
    ent = _normalized_entropy(usage_counts)
    return {
        "tf_coverage": float(tf_cov),
        "tfbs_coverage": float(tfbs_cov),
        "tfbs_entropy": float(ent) if ent is not None else None,
        "used_tf_count": int(used_tf_count),
        "library_tf_count": int(lib_tf_count),
        "used_tfbs_count": int(used_tfbs_count),
        "library_tfbs_count": int(lib_tfbs_count),
    }


def _leaderboard_snapshot(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]],
    *,
    input_name: str,
    plan_name: str,
    library_tfs: list[str],
    library_tfbs: list[str],
    top: int = 5,
) -> dict[str, object]:
    return {
        "tf": _leaderboard_items(tf_usage_counts, top=top),
        "tfbs": _leaderboard_items(usage_counts, top=top),
        "failed_tfbs": _failure_leaderboard_items(failure_counts, input_name=input_name, plan_name=plan_name, top=top),
        "diversity": _diversity_snapshot(
            usage_counts,
            tf_usage_counts,
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
        ),
    }


def _apply_gap_fill_offsets(used_tfbs_detail: list[dict], gap_meta: dict) -> list[dict]:
    pad_left = 0
    if gap_meta.get("used") and gap_meta.get("end") == "5prime":
        pad_left = int(gap_meta.get("bases") or 0)
    for entry in used_tfbs_detail:
        offset_raw = int(entry.get("offset_raw", entry.get("offset", 0)))
        length = int(entry.get("length", len(entry.get("tfbs") or "")))
        offset = offset_raw + pad_left
        entry["offset_raw"] = offset_raw
        entry["pad_left"] = pad_left
        entry["length"] = length
        entry["offset"] = offset
        entry["end"] = offset + length
    return used_tfbs_detail


def _hash_library(
    library_for_opt: list[str],
    regulator_labels: list[str] | None,
    site_id_by_index: list[str | None] | None,
    source_by_index: list[str | None] | None,
) -> str:
    parts: list[str] = []
    for idx, motif in enumerate(library_for_opt):
        label = ""
        if regulator_labels is not None and idx < len(regulator_labels):
            label = str(regulator_labels[idx])
        site_id = None
        if site_id_by_index is not None and idx < len(site_id_by_index):
            site_id = site_id_by_index[idx]
        source = None
        if source_by_index is not None and idx < len(source_by_index):
            source = source_by_index[idx]
        payload = "\t".join(
            [
                str(motif),
                label,
                str(site_id) if site_id is not None else "None",
                str(source) if source is not None else "None",
            ]
        )
        parts.append(payload)
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def build_library_for_plan(
    *,
    source_label: str,
    plan_item: ResolvedPlanItem,
    pool: PoolData,
    sampling_cfg: object,
    seq_len: int,
    min_count_per_tf: int,
    usage_counts: dict[tuple[str, str], int],
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None,
    rng: random.Random,
    np_rng: np.random.Generator,
    schema_is_22: bool,
    library_index_start: int,
) -> tuple[list[str], list[str], list[str], dict]:
    pool_strategy = str(getattr(sampling_cfg, "pool_strategy", "subsample"))
    library_size = int(getattr(sampling_cfg, "library_size", 0))
    subsample_over = int(getattr(sampling_cfg, "subsample_over_length_budget_by", 0))
    library_sampling_strategy = str(getattr(sampling_cfg, "library_sampling_strategy", "tf_balanced"))
    cover_all_tfs = bool(getattr(sampling_cfg, "cover_all_regulators", True))
    unique_binding_sites = bool(getattr(sampling_cfg, "unique_binding_sites", True))
    max_sites_per_tf = getattr(sampling_cfg, "max_sites_per_regulator", None)
    relax_on_exhaustion = bool(getattr(sampling_cfg, "relax_on_exhaustion", False))
    allow_incomplete_coverage = bool(getattr(sampling_cfg, "allow_incomplete_coverage", False))
    iterative_max_libraries = int(getattr(sampling_cfg, "iterative_max_libraries", 0))
    iterative_min_new_solutions = int(getattr(sampling_cfg, "iterative_min_new_solutions", 0))

    data_entries = list(pool.sequences or [])
    meta_df = pool.df if pool.pool_mode == POOL_MODE_TFBS else None

    fixed_elements = plan_item.fixed_elements
    required_regulators = list(dict.fromkeys(plan_item.required_regulators or []))
    min_required_regulators = plan_item.min_required_regulators
    plan_min_count_by_regulator = dict(plan_item.min_count_by_regulator or {})
    k_required = int(min_required_regulators) if min_required_regulators is not None else None
    k_of_required = bool(required_regulators) and k_required is not None
    if k_of_required and k_required > len(required_regulators):
        raise ValueError(
            "min_required_regulators cannot exceed required_regulators size "
            f"({k_required} > {len(required_regulators)})."
        )
    side_left, side_right = _extract_side_biases(fixed_elements)
    required_bias_motifs = list(dict.fromkeys([*side_left, *side_right]))

    libraries_built = int(library_index_start)

    def _finalize(
        library: list[str],
        parts: list[str],
        reg_labels: list[str],
        info: dict,
        *,
        site_id_by_index: list[str | None] | None,
        source_by_index: list[str | None] | None,
        tfbs_id_by_index: list[str | None] | None,
        motif_id_by_index: list[str | None] | None,
    ) -> tuple[list[str], list[str], list[str], dict]:
        nonlocal libraries_built
        libraries_built += 1
        info["library_index"] = libraries_built
        info["library_hash"] = _hash_library(library, reg_labels, site_id_by_index, source_by_index)
        info["site_id_by_index"] = site_id_by_index
        info["source_by_index"] = source_by_index
        info["tfbs_id_by_index"] = tfbs_id_by_index
        info["motif_id_by_index"] = motif_id_by_index
        return library, parts, reg_labels, info

    if meta_df is not None and isinstance(meta_df, pd.DataFrame):
        available_tfs = set(meta_df["tf"].tolist())
        tfbs_counts = (
            meta_df.groupby("tf")["tfbs"].nunique() if unique_binding_sites else meta_df.groupby("tf")["tfbs"].size()
        )
        missing = [t for t in required_regulators if t not in available_tfs]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"Required regulators not found in input: {preview}")
        if plan_min_count_by_regulator:
            missing_counts = [t for t in plan_min_count_by_regulator if t not in available_tfs]
            if missing_counts:
                preview = ", ".join(missing_counts[:10])
                raise ValueError(f"min_count_by_regulator TFs not found in input: {preview}")
            for tf, min_count in plan_min_count_by_regulator.items():
                max_allowed = int(tfbs_counts.get(tf, 0))
                if max_sites_per_tf is not None:
                    max_allowed = min(max_allowed, int(max_sites_per_tf))
                if library_size > 0:
                    max_allowed = min(max_allowed, int(library_size))
                if int(min_count) > max_allowed:
                    raise ValueError(
                        f"min_count_by_regulator[{tf}]={min_count} exceeds available sites ({max_allowed}). "
                        "Increase library_size, relax min_count_by_regulator, or allow non-unique binding sites."
                    )
        if min_required_regulators is not None:
            if not required_regulators and min_required_regulators > len(available_tfs):
                raise ValueError(
                    f"min_required_regulators={min_required_regulators} exceeds available regulators "
                    f"({len(available_tfs)})."
                )
        if pool_strategy in {"subsample", "iterative_subsample"} and cover_all_tfs and not allow_incomplete_coverage:
            if library_size > 0 and library_size < len(available_tfs):
                raise ValueError(
                    "library_size is too small to cover all regulators. "
                    f"library_size={library_size} but available_tfs={len(available_tfs)}. "
                    "Increase library_size or allow_incomplete_coverage."
                )

        if pool_strategy == "full":
            lib_df = meta_df.copy()
            if unique_binding_sites:
                lib_df = lib_df.drop_duplicates(["tf", "tfbs"])
            if required_bias_motifs:
                missing_bias = [m for m in required_bias_motifs if m not in set(lib_df["tfbs"])]
                if missing_bias:
                    preview = ", ".join(missing_bias[:10])
                    raise ValueError(f"Required side-bias motifs not found in input: {preview}")
            lib_df = lib_df.reset_index(drop=True)
            library = lib_df["tfbs"].tolist()
            reg_labels = lib_df["tf"].tolist()
            parts = [f"{tf}:{tfbs}" for tf, tfbs in zip(reg_labels, lib_df["tfbs"].tolist())]
            site_id_by_index = lib_df["site_id"].tolist() if "site_id" in lib_df.columns else None
            source_by_index = lib_df["source"].tolist() if "source" in lib_df.columns else None
            tfbs_id_by_index = lib_df["tfbs_id"].tolist() if "tfbs_id" in lib_df.columns else None
            motif_id_by_index = lib_df["motif_id"].tolist() if "motif_id" in lib_df.columns else None
            info = {
                "target_length": seq_len + subsample_over,
                "achieved_length": sum(len(s) for s in library),
                "relaxed_cap": False,
                "final_cap": None,
                "pool_strategy": pool_strategy,
                "library_size": len(library),
                "iterative_max_libraries": iterative_max_libraries,
                "iterative_min_new_solutions": iterative_min_new_solutions,
            }
            return _finalize(
                library,
                parts,
                reg_labels,
                info,
                site_id_by_index=site_id_by_index,
                source_by_index=source_by_index,
                tfbs_id_by_index=tfbs_id_by_index,
                motif_id_by_index=motif_id_by_index,
            )

        sampler = TFSampler(meta_df, np_rng)
        required_regulators_selected = required_regulators
        if k_of_required:
            candidates = sorted(required_regulators)
            if k_required is not None and k_required < len(candidates):
                chosen = np_rng.choice(len(candidates), size=k_required, replace=False)
                required_regulators_selected = sorted([candidates[int(i)] for i in chosen])
            else:
                required_regulators_selected = candidates
        required_tfs_for_library = list(
            dict.fromkeys([*required_regulators_selected, *plan_min_count_by_regulator.keys()])
        )
        if min_required_regulators is not None and not required_regulators:
            if pool_strategy in {"subsample", "iterative_subsample"}:
                if library_size < int(min_required_regulators):
                    raise ValueError(
                        "library_size is too small to satisfy min_required_regulators when "
                        f"required_regulators is empty. library_size={library_size} "
                        f"min_required_regulators={min_required_regulators}. "
                        "Increase library_size or lower min_required_regulators."
                    )
        if pool_strategy in {"subsample", "iterative_subsample"}:
            required_slots = len(required_bias_motifs) + len(required_tfs_for_library)
            if library_size < required_slots:
                raise ValueError(
                    "library_size is too small for required motifs. "
                    f"library_size={library_size} but required_tfbs={len(required_bias_motifs)} "
                    f"+ required_tfs={len(required_tfs_for_library)} "
                    f"(min_required_regulators={min_required_regulators}). "
                    "Increase library_size or relax required constraints."
                )
        if schema_is_22 and pool_strategy in {"subsample", "iterative_subsample"}:
            failure_counts_by_tfbs: dict[tuple[str, str], int] | None = None
            if library_sampling_strategy == "coverage_weighted" and getattr(sampling_cfg, "avoid_failed_motifs", False):
                failure_counts_by_tfbs = _aggregate_failure_counts_for_sampling(
                    failure_counts,
                    input_name=source_label,
                    plan_name=plan_item.name,
                )
            library, parts, reg_labels, info = sampler.generate_binding_site_library(
                library_size,
                sequence_length=seq_len,
                budget_overhead=subsample_over,
                required_tfbs=required_bias_motifs,
                required_tfs=required_tfs_for_library,
                cover_all_tfs=cover_all_tfs,
                unique_binding_sites=unique_binding_sites,
                max_sites_per_tf=max_sites_per_tf,
                relax_on_exhaustion=relax_on_exhaustion,
                allow_incomplete_coverage=allow_incomplete_coverage,
                sampling_strategy=library_sampling_strategy,
                usage_counts=usage_counts if library_sampling_strategy == "coverage_weighted" else None,
                coverage_boost_alpha=float(getattr(sampling_cfg, "coverage_boost_alpha", 0.15)),
                coverage_boost_power=float(getattr(sampling_cfg, "coverage_boost_power", 1.0)),
                failure_counts=failure_counts_by_tfbs,
                avoid_failed_motifs=bool(getattr(sampling_cfg, "avoid_failed_motifs", False)),
                failure_penalty_alpha=float(getattr(sampling_cfg, "failure_penalty_alpha", 0.5)),
                failure_penalty_power=float(getattr(sampling_cfg, "failure_penalty_power", 1.0)),
            )
        else:
            library, parts, reg_labels, info = sampler.generate_binding_site_subsample(
                seq_len,
                subsample_over,
                required_tfbs=required_bias_motifs,
                required_tfs=required_tfs_for_library,
                cover_all_tfs=cover_all_tfs,
                unique_binding_sites=unique_binding_sites,
                max_sites_per_tf=max_sites_per_tf,
                relax_on_exhaustion=relax_on_exhaustion,
                allow_incomplete_coverage=allow_incomplete_coverage,
            )
        info.update(
            {
                "pool_strategy": pool_strategy,
                "library_size": library_size,
                "library_sampling_strategy": library_sampling_strategy,
                "coverage_boost_alpha": float(getattr(sampling_cfg, "coverage_boost_alpha", 0.15)),
                "coverage_boost_power": float(getattr(sampling_cfg, "coverage_boost_power", 1.0)),
                "iterative_max_libraries": iterative_max_libraries,
                "iterative_min_new_solutions": iterative_min_new_solutions,
                "required_regulators_selected": required_regulators_selected if k_of_required else None,
            }
        )
        site_id_by_index = info.get("site_id_by_index")
        source_by_index = info.get("source_by_index")
        tfbs_id_by_index = info.get("tfbs_id_by_index")
        motif_id_by_index = info.get("motif_id_by_index")
        return _finalize(
            library,
            parts,
            reg_labels,
            info,
            site_id_by_index=site_id_by_index,
            source_by_index=source_by_index,
            tfbs_id_by_index=tfbs_id_by_index,
            motif_id_by_index=motif_id_by_index,
        )

    if required_regulators or plan_min_count_by_regulator or min_required_regulators is not None:
        preview = ", ".join(required_regulators[:10]) if required_regulators else "n/a"
        raise ValueError(
            "Regulator constraints are set (required/min_count/min_required) "
            "but the input does not provide regulators. "
            f"required_regulators={preview}."
        )
    all_sequences = [s for s in data_entries]
    if not all_sequences:
        raise ValueError(f"No sequences found for source {source_label}")
    pool = list(dict.fromkeys(all_sequences)) if unique_binding_sites else list(all_sequences)
    if pool_strategy == "full":
        if required_bias_motifs:
            missing = [m for m in required_bias_motifs if m not in pool]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required side-bias motifs not found in sequences input: {preview}")
        library = pool
    else:
        if library_size > len(pool):
            raise ValueError(f"library_size={library_size} exceeds available unique sequences ({len(pool)}).")
        take = min(max(1, int(library_size)), len(pool))
        if required_bias_motifs:
            missing = [m for m in required_bias_motifs if m not in pool]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required side-bias motifs not found in sequences input: {preview}")
            if take < len(required_bias_motifs):
                raise ValueError(
                    f"library_size={take} is smaller than required side_biases ({len(required_bias_motifs)})."
                )
            required_set = set(required_bias_motifs)
            remaining = [s for s in pool if s not in required_set]
            library = list(required_bias_motifs) + rng.sample(remaining, take - len(required_bias_motifs))
        else:
            library = rng.sample(pool, take)
    tf_parts: list[str] = []
    reg_labels: list[str] = []
    info = {
        "target_length": seq_len + subsample_over,
        "achieved_length": sum(len(s) for s in library),
        "relaxed_cap": False,
        "final_cap": None,
        "pool_strategy": pool_strategy,
        "library_size": len(library) if pool_strategy == "full" else library_size,
        "iterative_max_libraries": iterative_max_libraries,
        "iterative_min_new_solutions": iterative_min_new_solutions,
    }
    tfbs_id_by_index = [
        hash_tfbs_id(motif_id=None, sequence=seq, scoring_backend="sequence_library") for seq in library
    ]
    return _finalize(
        library,
        tf_parts,
        reg_labels,
        info,
        site_id_by_index=None,
        source_by_index=None,
        tfbs_id_by_index=tfbs_id_by_index,
        motif_id_by_index=None,
    )


def _compute_sampling_fraction(
    library: list[str],
    *,
    input_tfbs_count: int,
    pool_strategy: str,
) -> float | None:
    if pool_strategy == "full":
        return 1.0
    if input_tfbs_count > 0:
        return len(set(library)) / float(input_tfbs_count)
    return None


def _compute_sampling_fraction_pairs(
    library: list[str],
    regulator_labels: list[str] | None,
    *,
    input_pair_count: int | None,
    pool_strategy: str,
) -> float | None:
    if input_pair_count is None or input_pair_count <= 0:
        return None
    if not regulator_labels:
        return None
    if pool_strategy == "full":
        return 1.0
    pairs = set(zip(regulator_labels[: len(library)], library))
    return len(pairs) / float(input_pair_count)


def _consolidate_parts(outputs_root: Path, *, part_glob: str, final_name: str) -> bool:
    parts = sorted(outputs_root.glob(part_glob))
    if not parts:
        return False
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required to consolidate parquet parts.") from exc
    final_path = outputs_root / final_name
    sources = [str(p) for p in parts]
    if final_path.exists():
        sources.insert(0, str(final_path))
    dataset = ds.dataset(sources, format="parquet")
    tmp_path = outputs_root / f".{final_name}.tmp"
    writer = pq.ParquetWriter(tmp_path, schema=dataset.schema)
    scanner = ds.Scanner.from_dataset(dataset, batch_size=4096)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        writer.write_table(pa.Table.from_batches([batch], schema=dataset.schema))
    writer.close()
    tmp_path.replace(final_path)
    for part in parts:
        part.unlink()
    return True


def _emit_event(events_path: Path, *, event: str, payload: dict) -> None:
    record = {"event": event, "created_at": datetime.now(timezone.utc).isoformat()}
    record.update(payload)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _dump_model(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=False)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return dict(value)


def _effective_sampling_caps(input_cfg, cfg_path: Path) -> dict | None:
    sampling = getattr(input_cfg, "sampling", None)
    if sampling is None:
        return None
    n_sites = getattr(sampling, "n_sites", None)
    oversample = getattr(sampling, "oversample_factor", None)
    requested = None
    if isinstance(n_sites, int) and isinstance(oversample, int):
        requested = int(n_sites) * int(oversample)
    backend = str(getattr(sampling, "scoring_backend", "densegen"))
    mining = getattr(sampling, "mining", None)
    return {
        "scoring_backend": backend,
        "requested_candidates": requested,
        "cap_candidates": getattr(mining, "max_candidates", None)
        if backend == "fimo"
        else getattr(sampling, "max_candidates", None),
        "cap_seconds": getattr(mining, "max_seconds", None)
        if backend == "fimo"
        else getattr(sampling, "max_seconds", None),
        "cap_batches": getattr(mining, "max_batches", None) if backend == "fimo" else None,
    }


def _write_effective_config(
    *,
    cfg,
    cfg_path: Path,
    run_root: Path,
    seeds: dict[str, int],
    outputs_root: Path,
) -> Path:
    resolved_inputs = []
    for inp in cfg.inputs:
        entry = {"name": inp.name, "type": getattr(inp, "type", None)}
        if hasattr(inp, "path"):
            entry["path"] = str(resolve_relative_path(cfg_path, getattr(inp, "path")))
        if hasattr(inp, "paths"):
            paths = getattr(inp, "paths", None)
            if isinstance(paths, list):
                entry["paths"] = [str(resolve_relative_path(cfg_path, p)) for p in paths]
        caps = _effective_sampling_caps(inp, cfg_path)
        if caps is not None:
            entry["sampling_caps"] = caps
        resolved_inputs.append(entry)

    payload = {
        "schema_version": cfg.schema_version,
        "run_id": cfg.run.id,
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "seeds": {k: int(v) for k, v in seeds.items()},
        "inputs": resolved_inputs,
        "config": _dump_model(cfg),
    }
    out_path = outputs_root / "meta" / "effective_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


ATTEMPTS_CHUNK_SIZE = 256


def _flush_attempts(outputs_root: Path, buffer: list[dict]) -> None:
    if not buffer:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to write attempt logs.") from exc

    schema = pa.schema(
        [
            pa.field("attempt_id", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("input_name", pa.string()),
            pa.field("plan_name", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("detail_json", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("sequence_hash", pa.string()),
            pa.field("output_id", pa.string()),
            pa.field("used_tf_counts_json", pa.string()),
            pa.field("used_tf_list", pa.list_(pa.string())),
            pa.field("sampling_library_index", pa.int64()),
            pa.field("sampling_library_hash", pa.string()),
            pa.field("solver_status", pa.string()),
            pa.field("solver_objective", pa.float64()),
            pa.field("solver_solve_time_s", pa.float64()),
            pa.field("dense_arrays_version", pa.string()),
            pa.field("dense_arrays_version_source", pa.string()),
            pa.field("library_tfbs", pa.list_(pa.string())),
            pa.field("library_tfs", pa.list_(pa.string())),
            pa.field("library_site_ids", pa.list_(pa.string())),
            pa.field("library_sources", pa.list_(pa.string())),
        ]
    )
    table = pa.Table.from_pylist(buffer, schema=schema)
    outputs_root.mkdir(parents=True, exist_ok=True)
    filename = f"attempts_part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, outputs_root / filename)
    buffer.clear()


def _load_failure_counts_from_attempts(
    outputs_root: Path,
) -> dict[tuple[str, str, str, str, str | None], dict[str, int]]:
    attempts_path = outputs_root / "attempts.parquet"
    if not attempts_path.exists():
        return {}
    try:
        df = pd.read_parquet(attempts_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] = {}
    for _, row in df.iterrows():
        status = str(row.get("status") or "")
        if status == "success":
            continue
        reason = str(row.get("reason") or "unknown")
        input_name = str(row.get("input_name") or "")
        plan_name = str(row.get("plan_name") or "")
        library_tfbs = row.get("library_tfbs") or []
        library_tfs = row.get("library_tfs") or []
        library_site_ids = row.get("library_site_ids") or []
        if isinstance(library_tfbs, str):
            try:
                library_tfbs = json.loads(library_tfbs)
            except Exception:
                library_tfbs = []
        if isinstance(library_tfs, str):
            try:
                library_tfs = json.loads(library_tfs)
            except Exception:
                library_tfs = []
        if isinstance(library_site_ids, str):
            try:
                library_site_ids = json.loads(library_site_ids)
            except Exception:
                library_site_ids = []
        for idx, tfbs in enumerate(library_tfbs or []):
            tf = str(library_tfs[idx]) if idx < len(library_tfs) else ""
            site_id_raw = library_site_ids[idx] if idx < len(library_site_ids) else None
            site_id = None
            if site_id_raw not in (None, "", "None"):
                site_id = str(site_id_raw)
            key = (input_name, plan_name, tf, str(tfbs), site_id)
            reasons = counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1
    return counts


def _load_existing_library_index(outputs_root: Path) -> int:
    attempts_path = outputs_root / "attempts.parquet"
    paths: list[Path] = []
    if attempts_path.exists():
        paths.append(attempts_path)
    paths.extend(sorted(outputs_root.glob("attempts_part-*.parquet")))
    if not paths:
        return 0
    max_idx = 0
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["sampling_library_index"])
        except Exception:
            continue
        if df.empty or "sampling_library_index" not in df.columns:
            continue
        try:
            current = int(pd.to_numeric(df["sampling_library_index"], errors="coerce").dropna().max() or 0)
        except Exception:
            continue
        max_idx = max(max_idx, current)
    return max_idx


def _append_attempt(
    outputs_root: Path,
    *,
    run_id: str,
    input_name: str,
    plan_name: str,
    status: str,
    reason: str,
    detail: dict | None,
    sequence: str | None,
    used_tf_counts: dict[str, int] | None,
    used_tf_list: list[str] | None,
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    output_id: str | None = None,
    library_tfbs: list[str] | None = None,
    library_tfs: list[str] | None = None,
    library_site_ids: list[str | None] | None = None,
    library_sources: list[str | None] | None = None,
    attempts_buffer: list[dict] | None = None,
) -> None:
    sequence_val = sequence or ""
    lib_tfbs = [str(x) for x in (library_tfbs or [])]
    lib_tfs = [str(x) for x in (library_tfs or [])]
    lib_site_ids = [str(x) if x is not None else "" for x in (library_site_ids or [])]
    lib_sources = [str(x) if x is not None else "" for x in (library_sources or [])]
    payload = {
        "attempt_id": uuid.uuid4().hex,
        "run_id": run_id,
        "input_name": input_name,
        "plan_name": plan_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "detail_json": json.dumps(detail or {}),
        "sequence": sequence_val,
        "sequence_hash": hashlib.sha256(sequence_val.encode("utf-8")).hexdigest() if sequence_val else "",
        "output_id": output_id,
        "used_tf_counts_json": json.dumps(used_tf_counts or {}),
        "used_tf_list": used_tf_list or [],
        "sampling_library_index": int(sampling_library_index),
        "sampling_library_hash": sampling_library_hash,
        "solver_status": solver_status,
        "solver_objective": solver_objective,
        "solver_solve_time_s": solver_solve_time_s,
        "dense_arrays_version": dense_arrays_version,
        "dense_arrays_version_source": dense_arrays_version_source,
        "library_tfbs": lib_tfbs,
        "library_tfs": lib_tfs,
        "library_site_ids": lib_site_ids,
        "library_sources": lib_sources,
    }
    if attempts_buffer is not None:
        attempts_buffer.append(payload)
        if len(attempts_buffer) >= ATTEMPTS_CHUNK_SIZE:
            _flush_attempts(outputs_root, attempts_buffer)
        return
    _flush_attempts(outputs_root, [payload])


def _log_rejection(
    outputs_root: Path,
    *,
    run_id: str,
    input_name: str,
    plan_name: str,
    reason: str,
    detail: dict | None,
    sequence: str,
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    library_tfbs: list[str] | None = None,
    library_tfs: list[str] | None = None,
    library_site_ids: list[str | None] | None = None,
    library_sources: list[str | None] | None = None,
    attempts_buffer: list[dict] | None = None,
) -> None:
    status = "duplicate" if reason == "output_duplicate" else "rejected"
    _append_attempt(
        outputs_root,
        run_id=run_id,
        input_name=input_name,
        plan_name=plan_name,
        status=status,
        reason=reason,
        detail=detail,
        sequence=sequence,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        sampling_library_index=sampling_library_index,
        sampling_library_hash=sampling_library_hash,
        solver_status=solver_status,
        solver_objective=solver_objective,
        solver_solve_time_s=solver_solve_time_s,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        library_site_ids=library_site_ids,
        library_sources=library_sources,
        attempts_buffer=attempts_buffer,
    )


def _assert_sink_alignment(sinks: list[SinkBase]) -> None:
    if len(sinks) <= 1:
        return
    sink_types = [type(s).__name__ for s in sinks]
    if len(set(sink_types)) != len(sink_types):
        raise RuntimeError("Duplicate sink types detected; output.targets must be unique.")
    digests = []
    for sink in sinks:
        digest = sink.alignment_digest()
        if digest is None:
            raise RuntimeError(
                f"Sink {type(sink).__name__} does not provide alignment digest; alignment requires digest support."
            )
        digests.append((type(sink).__name__, digest))

    baseline_name, baseline = digests[0]
    mismatches = []
    for name, digest in digests[1:]:
        if digest != baseline:
            mismatches.append((name, digest.id_count, digest.xor_hash))
    if mismatches:
        details = "; ".join(f"{name}: count={cnt} hash={h}" for name, cnt, h in mismatches)
        raise RuntimeError(
            "Output sinks are out of sync before run. "
            f"Baseline={baseline_name} count={baseline.id_count} hash={baseline.xor_hash}. "
            f"Differences: {details}. Remove stale outputs or run with a single target to rebuild."
        )


def _write_to_sinks(sinks: list[SinkBase], record: OutputRecord) -> bool:
    results = [sink.add(record) for sink in sinks]
    if not results:
        raise RuntimeError("No output sinks configured.")
    if all(results):
        return True
    if not any(results):
        return False
    raise RuntimeError("Output sinks are inconsistent (some accepted, some rejected).")


def _process_plan_for_source(
    source_cfg,
    plan_item: ResolvedPlanItem,
    global_cfg,
    sinks,
    *,
    chosen_solver: str | None,
    deps: PipelineDeps,
    rng: random.Random,
    np_rng: np.random.Generator,
    cfg_path: Path,
    run_id: str,
    run_root: str,
    run_config_path: str,
    run_config_sha256: str,
    random_seed: int,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    output_bio_type: str,
    output_alphabet: str,
    one_subsample_only: bool = False,
    already_generated: int = 0,
    inputs_manifest: dict[str, dict] | None = None,
    existing_usage_counts: dict[tuple[str, str], int] | None = None,
    state_counts: dict[tuple[str, str], int] | None = None,
    checkpoint_every: int = 0,
    write_state: Callable[[], None] | None = None,
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None = None,
    source_cache: dict[str, PoolData] | None = None,
    library_build_rows: list[dict] | None = None,
    library_member_rows: list[dict] | None = None,
    composition_rows: list[dict] | None = None,
    events_path: Path | None = None,
) -> tuple[int, dict]:
    source_label = source_cfg.name
    plan_name = plan_item.name
    quota = int(plan_item.quota)

    gen = global_cfg.generation
    seq_len = int(gen.sequence_length)
    sampling_cfg = gen.sampling

    def _record_library_build(
        *,
        sampling_info: dict,
        library_tfbs: list[str],
        library_tfs: list[str],
        library_tfbs_ids: list[str],
        library_motif_ids: list[str],
        library_site_ids: list[str | None],
        library_sources: list[str | None],
    ) -> None:
        if library_build_rows is None or library_member_rows is None:
            return
        library_index = int(sampling_info.get("library_index") or 0)
        library_hash = str(sampling_info.get("library_hash") or "")
        library_id = library_hash or f"{source_label}:{plan_name}:{library_index}"
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_name": source_label,
            "plan_name": plan_name,
            "library_index": library_index,
            "library_id": library_id,
            "library_hash": library_hash,
            "pool_strategy": sampling_info.get("pool_strategy"),
            "library_sampling_strategy": sampling_info.get("library_sampling_strategy"),
            "library_size": int(sampling_info.get("library_size") or len(library_tfbs)),
            "target_length": sampling_info.get("target_length"),
            "achieved_length": sampling_info.get("achieved_length"),
            "relaxed_cap": sampling_info.get("relaxed_cap"),
            "final_cap": sampling_info.get("final_cap"),
            "iterative_max_libraries": sampling_info.get("iterative_max_libraries"),
            "iterative_min_new_solutions": sampling_info.get("iterative_min_new_solutions"),
            "required_regulators_selected": sampling_info.get("required_regulators_selected"),
        }
        library_build_rows.append(row)
        if events_path is not None:
            try:
                _emit_event(
                    events_path,
                    event="LIBRARY_BUILT",
                    payload={
                        "input_name": source_label,
                        "plan_name": plan_name,
                        "library_index": library_index,
                        "library_hash": library_hash,
                        "library_size": int(row.get("library_size") or len(library_tfbs)),
                    },
                )
            except Exception:
                log.debug("Failed to emit LIBRARY_BUILT event.", exc_info=True)
        for idx, tfbs in enumerate(library_tfbs):
            library_member_rows.append(
                {
                    "library_id": library_id,
                    "library_hash": library_hash,
                    "library_index": library_index,
                    "input_name": source_label,
                    "plan_name": plan_name,
                    "position": int(idx),
                    "tf": library_tfs[idx] if idx < len(library_tfs) else "",
                    "tfbs": tfbs,
                    "tfbs_id": library_tfbs_ids[idx] if idx < len(library_tfbs_ids) else None,
                    "motif_id": library_motif_ids[idx] if idx < len(library_motif_ids) else None,
                    "site_id": library_site_ids[idx] if idx < len(library_site_ids) else None,
                    "source": library_sources[idx] if idx < len(library_sources) else None,
                }
            )

    pool_strategy = str(sampling_cfg.pool_strategy)
    library_sampling_strategy = str(sampling_cfg.library_sampling_strategy)
    iterative_max_libraries = int(sampling_cfg.iterative_max_libraries)
    iterative_min_new_solutions = int(sampling_cfg.iterative_min_new_solutions)
    schema_is_22 = schema_version_at_least(global_cfg.schema_version, major=2, minor=2)

    runtime_cfg = global_cfg.runtime
    max_per_subsample = int(runtime_cfg.arrays_generated_before_resample)
    min_count_per_tf = int(runtime_cfg.min_count_per_tf)
    max_dupes = int(runtime_cfg.max_duplicate_solutions)
    max_resample_attempts = int(runtime_cfg.max_resample_attempts)
    stall_seconds = int(runtime_cfg.stall_seconds_before_resample)
    stall_warn_every = int(runtime_cfg.stall_warning_every_seconds)
    max_total_resamples = int(runtime_cfg.max_total_resamples)
    max_seconds_per_plan = int(runtime_cfg.max_seconds_per_plan)
    max_failed_solutions = int(runtime_cfg.max_failed_solutions)
    leaderboard_every = int(runtime_cfg.leaderboard_every)
    checkpoint_every = int(checkpoint_every or 0)

    policy = RuntimePolicy(
        pool_strategy=pool_strategy,
        schema_is_22=schema_is_22,
        arrays_generated_before_resample=max_per_subsample,
        stall_seconds_before_resample=stall_seconds,
        stall_warning_every_seconds=stall_warn_every,
        max_resample_attempts=max_resample_attempts,
        max_total_resamples=max_total_resamples,
        max_seconds_per_plan=max_seconds_per_plan,
    )

    post = global_cfg.postprocess
    gap_cfg = post.gap_fill
    fill_gap = gap_cfg.mode != "off"
    fill_mode = gap_cfg.mode
    fill_end = gap_cfg.end
    fill_gc_min = float(gap_cfg.gc_min)
    fill_gc_max = float(gap_cfg.gc_max)
    fill_max_tries = int(gap_cfg.max_tries)

    solver_cfg = global_cfg.solver
    solver_opts = list(solver_cfg.options)
    solver_strategy = str(solver_cfg.strategy)
    solver_strands = str(solver_cfg.strands)

    log_cfg = global_cfg.logging
    print_visual = bool(log_cfg.print_visual)
    progress_style = str(getattr(log_cfg, "progress_style", "stream"))
    progress_every = int(getattr(log_cfg, "progress_every", 1))
    progress_refresh_seconds = float(getattr(log_cfg, "progress_refresh_seconds", 1.0))
    screen_console = Console() if progress_style == "screen" else None
    last_screen_refresh = 0.0
    latest_failure_totals: str | None = None

    policy_gc_fill = str(fill_mode)
    policy_sampling = pool_strategy
    policy_solver = solver_strategy

    plan_start = time.monotonic()
    total_resamples = 0
    failed_solutions = 0
    duplicate_records = 0
    stall_events = 0
    failed_min_count_per_tf = 0
    failed_required_regulators = 0
    failed_min_count_by_regulator = 0
    failed_min_required_regulators = 0
    duplicate_solutions = 0
    usage_counts: dict[tuple[str, str], int] = dict(existing_usage_counts or {})
    tf_usage_counts: dict[str, int] = {}
    for (tf, _tfbs), count in usage_counts.items():
        tf_usage_counts[tf] = tf_usage_counts.get(tf, 0) + int(count)
    track_failures = site_failure_counts is not None
    failure_counts = site_failure_counts if site_failure_counts is not None else {}
    attempts_buffer: list[dict] = []
    run_root_path = Path(run_root)
    outputs_root = run_root_path / "outputs"
    existing_library_builds = _load_existing_library_index(outputs_root)

    # Load source (cache PWM sampling results across round-robin passes).
    cache_key = source_label
    cached = source_cache.get(cache_key) if source_cache is not None else None
    if cached is None:
        src_obj = deps.source_factory(source_cfg, cfg_path)
        data_entries, meta_df = src_obj.load_data(rng=np_rng, outputs_root=outputs_root)
        if meta_df is not None and isinstance(meta_df, pd.DataFrame):
            sequences = meta_df["tfbs"].tolist() if "tfbs" in meta_df.columns else list(data_entries or [])
            pool = PoolData(
                name=source_label,
                input_type=str(getattr(source_cfg, "type", "")),
                pool_mode=POOL_MODE_TFBS,
                df=meta_df,
                sequences=sequences,
                pool_path=Path("."),
            )
        else:
            pool = PoolData(
                name=source_label,
                input_type=str(getattr(source_cfg, "type", "")),
                pool_mode=POOL_MODE_SEQUENCE,
                df=None,
                sequences=list(data_entries or []),
                pool_path=Path("."),
            )
        if source_cache is not None:
            source_cache[cache_key] = pool
    else:
        pool = cached
    data_entries = pool.sequences
    meta_df = pool.df
    input_meta = _input_metadata(source_cfg, cfg_path)
    input_tf_tfbs_pair_count: int | None = None
    if meta_df is not None and isinstance(meta_df, pd.DataFrame):
        input_row_count = int(len(meta_df))
        input_tf_count = int(meta_df["tf"].nunique()) if "tf" in meta_df.columns else 0
        input_tfbs_count = int(meta_df["tfbs"].nunique()) if "tfbs" in meta_df.columns else 0
        if "tf" in meta_df.columns and "tfbs" in meta_df.columns:
            input_tf_tfbs_pair_count = int(meta_df.drop_duplicates(["tf", "tfbs"]).shape[0])
    else:
        input_row_count = int(len(data_entries))
        input_tf_count = 0
        input_tfbs_count = int(len(set(data_entries))) if data_entries else 0
        input_tf_tfbs_pair_count = None
    input_meta.update(
        {
            "input_row_count": input_row_count,
            "input_tf_count": input_tf_count,
            "input_tfbs_count": input_tfbs_count,
            "input_tf_tfbs_pair_count": input_tf_tfbs_pair_count,
            "sampling_fraction": None,
            "sampling_fraction_pairs": None,
        }
    )
    pair_label = str(input_tf_tfbs_pair_count) if input_tf_tfbs_pair_count is not None else "-"
    log.info(
        "[%s/%s] Input summary: mode=%s rows=%d tfs=%d tfbs=%d pairs=%s",
        source_label,
        plan_name,
        input_meta.get("input_mode"),
        input_row_count,
        input_tf_count,
        input_tfbs_count,
        pair_label,
    )
    source_type = getattr(source_cfg, "type", None)
    if source_type in PWM_INPUT_TYPES and meta_df is not None and "tf" in meta_df.columns:
        input_meta["input_pwm_ids"] = sorted(set(meta_df["tf"].tolist()))
        if inputs_manifest is not None and source_label not in inputs_manifest:
            input_sampling_cfg = getattr(source_cfg, "sampling", None)
            strategy = _sampling_attr(input_sampling_cfg, "strategy")
            n_sites = _sampling_attr(input_sampling_cfg, "n_sites")
            oversample = _sampling_attr(input_sampling_cfg, "oversample_factor")
            max_candidates = _sampling_attr(input_sampling_cfg, "max_candidates")
            max_seconds = _sampling_attr(input_sampling_cfg, "max_seconds")
            score_threshold = _sampling_attr(input_sampling_cfg, "score_threshold")
            score_percentile = _sampling_attr(input_sampling_cfg, "score_percentile")
            scoring_backend = _sampling_attr(input_sampling_cfg, "scoring_backend") or "densegen"
            pvalue_threshold = _sampling_attr(input_sampling_cfg, "pvalue_threshold")
            selection_policy = _sampling_attr(input_sampling_cfg, "selection_policy")
            length_policy = _sampling_attr(input_sampling_cfg, "length_policy")
            length_range = _sampling_attr(input_sampling_cfg, "length_range")
            mining_cfg = _sampling_attr(input_sampling_cfg, "mining")
            mining_batch_size = _mining_attr(mining_cfg, "batch_size")
            mining_max_batches = _mining_attr(mining_cfg, "max_batches")
            mining_max_candidates = _mining_attr(mining_cfg, "max_candidates")
            mining_max_seconds = _mining_attr(mining_cfg, "max_seconds")
            mining_retain_bins = _mining_attr(mining_cfg, "retain_bin_ids")
            if length_range is not None:
                length_range = list(length_range)
            score_label = "-"
            if scoring_backend == "fimo" and pvalue_threshold is not None:
                comparator = ">=" if str(strategy) == "background" else "<="
                score_label = f"pvalue{comparator}{pvalue_threshold}"
            elif score_threshold is not None:
                score_label = f"threshold={score_threshold}"
            elif score_percentile is not None:
                score_label = f"percentile={score_percentile}"
            bins_label = "-"
            if scoring_backend == "fimo":
                bins_label = "canonical" if _sampling_attr(input_sampling_cfg, "pvalue_bins") is None else "custom"
                bin_ids = mining_retain_bins
                if bin_ids:
                    bins_label = f"{bins_label} retain={sorted(list(bin_ids))}"
            length_label = str(length_policy)
            if length_policy == "range" and length_range:
                length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"
            cap_label = "-"
            if isinstance(n_sites, int) and isinstance(oversample, int):
                requested = n_sites * oversample
                if scoring_backend == "fimo":
                    if mining_max_candidates is not None:
                        cap_label = f"{mining_max_candidates} (requested={requested})"
                    if mining_max_seconds is not None:
                        cap_label = (
                            f"{cap_label}; max_seconds={mining_max_seconds}s"
                            if cap_label != "-"
                            else f"{mining_max_seconds}s"
                        )
                else:
                    if max_candidates is not None:
                        cap_label = f"{max_candidates} (requested={requested})"
                    if max_seconds is not None:
                        cap_label = f"{cap_label}; max_seconds={max_seconds}" if cap_label != "-" else f"{max_seconds}s"
            counts_label = _summarize_tf_counts(meta_df["tf"].tolist())
            selection_label = selection_policy if scoring_backend == "fimo" else "-"
            mining_label = "-"
            if scoring_backend == "fimo" and mining_cfg is not None:
                parts = []
                if mining_batch_size is not None:
                    parts.append(f"batch={mining_batch_size}")
                if mining_max_batches is not None:
                    parts.append(f"max_batches={mining_max_batches}")
                if mining_max_candidates is not None:
                    parts.append(f"max_candidates={mining_max_candidates}")
                if mining_max_seconds is not None:
                    parts.append(f"max_seconds={mining_max_seconds}s")
                mining_label = ", ".join(parts) if parts else "enabled"
            log.info(
                "PWM input sampling for %s: motifs=%d | sites=%s | strategy=%s | backend=%s | score=%s | "
                "selection=%s | bins=%s | mining=%s | oversample=%s | max_candidates=%s | length=%s",
                source_label,
                len(input_meta.get("input_pwm_ids") or []),
                counts_label or "-",
                strategy,
                scoring_backend,
                score_label,
                selection_label,
                bins_label,
                mining_label,
                oversample,
                cap_label,
                length_label,
            )
            inputs_manifest[source_label] = _build_input_manifest_entry(
                source_cfg=source_cfg,
                cfg_path=cfg_path,
                input_meta=input_meta,
                input_row_count=input_row_count,
                input_tf_count=input_tf_count,
                input_tfbs_count=input_tfbs_count,
                input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
                meta_df=meta_df,
            )
    elif inputs_manifest is not None and source_label not in inputs_manifest:
        inputs_manifest[source_label] = _build_input_manifest_entry(
            source_cfg=source_cfg,
            cfg_path=cfg_path,
            input_meta=input_meta,
            input_row_count=input_row_count,
            input_tf_count=input_tf_count,
            input_tfbs_count=input_tfbs_count,
            input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
            meta_df=meta_df,
        )
    fixed_elements = plan_item.fixed_elements
    required_regulators = list(dict.fromkeys(plan_item.required_regulators or []))
    min_required_regulators = plan_item.min_required_regulators
    plan_min_count_by_regulator = dict(plan_item.min_count_by_regulator or {})
    k_required = int(min_required_regulators) if min_required_regulators is not None else None
    k_of_required = bool(required_regulators) and k_required is not None
    if k_of_required and k_required > len(required_regulators):
        raise ValueError(
            "min_required_regulators cannot exceed required_regulators size "
            f"({k_required} > {len(required_regulators)})."
        )
    metadata_min_counts = {tf: max(min_count_per_tf, int(val)) for tf, val in plan_min_count_by_regulator.items()}
    fixed_elements_dump = _fixed_elements_dump(fixed_elements)
    fixed_elements_max_len = _max_fixed_element_len(fixed_elements_dump)

    # Build initial library
    library_for_opt: List[str]
    tfbs_parts: List[str]
    libraries_built = existing_library_builds
    libraries_built_start = existing_library_builds

    if pool_strategy != "iterative_subsample" and not one_subsample_only:
        max_per_subsample = quota
    library_for_opt, tfbs_parts, regulator_labels, sampling_info = build_library_for_plan(
        source_label=source_label,
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=seq_len,
        min_count_per_tf=min_count_per_tf,
        usage_counts=usage_counts,
        failure_counts=failure_counts if failure_counts else None,
        rng=rng,
        np_rng=np_rng,
        schema_is_22=schema_is_22,
        library_index_start=libraries_built,
    )
    libraries_built = int(sampling_info.get("library_index", libraries_built))
    site_id_by_index = sampling_info.get("site_id_by_index")
    source_by_index = sampling_info.get("source_by_index")
    tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
    motif_id_by_index = sampling_info.get("motif_id_by_index")
    sampling_library_index = sampling_info.get("library_index", 0)
    sampling_library_hash = sampling_info.get("library_hash", "")
    library_tfbs = list(library_for_opt)
    library_tfs = list(regulator_labels) if regulator_labels else []
    library_site_ids = list(site_id_by_index) if site_id_by_index else []
    library_sources = list(source_by_index) if source_by_index else []
    library_tfbs_ids = list(tfbs_id_by_index) if tfbs_id_by_index else []
    library_motif_ids = list(motif_id_by_index) if motif_id_by_index else []
    _record_library_build(
        sampling_info=sampling_info,
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        library_tfbs_ids=library_tfbs_ids,
        library_motif_ids=library_motif_ids,
        library_site_ids=library_site_ids,
        library_sources=library_sources,
    )
    max_tfbs_len = max((len(str(m)) for m in library_tfbs), default=0)
    required_len = max(max_tfbs_len, fixed_elements_max_len)
    if seq_len < required_len:
        raise ValueError(
            "generation.sequence_length is shorter than the widest required motif "
            f"(sequence_length={seq_len}, max_library_motif={max_tfbs_len}, "
            f"max_fixed_element={fixed_elements_max_len}). "
            "Increase densegen.generation.sequence_length or reduce motif lengths "
            "(e.g., adjust PWM sampling length_range or fixed-element motifs)."
        )

    def _current_leaderboard_snapshot() -> dict[str, object]:
        return _leaderboard_snapshot(
            usage_counts,
            tf_usage_counts,
            failure_counts,
            input_name=source_label,
            plan_name=plan_name,
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
        )

    def _log_leaderboard_snapshot() -> None:
        log.info(
            "[%s/%s] Leaderboard (TF): %s",
            source_label,
            plan_name,
            _summarize_leaderboard(tf_usage_counts, top=5),
        )
        log.info(
            "[%s/%s] Leaderboard (TFBS): %s",
            source_label,
            plan_name,
            _summarize_leaderboard(usage_counts, top=5),
        )
        log.info(
            "[%s/%s] Failed TFBS: %s",
            source_label,
            plan_name,
            _summarize_failure_leaderboard(
                failure_counts,
                input_name=source_label,
                plan_name=plan_name,
                top=5,
            ),
        )
        log.info(
            "[%s/%s] Diversity: %s",
            source_label,
            plan_name,
            _summarize_diversity(
                usage_counts,
                tf_usage_counts,
                library_tfs=library_tfs,
                library_tfbs=library_tfbs,
            ),
        )

    def _record_site_failures(reason: str) -> None:
        if not track_failures:
            return
        if not library_tfbs:
            return
        for idx, tfbs in enumerate(library_tfbs):
            tf = library_tfs[idx] if idx < len(library_tfs) else ""
            site_id = None
            if library_site_ids and idx < len(library_site_ids):
                raw = library_site_ids[idx]
                if raw not in (None, "", "None"):
                    site_id = str(raw)
            key = (source_label, plan_name, tf, tfbs, site_id)
            reasons = failure_counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1

    # Alignment (7): sampling_fraction uses unique TFBS strings and is bounded.
    sampling_fraction = _compute_sampling_fraction(
        library_for_opt,
        input_tfbs_count=input_tfbs_count,
        pool_strategy=pool_strategy,
    )
    input_meta["sampling_fraction"] = sampling_fraction
    sampling_fraction_pairs = _compute_sampling_fraction_pairs(
        library_for_opt,
        regulator_labels,
        input_pair_count=input_tf_tfbs_pair_count,
        pool_strategy=pool_strategy,
    )
    input_meta["sampling_fraction_pairs"] = sampling_fraction_pairs
    # Library summary (succinct)
    tf_summary = _summarize_tf_counts(regulator_labels)
    library_index = sampling_info.get("library_index")
    strategy_label = sampling_info.get("library_sampling_strategy", library_sampling_strategy)
    pool_label = sampling_info.get("pool_strategy")
    target_len = sampling_info.get("target_length")
    achieved_len = sampling_info.get("achieved_length")
    header = f"Stage B library for {source_label}/{plan_name}"
    if library_index is not None:
        header = f"{header} (build {library_index})"
    if tf_summary:
        log.info(
            "%s: %d motifs | TF counts: %s | target=%s achieved=%s pool=%s sampling=%s",
            header,
            len(library_for_opt),
            tf_summary,
            target_len,
            achieved_len,
            pool_label,
            strategy_label,
        )
    else:
        log.info(
            "%s: %d motifs | target=%s achieved=%s pool=%s sampling=%s",
            header,
            len(library_for_opt),
            target_len,
            achieved_len,
            pool_label,
            strategy_label,
        )

    solver_min_counts: dict[str, int] | None = None

    def _make_generator(_library_for_opt: List[str], _regulator_labels: List[str]):
        nonlocal solver_min_counts
        regulator_by_index = list(_regulator_labels) if _regulator_labels else None
        base_min_counts = _min_count_by_regulator(regulator_by_index, min_count_per_tf)
        solver_min_counts = _merge_min_counts(base_min_counts, plan_min_count_by_regulator)
        fe_dict = fixed_elements.model_dump() if hasattr(fixed_elements, "model_dump") else fixed_elements
        solver_required_regs = required_regulators
        if k_of_required and regulator_by_index:
            available = set(regulator_by_index)
            solver_required_regs = [tf for tf in required_regulators if tf in available]
            if k_required is not None and len(solver_required_regs) < k_required:
                raise ValueError(
                    "Required regulator candidate set is smaller than min_required_regulators "
                    f"after library sampling ({len(solver_required_regs)} < {k_required}). "
                    "Increase library_size or relax required_regulators/min_required_regulators."
                )
        if min_required_regulators is not None and not required_regulators:
            solver_required_regs = None
        run = deps.optimizer.build(
            library=_library_for_opt,
            sequence_length=seq_len,
            solver=chosen_solver,
            strategy=solver_strategy,
            solver_options=solver_opts,
            fixed_elements=fe_dict,
            strands=solver_strands,
            regulator_by_index=regulator_by_index,
            required_regulators=solver_required_regs,
            min_count_by_regulator=solver_min_counts,
            min_required_regulators=min_required_regulators,
        )
        return run

    run = _make_generator(library_for_opt, regulator_labels)
    opt = run.optimizer
    generator = run.generator
    forbid_each = run.forbid_each

    global_generated = already_generated
    produced_total_this_call = 0

    while global_generated < quota:
        if policy.plan_timed_out(now=time.monotonic(), plan_started=plan_start):
            raise RuntimeError(f"[{source_label}/{plan_name}] Exceeded max_seconds_per_plan={max_seconds_per_plan}.")
        local_generated = 0
        resamples_in_try = 0

        while local_generated < max_per_subsample and global_generated < quota:
            fingerprints = set()
            consecutive_dup = 0
            subsample_started = time.monotonic()
            last_log_warn = subsample_started
            produced_this_library = 0
            stall_triggered = False

            for sol in generator:
                now = time.monotonic()
                if policy.should_trigger_stall(
                    now=now,
                    subsample_started=subsample_started,
                    produced_this_library=produced_this_library,
                ):
                    log.info(
                        "[%s/%s] Stall (> %ds) with no solutions; will resample.",
                        source_label,
                        plan_name,
                        stall_seconds,
                    )
                    stall_events += 1
                    if events_path is not None:
                        try:
                            _emit_event(
                                events_path,
                                event="STALL_DETECTED",
                                payload={
                                    "input_name": source_label,
                                    "plan_name": plan_name,
                                    "stall_seconds": float(now - subsample_started),
                                    "library_index": int(sampling_library_index),
                                    "library_hash": str(sampling_library_hash),
                                },
                            )
                        except Exception:
                            log.debug("Failed to emit STALL_DETECTED event.", exc_info=True)
                    stall_triggered = True
                    break
                if policy.should_warn_stall(
                    now=now,
                    last_warn=last_log_warn,
                    produced_this_library=produced_this_library,
                ):
                    log.info(
                        "[%s/%s] Still working... %.1fs on current library.",
                        source_label,
                        plan_name,
                        now - subsample_started,
                    )
                    last_log_warn = now

                if forbid_each:
                    opt.forbid(sol)
                seq = sol.sequence
                if seq in fingerprints:
                    duplicate_solutions += 1
                    consecutive_dup += 1
                    if consecutive_dup >= max_dupes:
                        log.info(
                            "[%s/%s] Duplicate guard (>= %d in a row); will resample.",
                            source_label,
                            plan_name,
                            max_dupes,
                        )
                        break
                    continue
                consecutive_dup = 0
                fingerprints.add(seq)

                used_tfbs, used_tfbs_detail, used_tf_counts, used_tf_list = _compute_used_tf_info(
                    sol,
                    library_for_opt,
                    regulator_labels,
                    fixed_elements,
                    site_id_by_index,
                    source_by_index,
                    tfbs_id_by_index,
                    motif_id_by_index,
                )
                tf_list_from_library = sorted(set(regulator_labels)) if regulator_labels else []
                solver_status = getattr(sol, "status", None)
                if solver_status is not None:
                    solver_status = str(solver_status)
                solver_objective = getattr(sol, "objective", None)
                if solver_objective is None:
                    solver_objective = getattr(sol, "objective_value", None)
                try:
                    solver_objective = float(solver_objective) if solver_objective is not None else None
                except (TypeError, ValueError):
                    solver_objective = None
                solver_solve_time_s = getattr(sol, "_densegen_solve_time_s", None)

                covers_all = True
                covers_required = True
                if min_count_per_tf > 0 and tf_list_from_library:
                    missing = [tf for tf in tf_list_from_library if used_tf_counts.get(tf, 0) < min_count_per_tf]
                    if missing:
                        covers_all = False
                        failed_solutions += 1
                        failed_min_count_per_tf += 1
                        _record_site_failures("min_count_per_tf")
                        _log_rejection(
                            outputs_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            reason="min_count_per_tf",
                            detail={
                                "min_count_per_tf": min_count_per_tf,
                                "missing_tfs": missing,
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                if required_regulators and not k_of_required:
                    missing = [tf for tf in required_regulators if used_tf_counts.get(tf, 0) < 1]
                    if missing:
                        covers_required = False
                        failed_solutions += 1
                        failed_required_regulators += 1
                        _record_site_failures("required_regulators")
                        _log_rejection(
                            outputs_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            reason="required_regulators",
                            detail={
                                "required_regulators": required_regulators,
                                "missing_tfs": missing,
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                if plan_min_count_by_regulator:
                    missing = [
                        tf
                        for tf, min_count in plan_min_count_by_regulator.items()
                        if used_tf_counts.get(tf, 0) < int(min_count)
                    ]
                    if missing:
                        failed_solutions += 1
                        failed_min_count_by_regulator += 1
                        _record_site_failures("min_count_by_regulator")
                        _log_rejection(
                            outputs_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            reason="min_count_by_regulator",
                            detail={
                                "min_count_by_regulator": [
                                    {
                                        "tf": tf,
                                        "min_count": int(plan_min_count_by_regulator[tf]),
                                        "found": int(used_tf_counts.get(tf, 0)),
                                    }
                                    for tf in missing
                                ]
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                if k_of_required:
                    present_required = [tf for tf in used_tf_list if tf in required_regulators]
                    missing_required = [tf for tf in required_regulators if tf not in present_required]
                    if len(present_required) < int(k_required or 0):
                        covers_required = False
                        failed_solutions += 1
                        failed_min_required_regulators += 1
                        _record_site_failures("min_required_regulators")
                        _log_rejection(
                            outputs_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            reason="min_required_regulators",
                            detail={
                                "required_regulators": required_regulators,
                                "min_required_regulators": int(k_required or 0),
                                "found_required_count": len(present_required),
                                "present_required": present_required,
                                "missing_required": missing_required,
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue
                elif min_required_regulators is not None:
                    if len(used_tf_list) < int(min_required_regulators):
                        covers_required = False
                        failed_solutions += 1
                        failed_min_required_regulators += 1
                        _record_site_failures("min_required_regulators")
                        _log_rejection(
                            outputs_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            reason="min_required_regulators",
                            detail={
                                "min_required_regulators": int(min_required_regulators),
                                "used_regulator_count": len(used_tf_list),
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                gap_meta = {"used": False}
                final_seq = seq
                if not fill_gap and len(final_seq) < seq_len:
                    raise RuntimeError(
                        f"[{source_label}/{plan_name}] Sequence shorter than target and gap_fill.mode=off."
                    )
                if fill_gap and len(final_seq) < seq_len:
                    gap = seq_len - len(final_seq)
                    rf = deps.gap_fill(
                        gap,
                        fill_gc_min,
                        fill_gc_max,
                        max_tries=fill_max_tries,
                        mode=fill_mode,
                        rng=rng,
                    )
                    if isinstance(rf, tuple) and len(rf) == 2:
                        pad, pad_info = rf
                        pad_info = pad_info or {}
                    else:
                        pad, pad_info = rf, {}
                    final_seq = (pad + final_seq) if fill_end == "5prime" else (final_seq + pad)
                    gap_meta = {
                        "used": True,
                        "bases": gap,
                        "end": fill_end,
                        "gc_min": pad_info.get("final_gc_min", fill_gc_min),
                        "gc_max": pad_info.get("final_gc_max", fill_gc_max),
                        "gc_target_min": pad_info.get("target_gc_min", fill_gc_min),
                        "gc_target_max": pad_info.get("target_gc_max", fill_gc_max),
                        "gc_actual": pad_info.get("gc_actual"),
                        "relaxed": pad_info.get("relaxed"),
                        "attempts": pad_info.get("attempts"),
                    }

                used_tfbs_detail = _apply_gap_fill_offsets(used_tfbs_detail, gap_meta)
                gc_core = _gc_fraction(seq)
                gc_total = _gc_fraction(final_seq)
                created_at = datetime.now(timezone.utc).isoformat()
                derived = build_metadata(
                    sol=sol,
                    plan_name=plan_name,
                    tfbs_parts=tfbs_parts,
                    regulator_labels=regulator_labels,
                    library_for_opt=library_for_opt,
                    fixed_elements=fixed_elements,
                    chosen_solver=chosen_solver,
                    solver_strategy=solver_strategy,
                    solver_options=solver_opts,
                    solver_strands=solver_strands,
                    seq_len=seq_len,
                    actual_length=len(final_seq),
                    gap_meta=gap_meta,
                    sampling_meta=sampling_info,
                    schema_version=str(global_cfg.schema_version),
                    created_at=created_at,
                    run_id=run_id,
                    run_root=run_root,
                    run_config_path=run_config_path,
                    run_config_sha256=run_config_sha256,
                    random_seed=random_seed,
                    policy_gc_fill=policy_gc_fill,
                    policy_sampling=policy_sampling,
                    policy_solver=policy_solver,
                    input_meta=input_meta,
                    fixed_elements_dump=fixed_elements_dump,
                    used_tfbs=used_tfbs,
                    used_tfbs_detail=used_tfbs_detail,
                    used_tf_counts=used_tf_counts,
                    used_tf_list=used_tf_list,
                    min_count_per_tf=min_count_per_tf,
                    covers_all_tfs_in_solution=covers_all,
                    required_regulators=required_regulators,
                    min_required_regulators=min_required_regulators,
                    min_count_by_regulator=metadata_min_counts,
                    covers_required_regulators=covers_required,
                    gc_total=gc_total,
                    gc_core=gc_core,
                    input_row_count=input_row_count,
                    input_tf_count=input_tf_count,
                    input_tfbs_count=input_tfbs_count,
                    input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
                    sampling_fraction=sampling_fraction,
                    sampling_fraction_pairs=sampling_fraction_pairs,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    solver_status=solver_status,
                    solver_objective=solver_objective,
                    solver_solve_time_s=solver_solve_time_s,
                )

                src_label = f"densegen:{source_label}:{plan_name}"
                record = OutputRecord.from_sequence(
                    sequence=final_seq,
                    meta=derived,
                    source=src_label,
                    bio_type=output_bio_type,
                    alphabet=output_alphabet,
                )
                accepted = _write_to_sinks(sinks, record)
                if not accepted:
                    failed_solutions += 1
                    duplicate_records += 1
                    _log_rejection(
                        outputs_root,
                        run_id=run_id,
                        input_name=source_label,
                        plan_name=plan_name,
                        reason="output_duplicate",
                        detail={},
                        sequence=final_seq,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        dense_arrays_version=dense_arrays_version,
                        dense_arrays_version_source=dense_arrays_version_source,
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        attempts_buffer=attempts_buffer,
                    )
                    if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                        raise RuntimeError(
                            f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                        )
                    log.info(
                        "[%s/%s] Output duplicate detected; skipping.",
                        source_label,
                        plan_name,
                    )
                    continue

                if composition_rows is not None:
                    for placement_index, entry in enumerate(used_tfbs_detail or []):
                        composition_rows.append(
                            {
                                "sequence_id": record.id,
                                "input_name": source_label,
                                "plan_name": plan_name,
                                "library_index": int(sampling_library_index),
                                "library_hash": str(sampling_library_hash),
                                "placement_index": int(placement_index),
                                "tf": entry.get("tf"),
                                "tfbs": entry.get("tfbs"),
                                "motif_id": entry.get("motif_id"),
                                "tfbs_id": entry.get("tfbs_id"),
                                "orientation": entry.get("orientation"),
                                "offset": entry.get("offset"),
                                "length": entry.get("length"),
                                "end": entry.get("end"),
                                "pad_left": entry.get("pad_left"),
                                "site_id": entry.get("site_id"),
                                "source": entry.get("source"),
                            }
                        )

                _append_attempt(
                    outputs_root,
                    run_id=run_id,
                    input_name=source_label,
                    plan_name=plan_name,
                    status="success",
                    reason="ok",
                    detail={},
                    sequence=final_seq,
                    used_tf_counts=used_tf_counts,
                    used_tf_list=used_tf_list,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    solver_status=solver_status,
                    solver_objective=solver_objective,
                    solver_solve_time_s=solver_solve_time_s,
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    output_id=record.id,
                    library_tfbs=library_tfbs,
                    library_tfs=library_tfs,
                    library_site_ids=library_site_ids,
                    library_sources=library_sources,
                    attempts_buffer=attempts_buffer,
                )

                _update_usage_counts(usage_counts, used_tfbs_detail)
                for tf, count in used_tf_counts.items():
                    tf_usage_counts[tf] = tf_usage_counts.get(tf, 0) + int(count)

                global_generated += 1
                local_generated += 1
                produced_this_library += 1
                produced_total_this_call += 1
                if state_counts is not None:
                    state_counts[(source_label, plan_name)] = int(global_generated)
                    if write_state is not None and checkpoint_every > 0:
                        if global_generated % checkpoint_every == 0:
                            write_state()

                pct = 100.0 * (global_generated / max(1, quota))
                bar = _format_progress_bar(global_generated, quota, width=24)
                cr = getattr(sol, "compression_ratio", float("nan"))
                should_log = progress_every > 0 and global_generated % max(1, progress_every) == 0
                if progress_style == "screen":
                    if should_log and screen_console is not None:
                        now = time.monotonic()
                        if (now - last_screen_refresh) >= progress_refresh_seconds:
                            screen_console.clear()
                            seq_preview = final_seq if len(final_seq) <= 120 else f"{final_seq[:117]}..."
                            screen_console.print(
                                f"[bold]{source_label}/{plan_name}[/] {bar} {global_generated}/{quota} ({pct:.2f}%)"
                            )
                            screen_console.print(
                                f"local {local_generated}/{max_per_subsample} | CR={cr:.3f} | "
                                f"resamples={total_resamples} dup_out={duplicate_records} "
                                f"dup_sol={duplicate_solutions} fails={failed_solutions} stalls={stall_events}"
                            )
                            if latest_failure_totals:
                                screen_console.print(f"failures: {latest_failure_totals}")
                            if tf_usage_counts:
                                screen_console.print(
                                    f"TF leaderboard: {_summarize_leaderboard(tf_usage_counts, top=5)}"
                                )
                            if usage_counts:
                                screen_console.print(f"TFBS leaderboard: {_summarize_leaderboard(usage_counts, top=5)}")
                            diversity_label = _summarize_diversity(
                                usage_counts,
                                tf_usage_counts,
                                library_tfs=library_tfs,
                                library_tfbs=library_tfbs,
                            )
                            screen_console.print(f"Diversity: {diversity_label}")
                            if print_visual:
                                screen_console.print(derived["visual"])
                            screen_console.print(f"sequence {seq_preview}")
                            last_screen_refresh = now
                elif progress_style == "summary":
                    pass
                else:
                    if should_log:
                        if print_visual:
                            log.info(
                                "╭─ %s/%s  %s  %d/%d (%.2f%%) — local %d/%d — CR=%.3f\n"
                                "%s\nsequence %s\n"
                                "╰────────────────────────────────────────────────────────",
                                source_label,
                                plan_name,
                                bar,
                                global_generated,
                                quota,
                                pct,
                                local_generated,
                                max_per_subsample,
                                cr,
                                derived["visual"],
                                final_seq,
                            )
                        else:
                            log.info(
                                "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f | seq %s",
                                source_label,
                                plan_name,
                                bar,
                                global_generated,
                                quota,
                                pct,
                                local_generated,
                                max_per_subsample,
                                cr,
                                final_seq,
                            )

                if leaderboard_every > 0 and global_generated % max(1, leaderboard_every) == 0:
                    failure_totals = _summarize_failure_totals(
                        failure_counts,
                        input_name=source_label,
                        plan_name=plan_name,
                    )
                    latest_failure_totals = failure_totals
                    if progress_style != "screen":
                        log.info(
                            "[%s/%s] Progress %s %d/%d (%.2f%%) | resamples=%d dup_out=%d "
                            "dup_sol=%d fails=%d stalls=%d | %s",
                            source_label,
                            plan_name,
                            bar,
                            global_generated,
                            quota,
                            pct,
                            total_resamples,
                            duplicate_records,
                            duplicate_solutions,
                            failed_solutions,
                            stall_events,
                            failure_totals,
                        )
                        log.info(
                            "[%s/%s] Leaderboard (TF): %s",
                            source_label,
                            plan_name,
                            _summarize_leaderboard(tf_usage_counts, top=5),
                        )
                        log.info(
                            "[%s/%s] Leaderboard (TFBS): %s",
                            source_label,
                            plan_name,
                            _summarize_leaderboard(usage_counts, top=5),
                        )
                        log.info(
                            "[%s/%s] Failed TFBS: %s",
                            source_label,
                            plan_name,
                            _summarize_failure_leaderboard(
                                failure_counts,
                                input_name=source_label,
                                plan_name=plan_name,
                                top=5,
                            ),
                        )
                        log.info(
                            "[%s/%s] Diversity: %s",
                            source_label,
                            plan_name,
                            _summarize_diversity(
                                usage_counts,
                                tf_usage_counts,
                                library_tfs=library_tfs,
                                library_tfbs=library_tfbs,
                            ),
                        )
                    log.info(
                        "[%s/%s] Example: %s",
                        source_label,
                        plan_name,
                        final_seq,
                    )

                if local_generated >= max_per_subsample or global_generated >= quota:
                    break

            if local_generated >= max_per_subsample or global_generated >= quota:
                break

            if produced_this_library == 0:
                reason = "stall_no_solution" if stall_triggered else "no_solution"
                _record_site_failures(reason)
                _append_attempt(
                    outputs_root,
                    run_id=run_id,
                    input_name=source_label,
                    plan_name=plan_name,
                    status="failed",
                    reason=reason,
                    detail={"stall_seconds": stall_seconds} if stall_triggered else {},
                    sequence=None,
                    used_tf_counts=None,
                    used_tf_list=[],
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    solver_status=None,
                    solver_objective=None,
                    solver_solve_time_s=None,
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    library_tfbs=library_tfbs,
                    library_tfs=library_tfs,
                    library_site_ids=library_site_ids,
                    library_sources=library_sources,
                    attempts_buffer=attempts_buffer,
                )

            if pool_strategy == "iterative_subsample" and iterative_min_new_solutions > 0:
                if produced_this_library < iterative_min_new_solutions:
                    log.info(
                        "[%s/%s] Library produced %d < iterative_min_new_solutions=%d; resampling.",
                        source_label,
                        plan_name,
                        produced_this_library,
                        iterative_min_new_solutions,
                    )

            resample_reason = "resample"
            if produced_this_library == 0:
                resample_reason = "stall_no_solution" if stall_triggered else "no_solution"
            elif pool_strategy == "iterative_subsample" and iterative_min_new_solutions > 0:
                if produced_this_library < iterative_min_new_solutions:
                    resample_reason = "min_new_solutions"

            # Resample
            # Alignment (2): allow reactive resampling for subsample under schema>=2.2.
            if not policy.allow_resample():
                raise RuntimeError(
                    f"[{source_label}/{plan_name}] pool_strategy={pool_strategy!r} does not allow resampling "
                    f"under schema_version={global_cfg.schema_version}. "
                    "Reduce quota or use iterative_subsample."
                )
            resamples_in_try += 1
            total_resamples += 1
            if events_path is not None:
                try:
                    _emit_event(
                        events_path,
                        event="RESAMPLE_TRIGGERED",
                        payload={
                            "input_name": source_label,
                            "plan_name": plan_name,
                            "reason": resample_reason,
                            "produced_this_library": int(produced_this_library),
                            "library_index": int(sampling_library_index),
                            "library_hash": str(sampling_library_hash),
                        },
                    )
                except Exception:
                    log.debug("Failed to emit RESAMPLE_TRIGGERED event.", exc_info=True)
            if policy.max_total_resamples > 0 and total_resamples > policy.max_total_resamples:
                raise RuntimeError(f"[{source_label}/{plan_name}] Exceeded max_total_resamples={max_total_resamples}.")
            if resamples_in_try > policy.max_resample_attempts:
                log.info(
                    "[%s/%s] Reached max_resample_attempts (%d) for this subsample try "
                    "(produced %d/%d here). Moving on.",
                    source_label,
                    plan_name,
                    policy.max_resample_attempts,
                    local_generated,
                    max_per_subsample,
                )
                break

            if iterative_max_libraries > 0 and libraries_built >= iterative_max_libraries:
                raise RuntimeError(
                    f"[{source_label}/{plan_name}] Exceeded iterative_max_libraries={iterative_max_libraries}."
                )

            # New library
            library_for_opt, tfbs_parts, regulator_labels, sampling_info = build_library_for_plan(
                source_label=source_label,
                plan_item=plan_item,
                pool=pool,
                sampling_cfg=sampling_cfg,
                seq_len=seq_len,
                min_count_per_tf=min_count_per_tf,
                usage_counts=usage_counts,
                failure_counts=failure_counts if failure_counts else None,
                rng=rng,
                np_rng=np_rng,
                schema_is_22=schema_is_22,
                library_index_start=libraries_built,
            )
            libraries_built = int(sampling_info.get("library_index", libraries_built))
            site_id_by_index = sampling_info.get("site_id_by_index")
            source_by_index = sampling_info.get("source_by_index")
            tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
            motif_id_by_index = sampling_info.get("motif_id_by_index")
            sampling_library_index = sampling_info.get("library_index", sampling_library_index)
            sampling_library_hash = sampling_info.get("library_hash", sampling_library_hash)
            library_tfbs = list(library_for_opt)
            library_tfs = list(regulator_labels) if regulator_labels else []
            library_site_ids = list(site_id_by_index) if site_id_by_index else []
            library_sources = list(source_by_index) if source_by_index else []
            library_tfbs_ids = list(tfbs_id_by_index) if tfbs_id_by_index else []
            library_motif_ids = list(motif_id_by_index) if motif_id_by_index else []
            _record_library_build(
                sampling_info=sampling_info,
                library_tfbs=library_tfbs,
                library_tfs=library_tfs,
                library_tfbs_ids=library_tfbs_ids,
                library_motif_ids=library_motif_ids,
                library_site_ids=library_site_ids,
                library_sources=library_sources,
            )
            # Alignment (7): sampling_fraction uses unique TFBS strings and is bounded.
            sampling_fraction = _compute_sampling_fraction(
                library_for_opt,
                input_tfbs_count=input_tfbs_count,
                pool_strategy=pool_strategy,
            )
            input_meta["sampling_fraction"] = sampling_fraction
            sampling_fraction_pairs = _compute_sampling_fraction_pairs(
                library_for_opt,
                regulator_labels,
                input_pair_count=input_tf_tfbs_pair_count,
                pool_strategy=pool_strategy,
            )
            input_meta["sampling_fraction_pairs"] = sampling_fraction_pairs
            tf_summary = _summarize_tf_counts(regulator_labels)
            if tf_summary:
                log.info(
                    "Resampled library for %s/%s: %d motifs | TF counts: %s",
                    source_label,
                    plan_name,
                    len(library_for_opt),
                    tf_summary,
                )
            else:
                log.info(
                    "Resampled library for %s/%s: %d motifs",
                    source_label,
                    plan_name,
                    len(library_for_opt),
                )

            run = _make_generator(library_for_opt, regulator_labels)
            opt = run.optimizer
            generator = run.generator
            forbid_each = run.forbid_each

        for sink in sinks:
            sink.flush()

        if one_subsample_only:
            _flush_attempts(outputs_root, attempts_buffer)
            if state_counts is not None:
                state_counts[(source_label, plan_name)] = int(global_generated)
                if write_state is not None:
                    write_state()
            snapshot = _current_leaderboard_snapshot()
            if global_generated >= quota and (usage_counts or tf_usage_counts or failure_counts):
                _log_leaderboard_snapshot()
            return produced_total_this_call, {
                "generated": produced_total_this_call,
                "duplicates_skipped": duplicate_records,
                "failed_solutions": failed_solutions,
                "total_resamples": total_resamples,
                "libraries_built": max(0, libraries_built - libraries_built_start),
                "stall_events": stall_events,
                "failed_min_count_per_tf": failed_min_count_per_tf,
                "failed_required_regulators": failed_required_regulators,
                "failed_min_count_by_regulator": failed_min_count_by_regulator,
                "failed_min_required_regulators": failed_min_required_regulators,
                "duplicate_solutions": duplicate_solutions,
                "leaderboard_latest": snapshot,
            }

    _flush_attempts(outputs_root, attempts_buffer)
    log.info("Completed %s/%s: %d/%d", source_label, plan_name, global_generated, quota)
    if state_counts is not None:
        state_counts[(source_label, plan_name)] = int(global_generated)
        if write_state is not None:
            write_state()
    snapshot = _current_leaderboard_snapshot()
    if usage_counts or tf_usage_counts or failure_counts:
        _log_leaderboard_snapshot()
    return produced_total_this_call, {
        "generated": produced_total_this_call,
        "duplicates_skipped": duplicate_records,
        "failed_solutions": failed_solutions,
        "total_resamples": total_resamples,
        "libraries_built": max(0, libraries_built - libraries_built_start),
        "stall_events": stall_events,
        "failed_min_count_per_tf": failed_min_count_per_tf,
        "failed_required_regulators": failed_required_regulators,
        "failed_min_count_by_regulator": failed_min_count_by_regulator,
        "failed_min_required_regulators": failed_min_required_regulators,
        "duplicate_solutions": duplicate_solutions,
        "leaderboard_latest": snapshot,
    }


def run_pipeline(loaded: LoadedConfig, *, deps: PipelineDeps | None = None) -> RunSummary:
    deps = deps or default_deps()
    cfg = loaded.root.densegen
    install_native_stderr_filters(suppress_solver_messages=bool(cfg.logging.suppress_solver_stderr))
    run_root = resolve_run_root(loaded.path, cfg.run.root)
    run_root_str = str(run_root)
    config_sha = hashlib.sha256(loaded.path.read_bytes()).hexdigest()
    try:
        run_cfg_path = str(loaded.path.relative_to(run_root))
    except ValueError:
        run_cfg_path = str(loaded.path)

    # Seed
    seed = int(cfg.runtime.random_seed)
    seeds = derive_seed_map(seed, ["stage_a", "stage_b", "solver"])
    rng = random.Random(seeds["stage_b"])
    np_rng_stage_a = np.random.default_rng(seeds["stage_a"])
    np_rng_stage_b = np.random.default_rng(seeds["stage_b"])

    # Plan & solver
    pl = cfg.generation.resolve_plan()
    chosen_solver = select_solver(
        cfg.solver.backend,
        deps.optimizer,
        strategy=str(cfg.solver.strategy),
        fallback_to_cbc=bool(cfg.solver.fallback_to_cbc),
    )
    dense_arrays_version, dense_arrays_version_source = _resolve_dense_arrays_version(loaded.path)

    # Build sinks
    sinks = list(deps.sink_factory(cfg, loaded.path))
    _assert_sink_alignment(sinks)
    output_bio_type, output_alphabet = resolve_bio_alphabet(cfg)

    total = 0
    per_plan: dict[tuple[str, str], int] = {}
    plan_stats: dict[tuple[str, str], dict[str, int]] = {}
    plan_order: list[tuple[str, str]] = []
    plan_leaderboards: dict[tuple[str, str], dict] = {}
    inputs_manifest_entries: dict[str, dict] = {}
    source_cache: dict[str, PoolData] = {}
    library_build_rows: list[dict] = []
    library_member_rows: list[dict] = []
    composition_rows: list[dict] = []
    outputs_root = run_outputs_root(run_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    events_path = outputs_root / "meta" / "events.jsonl"
    try:
        _write_effective_config(
            cfg=cfg, cfg_path=loaded.path, run_root=run_root, seeds=seeds, outputs_root=outputs_root
        )
    except Exception:
        log.debug("Failed to write effective_config.json.", exc_info=True)
    pool_dir = outputs_root / "pools"
    try:
        _pool_artifact, pool_data = build_pool_artifact(
            cfg=cfg,
            cfg_path=loaded.path,
            deps=deps,
            rng=np_rng_stage_a,
            outputs_root=outputs_root,
            out_dir=pool_dir,
            overwrite=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build Stage-A TFBS pools: {exc}") from exc
    try:
        _emit_event(
            events_path,
            event="POOL_BUILT",
            payload={
                "inputs": [
                    {
                        "name": pool.name,
                        "input_type": pool.input_type,
                        "pool_mode": pool.pool_mode,
                        "rows": int(pool.df.shape[0]) if pool.df is not None else int(len(pool.sequences)),
                    }
                    for pool in pool_data.values()
                ]
            },
        )
    except Exception:
        log.debug("Failed to emit POOL_BUILT event.", exc_info=True)
    for name, pool in pool_data.items():
        source_cache[name] = pool
    ensure_run_meta_dir(run_root)
    state_path = run_state_path(run_root)
    state_created_at = datetime.now(timezone.utc).isoformat()
    if state_path.exists():
        try:
            existing_state = load_run_state(state_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read run_state.json: {exc}") from exc
        if existing_state.run_id and existing_state.run_id != cfg.run.id:
            raise RuntimeError(
                "Existing run_state.json was created with a different run_id. "
                "Remove run_state.json or stage a new run root to start fresh."
            )
        if existing_state.config_sha256 and existing_state.config_sha256 != config_sha:
            raise RuntimeError(
                "Existing run_state.json was created with a different config. "
                "Remove run_state.json or stage a new run root to start fresh."
            )
        if existing_state.created_at:
            state_created_at = existing_state.created_at

    # Resume from existing outputs if present and aligned with config/run.
    existing_counts: dict[tuple[str, str], int] = {}
    existing_usage_by_plan: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    site_failure_counts = _load_failure_counts_from_attempts(outputs_root)
    if cfg.output.targets:
        try:
            df_existing, _ = load_records_from_config(
                loaded.root,
                loaded.path,
                columns=[
                    "densegen__run_config_sha256",
                    "densegen__run_id",
                    "densegen__input_name",
                    "densegen__plan",
                    "densegen__used_tfbs_detail",
                ],
            )
        except Exception:
            df_existing = None
        if df_existing is not None and not df_existing.empty:
            if "densegen__run_config_sha256" in df_existing.columns:
                mismatched = df_existing["densegen__run_config_sha256"].dropna().unique().tolist()
                if mismatched and any(val != config_sha for val in mismatched):
                    raise RuntimeError(
                        "Existing outputs were produced with a different config. "
                        "Remove outputs/ or stage a new run root to start fresh."
                    )
            if "densegen__run_id" in df_existing.columns:
                run_ids = df_existing["densegen__run_id"].dropna().unique().tolist()
                if run_ids and any(val != cfg.run.id for val in run_ids):
                    raise RuntimeError(
                        "Existing outputs were produced with a different run_id. "
                        "Remove outputs/ or stage a new run root to start fresh."
                    )
            if {"densegen__input_name", "densegen__plan"} <= set(df_existing.columns):
                counts = df_existing.groupby(["densegen__input_name", "densegen__plan"]).size().astype(int).to_dict()
                existing_counts = {(str(k[0]), str(k[1])): int(v) for k, v in counts.items()}
            if "densegen__used_tfbs_detail" in df_existing.columns:
                for _, row in df_existing.iterrows():
                    input_name = str(row.get("densegen__input_name") or "")
                    plan_name = str(row.get("densegen__plan") or "")
                    if not input_name or not plan_name:
                        continue
                    key = (input_name, plan_name)
                    counts = existing_usage_by_plan.setdefault(key, {})
                    used = _parse_used_tfbs_detail(row.get("densegen__used_tfbs_detail"))
                    _update_usage_counts(counts, used)
            if existing_counts:
                total = sum(existing_counts.values())
                per_plan = dict(existing_counts)
                log.info(
                    "Resuming from existing outputs: %d sequences across %d plan(s).",
                    total,
                    len(existing_counts),
                )

    def _accumulate_stats(key: tuple[str, str], stats: dict) -> None:
        if key not in plan_stats:
            plan_stats[key] = {
                "generated": 0,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 0,
                "stall_events": 0,
                "failed_min_count_per_tf": 0,
                "failed_required_regulators": 0,
                "failed_min_count_by_regulator": 0,
                "failed_min_required_regulators": 0,
                "duplicate_solutions": 0,
            }
            plan_order.append(key)
        dest = plan_stats[key]
        for field in dest:
            dest[field] += int(stats.get(field, 0))

    # Round-robin scheduler
    round_robin = bool(cfg.runtime.round_robin)
    if round_robin and str(cfg.generation.sampling.pool_strategy) == "iterative_subsample":
        log.warning(
            "round_robin=true with pool_strategy=iterative_subsample will rebuild libraries more frequently; "
            "expect higher runtime for multi-plan runs."
        )
    inputs = cfg.inputs
    checkpoint_every = int(cfg.runtime.checkpoint_every)
    state_counts: dict[tuple[str, str], int] = {}
    for s in inputs:
        for item in pl:
            state_counts[(s.name, item.name)] = int(existing_counts.get((s.name, item.name), 0))

    if state_path.exists() and not existing_counts:
        # run_state exists but no outputs; avoid accidental double-counting.
        existing_state = load_run_state(state_path)
        if existing_state.items and sum(item.generated for item in existing_state.items) > 0:
            raise RuntimeError(
                "run_state.json indicates prior progress, but no outputs were found. "
                "Restore outputs or delete run_state.json before resuming."
            )

    def _write_state() -> None:
        _write_run_state(
            state_path,
            run_id=cfg.run.id,
            schema_version=str(cfg.schema_version),
            config_sha256=config_sha,
            run_root=str(run_root),
            counts=state_counts,
            created_at=state_created_at,
        )

    _write_state()
    # Seed plan stats with any existing outputs to keep manifests aligned on resume.
    for s in inputs:
        for item in pl:
            key = (s.name, item.name)
            if key not in plan_stats:
                plan_stats[key] = {
                    "generated": int(existing_counts.get(key, 0)),
                    "duplicates_skipped": 0,
                    "failed_solutions": 0,
                    "total_resamples": 0,
                    "libraries_built": 0,
                    "stall_events": 0,
                    "failed_min_count_per_tf": 0,
                    "failed_required_regulators": 0,
                    "failed_min_count_by_regulator": 0,
                    "failed_min_required_regulators": 0,
                    "duplicate_solutions": 0,
                }
                plan_order.append(key)

    if not round_robin:
        for s in inputs:
            for item in pl:
                produced, stats = _process_plan_for_source(
                    s,
                    item,
                    cfg,
                    sinks,
                    chosen_solver=chosen_solver,
                    deps=deps,
                    rng=rng,
                    np_rng=np_rng_stage_b,
                    cfg_path=loaded.path,
                    run_id=cfg.run.id,
                    run_root=run_root_str,
                    run_config_path=run_cfg_path,
                    run_config_sha256=config_sha,
                    random_seed=seed,
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    output_bio_type=output_bio_type,
                    output_alphabet=output_alphabet,
                    one_subsample_only=False,
                    already_generated=int(existing_counts.get((s.name, item.name), 0)),
                    inputs_manifest=inputs_manifest_entries,
                    existing_usage_counts=existing_usage_by_plan.get((s.name, item.name)),
                    state_counts=state_counts,
                    checkpoint_every=checkpoint_every,
                    write_state=_write_state,
                    site_failure_counts=site_failure_counts,
                    source_cache=source_cache,
                    library_build_rows=library_build_rows,
                    library_member_rows=library_member_rows,
                    composition_rows=composition_rows,
                    events_path=events_path,
                )
                per_plan[(s.name, item.name)] = per_plan.get((s.name, item.name), 0) + produced
                total += produced
                leaderboard_latest = stats.get("leaderboard_latest")
                if leaderboard_latest is not None:
                    plan_leaderboards[(s.name, item.name)] = leaderboard_latest
                _accumulate_stats((s.name, item.name), stats)
    else:
        produced_counts: dict[tuple[str, str], int] = dict(existing_counts)
        done = False
        while not done:
            done = True
            for s in inputs:
                for item in pl:
                    key = (s.name, item.name)
                    current = produced_counts.get(key, 0)
                    quota = int(item.quota)
                    if current >= quota:
                        continue
                    done = False
                    produced, stats = _process_plan_for_source(
                        s,
                        item,
                        cfg,
                        sinks,
                        chosen_solver=chosen_solver,
                        deps=deps,
                        rng=rng,
                        np_rng=np_rng_stage_b,
                        cfg_path=loaded.path,
                        run_id=cfg.run.id,
                        run_root=run_root_str,
                        run_config_path=run_cfg_path,
                        run_config_sha256=config_sha,
                        random_seed=seed,
                        dense_arrays_version=dense_arrays_version,
                        dense_arrays_version_source=dense_arrays_version_source,
                        output_bio_type=output_bio_type,
                        output_alphabet=output_alphabet,
                        one_subsample_only=True,
                        already_generated=current,
                        inputs_manifest=inputs_manifest_entries,
                        existing_usage_counts=existing_usage_by_plan.get((s.name, item.name)),
                        state_counts=state_counts,
                        checkpoint_every=checkpoint_every,
                        write_state=_write_state,
                        site_failure_counts=site_failure_counts,
                        source_cache=source_cache,
                        library_build_rows=library_build_rows,
                        library_member_rows=library_member_rows,
                        composition_rows=composition_rows,
                        events_path=events_path,
                    )
                    produced_counts[key] = current + produced
                    leaderboard_latest = stats.get("leaderboard_latest")
                    if leaderboard_latest is not None:
                        plan_leaderboards[key] = leaderboard_latest
                    _accumulate_stats(key, stats)
        per_plan = produced_counts
        total = sum(per_plan.values())

    for sink in sinks:
        sink.finalize()

    outputs_root = run_outputs_root(run_root)
    _consolidate_parts(outputs_root, part_glob="attempts_part-*.parquet", final_name="attempts.parquet")

    if library_build_rows:
        libraries_dir = outputs_root / "libraries"
        existing_builds: list[dict] = []
        existing_members: list[dict] = []
        builds_path = libraries_dir / "library_builds.parquet"
        members_path = libraries_dir / "library_members.parquet"
        if builds_path.exists():
            try:
                existing_builds = pd.read_parquet(builds_path).to_dict("records")
            except Exception:
                log.warning("Failed to read existing library_builds.parquet; overwriting.", exc_info=True)
                existing_builds = []
        if members_path.exists():
            try:
                existing_members = pd.read_parquet(members_path).to_dict("records")
            except Exception:
                log.warning("Failed to read existing library_members.parquet; overwriting.", exc_info=True)
                existing_members = []

        existing_indices = {
            int(row.get("library_index") or 0) for row in existing_builds if row.get("library_index") is not None
        }
        new_builds = [row for row in library_build_rows if int(row.get("library_index") or 0) not in existing_indices]
        build_rows = existing_builds + new_builds

        existing_member_keys = {
            (
                int(row.get("library_index") or 0),
                int(row.get("position") or 0),
            )
            for row in existing_members
        }
        new_members = [
            row
            for row in library_member_rows
            if (int(row.get("library_index") or 0), int(row.get("position") or 0)) not in existing_member_keys
        ]
        member_rows = existing_members + new_members

        try:
            write_library_artifact(
                out_dir=libraries_dir,
                builds=build_rows,
                members=member_rows,
                cfg_path=loaded.path,
                run_id=str(cfg.run.id),
                run_root=run_root,
                overwrite=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write library artifacts: {exc}") from exc

    if composition_rows:
        composition_path = outputs_root / "composition.parquet"
        existing_rows: list[dict] = []
        if composition_path.exists():
            try:
                existing_rows = pd.read_parquet(composition_path).to_dict("records")
            except Exception:
                log.warning("Failed to read existing composition.parquet; overwriting.", exc_info=True)
                existing_rows = []
        existing_keys = {
            (str(row.get("sequence_id") or ""), int(row.get("placement_index") or 0)) for row in existing_rows
        }
        new_rows = [
            row
            for row in composition_rows
            if (str(row.get("sequence_id") or ""), int(row.get("placement_index") or 0)) not in existing_keys
        ]
        pd.DataFrame(existing_rows + new_rows).to_parquet(composition_path, index=False)

    manifest_items = [
        PlanManifest(
            input_name=key[0],
            plan_name=key[1],
            generated=plan_stats[key]["generated"],
            duplicates_skipped=plan_stats[key]["duplicates_skipped"],
            failed_solutions=plan_stats[key]["failed_solutions"],
            total_resamples=plan_stats[key]["total_resamples"],
            libraries_built=plan_stats[key]["libraries_built"],
            stall_events=plan_stats[key]["stall_events"],
            failed_min_count_per_tf=plan_stats[key]["failed_min_count_per_tf"],
            failed_required_regulators=plan_stats[key]["failed_required_regulators"],
            failed_min_count_by_regulator=plan_stats[key]["failed_min_count_by_regulator"],
            failed_min_required_regulators=plan_stats[key]["failed_min_required_regulators"],
            duplicate_solutions=plan_stats[key]["duplicate_solutions"],
            leaderboard_latest=plan_leaderboards.get(key),
        )
        for key in plan_order
    ]
    manifest = RunManifest(
        run_id=cfg.run.id,
        created_at=datetime.now(timezone.utc).isoformat(),
        schema_version=str(cfg.schema_version),
        config_sha256=config_sha,
        run_root=run_root_str,
        random_seed=seed,
        seed_stage_a=seeds.get("stage_a"),
        seed_stage_b=seeds.get("stage_b"),
        seed_solver=seeds.get("solver"),
        solver_backend=chosen_solver,
        solver_strategy=str(cfg.solver.strategy),
        solver_options=list(cfg.solver.options),
        solver_strands=str(cfg.solver.strands),
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        items=manifest_items,
    )
    manifest_path = run_manifest_path(run_root)
    manifest.write_json(manifest_path)

    if inputs_manifest_entries:
        payload = {
            "schema_version": "1.0",
            "run_id": cfg.run.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "inputs": [
                inputs_manifest_entries.get(inp.name) for inp in cfg.inputs if inp.name in inputs_manifest_entries
            ],
            "library_sampling": cfg.generation.sampling.model_dump(),
        }
        inputs_manifest = inputs_manifest_path(run_root)
        inputs_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
        log.info("Inputs manifest written: %s", inputs_manifest)

    _write_state()

    return RunSummary(total_generated=total, per_plan=per_plan)
