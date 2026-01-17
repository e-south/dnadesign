"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample_workflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import signal
import time
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.elites import find_elites_parquet
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.app.run_service import (
    drop_run_index_entries,
    update_run_index_from_manifest,
    update_run_index_from_status,
)
from dnadesign.cruncher.app.target_service import (
    has_blocking_target_errors,
    target_statuses,
)
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    config_used_path,
    elites_json_path,
    elites_path,
    elites_yaml_path,
    ensure_run_dirs,
    live_metrics_path,
    run_group_label,
    sequences_path,
    status_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, load_manifest, write_manifest
from dnadesign.cruncher.artifacts.status import RunStatusWriter
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import (
    AutoOptConfig,
    BetaLadderFixed,
    BetaLadderGeometric,
    CoolingFixed,
    CoolingLinear,
    CoolingPiecewise,
    CruncherConfig,
    SampleConfig,
)
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.labels import format_regulator_slug, regulator_sets
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.sequence import canon_int, dsdna_hamming, hamming_distance
from dnadesign.cruncher.core.state import SequenceState
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import (
    read_lockfile,
    validate_lockfile,
    verify_lockfile_hashes,
)
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)


@contextmanager
def _sigterm_as_keyboard_interrupt():
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None:
        yield
        return
    previous = signal.getsignal(sigterm)

    def _handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(sigterm, _handler)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)


@dataclass
class AutoOptCandidate:
    kind: str
    length: int | None
    budget: int | None
    cooling_boost: float
    run_dir: Path
    run_dirs: list[Path]
    best_score: float | None
    top_k_median_final: float | None
    best_score_final: float | None
    top_k_ci_low: float | None
    top_k_ci_high: float | None
    rhat: float | None
    ess: float | None
    unique_fraction: float | None
    balance_median: float | None
    diversity: float | None
    improvement: float | None
    acceptance_b: float | None
    acceptance_m: float | None
    acceptance_mh: float | None
    swap_rate: float | None
    status: str
    quality: str
    warnings: list[str]
    diagnostics: dict[str, object]


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        resolve_catalog_root(config_path, cfg.motif_store.catalog_root),
        pwm_source=cfg.motif_store.pwm_source,
        site_kinds=cfg.motif_store.site_kinds,
        combine_sites=cfg.motif_store.combine_sites,
        site_window_lengths=cfg.motif_store.site_window_lengths,
        site_window_center=cfg.motif_store.site_window_center,
        pwm_window_lengths=cfg.motif_store.pwm_window_lengths,
        pwm_window_strategy=cfg.motif_store.pwm_window_strategy,
        min_sites_for_pwm=cfg.motif_store.min_sites_for_pwm,
        allow_low_sites=cfg.motif_store.allow_low_sites,
        pseudocounts=cfg.motif_store.pseudocounts,
    )


def _lockmap_for(cfg: CruncherConfig, config_path: Path) -> dict[str, object]:
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    lock_path = resolve_lock_path(config_path)
    if not lock_path.exists():
        raise ValueError(f"Lockfile is required: {lock_path}. Run `cruncher lock {config_path.name}`.")
    lockfile = read_lockfile(lock_path)
    required = {tf for group in cfg.regulator_sets for tf in group}
    validate_lockfile(
        lockfile,
        expected_pwm_source=cfg.motif_store.pwm_source,
        expected_site_kinds=cfg.motif_store.site_kinds,
        expected_combine_sites=cfg.motif_store.combine_sites,
        required_tfs=required,
    )
    verify_lockfile_hashes(
        lockfile=lockfile,
        catalog_root=catalog_root,
        expected_pwm_source=cfg.motif_store.pwm_source,
    )
    return lockfile.resolved


def _load_pwms_for_set(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    tfs: list[str],
    lockmap: dict[str, object],
) -> dict[str, PWM]:
    store = _store(cfg, config_path)
    pwms: dict[str, PWM] = {}
    for tf in sorted(tfs):
        logger.debug("  Loading PWM for %s", tf)
        entry = lockmap.get(tf)
        if entry is None:
            raise ValueError(f"Missing lock entry for TF '{tf}'")
        ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
        pwms[tf] = store.get_pwm(ref)
    return pwms


def _stable_pilot_seed(
    *,
    pilot_cfg: SampleConfig,
    tfs: list[str],
    lockmap: dict[str, object],
    label: str,
    kind: str,
) -> int:
    moves = resolve_move_config(pilot_cfg.moves)
    payload = {
        "label": label,
        "kind": kind,
        "seed": pilot_cfg.rng.seed,
        "mode": pilot_cfg.mode,
        "objective": pilot_cfg.objective.model_dump(),
        "init": pilot_cfg.init.model_dump(),
        "budget": pilot_cfg.budget.model_dump(),
        "elites": pilot_cfg.elites.model_dump(),
        "moves": moves.model_dump(),
        "optimizer": pilot_cfg.optimizer.model_dump(),
        "optimizers": pilot_cfg.optimizers.model_dump(),
        "tfs": sorted(tfs),
        "locks": {tf: lockmap[tf].sha256 for tf in sorted(tfs)},
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return int(digest[:8], 16)


def _candidate_lengths(sample_cfg: SampleConfig, auto_cfg: AutoOptConfig, pwms: dict[str, PWM]) -> list[int]:
    max_w = max(pwm.length for pwm in pwms.values()) if pwms else sample_cfg.init.length
    sum_w = sum(pwm.length for pwm in pwms.values()) if pwms else sample_cfg.init.length
    length_cfg = auto_cfg.length
    if not length_cfg.enabled:
        return [sample_cfg.init.length]
    if length_cfg.mode == "ladder":
        min_len = length_cfg.min_length if length_cfg.min_length is not None else max_w
        max_len = length_cfg.max_length if length_cfg.max_length is not None else sum_w
        min_len = max(min_len, max_w)
        max_len = max(max_len, min_len)
        lengths = list(range(min_len, max_len + 1, 1))
    else:
        min_len = length_cfg.min_length if length_cfg.min_length is not None else max_w
        max_len = length_cfg.max_length if length_cfg.max_length is not None else sample_cfg.init.length
        min_len = max(min_len, max_w)
        max_len = max(max_len, min_len)
        step = max(1, length_cfg.step)
        lengths = list(range(min_len, max_len + 1, step))
        if sample_cfg.init.length not in lengths and min_len <= sample_cfg.init.length <= max_len:
            lengths.append(sample_cfg.init.length)
        lengths = sorted(set(lengths))
        if length_cfg.max_candidates and len(lengths) > length_cfg.max_candidates:
            lengths = lengths[: length_cfg.max_candidates]
    if not lengths:
        raise ValueError("Auto-opt length selection produced no valid candidates.")
    return lengths


_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_sequence_string(seq: str) -> np.ndarray:
    clean = seq.strip().upper()
    try:
        return np.array([_BASE_TO_INT[ch] for ch in clean], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"Invalid base in sequence '{seq}'") from exc


def _sample_insert_base(pad_with: str | None, rng: np.random.Generator) -> np.int8:
    if pad_with is None or pad_with == "background":
        return np.int8(rng.integers(0, 4))
    base = pad_with.upper()
    if base not in _BASE_TO_INT:
        raise ValueError(f"Invalid pad_with='{pad_with}' for warm-start insertion")
    return np.int8(_BASE_TO_INT[base])


def _extend_sequence_by_one(seq_arr: np.ndarray, *, rng: np.random.Generator, pad_with: str | None) -> np.ndarray:
    insert_pos = int(rng.integers(0, seq_arr.size + 1))
    insert_base = _sample_insert_base(pad_with, rng)
    return np.concatenate([seq_arr[:insert_pos], np.array([insert_base], dtype=np.int8), seq_arr[insert_pos:]])


def _warm_start_seeds_from_elites(
    run_dir: Path,
    *,
    target_length: int,
    rng: np.random.Generator,
    pad_with: str | None,
    max_seeds: int | None = None,
) -> list[np.ndarray]:
    try:
        elite_path = find_elites_parquet(run_dir)
        elites_df = read_parquet(elite_path)
    except Exception as exc:
        logger.warning("Warm-start: unable to load elites from %s (%s).", run_dir, exc)
        return []
    if "sequence" not in elites_df.columns:
        return []
    seeds: list[np.ndarray] = []
    for seq_str in elites_df["sequence"].astype(str).tolist():
        seq_arr = _encode_sequence_string(seq_str)
        if seq_arr.size == target_length:
            seeds.append(seq_arr.copy())
        elif seq_arr.size == target_length - 1:
            seeds.append(_extend_sequence_by_one(seq_arr, rng=rng, pad_with=pad_with))
        else:
            continue
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    return seeds


def _warm_start_seeds_from_sequences(
    run_dir: Path,
    *,
    target_length: int,
    rng: np.random.Generator,
    pad_with: str | None,
    max_seeds: int | None = None,
) -> list[np.ndarray]:
    try:
        seq_path = sequences_path(run_dir)
        seq_df = read_parquet(seq_path)
    except Exception as exc:
        logger.warning("Warm-start: unable to load sequences from %s (%s).", run_dir, exc)
        return []
    if "sequence" not in seq_df.columns:
        return []
    if "phase" in seq_df.columns:
        seq_df = seq_df[seq_df["phase"] == "draw"].copy()

    score_cols = [col for col in seq_df.columns if col.startswith("score_")]
    use_combined = "combined_score_final" in seq_df.columns

    candidates: list[tuple[float, np.ndarray]] = []
    for _, row in seq_df.iterrows():
        seq_str = str(row["sequence"])
        seq_arr = _encode_sequence_string(seq_str)
        score = 0.0
        if use_combined:
            raw = row.get("combined_score_final")
            score = float(raw) if raw is not None else 0.0
        elif score_cols:
            vals = [row.get(col) for col in score_cols]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                score = float(np.mean(vals))
        if seq_arr.size == target_length:
            candidates.append((score, seq_arr.copy()))
        elif seq_arr.size == target_length - 1:
            candidates.append((score, _extend_sequence_by_one(seq_arr, rng=rng, pad_with=pad_with)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    seeds: list[np.ndarray] = []
    for _, seq_arr in candidates:
        seeds.append(seq_arr)
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    return seeds


def _elite_filter_passes(
    *,
    norm_map: dict[str, float],
    min_norm: float,
    sum_norm: float,
    min_per_tf_norm: float | None,
    require_all_tfs_over_min_norm: bool,
    pwm_sum_min: float,
) -> bool:
    if min_per_tf_norm is not None:
        if require_all_tfs_over_min_norm:
            if not all(score >= min_per_tf_norm for score in norm_map.values()):
                return False
        else:
            if min_norm < min_per_tf_norm:
                return False
    if pwm_sum_min > 0 and sum_norm < pwm_sum_min:
        return False
    return True


def _filter_elite_candidates(
    candidates: list["_EliteCandidate"],
    *,
    min_per_tf_norm: float | None,
    require_all_tfs_over_min_norm: bool,
    pwm_sum_min: float,
) -> list["_EliteCandidate"]:
    filtered: list[_EliteCandidate] = []
    for cand in candidates:
        if _elite_filter_passes(
            norm_map=cand.norm_map,
            min_norm=cand.min_norm,
            sum_norm=cand.sum_norm,
            min_per_tf_norm=min_per_tf_norm,
            require_all_tfs_over_min_norm=require_all_tfs_over_min_norm,
            pwm_sum_min=pwm_sum_min,
        ):
            filtered.append(cand)
    return filtered


def _elite_rank_key(combined_score: float, min_norm: float, sum_norm: float) -> tuple[float, float, float]:
    return (combined_score, min_norm, sum_norm)


def _resolve_final_softmin_beta(optimizer: object, sample_cfg: SampleConfig) -> float | None:
    if hasattr(optimizer, "final_softmin_beta") and callable(getattr(optimizer, "final_softmin_beta")):
        try:
            return getattr(optimizer, "final_softmin_beta")()
        except Exception:
            pass
    if hasattr(optimizer, "stats") and callable(getattr(optimizer, "stats")):
        try:
            stats = getattr(optimizer, "stats")()
            if isinstance(stats, dict):
                value = stats.get("final_softmin_beta")
                if isinstance(value, (int, float)):
                    return float(value)
        except Exception:
            pass
    softmin_cfg = sample_cfg.objective.softmin
    if softmin_cfg.enabled:
        total = sample_cfg.budget.tune + sample_cfg.budget.draws
        softmin_sched = {k: v for k, v in softmin_cfg.model_dump().items() if k in ("kind", "beta", "stages")}
        return make_beta_scheduler(softmin_sched, total)(total - 1)
    return None


_AUTO_OPT_BOOTSTRAP_SAMPLES = 300
_AUTO_OPT_BOOTSTRAP_PCT = (5.0, 95.0)


def _draw_scores_from_sequences(seq_df: pd.DataFrame) -> np.ndarray:
    if "combined_score_final" not in seq_df.columns:
        return np.array([], dtype=float)
    df = seq_df
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"]
    series = pd.to_numeric(df["combined_score_final"], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return np.array([], dtype=float)
    return series.to_numpy(dtype=float)


def _best_score_final_from_sequences(seq_df: pd.DataFrame) -> float | None:
    scores = _draw_scores_from_sequences(seq_df)
    if scores.size == 0:
        return None
    return float(np.max(scores))


def _top_k_median_from_scores(scores: np.ndarray, k: int) -> float | None:
    if scores.size == 0:
        return None
    k = max(1, min(int(k), int(scores.size)))
    if k >= scores.size:
        return float(np.median(scores))
    topk = np.partition(scores, scores.size - k)[scores.size - k :]
    return float(np.median(topk))


def _top_k_median_from_sequences(seq_df: pd.DataFrame, *, k: int) -> float | None:
    scores = _draw_scores_from_sequences(seq_df)
    return _top_k_median_from_scores(scores, k)


def _bootstrap_top_k_ci(
    scores: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
    n_boot: int = _AUTO_OPT_BOOTSTRAP_SAMPLES,
) -> tuple[float, float] | None:
    if scores.size == 0:
        return None
    if scores.size == 1:
        value = float(scores[0])
        return value, value
    n_boot = max(1, int(n_boot))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, scores.size, size=scores.size)
        sample = scores[idx]
        boot[i] = _top_k_median_from_scores(sample, k) or float("nan")
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return None
    low, high = np.percentile(boot, _AUTO_OPT_BOOTSTRAP_PCT)
    return float(low), float(high)


def _bootstrap_seed_payload(payload: dict[str, object]) -> int:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _bootstrap_seed(
    *,
    manifest: dict[str, object],
    run_dir: Path,
    kind: str,
) -> int:
    _ = run_dir  # unused but retained for call sites; keep signature stable
    auto_meta = manifest.get("auto_opt") or {}
    regulator_set = manifest.get("regulator_set") or {}
    payload = {
        "seed": manifest.get("seed"),
        "kind": kind,
        "attempt": auto_meta.get("attempt"),
        "candidate": auto_meta.get("candidate"),
        "budget": auto_meta.get("budget"),
        "replicate": auto_meta.get("replicate"),
        "length": manifest.get("sequence_length"),
        "tfs": regulator_set.get("tfs"),
    }
    return _bootstrap_seed_payload(payload)


def _pooled_bootstrap_seed(
    *,
    manifests: list[dict[str, object]],
    kind: str,
    length: int | None,
    budget: int | None,
) -> int | None:
    if not manifests:
        return None
    replicates: list[dict[str, object]] = []
    for manifest in manifests:
        auto_meta = manifest.get("auto_opt") or {}
        replicates.append(
            {
                "seed": manifest.get("seed"),
                "attempt": auto_meta.get("attempt"),
                "candidate": auto_meta.get("candidate"),
                "budget": auto_meta.get("budget"),
                "replicate": auto_meta.get("replicate"),
                "length": manifest.get("sequence_length"),
            }
        )
    payload = {
        "kind": kind,
        "length": length,
        "budget": budget,
        "replicates": sorted(replicates, key=lambda item: json.dumps(item, sort_keys=True)),
    }
    return _bootstrap_seed_payload(payload)


def _polish_sequence(
    seq_arr: np.ndarray,
    *,
    evaluator: SequenceEvaluator,
    beta_softmin_final: float | None,
    max_rounds: int,
    improvement_tol: float,
    max_evals: int | None,
) -> np.ndarray:
    seq = seq_arr.copy()
    evals = 0

    def _score() -> float:
        nonlocal evals
        evals += 1
        return evaluator.combined(SequenceState(seq), beta=beta_softmin_final)

    best_score = _score()
    for _ in range(max_rounds):
        improved = False
        for i in range(seq.size):
            old_base = seq[i]
            best_base = old_base
            best_local = best_score
            for b in range(4):
                if b == old_base:
                    continue
                seq[i] = b
                score = _score()
                if score > best_local + improvement_tol:
                    best_local = score
                    best_base = b
                if max_evals is not None and evals >= max_evals:
                    seq[i] = best_base
                    return seq
            seq[i] = best_base
            if best_base != old_base:
                best_score = best_local
                improved = True
            if max_evals is not None and evals >= max_evals:
                return seq
        if not improved:
            break
    return seq


@dataclass
class _EliteCandidate:
    seq_arr: np.ndarray
    chain_id: int
    draw_idx: int
    combined_score: float
    min_norm: float
    sum_norm: float
    per_tf_map: dict[str, float]
    norm_map: dict[str, float]


def _write_length_ladder_table(rows: list[dict[str, object]], *, pilot_root: Path) -> Path:
    tables_dir = pilot_root / "analysis" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "length_ladder.csv"
    fieldnames = ["length", "best_score", "balance", "diversity", "unique_fraction", "runtime_sec"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return out_path


def _save_config(
    cfg: CruncherConfig,
    batch_dir: Path,
    config_path: Path,
    *,
    tfs: list[str],
    set_index: int,
    sample_cfg: SampleConfig | None = None,
    log_fn: Callable[..., None] | None = None,
) -> None:
    """
    Save the exact Pydantic-validated config into <batch_dir>/meta/config_used.yaml,
    plus, for each TF:
      - alphabet: ["A","C","G","T"]
      - pwm_matrix: a list of [p_A, p_C, p_G, p_T] for each position
      - consensus: consensus sequence string
    """
    cfg_path = config_used_path(batch_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json")
    if sample_cfg is not None:
        data["sample"] = sample_cfg.model_dump(mode="json")

    store = _store(cfg, config_path)
    pwms_info: dict[str, dict[str, object]] = {}

    logger.debug("Saving PWM info for config_used.yaml")
    lockmap = _lockmap_for(cfg, config_path)
    for tf in sorted(tfs):
        entry = lockmap.get(tf)
        if entry is None:
            raise ValueError(f"Missing lock entry for TF '{tf}'")
        ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
        pwm: PWM = store.get_pwm(ref)

        # Build consensus: argmax over each column
        cons_vec = np.argmax(pwm.matrix, axis=1)
        consensus = "".join("ACGT"[i] for i in cons_vec)

        pwm_probs_rounded: list[list[float]] = []
        for row in pwm.matrix:
            rounded_row = [round(float(p), 6) for p in row]
            pwm_probs_rounded.append(rounded_row)

        pwms_info[tf] = {
            "alphabet": ["A", "C", "G", "T"],
            "pwm_matrix": pwm_probs_rounded,
            "consensus": consensus,
        }

    data["pwms_info"] = pwms_info
    data["active_regulator_set"] = {"index": set_index, "tfs": tfs}
    with cfg_path.open("w") as fh:
        yaml.safe_dump({"cruncher": data}, fh, sort_keys=False, default_flow_style=False)
    log_fn = log_fn or logger.info
    log_fn("Wrote config_used.yaml to %s", cfg_path.relative_to(batch_dir.parent))


def _write_parquet_rows(
    path: Path,
    rows: Iterable[dict[str, object]],
    *,
    chunk_size: int = 10000,
    schema: Any | None = None,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, object]] = []
    count = 0
    for row in rows:
        buffer.append(row)
        if len(buffer) < chunk_size:
            continue
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(str(path), table.schema)
        writer.write_table(table)
        count += len(buffer)
        buffer.clear()
    if buffer:
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(str(path), table.schema)
        writer.write_table(table)
        count += len(buffer)
    if writer is not None:
        writer.close()
    elif schema is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        empty = pa.Table.from_pylist([], schema=schema)
        pq.write_table(empty, str(path))
    return count


def _elite_parquet_schema(tfs: Iterable[str], *, include_canonical: bool) -> Any:
    import pyarrow as pa

    fields = [
        pa.field("sequence", pa.string()),
        pa.field("rank", pa.int64()),
        pa.field("norm_sum", pa.float64()),
        pa.field("min_norm", pa.float64()),
        pa.field("sum_norm", pa.float64()),
        pa.field("combined_score_final", pa.float64()),
        pa.field("chain", pa.int64()),
        pa.field("chain_1based", pa.int64()),
        pa.field("draw_idx", pa.int64()),
        pa.field("draw_in_phase", pa.int64()),
        pa.field("meta_type", pa.string()),
        pa.field("meta_source", pa.string()),
        pa.field("meta_date", pa.string()),
        pa.field("per_tf_json", pa.string()),
    ]
    if include_canonical:
        fields.append(pa.field("canonical_sequence", pa.string()))
    for tf_name in sorted(tfs):
        fields.append(pa.field(f"score_{tf_name}", pa.float64()))
        fields.append(pa.field(f"norm_{tf_name}", pa.float64()))
    return pa.schema(fields)


def _norm_map_for_elites(
    seq_arr: np.ndarray,
    per_tf_map: dict[str, float],
    *,
    scorer: Scorer,
    score_scale: str,
) -> dict[str, float]:
    if score_scale.lower() == "normalized-llr":
        return {tf: float(per_tf_map.get(tf, 0.0)) for tf in scorer.tf_names}
    return scorer.normalized_llr_map(seq_arr)


def _default_beta_ladder(chains: int, beta_min: float, beta_max: float) -> list[float]:
    if chains < 2:
        raise ValueError("PT beta ladder requires at least 2 chains")
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("PT geometric ladder requires beta_min>0 and beta_max>0")
    ladder = np.geomspace(beta_min, beta_max, chains, dtype=float)
    return [float(beta) for beta in ladder]


def _resolve_beta_ladder(pt_cfg) -> tuple[list[float], list[str]]:
    notes: list[str] = []
    ladder = pt_cfg.beta_ladder
    if isinstance(ladder, BetaLadderFixed):
        return [float(ladder.beta)], notes
    if isinstance(ladder, BetaLadderGeometric):
        if ladder.betas is not None:
            return [float(b) for b in ladder.betas], notes
        notes.append("derived PT beta ladder from beta_min/beta_max/n_temps")
        return _default_beta_ladder(int(ladder.n_temps), float(ladder.beta_min), float(ladder.beta_max)), notes
    raise ValueError("Unsupported beta_ladder configuration")


def _resolve_optimizer_kind(sample_cfg: SampleConfig) -> str:
    kind = sample_cfg.optimizer.name
    if kind == "auto":
        raise ValueError("sample.optimizer.name must be 'gibbs' or 'pt' for a concrete run.")
    return kind


def _effective_chain_count(sample_cfg: SampleConfig, *, kind: str) -> int:
    if kind == "gibbs":
        return sample_cfg.budget.restarts
    if kind == "pt":
        ladder, _ = _resolve_beta_ladder(sample_cfg.optimizers.pt)
        return len(ladder)
    raise ValueError(f"Unknown optimizer kind '{kind}'")


def _boost_cooling(cooling: Any, factor: float) -> tuple[Any, list[str]]:
    if factor <= 1:
        return cooling, []
    notes = [f"boosted cooling by x{factor:g} for stabilization"]
    if isinstance(cooling, CoolingLinear):
        beta = [float(cooling.beta[0]) * factor, float(cooling.beta[1]) * factor]
        return CoolingLinear(beta=beta), notes
    if isinstance(cooling, CoolingFixed):
        return CoolingFixed(beta=float(cooling.beta) * factor), notes
    if isinstance(cooling, CoolingPiecewise):
        stages = [{"sweeps": stage.sweeps, "beta": float(stage.beta) * factor} for stage in cooling.stages]
        return CoolingPiecewise(stages=stages), notes
    return cooling, notes


def _pilot_budget_levels(base_cfg: SampleConfig, auto_cfg: AutoOptConfig) -> list[int]:
    base_total = base_cfg.budget.tune + base_cfg.budget.draws
    if base_total < 4:
        raise ValueError("auto_opt requires sample.budget.tune+draws >= 4")
    budgets = sorted({min(level, base_total) for level in auto_cfg.budget_levels})
    budgets = [b for b in budgets if b >= 4]
    if not budgets:
        raise ValueError("auto_opt.budget_levels produced no valid pilot budgets")
    return budgets


def _budget_to_tune_draws(base_cfg: SampleConfig, total_sweeps: int) -> tuple[int, int]:
    base_total = base_cfg.budget.tune + base_cfg.budget.draws
    ratio = base_cfg.budget.tune / base_total if base_total > 0 else 0.0
    tune = int(round(total_sweeps * ratio))
    draws = total_sweeps - tune
    if draws < 4:
        draws = 4
        tune = max(0, total_sweeps - draws)
    return tune, draws


def _build_pilot_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    auto_cfg: AutoOptConfig,
    total_sweeps: int,
    cooling_boost: float = 1.0,
    length_override: int | None = None,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    pilot = base_cfg.model_copy(deep=True)
    tune, draws = _budget_to_tune_draws(base_cfg, total_sweeps)
    pilot.budget.tune = tune
    pilot.budget.draws = draws
    if base_cfg.budget.tune != tune or base_cfg.budget.draws != draws:
        notes.append("overrode tune/draws for pilot budget")

    pilot.output.trace.save = True
    if not base_cfg.output.trace.save:
        notes.append("enabled output.trace.save for pilot diagnostics")
    pilot.output.save_sequences = True
    if not base_cfg.output.save_sequences:
        notes.append("enabled output.save_sequences for pilot diagnostics")
    pilot.output.trace.include_tune = False
    if base_cfg.output.trace.include_tune:
        notes.append("disabled output.trace.include_tune for pilots")
    pilot.output.live_metrics = False
    if base_cfg.output.live_metrics:
        notes.append("disabled output.live_metrics for pilots")
    pilot.ui.progress_bar = False
    pilot.ui.progress_every = 0
    if not auto_cfg.allow_trim_polish_in_pilots:
        if pilot.output.trim.enabled:
            pilot.output.trim.enabled = False
            notes.append("disabled output.trim.enabled for pilots")
        if pilot.output.polish.enabled:
            pilot.output.polish.enabled = False
            notes.append("disabled output.polish.enabled for pilots")

    if length_override is not None and pilot.init.length != length_override:
        pilot.init.length = int(length_override)
        notes.append(f"overrode init.length={pilot.init.length} for auto-opt length")

    pilot.optimizer.name = kind
    if kind == "pt" and pilot.budget.restarts != 1:
        pilot.budget.restarts = 1
        notes.append("forced budget.restarts=1 for PT pilots")

    if kind == "gibbs" and cooling_boost > 1:
        schedule = pilot.optimizers.gibbs.beta_schedule
        boosted, boost_notes = _boost_cooling(schedule, cooling_boost)
        pilot.optimizers.gibbs.beta_schedule = boosted
        notes.extend(boost_notes)

    return pilot, notes


def _build_final_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    length_override: int | None = None,
    cooling_boost: float = 1.0,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    final_cfg = base_cfg.model_copy(deep=True)
    if length_override is not None and final_cfg.init.length != length_override:
        final_cfg.init.length = int(length_override)
        notes.append(f"set init.length={final_cfg.init.length} from auto-opt selection")
    final_cfg.optimizer.name = kind
    if kind == "pt" and final_cfg.budget.restarts != 1:
        final_cfg.budget.restarts = 1
        notes.append("forced budget.restarts=1 for PT final run")
    if kind == "gibbs" and cooling_boost > 1:
        schedule = final_cfg.optimizers.gibbs.beta_schedule
        boosted, boost_notes = _boost_cooling(schedule, cooling_boost)
        final_cfg.optimizers.gibbs.beta_schedule = boosted
        notes.extend(boost_notes)
    return final_cfg, notes


def _assess_candidate_quality(
    candidate: AutoOptCandidate,
    auto_cfg: AutoOptConfig,
    *,
    mode: str,
) -> list[str]:
    _ = auto_cfg
    _ = mode
    notes: list[str] = []
    if candidate.status == "fail":
        candidate.quality = "fail"
        return notes
    candidate.quality = "ok"
    return notes


def _run_auto_optimize_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    auto_cfg: AutoOptConfig,
) -> Path:
    logger.info("Auto-optimize enabled: running pilot sweeps (gibbs + pt).")
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    pwms = _load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)
    lengths = _candidate_lengths(sample_cfg, auto_cfg, pwms)
    budgets = _pilot_budget_levels(sample_cfg, auto_cfg)
    logger.info("Auto-opt length candidates: %s", ", ".join(str(L) for L in lengths))
    logger.info("Auto-opt budget levels: %s", ", ".join(str(b) for b in budgets))
    logger.info("Auto-opt keep_pilots: %s", auto_cfg.keep_pilots)

    length_cfg = auto_cfg.length
    candidate_specs = [(kind, length) for length in lengths for kind in ("gibbs", "pt")]
    seed_rng = np.random.default_rng(sample_cfg.rng.seed)

    def _run_candidate(
        *,
        kind: str,
        length: int,
        budget: int,
        label: str,
        cooling_boost: float,
        init_seeds: list[np.ndarray] | None = None,
    ) -> AutoOptCandidate:
        runs: list[AutoOptCandidate] = []
        for rep in range(auto_cfg.replicates):
            pilot_cfg, notes = _build_pilot_sample_cfg(
                sample_cfg,
                kind=kind,
                auto_cfg=auto_cfg,
                total_sweeps=budget,
                cooling_boost=cooling_boost,
                length_override=length,
            )
            seed_label = f"{label}_R{rep + 1}"
            if sample_cfg.rng.deterministic:
                seed = _stable_pilot_seed(pilot_cfg=pilot_cfg, tfs=tfs, lockmap=lockmap, label=seed_label, kind=kind)
            else:
                seed = int(seed_rng.integers(0, 2**31 - 1))
            pilot_cfg.rng.seed = seed
            notes.append(f"set pilot seed={seed}")
            if notes:
                logger.debug(
                    "Auto-opt %s %s (L=%d B=%d R=%d) overrides: %s",
                    label,
                    kind,
                    pilot_cfg.init.length,
                    budget,
                    rep + 1,
                    "; ".join(notes),
                )
            pilot_meta = {
                "mode": "pilot",
                "attempt": label,
                "candidate": kind,
                "length": pilot_cfg.init.length,
                "seed": pilot_cfg.rng.seed,
                "draws": pilot_cfg.budget.draws,
                "tune": pilot_cfg.budget.tune,
                "restarts": pilot_cfg.budget.restarts,
                "budget": budget,
                "replicate": rep + 1,
                "cooling_boost": cooling_boost,
            }
            pilot_run_dir = _run_sample_for_set(
                cfg,
                config_path,
                set_index=set_index,
                set_count=set_count,
                include_set_index=include_set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=pilot_cfg,
                stage="auto_opt",
                run_kind=f"auto_opt_{label}_{kind}_L{pilot_cfg.init.length}_B{budget}_R{rep + 1}",
                auto_opt_meta=pilot_meta,
                init_seeds=init_seeds,
            )
            try:
                candidate = _evaluate_pilot_run(
                    pilot_run_dir,
                    kind,
                    budget=budget,
                    mode=sample_cfg.mode,
                    scorecard_top_k=auto_cfg.policy.scorecard.top_k,
                    cooling_boost=cooling_boost,
                )
                if candidate.length is None:
                    candidate.length = pilot_cfg.init.length
            except Exception as exc:
                logger.warning("Auto-opt %s %s failed: %s", label, kind, exc)
                candidate = AutoOptCandidate(
                    kind=kind,
                    length=pilot_cfg.init.length,
                    budget=budget,
                    cooling_boost=cooling_boost,
                    run_dir=pilot_run_dir,
                    run_dirs=[pilot_run_dir],
                    best_score=None,
                    top_k_median_final=None,
                    best_score_final=None,
                    top_k_ci_low=None,
                    top_k_ci_high=None,
                    rhat=None,
                    ess=None,
                    unique_fraction=None,
                    balance_median=None,
                    diversity=None,
                    improvement=None,
                    acceptance_b=None,
                    acceptance_m=None,
                    acceptance_mh=None,
                    swap_rate=None,
                    status="fail",
                    quality="fail",
                    warnings=[str(exc)],
                    diagnostics={},
                )
            runs.append(candidate)
        return _aggregate_candidate_runs(runs, budget=budget, scorecard_top_k=auto_cfg.policy.scorecard.top_k)

    def _log_candidate(candidate: AutoOptCandidate, *, label: str) -> None:
        diag_status = "n/a"
        if isinstance(candidate.diagnostics, dict):
            diag_status = candidate.diagnostics.get("status") or "n/a"
        logger.debug(
            (
                "Auto-opt %s %s L=%s B=%s boost=%s: scorecard=%s diagnostics=%s top_k_median=%s best=%s "
                "balance=%s diversity=%s rhat=%s ess=%s unique_fraction=%s"
            ),
            label,
            candidate.kind,
            candidate.length if candidate.length is not None else "n/a",
            candidate.budget if candidate.budget is not None else "n/a",
            f"{candidate.cooling_boost:g}",
            candidate.quality,
            diag_status,
            f"{candidate.top_k_median_final:.3f}" if candidate.top_k_median_final is not None else "n/a",
            f"{candidate.best_score_final:.3f}" if candidate.best_score_final is not None else "n/a",
            f"{candidate.balance_median:.3f}" if candidate.balance_median is not None else "n/a",
            f"{candidate.diversity:.3f}" if candidate.diversity is not None else "n/a",
            f"{candidate.rhat:.3f}" if candidate.rhat is not None else "n/a",
            f"{candidate.ess:.1f}" if candidate.ess is not None else "n/a",
            f"{candidate.unique_fraction:.2f}" if candidate.unique_fraction is not None else "n/a",
        )
        if candidate.warnings:
            logger.warning("Auto-opt %s %s warnings: %s", label, candidate.kind, "; ".join(candidate.warnings))

    all_candidates: list[AutoOptCandidate] = []
    final_level_candidates: list[AutoOptCandidate] = []
    length_summary_rows: list[dict[str, object]] = []

    def _scaled_budgets(base_budgets: list[int], *, scale: float) -> list[int]:
        base_total = sample_cfg.budget.tune + sample_cfg.budget.draws
        return [min(base_total, max(4, int(round(b * scale)))) for b in base_budgets]

    if length_cfg.mode == "ladder":
        previous_best_run: Path | None = None
        final_level_candidates_all: list[AutoOptCandidate] = []
        for length_idx, length in enumerate(lengths):
            length_budgets = budgets
            if length_idx > 0:
                length_budgets = _scaled_budgets(budgets, scale=length_cfg.ladder_budget_scale)
            init_seeds: list[np.ndarray] | None = None
            if length_cfg.warm_start and previous_best_run is not None:
                init_seeds = _warm_start_seeds_from_sequences(
                    previous_best_run,
                    target_length=length,
                    rng=seed_rng,
                    pad_with=sample_cfg.init.pad_with,
                    max_seeds=sample_cfg.elites.k,
                )
                if not init_seeds:
                    init_seeds = _warm_start_seeds_from_elites(
                        previous_best_run,
                        target_length=length,
                        rng=seed_rng,
                        pad_with=sample_cfg.init.pad_with,
                        max_seeds=sample_cfg.elites.k,
                    )
                if not init_seeds:
                    logger.warning("Warm-start: no usable seeds for length %d; falling back to fresh init.", length)
            current_specs = [("gibbs", length), ("pt", length)]
            level_candidates: list[AutoOptCandidate] = []
            length_final_candidates: list[AutoOptCandidate] = []
            start_time = time.perf_counter()
            for idx, budget in enumerate(length_budgets):
                level_candidates = []
                for kind, _ in current_specs:
                    label = f"pilot_{idx + 1}_L{length}_B{budget}"
                    candidate = _run_candidate(
                        kind=kind,
                        length=length,
                        budget=budget,
                        label=label,
                        cooling_boost=1.0,
                        init_seeds=init_seeds,
                    )
                    candidate_notes = _assess_candidate_quality(candidate, auto_cfg, mode=sample_cfg.mode)
                    if candidate_notes:
                        candidate.warnings.extend(candidate_notes)
                    _log_candidate(candidate, label=label)
                    level_candidates.append(candidate)
                    all_candidates.append(candidate)
                confident, _, _ = _confidence_from_candidates(level_candidates, auto_cfg)
                if confident:
                    length_final_candidates = level_candidates
                    if idx < len(length_budgets) - 1:
                        logger.info(
                            "Auto-opt length %s: confident at budget %s; skipping remaining budget levels.",
                            length,
                            budget,
                        )
                    break
                if idx < len(length_budgets) - 1:
                    ranked = _rank_auto_opt_candidates(level_candidates, auto_cfg)
                    current_specs = [(c.kind, c.length) for c in ranked]
                else:
                    length_final_candidates = level_candidates
            runtime_sec = time.perf_counter() - start_time
            if not length_final_candidates:
                raise ValueError("Auto-optimize failed: no final-level candidates were evaluated.")
            ranked_length = _rank_auto_opt_candidates(length_final_candidates, auto_cfg)
            best_for_length = ranked_length[0]
            previous_best_run = best_for_length.run_dir
            length_summary_rows.append(
                {
                    "length": length,
                    "best_score": best_for_length.top_k_median_final,
                    "balance": best_for_length.balance_median,
                    "diversity": best_for_length.diversity,
                    "unique_fraction": best_for_length.unique_fraction,
                    "runtime_sec": round(runtime_sec, 3),
                }
            )
            final_level_candidates_all.extend(length_final_candidates)
        final_level_candidates = final_level_candidates_all
    else:
        current_specs = list(candidate_specs)
        for idx, budget in enumerate(budgets):
            level_candidates = []
            for kind, length in current_specs:
                label = f"pilot_{idx + 1}_L{length}_B{budget}"
                candidate = _run_candidate(
                    kind=kind,
                    length=length,
                    budget=budget,
                    label=label,
                    cooling_boost=1.0,
                )
                candidate_notes = _assess_candidate_quality(candidate, auto_cfg, mode=sample_cfg.mode)
                if candidate_notes:
                    candidate.warnings.extend(candidate_notes)
                _log_candidate(candidate, label=label)
                level_candidates.append(candidate)
                all_candidates.append(candidate)

            confident, _, _ = _confidence_from_candidates(level_candidates, auto_cfg)
            if confident:
                final_level_candidates = level_candidates
                if idx < len(budgets) - 1:
                    logger.info(
                        "Auto-opt: confident at budget %s; skipping remaining budget levels.",
                        budget,
                    )
                break
            if idx < len(budgets) - 1:
                ranked = _rank_auto_opt_candidates(level_candidates, auto_cfg)
                current_specs = [(c.kind, c.length) for c in ranked]
            else:
                final_level_candidates = level_candidates

    if not final_level_candidates:
        raise ValueError("Auto-optimize failed: no final-level candidates were evaluated.")

    decision_notes: list[str] = []
    viable, _ = _validate_auto_opt_candidates(
        final_level_candidates,
        allow_warn=auto_cfg.policy.allow_warn,
    )

    confident, _best_candidate, _second_candidate = _confidence_from_candidates(viable, auto_cfg)
    selection_confident = bool(confident)
    if selection_confident:
        decision_notes.append("confidence=high")
    else:
        decision_notes.append("confidence=low")
        if auto_cfg.policy.allow_warn:
            logger.warning(
                "Auto-optimize: could not confidently separate top candidates at max budget; "
                "selecting best available candidate."
            )
        else:
            raise ValueError(
                "Auto-optimize failed: could not confidently separate top candidates at the maximum "
                "configured budgets/replicates. Increase auto_opt.budget_levels and/or auto_opt.replicates."
            )

    winner = _select_auto_opt_candidate(viable, auto_cfg, allow_fail=False)

    if winner.quality == "ok":
        logger.info(
            (
                "Auto-optimize selected %s L=%s (top_k_median=%s, best=%s, "
                "balance=%s, rhat=%s, ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            f"{winner.top_k_median_final:.3f}" if winner.top_k_median_final is not None else "n/a",
            f"{winner.best_score_final:.3f}" if winner.best_score_final is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    else:
        logger.warning(
            (
                "Auto-optimize selected %s L=%s (quality=%s top_k_median=%s, best=%s, balance=%s, rhat=%s, "
                "ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            winner.quality,
            f"{winner.top_k_median_final:.3f}" if winner.top_k_median_final is not None else "n/a",
            f"{winner.best_score_final:.3f}" if winner.best_score_final is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    if winner.cooling_boost > 1:
        decision_notes.append(f"cooling_boost={winner.cooling_boost:g}")
    if decision_notes:
        logger.warning("Auto-opt notes: %s", ", ".join(decision_notes))

    final_cfg, notes = _build_final_sample_cfg(
        sample_cfg,
        kind=winner.kind,
        length_override=winner.length,
        cooling_boost=winner.cooling_boost,
    )
    if notes:
        logger.info("Auto-opt final overrides: %s", "; ".join(notes))
    length_aware = auto_cfg.length.enabled and len({c.length for c in all_candidates if c.length is not None}) > 1
    ranking_label = (
        "top_k_median -> best_score_final -> balance -> diversity -> improvement"
        if length_aware
        else "top_k_median -> best_score_final -> balance -> diversity -> improvement"
    )
    logger.info("Auto-opt selection ranking: %s.", ranking_label)
    logger.info("Auto-opt final config: %s", _format_auto_opt_config_summary(final_cfg))
    pilot_root = all_candidates[0].run_dir.parent if all_candidates else None
    best_marker_path = _auto_opt_best_marker_path(pilot_root, run_group=run_group) if pilot_root is not None else None
    length_ladder_path: Path | None = None
    if length_cfg.mode == "ladder" and length_summary_rows and pilot_root is not None:
        length_ladder_path = _write_length_ladder_table(length_summary_rows, pilot_root=pilot_root)
        logger.info(
            "Length ladder summary -> %s",
            _format_run_path(length_ladder_path, base=config_path.parent),
        )
    if pilot_root is not None:
        logger.info(
            "Auto-opt details: pilot manifests under %s; best marker -> %s; final config in config_used.yaml.",
            _format_run_path(pilot_root, base=config_path.parent),
            _format_run_path(best_marker_path, base=config_path.parent) if best_marker_path else "n/a",
        )

    decision_payload = {
        "mode": "final",
        "selected": winner.kind,
        "selected_length": winner.length,
        "selection_quality": winner.quality,
        "selection_confident": selection_confident,
        "selection_confidence": "high" if selection_confident else "low",
        "notes": decision_notes,
        "config_summary": _format_auto_opt_config_summary(final_cfg),
        "selected_config": _selected_config_payload(final_cfg),
        "selection_metrics": {
            "top_k_median_final": winner.top_k_median_final,
            "best_score_final": winner.best_score_final,
            "top_k_ci_low": winner.top_k_ci_low,
            "top_k_ci_high": winner.top_k_ci_high,
        },
        "length_config": auto_cfg.length.model_dump(),
        "length_ladder_table": str(length_ladder_path) if length_ladder_path is not None else None,
        "auto_opt_config": {
            "budget_levels": auto_cfg.budget_levels,
            "replicates": auto_cfg.replicates,
            "keep_pilots": auto_cfg.keep_pilots,
            "prefer_simpler_if_close": auto_cfg.prefer_simpler_if_close,
            "tolerance": auto_cfg.tolerance.model_dump(),
            "scorecard_top_k": auto_cfg.policy.scorecard.top_k,
        },
        "pilot_root": str(pilot_root) if pilot_root is not None else None,
        "best_marker": str(best_marker_path) if best_marker_path is not None else None,
        "candidates": [
            {
                "kind": candidate.kind,
                "length": candidate.length,
                "budget": candidate.budget,
                "cooling_boost": candidate.cooling_boost,
                "runs": [path.name for path in candidate.run_dirs],
                "status": candidate.status,
                "quality": candidate.quality,
                "best_score": candidate.best_score,
                "top_k_median_final": candidate.top_k_median_final,
                "best_score_final": candidate.best_score_final,
                "top_k_ci_low": candidate.top_k_ci_low,
                "top_k_ci_high": candidate.top_k_ci_high,
                "balance_median": candidate.balance_median,
                "diversity": candidate.diversity,
                "improvement": candidate.improvement,
                "acceptance_b": candidate.acceptance_b,
                "acceptance_m": candidate.acceptance_m,
                "acceptance_mh": candidate.acceptance_mh,
                "swap_rate": candidate.swap_rate,
                "rhat": candidate.rhat,
                "ess": candidate.ess,
                "unique_fraction": candidate.unique_fraction,
            }
            for candidate in all_candidates
        ],
    }
    final_run_dir = _run_sample_for_set(
        cfg,
        config_path,
        set_index=set_index,
        set_count=set_count,
        include_set_index=include_set_index,
        tfs=tfs,
        lockmap=lockmap,
        sample_cfg=final_cfg,
        stage="sample",
        run_kind="auto_opt_final",
        auto_opt_meta=decision_payload,
    )
    if pilot_root is not None:
        marker_payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "selected_candidate": {
                "kind": winner.kind,
                "length": winner.length,
                "cooling_boost": winner.cooling_boost,
                "pilot_run": winner.run_dir.name,
            },
            "final_sample_run": final_run_dir.name,
            "config_summary": _format_auto_opt_config_summary(final_cfg),
        }
        marker_path = _write_auto_opt_best_marker(pilot_root, marker_payload, run_group=run_group)
        logger.info(
            "Auto-opt best marker updated -> %s",
            _format_run_path(marker_path, base=config_path.parent),
        )
        removed = _prune_auto_opt_runs(
            config_path=config_path,
            pilot_root=pilot_root,
            candidates=all_candidates,
            winner=winner,
            keep_mode=auto_cfg.keep_pilots,
            catalog_root=cfg.motif_store.catalog_root,
        )
        if removed:
            logger.info(
                "Auto-opt pruned %d pilot run(s) (keep_pilots=%s).",
                len(removed),
                auto_cfg.keep_pilots,
            )
    logger.info(
        "Auto-opt final config saved -> %s",
        _format_run_path(config_used_path(final_run_dir), base=config_path.parent),
    )
    return final_run_dir


def _evaluate_pilot_run(
    run_dir: Path,
    kind: str,
    *,
    scorecard_top_k: int | None = None,
    budget: int | None = None,
    mode: str | None = None,
    cooling_boost: float = 1.0,
) -> AutoOptCandidate:
    manifest = load_manifest(run_dir)
    length = manifest.get("sequence_length")
    trace_file = trace_path(run_dir)
    seq_path = sequences_path(run_dir)
    if not trace_file.exists():
        raise FileNotFoundError(f"Missing trace.nc in {run_dir}")
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequences.parquet in {run_dir}")
    elite_path = find_elites_parquet(run_dir)
    seq_df = read_parquet(seq_path)
    elites_df = read_parquet(elite_path)
    tf_names = [m["tf_name"] for m in manifest.get("motifs", [])]
    top_k = scorecard_top_k
    if top_k is None:
        top_k = int(manifest.get("top_k") or 10)
    draw_scores = _draw_scores_from_sequences(seq_df)
    top_k_median_final = _top_k_median_from_scores(draw_scores, top_k)
    best_score_final = float(np.max(draw_scores)) if draw_scores.size else None
    best_score = top_k_median_final
    bootstrap_ci: tuple[float, float] | None = None
    if draw_scores.size:
        seed = _bootstrap_seed(manifest=manifest, run_dir=run_dir, kind=kind)
        rng = np.random.default_rng(seed)
        bootstrap_ci = _bootstrap_top_k_ci(draw_scores, k=top_k, rng=rng)

    import arviz as az

    trace_idata = az.from_netcdf(trace_file)
    sample_meta = None
    if scorecard_top_k is not None:
        sample_meta = {"top_k": scorecard_top_k}
    elif "top_k" in manifest:
        sample_meta = {"top_k": manifest.get("top_k")}
    if sample_meta is None:
        sample_meta = {}
    if mode is not None:
        sample_meta["mode"] = mode
    sample_meta["optimizer_kind"] = kind
    elites_cfg = manifest.get("elites")
    if isinstance(elites_cfg, dict):
        sample_meta["dsdna_canonicalize"] = elites_cfg.get("dsDNA_canonicalize", False)
        sample_meta["dsdna_hamming"] = elites_cfg.get("dsDNA_hamming", False)
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=trace_idata,
        sequences_df=seq_df,
        elites_df=elites_df,
        tf_names=tf_names,
        optimizer=manifest.get("optimizer", {}),
        optimizer_stats=manifest.get("optimizer_stats", {}),
        sample_meta=sample_meta,
    )
    metrics = diagnostics.get("metrics", {}) if isinstance(diagnostics, dict) else {}
    trace_metrics = metrics.get("trace", {})
    seq_metrics = metrics.get("sequences", {})
    elites_metrics = metrics.get("elites", {})
    optimizer_metrics = metrics.get("optimizer", {})
    rhat = trace_metrics.get("rhat")
    ess = trace_metrics.get("ess")
    unique_fraction = seq_metrics.get("unique_fraction")
    balance_median = elites_metrics.get("normalized_min_median") or elites_metrics.get("balance_index_median")
    diversity = elites_metrics.get("diversity_hamming")
    improvement = trace_metrics.get("best_so_far_slope") or trace_metrics.get("score_delta")
    acceptance_b = None
    acceptance_m = None
    acceptance_mh = None
    if isinstance(optimizer_metrics, dict):
        acc = optimizer_metrics.get("acceptance_rate") or {}
        if isinstance(acc, dict):
            acceptance_b = acc.get("B")
            acceptance_m = acc.get("M")
        acceptance_mh = optimizer_metrics.get("acceptance_rate_mh")
    swap_rate = optimizer_metrics.get("swap_acceptance_rate") if isinstance(optimizer_metrics, dict) else None
    warnings = diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []
    status = "ok"
    if isinstance(diagnostics, dict):
        diag_status = diagnostics.get("status", "ok")
        if diag_status and diag_status != "ok":
            warnings = list(warnings)
            warnings.append(f"Diagnostics status={diag_status}; see diagnostics.json for details.")

    if draw_scores.size == 0:
        warnings = list(warnings)
        warnings.append("Missing pilot metric: combined_score_final draw-phase values.")
        status = "fail"

    unique_draws: int | None = None
    if "sequence" in seq_df.columns:
        df_unique = seq_df
        if "phase" in df_unique.columns:
            df_unique = df_unique[df_unique["phase"] == "draw"]
        try:
            unique_draws = int(df_unique["sequence"].nunique())
        except Exception:
            unique_draws = None

    if kind == "gibbs" and status != "fail":
        moved = unique_draws is not None and unique_draws > 1
        accepted = acceptance_mh is not None and acceptance_mh > 0
        if not accepted and not moved:
            warnings = list(warnings)
            warnings.append("Gibbs pilot shows no accepted MH moves and no unique sequences.")
            status = "fail"

    if kind == "pt" and isinstance(optimizer_metrics, dict):
        swap_attempts = optimizer_metrics.get("swap_attempts")
        swap_prob = None
        optimizer_cfg = manifest.get("optimizer", {})
        if isinstance(optimizer_cfg, dict):
            pt_cfg = optimizer_cfg.get("pt", {})
            if isinstance(pt_cfg, dict):
                swap_prob = pt_cfg.get("swap_prob")
        if swap_prob and swap_attempts == 0:
            warnings = list(warnings)
            warnings.append("PT pilot recorded swap_prob>0 but swap_attempts=0.")

    return AutoOptCandidate(
        kind=kind,
        length=int(length) if isinstance(length, (int, float)) else None,
        budget=budget,
        cooling_boost=cooling_boost,
        run_dir=run_dir,
        run_dirs=[run_dir],
        best_score=best_score,
        top_k_median_final=top_k_median_final,
        best_score_final=best_score_final,
        top_k_ci_low=bootstrap_ci[0] if bootstrap_ci else None,
        top_k_ci_high=bootstrap_ci[1] if bootstrap_ci else None,
        rhat=rhat,
        ess=ess,
        unique_fraction=unique_fraction,
        balance_median=balance_median,
        diversity=diversity,
        improvement=improvement,
        acceptance_b=acceptance_b,
        acceptance_m=acceptance_m,
        acceptance_mh=acceptance_mh,
        swap_rate=swap_rate,
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
    )


def _aggregate_candidate_runs(
    runs: list[AutoOptCandidate],
    *,
    budget: int,
    scorecard_top_k: int,
) -> AutoOptCandidate:
    if not runs:
        raise ValueError("Auto-opt aggregation requires at least one pilot run")

    def _median(values: list[float | None]) -> float | None:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return float(np.median(filtered))

    kind = runs[0].kind
    length = runs[0].length
    run_dirs = [run.run_dir for run in runs]
    status = "ok"
    if all(run.status == "fail" for run in runs):
        status = "fail"

    warnings: list[str] = []
    for run in runs:
        warnings.extend(run.warnings)
    if status != "ok":
        warnings.append("One or more pilot replicates failed")

    diagnostics = {
        "replicates": [run.diagnostics for run in runs],
        "run_dirs": [str(path) for path in run_dirs],
    }
    best_run = max(
        runs,
        key=lambda run: (
            float(run.top_k_median_final) if run.top_k_median_final is not None else float("-inf"),
            float(run.best_score_final) if run.best_score_final is not None else float("-inf"),
            float(run.balance_median) if run.balance_median is not None else float("-inf"),
        ),
    )

    pooled_scores: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    for run_dir in run_dirs:
        seq_path = sequences_path(run_dir)
        if not seq_path.exists():
            warnings.append(f"Missing sequences.parquet for replicate {run_dir.name}")
            continue
        try:
            seq_df = read_parquet(seq_path)
        except Exception as exc:
            warnings.append(f"Failed to read sequences.parquet for {run_dir.name}: {exc}")
            continue
        scores = _draw_scores_from_sequences(seq_df)
        if scores.size:
            pooled_scores.append(scores)
        try:
            manifest = load_manifest(run_dir)
        except Exception as exc:
            warnings.append(f"Failed to read manifest for {run_dir.name}: {exc}")
        else:
            manifests.append(manifest)

    combined_scores = np.concatenate(pooled_scores) if pooled_scores else np.array([], dtype=float)
    if combined_scores.size:
        top_k_median_final = _top_k_median_from_scores(combined_scores, scorecard_top_k)
        best_score_final = float(np.max(combined_scores))
        best_score = top_k_median_final
    else:
        top_k_median_final = _median([run.top_k_median_final for run in runs])
        best_score_final = max(
            [float(run.best_score_final) for run in runs if run.best_score_final is not None],
            default=None,
        )
        best_score = top_k_median_final
    bootstrap_ci: tuple[float, float] | None = None
    if combined_scores.size:
        seed = _pooled_bootstrap_seed(manifests=manifests, kind=kind, length=length, budget=budget)
        if seed is None:
            payload = {"kind": kind, "length": length, "budget": budget, "replicate_count": len(run_dirs)}
            seed = _bootstrap_seed_payload(payload)
        rng = np.random.default_rng(seed)
        bootstrap_ci = _bootstrap_top_k_ci(combined_scores, k=scorecard_top_k, rng=rng)
    else:
        lows = [run.top_k_ci_low for run in runs if run.top_k_ci_low is not None]
        highs = [run.top_k_ci_high for run in runs if run.top_k_ci_high is not None]
        if lows and highs:
            bootstrap_ci = (float(min(lows)), float(max(highs)))

    return AutoOptCandidate(
        kind=kind,
        length=length,
        budget=budget,
        cooling_boost=float(np.median([run.cooling_boost for run in runs])),
        run_dir=best_run.run_dir,
        run_dirs=run_dirs,
        best_score=best_score,
        top_k_median_final=top_k_median_final,
        best_score_final=best_score_final,
        top_k_ci_low=bootstrap_ci[0] if bootstrap_ci else None,
        top_k_ci_high=bootstrap_ci[1] if bootstrap_ci else None,
        rhat=_median([run.rhat for run in runs]),
        ess=_median([run.ess for run in runs]),
        unique_fraction=_median([run.unique_fraction for run in runs]),
        balance_median=_median([run.balance_median for run in runs]),
        diversity=_median([run.diversity for run in runs]),
        improvement=_median([run.improvement for run in runs]),
        acceptance_b=_median([run.acceptance_b for run in runs]),
        acceptance_m=_median([run.acceptance_m for run in runs]),
        acceptance_mh=_median([run.acceptance_mh for run in runs]),
        swap_rate=_median([run.swap_rate for run in runs]),
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _rank_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
    auto_cfg: AutoOptConfig,
) -> list[AutoOptCandidate]:
    ranked: list[tuple[tuple[float, float, float, float, float, float, float], AutoOptCandidate]] = []
    status_rank = {"ok": 2, "warn": 1, "fail": 0}
    for candidate in candidates:
        score = candidate.top_k_median_final if candidate.top_k_median_final is not None else candidate.best_score
        if score is None:
            score = float("-inf")
        secondary = (
            candidate.best_score_final
            if candidate.best_score_final is not None
            else (candidate.best_score if candidate.best_score is not None else float("-inf"))
        )
        balance = candidate.balance_median if candidate.balance_median is not None else float("-inf")
        diversity = candidate.diversity if candidate.diversity is not None else float("-inf")
        improvement = candidate.improvement if candidate.improvement is not None else float("-inf")
        unique_fraction = candidate.unique_fraction if candidate.unique_fraction is not None else float("-inf")
        rank = (
            float(score),
            float(secondary),
            float(balance),
            float(diversity),
            float(improvement),
            float(unique_fraction),
            float(status_rank.get(candidate.quality, status_rank.get(candidate.status, 0))),
        )
        ranked.append((rank, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def _confidence_from_candidates(
    candidates: list[AutoOptCandidate],
    auto_cfg: AutoOptConfig,
) -> tuple[bool, AutoOptCandidate | None, AutoOptCandidate | None]:
    if not candidates:
        return False, None, None
    ranked = _rank_auto_opt_candidates(candidates, auto_cfg)
    best = ranked[0]
    if len(ranked) == 1:
        return True, best, None
    second = ranked[1]
    if best.top_k_ci_low is None or second.top_k_ci_high is None:
        return False, best, second
    return best.top_k_ci_low > second.top_k_ci_high, best, second


def _validate_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
    *,
    allow_warn: bool,
) -> tuple[list[AutoOptCandidate], bool]:
    _ = allow_warn
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates.")
    viable = [c for c in candidates if c.status != "fail"]
    if not viable:
        raise ValueError(
            "Auto-optimize failed: all pilot candidates failed catastrophic checks "
            "(missing scores or no movement). Re-run with larger budgets or fix the model inputs."
        )
    return viable, False


def _select_auto_opt_candidate(
    candidates: list[AutoOptCandidate],
    auto_cfg: AutoOptConfig,
    *,
    allow_fail: bool = False,
) -> AutoOptCandidate:
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates")

    ok_candidates = [c for c in candidates if c.status != "fail"]
    if not ok_candidates and allow_fail:
        ok_candidates = list(candidates)
    if auto_cfg.length.enabled and auto_cfg.length.prefer_shortest:
        lengths = [c.length for c in ok_candidates if c.length is not None]
        if lengths:
            min_len = min(lengths)
            ok_candidates = [c for c in ok_candidates if c.length == min_len]

    ranked = _rank_auto_opt_candidates(ok_candidates, auto_cfg)
    winner = ranked[0]
    if winner.status == "fail" and not allow_fail:
        raise ValueError("Auto-optimize failed: all pilot candidates reported missing diagnostics.")

    if auto_cfg.prefer_simpler_if_close and winner.kind == "pt":
        gibbs = [c for c in ranked if c.kind == "gibbs"]
        if gibbs and winner.top_k_median_final is not None and gibbs[0].top_k_median_final is not None:
            if gibbs[0].top_k_median_final >= winner.top_k_median_final - auto_cfg.tolerance.score:
                return gibbs[0]
    return winner


def _format_run_path(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _auto_opt_best_marker_path(pilot_root: Path, *, run_group: str) -> Path:
    suffix = run_group or "best"
    return pilot_root / f"best_{suffix}.json"


def _write_auto_opt_best_marker(pilot_root: Path, payload: dict[str, object], *, run_group: str) -> Path:
    path = _auto_opt_best_marker_path(pilot_root, run_group=run_group)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _prune_auto_opt_runs(
    *,
    config_path: Path,
    pilot_root: Path,
    candidates: list[AutoOptCandidate],
    winner: AutoOptCandidate,
    keep_mode: str,
    catalog_root: Path | str | None,
) -> list[str]:
    if keep_mode == "all":
        return []
    keep_dirs: set[Path] = {winner.run_dir}
    if keep_mode == "ok":
        for candidate in candidates:
            if candidate.quality == "ok":
                keep_dirs.update(candidate.run_dirs)
    keep_names: set[str] = {path.name for path in keep_dirs}
    remove_dirs: list[Path] = []
    for candidate in candidates:
        for run_dir in candidate.run_dirs:
            if run_dir.name not in keep_names:
                remove_dirs.append(run_dir)
    removed_names: list[str] = []
    for run_dir in remove_dirs:
        if not run_dir.exists():
            continue
        if pilot_root not in run_dir.parents:
            logger.warning("Auto-opt prune skipped unexpected path: %s", run_dir)
            continue
        shutil.rmtree(run_dir)
        removed_names.append(run_dir.name)
    if removed_names:
        drop_run_index_entries(config_path, removed_names, catalog_root=catalog_root)
    return removed_names


def _format_cooling_summary(cooling: object) -> str:
    if isinstance(cooling, CoolingLinear):
        beta0, beta1 = cooling.beta
        return f"linear({beta0:g}->{beta1:g})"
    if isinstance(cooling, CoolingFixed):
        return f"fixed({cooling.beta:g})"
    if isinstance(cooling, CoolingPiecewise):
        stages = ",".join(f"{stage.sweeps}@{stage.beta:g}" for stage in cooling.stages)
        return f"piecewise([{stages}])"
    return str(cooling)


def _format_beta_ladder_summary(pt_cfg: object) -> str:
    betas, _ = _resolve_beta_ladder(pt_cfg)
    if isinstance(pt_cfg.beta_ladder, BetaLadderFixed):
        return f"fixed({betas[0]:g})"
    beta = ",".join(f"{b:g}" for b in betas)
    return f"geometric([{beta}])"


def _format_move_probs(move_probs: dict[str, float]) -> str:
    parts = [f"{key}={val:.2f}" for key, val in move_probs.items() if val > 0]
    return ",".join(parts) if parts else "none"


def _format_auto_opt_config_summary(cfg: SampleConfig) -> str:
    kind = cfg.optimizer.name
    moves = _format_move_probs(resolve_move_config(cfg.moves).move_probs)
    combine = cfg.objective.combine or ("sum" if cfg.objective.score_scale == "consensus-neglop-sum" else "min")
    if kind == "gibbs":
        cooling = _format_cooling_summary(cfg.optimizers.gibbs.beta_schedule)
        chains = cfg.budget.restarts
    else:
        cooling = _format_beta_ladder_summary(cfg.optimizers.pt)
        chains = len(_resolve_beta_ladder(cfg.optimizers.pt)[0])
    return (
        f"optimizer={kind} scorer={cfg.objective.score_scale} "
        f"combine={combine} length={cfg.init.length} chains={chains} tune={cfg.budget.tune} draws={cfg.budget.draws} "
        f"cooling={cooling} moves={moves} progress_every={cfg.ui.progress_every}"
    )


def _selected_config_payload(cfg: SampleConfig) -> dict[str, object]:
    return {
        "optimizer": cfg.optimizer.name,
        "objective": cfg.objective.model_dump(),
        "optimizers": cfg.optimizers.model_dump(),
        "moves": resolve_move_config(cfg.moves).model_dump(),
        "init_length": cfg.init.length,
        "budget": cfg.budget.model_dump(),
        "elites": cfg.elites.model_dump(),
    }


def _run_sample_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    stage: str = "sample",
    run_kind: str | None = None,
    auto_opt_meta: dict[str, object] | None = None,
    init_seeds: list[np.ndarray] | None = None,
) -> Path:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each chain gets its own independent seed (random/consensus/consensus_mix) unless
    init_seeds are provided for warm-started runs.
    """
    base_out = config_path.parent / Path(cfg.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = build_run_dir(
        config_path=config_path,
        out_dir=cfg.out_dir,
        stage=stage,
        tfs=tfs,
        set_index=set_index,
        include_set_index=include_set_index,
    )
    ensure_run_dirs(out_dir, meta=True, artifacts=True, live=sample_cfg.output.live_metrics)
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    run_logger = logger.debug if stage == "auto_opt" else logger.info
    stage_label = stage.upper().replace("_", "-")
    run_logger("=== RUN %s: %s ===", stage_label, out_dir)
    logger.debug("Full sample config: %s", sample_cfg.model_dump_json())

    metrics_path = live_metrics_path(out_dir) if sample_cfg.output.live_metrics else None
    optimizer_kind = _resolve_optimizer_kind(sample_cfg)
    chain_count = _effective_chain_count(sample_cfg, kind=optimizer_kind)
    status_writer = RunStatusWriter(
        path=status_path(out_dir),
        stage=stage,
        run_dir=out_dir,
        metrics_path=metrics_path,
        payload={
            "config_path": str(config_path.resolve()),
            "status_message": "initializing",
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
        },
    )
    update_run_index_from_status(
        config_path,
        out_dir,
        status_writer.payload,
        catalog_root=cfg.motif_store.catalog_root,
    )
    status_writer.update(
        status_message="loading_pwms",
        draws=sample_cfg.budget.draws,
        tune=sample_cfg.budget.tune,
        chains=chain_count,
        optimizer=optimizer_kind,
    )

    # 1) LOAD all required PWMs
    pwms = _load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)

    if sample_cfg.init.kind == "consensus":
        regulator = sample_cfg.init.regulator
        if regulator not in tfs:
            raise ValueError(
                f"init.regulator='{regulator}' is not in regulator_sets[{set_index}] ({tfs}). "
                "Use a regulator within the active set or switch to init.kind='consensus_mix'."
            )

    max_w = max(pwm.length for pwm in pwms.values())
    if sample_cfg.init.length < max_w:
        names = ", ".join(f"{tf}:{pwms[tf].length}" for tf in sorted(pwms))
        raise ValueError(
            f"init.length={sample_cfg.init.length} is shorter than the widest PWM (max={max_w}). "
            f"Per-TF lengths: {names}. Increase sample.init.length."
        )

    # 2) INSTANTIATE SCORER and SequenceEvaluator
    scale = sample_cfg.objective.score_scale
    combine_cfg = sample_cfg.objective.combine
    logger.debug("Using score_scale = %r", scale)
    if scale == "llr" and len(tfs) > 1:
        logger.warning(
            "score_scale='llr' is not comparable across PWMs in multi-TF runs. "
            "Consider normalized-llr or logp, or set objective.allow_unscaled_llr=true to silence this warning."
        )
    if sample_cfg.objective.bidirectional and not (
        sample_cfg.elites.dsDNA_canonicalize or sample_cfg.elites.dsDNA_hamming
    ):
        logger.warning(
            "Bidirectional scoring is enabled but dsDNA equivalence is disabled "
            "(reverse complements will be treated as distinct for diversity/uniqueness). "
            "Consider setting sample.elites.dsDNA_canonicalize=true."
        )

    def _sum_combine(values):
        return float(sum(values))

    combiner = None
    combine_resolved = "min"
    if combine_cfg == "sum":
        combiner = _sum_combine
        combine_resolved = "sum"
        if sample_cfg.objective.softmin.enabled:
            logger.warning("objective.combine='sum' disables softmin; softmin schedule will be ignored.")
    elif combine_cfg == "min":
        if scale == "consensus-neglop-sum":
            combiner = min
        combine_resolved = "min"
    elif scale == "consensus-neglop-sum":
        combiner = _sum_combine
        combine_resolved = "sum"
        if sample_cfg.objective.softmin.enabled:
            logger.warning(
                "score_scale='consensus-neglop-sum' defaults to sum() and disables softmin. "
                "Set objective.combine='min' to enforce weakest-TF optimization."
            )
    logger.debug("Building Scorer and SequenceEvaluator with scale=%r", scale)
    scorer = Scorer(
        pwms,
        bidirectional=sample_cfg.objective.bidirectional,
        scale=scale,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
    )
    length_penalty_lambda = sample_cfg.objective.length_penalty_lambda
    length_penalty_ref = None
    if length_penalty_lambda > 0:
        if (
            sample_cfg.auto_opt is not None
            and sample_cfg.auto_opt.length is not None
            and sample_cfg.auto_opt.length.min_length is not None
        ):
            length_penalty_ref = sample_cfg.auto_opt.length.min_length
        else:
            length_penalty_ref = max_w
    evaluator = SequenceEvaluator(
        pwms=pwms,
        scale=scale,
        combiner=combiner,  # defaults to min(...) for llr/logp/normalized-llr/z
        scorer=scorer,
        bidirectional=sample_cfg.objective.bidirectional,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
        length_penalty_lambda=length_penalty_lambda,
        length_penalty_ref=length_penalty_ref,
    )

    logger.debug("Scorer and SequenceEvaluator instantiated")
    logger.debug("  Scorer.scale = %r", scorer.scale)
    logger.debug("  SequenceEvaluator.scale = %r", evaluator._scale)
    logger.debug("  SequenceEvaluator.combiner = %r", evaluator._combiner)

    # 3) FLATTEN optimizer config for Gibbs/PT
    moves = resolve_move_config(sample_cfg.moves)
    opt_cfg: dict[str, object] = {
        "draws": sample_cfg.budget.draws,
        "tune": sample_cfg.budget.tune,
        "chains": chain_count,
        "min_dist": sample_cfg.elites.min_hamming,
        "top_k": sample_cfg.elites.k,
        "bidirectional": sample_cfg.objective.bidirectional,
        "dsdna_hamming": bool(sample_cfg.elites.dsDNA_hamming),
        "record_tune": sample_cfg.output.trace.include_tune,
        "progress_bar": sample_cfg.ui.progress_bar,
        "progress_every": sample_cfg.ui.progress_every,
        "early_stop": sample_cfg.early_stop.model_dump(),
        **moves.model_dump(),
        "softmin": sample_cfg.objective.softmin.model_dump(),
    }
    if init_seeds:
        opt_cfg["init_seeds"] = init_seeds

    if optimizer_kind == "gibbs":
        schedule = sample_cfg.optimizers.gibbs.beta_schedule
        opt_cfg.update(schedule.model_dump())
        opt_cfg["adaptive_beta"] = sample_cfg.optimizers.gibbs.adaptive_beta.model_dump()
        opt_cfg["apply_during"] = sample_cfg.optimizers.gibbs.apply_during
        opt_cfg["schedule_scope"] = sample_cfg.optimizers.gibbs.schedule_scope
    elif optimizer_kind == "pt":
        ladder_cfg = sample_cfg.optimizers.pt.beta_ladder
        betas, ladder_notes = _resolve_beta_ladder(sample_cfg.optimizers.pt)
        if ladder_notes:
            logger.info("PT ladder notes: %s", "; ".join(ladder_notes))
        if isinstance(ladder_cfg, BetaLadderFixed):
            opt_cfg["kind"] = "fixed"
            opt_cfg["beta"] = float(betas[0])
        else:
            opt_cfg["kind"] = "geometric"
            opt_cfg["beta"] = betas
        opt_cfg["swap_prob"] = sample_cfg.optimizers.pt.swap_prob
        opt_cfg["adaptive_swap"] = sample_cfg.optimizers.pt.ladder_adapt.model_dump()
        logger.debug("β-ladder (%d levels): %s", len(betas), ", ".join(f"{b:g}" for b in betas))
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_kind}'")

    logger.debug("Optimizer config flattened: %s", opt_cfg)

    # 4) INSTANTIATE OPTIMIZER (Gibbs or PT), passing in init_cfg and pwms
    from dnadesign.cruncher.core.optimizers.registry import get_optimizer

    optimizer_factory = get_optimizer(optimizer_kind)
    run_logger("Instantiating optimizer: %s", optimizer_kind)
    rng = np.random.default_rng(sample_cfg.rng.seed + set_index - 1)
    optimizer = optimizer_factory(
        evaluator=evaluator,
        cfg=opt_cfg,
        rng=rng,
        pwms=pwms,
        init_cfg=sample_cfg.init,
        status_writer=status_writer,
    )

    # 5) RUN the MCMC sampling
    status_writer.update(status_message="sampling")
    run_logger("Starting MCMC sampling …")
    try:
        optimizer.optimise()
    except KeyboardInterrupt as exc:
        status_writer.finish(status="aborted", status_message="aborted", error=str(exc) or "KeyboardInterrupt")
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.motif_store.catalog_root,
        )
        raise
    except Exception as exc:  # pragma: no cover - ensures run status is updated on failure
        status_writer.finish(status="failed", status_message="failed", error=str(exc))
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.motif_store.catalog_root,
        )
        raise
    if status_writer is not None and hasattr(optimizer, "best_score"):
        best_meta = getattr(optimizer, "best_meta", None)
        status_writer.update(
            best_score=getattr(optimizer, "best_score", None),
            best_chain=(best_meta[0] + 1) if best_meta else None,
            best_draw=(best_meta[1]) if best_meta else None,
        )
    run_logger("MCMC sampling complete.")
    status_writer.update(status_message="sampling complete")
    status_writer.update(status_message="saving_artifacts")

    # 6) SAVE enriched config
    _save_config(cfg, out_dir, config_path, tfs=tfs, set_index=set_index, sample_cfg=sample_cfg, log_fn=run_logger)

    artifacts: list[dict[str, object]] = [
        artifact_entry(
            config_used_path(out_dir),
            out_dir,
            kind="config",
            label="Resolved config (config_used.yaml)",
            stage=stage,
        )
    ]

    # 7) SAVE trace.nc
    if sample_cfg.output.trace.save and hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.artifacts.traces import save_trace

        save_trace(optimizer.trace_idata, trace_path(out_dir))
        artifacts.append(
            artifact_entry(
                trace_path(out_dir),
                out_dir,
                kind="trace",
                label="Trace (NetCDF)",
                stage=stage,
            )
        )
        logger.debug("Saved MCMC trace -> %s", trace_path(out_dir).relative_to(out_dir.parent))
    elif not sample_cfg.output.trace.save:
        logger.debug("Skipping trace.nc (sample.output.trace.save=false)")

    # Resolve final soft-min beta for polishing/trimming/elite ranking (from optimizer schedule)
    beta_softmin_final: float | None = _resolve_final_softmin_beta(optimizer, sample_cfg)

    # 8) SAVE sequences.parquet (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.output.save_sequences
        and hasattr(optimizer, "all_samples")
        and hasattr(optimizer, "all_meta")
        and hasattr(optimizer, "all_scores")
    ):
        seq_parquet = sequences_path(out_dir)
        tf_order = sorted(tfs)
        want_canonical_all = bool(sample_cfg.elites.dsDNA_canonicalize)

        def _sequence_rows() -> Iterable[dict[str, object]]:
            for (chain_id, draw_i), seq_arr, per_tf_map in zip(
                optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
            ):
                if draw_i < sample_cfg.budget.tune:
                    phase = "tune"
                    draw_i_to_write = draw_i
                    draw_in_phase = draw_i
                else:
                    phase = "draw"
                    draw_i_to_write = draw_i
                    draw_in_phase = draw_i - sample_cfg.budget.tune
                seq_str = SequenceState(seq_arr).to_string()
                row: dict[str, object] = {
                    "chain": int(chain_id),
                    "chain_1based": int(chain_id) + 1,
                    "draw": int(draw_i_to_write),
                    "draw_in_phase": int(draw_in_phase),
                    "phase": phase,
                    "sequence": seq_str,
                }
                if want_canonical_all:
                    row["canonical_sequence"] = SequenceState(canon_int(seq_arr)).to_string()
                row["combined_score_final"] = float(
                    evaluator.combined_from_scores(per_tf_map, beta=beta_softmin_final, length=seq_arr.size)
                )
                for tf in tf_order:
                    row[f"score_{tf}"] = float(per_tf_map[tf])
                yield row

        _write_parquet_rows(seq_parquet, _sequence_rows())
        artifacts.append(
            artifact_entry(
                seq_parquet,
                out_dir,
                kind="table",
                label="Sequences with per-TF scores (Parquet)",
                stage=stage,
            )
        )
        logger.debug(
            "Saved all sequences with per-TF scores -> %s",
            seq_parquet.relative_to(out_dir.parent),
        )

    # 9) BUILD elites list — filter (representativeness), rank (objective), diversify
    filters = sample_cfg.elites.filters
    min_per_tf_norm = filters.min_per_tf_norm
    require_all = filters.require_all_tfs_over_min_norm
    pwm_sum_min = filters.pwm_sum_min
    min_dist: int = sample_cfg.elites.min_hamming
    use_dsdna_hamming = bool(sample_cfg.elites.dsDNA_hamming)
    raw_elites: list[_EliteCandidate] = []
    norm_sums: list[float] = []
    min_norms: list[float] = []
    total_draws_seen = len(optimizer.all_samples)

    for (chain_id, draw_idx), seq_arr, per_tf_map in zip(
        optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
    ):
        norm_map = _norm_map_for_elites(
            seq_arr,
            per_tf_map,
            scorer=scorer,
            score_scale=sample_cfg.objective.score_scale,
        )
        min_norm = min(norm_map.values()) if norm_map else 0.0
        sum_norm = float(sum(norm_map.values()))
        norm_sums.append(sum_norm)
        min_norms.append(min_norm)

        if not _elite_filter_passes(
            norm_map=norm_map,
            min_norm=min_norm,
            sum_norm=sum_norm,
            min_per_tf_norm=min_per_tf_norm,
            require_all_tfs_over_min_norm=require_all,
            pwm_sum_min=pwm_sum_min,
        ):
            continue

        combined_score = evaluator.combined_from_scores(
            per_tf_map,
            beta=beta_softmin_final,
            length=seq_arr.size,
        )
        raw_elites.append(
            _EliteCandidate(
                seq_arr=seq_arr,
                chain_id=chain_id,
                draw_idx=draw_idx,
                combined_score=float(combined_score),
                min_norm=float(min_norm),
                sum_norm=float(sum_norm),
                per_tf_map=per_tf_map,
                norm_map=norm_map,
            )
        )

    # -------- percentile diagnostics ----------------------------------------
    if norm_sums:
        p50, p90 = np.percentile(norm_sums, [50, 90])
        n_tf = scorer.pwm_count
        avg_thr_pct = 100 * pwm_sum_min / n_tf if n_tf else 0.0
        logger.debug("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        if pwm_sum_min > 0:
            logger.debug(
                "Threshold %.2f ⇒ ~%.0f%%-of-consensus on average per TF (%d regulators)",
                pwm_sum_min,
                avg_thr_pct,
                n_tf,
            )
        logger.debug(
            "Typical draw: med %.2f (≈ %.0f%%/TF); top-10%% %.2f (≈ %.0f%%/TF)",
            p50,
            100 * p50 / n_tf if n_tf else 0.0,
            p90,
            100 * p90 / n_tf if n_tf else 0.0,
        )
    if min_norms and min_per_tf_norm is not None:
        p50_min, p90_min = np.percentile(min_norms, [50, 90])
        logger.debug("Normalised-min percentiles |  median %.2f   90%% %.2f", p50_min, p90_min)

    # -------- rank raw_elites by objective ----------------------------------
    raw_elites.sort(
        key=lambda cand: _elite_rank_key(cand.combined_score, cand.min_norm, cand.sum_norm),
        reverse=True,
    )

    # -------- apply diversity filter ----------------------------------------
    kept_elites: list[_EliteCandidate] = []
    kept_seqs: list[np.ndarray] = []
    dist_fn = dsdna_hamming if use_dsdna_hamming else hamming_distance

    for cand in raw_elites:
        seq_arr = cand.seq_arr
        if all(dist_fn(seq_arr, s) >= min_dist for s in kept_seqs):
            kept_elites.append(cand)
            kept_seqs.append(seq_arr)

    if min_dist > 0:
        logger.debug(
            "Diversity filter (≥%d mismatches): kept %d / %d candidates",
            min_dist,
            len(kept_elites),
            len(raw_elites),
        )
    else:
        kept_elites = raw_elites  # no filtering

    kept_after_diversity_pre_polish = len(kept_elites)
    passed_post_polish_filter = kept_after_diversity_pre_polish
    kept_after_diversity_final = kept_after_diversity_pre_polish

    if not kept_elites and (pwm_sum_min > 0 or min_per_tf_norm is not None):
        logger.warning("Elite filters removed all candidates; relax min_per_tf_norm/pwm_sum_min or min_hamming.")

    # Optional deterministic polish + trimming
    if kept_elites and (sample_cfg.output.polish.enabled or sample_cfg.output.trim.enabled):
        max_w = max(pwm.length for pwm in pwms.values())
        polish_cfg = sample_cfg.output.polish
        polish_cap = polish_cfg.max_elites

        def _trim(seq_arr: np.ndarray) -> np.ndarray:
            L = seq_arr.size
            min_start = L
            max_end = 0
            for tf_name in scorer.tf_names:
                _llr, offset, strand = scorer.best_llr(seq_arr, tf_name)
                width = scorer.pwm_width(tf_name)
                if strand == "-":
                    offset = L - width - offset
                min_start = min(min_start, offset)
                max_end = max(max_end, offset + width)
            start = max(0, min_start - sample_cfg.output.trim.padding)
            end = min(L, max_end + sample_cfg.output.trim.padding)
            if end - start < max_w:
                raise ValueError(
                    "Trimmed length would be shorter than the widest PWM "
                    f"(trimmed={end - start}, max_w={max_w}). "
                    "Increase output.trim.padding or disable output.trim."
                )
            trimmed = seq_arr[start:end].copy()
            if sample_cfg.output.trim.require_non_decreasing:
                old_score = evaluator.combined(SequenceState(seq_arr), beta=beta_softmin_final)
                new_score = evaluator.combined(SequenceState(trimmed), beta=beta_softmin_final)
                if new_score < old_score:
                    return seq_arr
            return trimmed

        updated: list[_EliteCandidate] = []
        for idx, cand in enumerate(kept_elites):
            seq_new = cand.seq_arr
            if sample_cfg.output.polish.enabled and (polish_cap is None or idx < polish_cap):
                seq_new = _polish_sequence(
                    seq_new,
                    evaluator=evaluator,
                    beta_softmin_final=beta_softmin_final,
                    max_rounds=polish_cfg.max_rounds,
                    improvement_tol=polish_cfg.improvement_tol,
                    max_evals=polish_cfg.max_evals,
                )
            if sample_cfg.output.trim.enabled:
                seq_new = _trim(seq_new)
            per_tf_map = evaluator(SequenceState(seq_new))
            norm_map = _norm_map_for_elites(
                seq_new,
                per_tf_map,
                scorer=scorer,
                score_scale=sample_cfg.objective.score_scale,
            )
            min_norm = min(norm_map.values()) if norm_map else 0.0
            sum_norm = float(sum(norm_map.values()))
            combined_score = evaluator.combined_from_scores(
                per_tf_map,
                beta=beta_softmin_final,
                length=seq_new.size,
            )
            updated.append(
                _EliteCandidate(
                    seq_arr=seq_new,
                    chain_id=cand.chain_id,
                    draw_idx=cand.draw_idx,
                    combined_score=float(combined_score),
                    min_norm=float(min_norm),
                    sum_norm=float(sum_norm),
                    per_tf_map=per_tf_map,
                    norm_map=norm_map,
                )
            )

        updated = _filter_elite_candidates(
            updated,
            min_per_tf_norm=min_per_tf_norm,
            require_all_tfs_over_min_norm=require_all,
            pwm_sum_min=pwm_sum_min,
        )
        passed_post_polish_filter = len(updated)
        updated.sort(
            key=lambda cand: _elite_rank_key(cand.combined_score, cand.min_norm, cand.sum_norm),
            reverse=True,
        )
        kept_elites = []
        kept_seqs = []
        for cand in updated:
            seq_arr = cand.seq_arr
            if all(dist_fn(seq_arr, s) >= min_dist for s in kept_seqs):
                kept_elites.append(cand)
                kept_seqs.append(seq_arr)
        kept_after_diversity_final = len(kept_elites)
        if not kept_elites and (pwm_sum_min > 0 or min_per_tf_norm is not None):
            logger.warning(
                "Post-polish/trim filters removed all candidates; relax min_per_tf_norm/pwm_sum_min or min_hamming."
            )

    # serialise elites
    elites: list[dict[str, object]] = []
    want_cons = bool(sample_cfg.output.include_consensus_in_elites)

    want_canonical = bool(sample_cfg.elites.dsDNA_canonicalize)
    for rank, cand in enumerate(kept_elites, 1):
        seq_arr = cand.seq_arr
        seq_str = SequenceState(seq_arr).to_string()
        canonical_seq = None
        if want_canonical:
            canonical_seq = SequenceState(canon_int(seq_arr)).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}
        norm_map = cand.norm_map
        per_tf_map = cand.per_tf_map

        for tf_name in scorer.tf_names:
            # best site in this *sequence*
            raw_llr, offset, strand = scorer.best_llr(seq_arr, tf_name)
            width = scorer.pwm_width(tf_name)
            if strand == "-":
                offset = len(seq_arr) - width - offset
            start_pos = offset + 1
            strand_label = f"{strand}1"
            motif_diag = f"{start_pos}_[{strand_label}]_{width}"

            # OPTIONAL – consensus of the PWM (window only, no padding)
            if want_cons:
                consensus = scorer.consensus_sequence(tf_name)
            else:
                consensus = None

            per_tf_details[tf_name] = {
                "raw_llr": float(raw_llr),
                "offset": offset,
                "strand": strand,
                "width": width,
                "motif_diagram": motif_diag,
                "scaled_score": float(per_tf_map[tf_name]),
                "normalized_llr": float(norm_map.get(tf_name, 0.0)),
            }
            if want_cons:
                per_tf_details[tf_name]["consensus"] = consensus

        entry = {
            "id": str(uuid.uuid4()),
            "sequence": seq_str,
            "rank": rank,
            "norm_sum": cand.sum_norm,
            "min_norm": cand.min_norm,
            "sum_norm": cand.sum_norm,
            "combined_score_final": cand.combined_score,
            "chain": cand.chain_id,
            "chain_1based": cand.chain_id + 1,
            "draw_idx": cand.draw_idx,
            "draw_in_phase": cand.draw_idx - sample_cfg.budget.tune
            if cand.draw_idx >= sample_cfg.budget.tune
            else cand.draw_idx,
            "per_tf": per_tf_details,
            "meta_type": "mcmc-elite",
            "meta_source": out_dir.name,
            "meta_date": datetime.now(timezone.utc).isoformat(),
        }
        if want_canonical:
            entry["canonical_sequence"] = canonical_seq
        elites.append(entry)

    run_logger(
        "Final elite count: %d (pwm_sum_min=%s, min_per_tf_norm=%s, min_dist=%d)",
        len(elites),
        f"{pwm_sum_min:.2f}" if pwm_sum_min else "off",
        f"{min_per_tf_norm:.2f}" if min_per_tf_norm is not None else "off",
        min_dist,
    )

    tf_label = format_regulator_slug(tfs)

    # 1)  parquet payload ----------------------------------------------------
    parquet_path = elites_path(out_dir)

    def _elite_rows() -> Iterable[dict[str, object]]:
        for entry in elites:
            row = dict(entry)
            per_tf = row.pop("per_tf", None)
            if per_tf is not None:
                row["per_tf_json"] = json.dumps(per_tf, sort_keys=True)
                for tf_name, details in per_tf.items():
                    row[f"score_{tf_name}"] = details.get("scaled_score")
                    row[f"norm_{tf_name}"] = details.get("normalized_llr")
            yield row

    elite_schema = _elite_parquet_schema(tfs, include_canonical=want_canonical)
    _write_parquet_rows(parquet_path, _elite_rows(), chunk_size=2000, schema=elite_schema)
    logger.debug("Saved elites Parquet -> %s", parquet_path.relative_to(out_dir.parent))

    json_path = elites_json_path(out_dir)
    json_path.write_text(json.dumps(elites, indent=2))
    logger.debug("Saved elites JSON -> %s", json_path.relative_to(out_dir.parent))

    # 2)  .yaml run-metadata -----------------------------------------------
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "threshold_norm_sum": pwm_sum_min,
        "min_per_tf_norm": min_per_tf_norm,
        "require_all_tfs_over_min_norm": require_all,
        "min_hamming_dist": min_dist,
        "dsdna_canonicalize": want_canonical,
        "dsdna_hamming": use_dsdna_hamming,
        "total_draws_seen": total_draws_seen,
        "passed_pre_filter": len(raw_elites),
        "kept_after_diversity_pre_polish": kept_after_diversity_pre_polish,
        "passed_post_polish_filter": passed_post_polish_filter,
        "kept_after_diversity_final": kept_after_diversity_final,
        "objective_combine": combine_resolved,
        "softmin_beta_final_resolved": beta_softmin_final,
        "indexing_note": (
            "chain is 0-based; chain_1based is 1-based; draw_idx is absolute sweep; draw_in_phase is phase-relative"
        ),
        "tf_label": tf_label,
        "sequence_length": sample_cfg.init.length,
        #  you can inline the full cfg if you prefer – this keeps it concise
        "config_file": str(config_used_path(out_dir).resolve()),
        "regulator_set": {"index": set_index, "tfs": tfs},
    }
    if elites:
        lengths = [len(entry["sequence"]) for entry in elites]
        meta["sequence_length_trimmed"] = {"min": min(lengths), "max": max(lengths)}
    yaml_path = elites_yaml_path(out_dir)
    with yaml_path.open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    logger.debug("Saved metadata -> %s", yaml_path.relative_to(out_dir.parent))

    artifacts.extend(
        [
            artifact_entry(
                parquet_path,
                out_dir,
                kind="table",
                label="Elite sequences (Parquet)",
                stage=stage,
            ),
            artifact_entry(
                json_path,
                out_dir,
                kind="json",
                label="Elite sequences (JSON)",
                stage=stage,
            ),
            artifact_entry(
                yaml_path,
                out_dir,
                kind="metadata",
                label="Elite metadata (YAML)",
                stage=stage,
            ),
        ]
    )

    # 10) RUN MANIFEST (for reporting + provenance)
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    lock_path = resolve_lock_path(config_path)
    manifest = build_run_manifest(
        stage=stage,
        cfg=cfg,
        config_path=config_path,
        lock_path=lock_path,
        lockmap={tf: lockmap[tf] for tf in tfs},
        catalog=catalog,
        run_dir=out_dir,
        artifacts=artifacts,
        extra={
            "sequence_length": sample_cfg.init.length,
            "seed": sample_cfg.rng.seed,
            "seed_effective": sample_cfg.rng.seed + set_index - 1,
            "record_tune": sample_cfg.output.trace.include_tune,
            "save_trace": sample_cfg.output.trace.save,
            "tune": sample_cfg.budget.tune,
            "draws": sample_cfg.budget.draws,
            "restarts": sample_cfg.budget.restarts,
            "early_stop": sample_cfg.early_stop.model_dump(),
            "top_k": sample_cfg.elites.k,
            "min_dist": sample_cfg.elites.min_hamming,
            "elites": sample_cfg.elites.model_dump(),
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
            "objective": {
                "score_scale": sample_cfg.objective.score_scale,
                "combine": combine_resolved,
                "bidirectional": sample_cfg.objective.bidirectional,
                "softmin": sample_cfg.objective.softmin.model_dump(),
                "softmin_beta_final_resolved": beta_softmin_final,
            },
            "optimizer": {
                "kind": optimizer_kind,
                "gibbs": sample_cfg.optimizers.gibbs.model_dump(),
                "pt": sample_cfg.optimizers.pt.model_dump(),
            },
            "optimizer_stats": optimizer.stats() if hasattr(optimizer, "stats") else {},
            "objective_schedule_summary": optimizer.objective_schedule_summary()
            if hasattr(optimizer, "objective_schedule_summary")
            else {},
        },
    )
    manifest_path = write_manifest(out_dir, manifest)
    update_run_index_from_manifest(
        config_path,
        out_dir,
        manifest,
        catalog_root=cfg.motif_store.catalog_root,
    )
    logger.debug("Wrote run manifest -> %s", manifest_path.relative_to(out_dir.parent))
    status_writer.finish(status="completed", artifacts=artifacts)
    update_run_index_from_status(
        config_path,
        out_dir,
        status_writer.payload,
        catalog_root=cfg.motif_store.catalog_root,
    )
    return out_dir


def run_sample(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    auto_opt_override: bool | None = None,
) -> None:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each regulator set is sampled independently for clear provenance.
    """
    if cfg.sample is None:
        raise ValueError("sample section is required for sample")
    with _sigterm_as_keyboard_interrupt():
        ensure_mpl_cache(resolve_catalog_root(config_path, cfg.motif_store.catalog_root))
        lockmap = _lockmap_for(cfg, config_path)
        statuses = target_statuses(cfg=cfg, config_path=config_path)
        sample_cfg = cfg.sample
        auto_cfg = sample_cfg.auto_opt
        if auto_opt_override is not None:
            if auto_opt_override:
                sample_cfg.optimizer.name = "auto"
                if auto_cfg is None:
                    auto_cfg = AutoOptConfig()
                else:
                    auto_cfg = auto_cfg.model_copy(update={"enabled": True})
            else:
                auto_cfg = None
                if sample_cfg.optimizer.name == "auto":
                    raise ValueError(
                        "auto-opt disabled but sample.optimizer.name='auto'; set optimizer.name to 'gibbs' or 'pt'."
                    )
        use_auto_opt = sample_cfg.optimizer.name == "auto" and auto_cfg is not None and auto_cfg.enabled
        groups = regulator_sets(cfg.regulator_sets)
        set_count = len(groups)
        include_set_index = set_count > 1
        for set_index, group in enumerate(groups, start=1):
            if not group:
                raise ValueError(f"regulator_sets[{set_index}] is empty.")
            seen: set[str] = set()
            tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]
            subset = [status for status in statuses if status.set_index == set_index]
            if has_blocking_target_errors(subset):
                details = "; ".join(f"{s.tf_name}:{s.status}" for s in subset if s.status not in {"ready", "warning"})
                raise ValueError(
                    f"Target readiness failed for set {set_index} ({details}). "
                    f"Run `cruncher targets status {config_path.name}` for details."
                )
            if use_auto_opt:
                run_dir = _run_auto_optimize_for_set(
                    cfg,
                    config_path,
                    set_index=set_index,
                    set_count=set_count,
                    include_set_index=include_set_index,
                    tfs=tfs,
                    lockmap=lockmap,
                    sample_cfg=sample_cfg,
                    auto_cfg=auto_cfg,
                )
            else:
                run_dir = _run_sample_for_set(
                    cfg,
                    config_path,
                    set_index=set_index,
                    set_count=set_count,
                    include_set_index=include_set_index,
                    tfs=tfs,
                    lockmap=lockmap,
                    sample_cfg=sample_cfg,
                )
            logger.info(
                "Sample outputs -> %s",
                _format_run_path(run_dir, base=config_path.parent),
            )
            logger.info(
                "Config used -> %s",
                _format_run_path(config_used_path(run_dir), base=config_path.parent),
            )
