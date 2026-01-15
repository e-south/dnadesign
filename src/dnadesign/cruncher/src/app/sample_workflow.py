"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample_workflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import signal
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
from dnadesign.cruncher.core.sequence import hamming_distance
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
    rhat: float | None
    ess: float | None
    unique_fraction: float | None
    balance_median: float | None
    diversity: float | None
    improvement: float | None
    acceptance_b: float | None
    acceptance_m: float | None
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
    length_cfg = auto_cfg.length
    if not length_cfg.enabled:
        return [sample_cfg.init.length]
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
    return count


def _default_beta_ladder(chains: int, beta_min: float, beta_max: float) -> list[float]:
    if chains < 2:
        raise ValueError("PT beta ladder requires at least 2 chains")
    ladder = np.linspace(beta_min, beta_max, chains, dtype=float)
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
    notes: list[str] = []
    if candidate.status == "fail":
        candidate.quality = "fail"
        return notes
    quality = "ok"
    policy = auto_cfg.policy
    scorecard = policy.scorecard

    if candidate.kind != "pt":
        if candidate.rhat is None:
            notes.append("missing pilot metric: rhat")
        elif candidate.rhat > policy.max_rhat:
            notes.append(f"rhat={candidate.rhat:.3f} exceeds auto_opt.policy.max_rhat={policy.max_rhat:.3f}")
            if mode == "sample":
                quality = "warn"
        if candidate.ess is None:
            notes.append("missing pilot metric: ess")
        elif candidate.ess < policy.min_ess:
            notes.append(f"ess={candidate.ess:.1f} below auto_opt.policy.min_ess={policy.min_ess:.1f}")
            if mode == "sample":
                quality = "warn"

    if candidate.unique_fraction is None:
        quality = "warn"
        notes.append("missing pilot metric: unique_fraction")
    else:
        if candidate.unique_fraction < policy.min_unique_fraction:
            quality = "warn"
            notes.append(
                (
                    f"unique_fraction={candidate.unique_fraction:.2f} below "
                    f"auto_opt.policy.min_unique_fraction={policy.min_unique_fraction:.2f}"
                )
            )
        if policy.max_unique_fraction is not None and candidate.unique_fraction > policy.max_unique_fraction:
            quality = "warn"
            notes.append(
                (
                    f"unique_fraction={candidate.unique_fraction:.2f} above "
                    f"auto_opt.policy.max_unique_fraction={policy.max_unique_fraction:.2f}"
                )
            )
    if candidate.balance_median is None:
        quality = "warn"
        notes.append("missing pilot metric: balance_median")
    elif candidate.balance_median < scorecard.min_balance:
        quality = "warn"
        notes.append(
            (
                f"balance_median={candidate.balance_median:.3f} below "
                f"auto_opt.policy.scorecard.min_balance={scorecard.min_balance:.3f}"
            )
        )
    if candidate.diversity is None:
        quality = "warn"
        notes.append("missing pilot metric: diversity")
    elif candidate.diversity < scorecard.min_diversity:
        quality = "warn"
        notes.append(
            (
                f"diversity={candidate.diversity:.3f} below "
                f"auto_opt.policy.scorecard.min_diversity={scorecard.min_diversity:.3f}"
            )
        )

    def _acceptance_note(label: str, rate: float | None, target: float, tol: float) -> None:
        nonlocal quality
        if rate is None:
            quality = "warn"
            notes.append(f"missing pilot metric: {label}")
            return
        if abs(rate - target) > tol:
            quality = "warn"
            notes.append(f"{label}={rate:.3f} outside target {target:.2f}±{tol:.2f}")

    _acceptance_note(
        "acceptance_B", candidate.acceptance_b, scorecard.acceptance_target, scorecard.acceptance_tolerance
    )
    _acceptance_note(
        "acceptance_M", candidate.acceptance_m, scorecard.acceptance_target, scorecard.acceptance_tolerance
    )
    if candidate.kind == "pt":
        _acceptance_note("swap_rate", candidate.swap_rate, scorecard.swap_target, scorecard.swap_tolerance)
    candidate.quality = quality
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

    candidate_specs = [(kind, length) for length in lengths for kind in ("gibbs", "pt")]
    seed_rng = np.random.default_rng(sample_cfg.rng.seed)

    def _run_candidate(
        *,
        kind: str,
        length: int,
        budget: int,
        label: str,
        cooling_boost: float,
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
                    rhat=None,
                    ess=None,
                    unique_fraction=None,
                    balance_median=None,
                    diversity=None,
                    improvement=None,
                    acceptance_b=None,
                    acceptance_m=None,
                    swap_rate=None,
                    status="fail",
                    quality="fail",
                    warnings=[str(exc)],
                    diagnostics={},
                )
            runs.append(candidate)
        return _aggregate_candidate_runs(runs, budget=budget)

    def _log_candidate(candidate: AutoOptCandidate, *, label: str) -> None:
        diag_status = "n/a"
        if isinstance(candidate.diagnostics, dict):
            diag_status = candidate.diagnostics.get("status") or "n/a"
        logger.debug(
            (
                "Auto-opt %s %s L=%s B=%s boost=%s: scorecard=%s diagnostics=%s best_score=%s balance=%s "
                "diversity=%s rhat=%s ess=%s unique_fraction=%s"
            ),
            label,
            candidate.kind,
            candidate.length if candidate.length is not None else "n/a",
            candidate.budget if candidate.budget is not None else "n/a",
            f"{candidate.cooling_boost:g}",
            candidate.quality,
            diag_status,
            f"{candidate.best_score:.3f}" if candidate.best_score is not None else "n/a",
            f"{candidate.balance_median:.3f}" if candidate.balance_median is not None else "n/a",
            f"{candidate.diversity:.3f}" if candidate.diversity is not None else "n/a",
            f"{candidate.rhat:.3f}" if candidate.rhat is not None else "n/a",
            f"{candidate.ess:.1f}" if candidate.ess is not None else "n/a",
            f"{candidate.unique_fraction:.2f}" if candidate.unique_fraction is not None else "n/a",
        )
        if candidate.warnings:
            logger.warning("Auto-opt %s %s warnings: %s", label, candidate.kind, "; ".join(candidate.warnings))

    all_candidates: list[AutoOptCandidate] = []
    current_specs = list(candidate_specs)
    final_level_candidates: list[AutoOptCandidate] = []

    for idx, budget in enumerate(budgets):
        level_candidates: list[AutoOptCandidate] = []
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

        if idx < len(budgets) - 1:
            ranked = _rank_auto_opt_candidates(level_candidates, auto_cfg)
            keep_n = max(1, int(math.ceil(len(ranked) / max(1, auto_cfg.eta))))
            current_specs = [(c.kind, c.length) for c in ranked[:keep_n]]
        else:
            final_level_candidates = level_candidates

    if not final_level_candidates:
        raise ValueError("Auto-optimize failed: no final-level candidates were evaluated.")

    decision_notes: list[str] = []
    ok_candidates = [candidate for candidate in final_level_candidates if candidate.quality == "ok"]
    if not ok_candidates and auto_cfg.policy.retry_on_warn:
        logger.info(
            "Auto-optimize: no pilot met thresholds; retrying at max budget with boosted cooling (x%s).",
            auto_cfg.policy.cooling_boost,
        )
        retry_candidates: list[AutoOptCandidate] = []
        for kind, length in current_specs:
            label = f"retry_L{length}_B{budgets[-1]}"
            candidate = _run_candidate(
                kind=kind,
                length=length,
                budget=budgets[-1],
                label=label,
                cooling_boost=auto_cfg.policy.cooling_boost,
            )
            candidate_notes = _assess_candidate_quality(candidate, auto_cfg, mode=sample_cfg.mode)
            if candidate_notes:
                candidate.warnings.extend(candidate_notes)
            _log_candidate(candidate, label=label)
            retry_candidates.append(candidate)
            all_candidates.append(candidate)
        final_level_candidates.extend(retry_candidates)
        ok_candidates = [candidate for candidate in final_level_candidates if candidate.quality == "ok"]

    viable, allow_fail = _validate_auto_opt_candidates(
        final_level_candidates,
        allow_warn=auto_cfg.policy.allow_warn,
    )
    if allow_fail:
        logger.warning(
            "Auto-optimize: all pilot candidates missing diagnostics; proceeding with best available candidate."
        )
    if not ok_candidates:
        summary = ", ".join(f"{c.kind}:{c.quality}" for c in final_level_candidates)
        logger.warning(
            "Auto-optimize: no pilot met thresholds (candidates=%s). Proceeding with best available candidate.",
            summary,
        )
        decision_notes.append("no_pilot_met_thresholds")
    else:
        logger.info(
            "Auto-optimize: %d candidate(s) met thresholds; selecting best score across %d viable candidates.",
            len(ok_candidates),
            len(viable),
        )

    winner = _select_auto_opt_candidate(viable, auto_cfg, allow_fail=allow_fail)

    if winner.quality == "ok":
        logger.info(
            "Auto-optimize selected %s L=%s (best_score=%s, balance=%s, rhat=%s, ess=%s, unique_fraction=%s).",
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            f"{winner.best_score:.3f}" if winner.best_score is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    else:
        logger.warning(
            (
                "Auto-optimize selected %s L=%s (quality=%s best_score=%s, balance=%s, rhat=%s, "
                "ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            winner.quality,
            f"{winner.best_score:.3f}" if winner.best_score is not None else "n/a",
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
        "balance -> best_score -> diversity -> improvement -> acceptance -> swap_rate -> diagnostics"
        if length_aware
        else "best_score -> balance -> diversity -> improvement -> acceptance -> swap_rate -> diagnostics"
    )
    logger.info("Auto-opt selection ranking: %s.", ranking_label)
    logger.info("Auto-opt final config: %s", _format_auto_opt_config_summary(final_cfg))
    pilot_root = all_candidates[0].run_dir.parent if all_candidates else None
    best_marker_path = _auto_opt_best_marker_path(pilot_root, run_group=run_group) if pilot_root is not None else None
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
        "thresholds_met": winner.quality == "ok",
        "notes": decision_notes,
        "config_summary": _format_auto_opt_config_summary(final_cfg),
        "selected_config": _selected_config_payload(final_cfg),
        "length_config": auto_cfg.length.model_dump(),
        "auto_opt_config": {
            "budget_levels": auto_cfg.budget_levels,
            "eta": auto_cfg.eta,
            "replicates": auto_cfg.replicates,
            "keep_pilots": auto_cfg.keep_pilots,
            "prefer_simpler_if_close": auto_cfg.prefer_simpler_if_close,
            "tolerance": auto_cfg.tolerance.model_dump(),
        },
        "thresholds": {
            "min_unique_fraction": auto_cfg.policy.min_unique_fraction,
            "max_unique_fraction": auto_cfg.policy.max_unique_fraction,
            "scorecard_top_k": auto_cfg.policy.scorecard.top_k,
            "min_balance": auto_cfg.policy.scorecard.min_balance,
            "min_diversity": auto_cfg.policy.scorecard.min_diversity,
            "acceptance_target": auto_cfg.policy.scorecard.acceptance_target,
            "acceptance_tolerance": auto_cfg.policy.scorecard.acceptance_tolerance,
            "swap_target": auto_cfg.policy.scorecard.swap_target,
            "swap_tolerance": auto_cfg.policy.scorecard.swap_tolerance,
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
                "balance_median": candidate.balance_median,
                "diversity": candidate.diversity,
                "improvement": candidate.improvement,
                "acceptance_b": candidate.acceptance_b,
                "acceptance_m": candidate.acceptance_m,
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
    status_payload = json.loads(status_path(run_dir).read_text()) if status_path(run_dir).exists() else {}
    best_score = status_payload.get("best_score")
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
    if isinstance(optimizer_metrics, dict):
        acc = optimizer_metrics.get("acceptance_rate") or {}
        if isinstance(acc, dict):
            acceptance_b = acc.get("B")
            acceptance_m = acc.get("M")
    swap_rate = optimizer_metrics.get("swap_acceptance_rate") if isinstance(optimizer_metrics, dict) else None
    warnings = diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []
    status = "ok"
    if isinstance(diagnostics, dict):
        diag_status = diagnostics.get("status", "ok")
        if diag_status and diag_status != "ok":
            warnings = list(warnings)
            warnings.append(f"Diagnostics status={diag_status}; see diagnostics.json for details.")

    if best_score is None:
        warnings = list(warnings)
        warnings.append("Missing pilot metric: best_score")
        status = "fail"

    return AutoOptCandidate(
        kind=kind,
        length=int(length) if isinstance(length, (int, float)) else None,
        budget=budget,
        cooling_boost=cooling_boost,
        run_dir=run_dir,
        run_dirs=[run_dir],
        best_score=best_score,
        rhat=rhat,
        ess=ess,
        unique_fraction=unique_fraction,
        balance_median=balance_median,
        diversity=diversity,
        improvement=improvement,
        acceptance_b=acceptance_b,
        acceptance_m=acceptance_m,
        swap_rate=swap_rate,
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
    )


def _aggregate_candidate_runs(runs: list[AutoOptCandidate], *, budget: int) -> AutoOptCandidate:
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
    elif any(run.status == "fail" for run in runs):
        status = "warn"

    warnings: list[str] = []
    for run in runs:
        warnings.extend(run.warnings)
    if status == "warn":
        warnings.append("One or more pilot replicates failed")

    diagnostics = {
        "replicates": [run.diagnostics for run in runs],
        "run_dirs": [str(path) for path in run_dirs],
    }

    return AutoOptCandidate(
        kind=kind,
        length=length,
        budget=budget,
        cooling_boost=float(np.median([run.cooling_boost for run in runs])),
        run_dir=run_dirs[0],
        run_dirs=run_dirs,
        best_score=_median([run.best_score for run in runs]),
        rhat=_median([run.rhat for run in runs]),
        ess=_median([run.ess for run in runs]),
        unique_fraction=_median([run.unique_fraction for run in runs]),
        balance_median=_median([run.balance_median for run in runs]),
        diversity=_median([run.diversity for run in runs]),
        improvement=_median([run.improvement for run in runs]),
        acceptance_b=_median([run.acceptance_b for run in runs]),
        acceptance_m=_median([run.acceptance_m for run in runs]),
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
    def _closeness(rate: float | None, target: float, tol: float) -> float:
        if rate is None:
            return float("-inf")
        return 1.0 - min(1.0, abs(rate - target) / max(tol, 1.0e-6))

    ranked: list[tuple[tuple[float, float, float, float, float, float, float], AutoOptCandidate]] = []
    status_rank = {"ok": 2, "warn": 1, "fail": 0}
    scorecard = auto_cfg.policy.scorecard
    lengths = {c.length for c in candidates if c.length is not None}
    length_aware = auto_cfg.length.enabled and len(lengths) > 1
    for candidate in candidates:
        score = candidate.best_score if candidate.best_score is not None else float("-inf")
        balance = candidate.balance_median if candidate.balance_median is not None else float("-inf")
        diversity = candidate.diversity if candidate.diversity is not None else float("-inf")
        improvement = candidate.improvement if candidate.improvement is not None else float("-inf")
        acceptance = np.mean(
            [
                _closeness(candidate.acceptance_b, scorecard.acceptance_target, scorecard.acceptance_tolerance),
                _closeness(candidate.acceptance_m, scorecard.acceptance_target, scorecard.acceptance_tolerance),
            ]
        )
        swap_score = (
            _closeness(candidate.swap_rate, scorecard.swap_target, scorecard.swap_tolerance)
            if candidate.kind == "pt"
            else 0.0
        )
        primary = balance if length_aware else score
        secondary = score if length_aware else balance
        rank = (
            float(primary),
            float(secondary),
            float(diversity),
            float(improvement),
            float(acceptance),
            float(swap_score),
            float(status_rank.get(candidate.quality, status_rank.get(candidate.status, 0))),
        )
        ranked.append((rank, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def _validate_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
    *,
    allow_warn: bool,
) -> tuple[list[AutoOptCandidate], bool]:
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates.")
    viable = [c for c in candidates if c.status != "fail"]
    if not viable:
        if not allow_warn:
            raise ValueError(
                "Auto-optimize failed: all pilot candidates missing diagnostics. "
                "Set auto_opt.policy.allow_warn=true to proceed, or disable auto-opt."
            )
        return list(candidates), True
    ok_candidates = [c for c in candidates if c.quality == "ok"]
    if not ok_candidates and not allow_warn:
        summary = ", ".join(f"{c.kind}:{c.quality}" for c in candidates)
        raise ValueError(
            "Auto-optimize failed: no pilot met thresholds "
            f"(candidates={summary}). Set auto_opt.policy.allow_warn=true to proceed, "
            "or increase auto_opt budgets/adjust thresholds."
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
        ok_quality = [c for c in ok_candidates if c.quality == "ok"]
        if ok_quality:
            ok_candidates = ok_quality
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
        if gibbs and winner.best_score is not None and gibbs[0].best_score is not None:
            if gibbs[0].best_score >= winner.best_score - auto_cfg.tolerance.score:
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
    if kind == "gibbs":
        cooling = _format_cooling_summary(cfg.optimizers.gibbs.beta_schedule)
        chains = cfg.budget.restarts
    else:
        cooling = _format_beta_ladder_summary(cfg.optimizers.pt)
        chains = len(_resolve_beta_ladder(cfg.optimizers.pt)[0])
    return (
        f"optimizer={kind} scorer={cfg.objective.score_scale} "
        f"length={cfg.init.length} chains={chains} tune={cfg.budget.tune} draws={cfg.budget.draws} "
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
) -> Path:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each chain gets its own independent seed (random/consensus/consensus_mix).
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
    logger.debug("Using score_scale = %r", scale)
    combiner = (lambda vs: sum(vs)) if scale == "consensus-neglop-sum" else None
    logger.debug("Building Scorer and SequenceEvaluator with scale=%r", scale)
    scorer = Scorer(
        pwms,
        bidirectional=sample_cfg.objective.bidirectional,
        scale=scale,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
    )
    evaluator = SequenceEvaluator(
        pwms=pwms,
        scale=scale,
        combiner=combiner,  # defaults to min(...) for llr/logp/normalized-llr/z
        scorer=scorer,
        bidirectional=sample_cfg.objective.bidirectional,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
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
        "record_tune": sample_cfg.output.trace.include_tune,
        "progress_bar": sample_cfg.ui.progress_bar,
        "progress_every": sample_cfg.ui.progress_every,
        "early_stop": sample_cfg.early_stop.model_dump(),
        **moves.model_dump(),
        "softmin": sample_cfg.objective.softmin.model_dump(),
    }

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

    # 8) SAVE sequences.parquet (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.output.save_sequences
        and hasattr(optimizer, "all_samples")
        and hasattr(optimizer, "all_meta")
        and hasattr(optimizer, "all_scores")
    ):
        seq_parquet = sequences_path(out_dir)
        tf_order = sorted(tfs)

        def _sequence_rows() -> Iterable[dict[str, object]]:
            for (chain_id, draw_i), seq_arr, per_tf_map in zip(
                optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
            ):
                if draw_i < sample_cfg.budget.tune:
                    phase = "tune"
                    draw_i_to_write = draw_i
                else:
                    phase = "draw"
                    draw_i_to_write = draw_i
                seq_str = SequenceState(seq_arr).to_string()
                row: dict[str, object] = {
                    "chain": int(chain_id),
                    "draw": int(draw_i_to_write),
                    "phase": phase,
                    "sequence": seq_str,
                }
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

    # Resolve final soft-min beta for polishing/trimming
    beta_softmin_final: float | None = None
    softmin_cfg = sample_cfg.objective.softmin
    if softmin_cfg.enabled:
        total = sample_cfg.budget.tune + sample_cfg.budget.draws
        softmin_sched = {k: v for k, v in softmin_cfg.model_dump().items() if k in ("kind", "beta", "stages")}
        beta_softmin_final = make_beta_scheduler(softmin_sched, total)(total - 1)

    # 9)  BUILD elites list — keep draws whose ∑ normalised ≥ pwm_sum_threshold
    #     + enforce a per-sequence Hamming-distance diversity filter
    thr_norm: float = sample_cfg.elites.filters.pwm_sum_min  # e.g. 1.50
    min_dist: int = sample_cfg.elites.min_hamming  # e.g. 1
    raw_elites: list[tuple[np.ndarray, int, int, float, dict[str, float]]] = []
    norm_sums: list[float] = []  # diagnostics only

    for (chain_id, draw_idx), seq_arr, per_tf_map in zip(
        optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
    ):
        # ---------- per-PWM normalisation --------------------------
        #   raw_llr -> frac = (raw_llr - mu_null) / (llr_consensus - mu_null)
        #   - mu_null = null_mean (expected background)
        #   - frac < 0 => treat as 0 (below background contributes nothing)
        #   - frac may exceed 1 if a window scores better than "column max"
        # -------------------------------------------------------------------
        fracs = scorer.normalized_llr_components(seq_arr)
        total_norm = float(sum(fracs))  # 0 … n_TF + ε
        norm_sums.append(total_norm)

        if total_norm >= thr_norm:
            raw_elites.append((seq_arr, chain_id, draw_idx, total_norm, per_tf_map))

    # -------- percentile diagnostics ----------------------------------------
    if norm_sums:
        p50, p90 = np.percentile(norm_sums, [50, 90])
        n_tf = scorer.pwm_count
        avg_thr_pct = 100 * thr_norm / n_tf
        logger.debug("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        logger.debug(
            "Threshold %.2f ⇒ ~%.0f%%-of-consensus on average per TF (%d regulators)",
            thr_norm,
            avg_thr_pct,
            n_tf,
        )
        logger.debug(
            "Typical draw: med %.2f (≈ %.0f%%/TF); top-10%% %.2f (≈ %.0f%%/TF)",
            p50,
            100 * p50 / n_tf,
            p90,
            100 * p90 / n_tf,
        )

    # -------- rank raw_elites by score --------------------------------------
    raw_elites.sort(key=lambda t: t[3], reverse=True)  # highest first

    # -------- apply Hamming diversity filter -------------------------------
    kept_elites: list[tuple[np.ndarray, int, int, float, dict[str, float]]] = []
    kept_seqs: list[np.ndarray] = []

    for tpl in raw_elites:
        seq_arr = tpl[0]
        if all(hamming_distance(seq_arr, s) >= min_dist for s in kept_seqs):
            kept_elites.append(tpl)
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

    # Optional deterministic polish + trimming
    if kept_elites and (sample_cfg.output.polish.enabled or sample_cfg.output.trim.enabled):
        max_w = max(pwm.length for pwm in pwms.values())

        def _polish(seq_arr: np.ndarray) -> np.ndarray:
            seq = seq_arr.copy()
            best_score = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin_final)
            for _ in range(sample_cfg.output.polish.max_rounds):
                improved = False
                for i in range(seq.size):
                    old_base = seq[i]
                    best_base = old_base
                    best_local = best_score
                    for b in range(4):
                        if b == old_base:
                            continue
                        seq[i] = b
                        score = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin_final)
                        if score > best_local + sample_cfg.output.polish.improvement_tol:
                            best_local = score
                            best_base = b
                    seq[i] = best_base
                    if best_base != old_base:
                        best_score = best_local
                        improved = True
                if not improved:
                    break
            return seq

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
                old_score = evaluator.combined(SequenceState(seq_arr.copy()), beta=beta_softmin_final)
                new_score = evaluator.combined(SequenceState(trimmed.copy()), beta=beta_softmin_final)
                if new_score < old_score:
                    return seq_arr
            return trimmed

        updated: list[tuple[np.ndarray, int, int, float, dict[str, float]]] = []
        for seq_arr, chain_id, draw_idx, _total_norm, _per_tf_map in kept_elites:
            seq_new = seq_arr
            if sample_cfg.output.polish.enabled:
                seq_new = _polish(seq_new)
            if sample_cfg.output.trim.enabled:
                seq_new = _trim(seq_new)
            per_tf_map = evaluator(SequenceState(seq_new.copy()))
            total_norm = float(sum(scorer.normalized_llr_components(seq_new)))
            updated.append((seq_new, chain_id, draw_idx, total_norm, per_tf_map))

        updated.sort(key=lambda t: t[3], reverse=True)
        kept_elites = []
        kept_seqs = []
        for tpl in updated:
            seq_arr = tpl[0]
            if all(hamming_distance(seq_arr, s) >= min_dist for s in kept_seqs):
                kept_elites.append(tpl)
                kept_seqs.append(seq_arr)

    # serialise elites
    elites: list[dict[str, object]] = []
    want_cons = bool(sample_cfg.output.include_consensus_in_elites)

    for rank, (seq_arr, chain_id, draw_idx, total_norm, per_tf_map) in enumerate(kept_elites, 1):
        seq_str = SequenceState(seq_arr).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}
        norm_map = scorer.normalized_llr_map(seq_arr)

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
                "motif_diagram": motif_diag,
                "scaled_score": float(per_tf_map[tf_name]),
                "normalized_llr": float(norm_map.get(tf_name, 0.0)),
            }
            if want_cons:
                per_tf_details[tf_name]["consensus"] = consensus

        elites.append(
            {
                "id": str(uuid.uuid4()),
                "sequence": seq_str,
                "rank": rank,
                "norm_sum": total_norm,
                "chain": chain_id,
                "draw_idx": draw_idx,
                "per_tf": per_tf_details,
                "meta_type": "mcmc-elite",
                "meta_source": out_dir.name,
                "meta_date": datetime.now(timezone.utc).isoformat(),
            }
        )

    run_logger(
        "Final elite count: %d (normalised-sum ≥ %.2f, min_dist ≥ %d)",
        len(elites),
        thr_norm,
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

    _write_parquet_rows(parquet_path, _elite_rows(), chunk_size=2000)
    logger.debug("Saved elites Parquet -> %s", parquet_path.relative_to(out_dir.parent))

    json_path = elites_json_path(out_dir)
    json_path.write_text(json.dumps(elites, indent=2))
    logger.debug("Saved elites JSON -> %s", json_path.relative_to(out_dir.parent))

    # 2)  .yaml run-metadata -----------------------------------------------
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "threshold_norm_sum": thr_norm,
        "min_hamming_dist": min_dist,
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
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
            "objective": {
                "score_scale": sample_cfg.objective.score_scale,
                "bidirectional": sample_cfg.objective.bidirectional,
                "softmin": sample_cfg.objective.softmin.model_dump(),
            },
            "optimizer": {
                "kind": optimizer_kind,
                "gibbs": sample_cfg.optimizers.gibbs.model_dump(),
                "pt": sample_cfg.optimizers.pt.model_dump(),
            },
            "optimizer_stats": optimizer.stats() if hasattr(optimizer, "stats") else {},
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
