"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workflows/sample_workflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dnadesign.cruncher.config.schema_v2 import (
    AutoOptimizeConfig,
    CoolingFixed,
    CoolingGeometric,
    CoolingLinear,
    CruncherConfig,
    SampleConfig,
)
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.state import SequenceState
from dnadesign.cruncher.services.run_service import (
    update_run_index_from_manifest,
    update_run_index_from_status,
)
from dnadesign.cruncher.services.target_service import (
    has_blocking_target_errors,
    target_statuses,
)
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import (
    read_lockfile,
    validate_lockfile,
    verify_lockfile_hashes,
)
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.artifacts import artifact_entry
from dnadesign.cruncher.utils.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.utils.elites import find_elites_parquet
from dnadesign.cruncher.utils.labels import format_regulator_slug, regulator_sets
from dnadesign.cruncher.utils.manifest import build_run_manifest, load_manifest, write_manifest
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from dnadesign.cruncher.utils.parquet import read_parquet
from dnadesign.cruncher.utils.run_layout import (
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
from dnadesign.cruncher.utils.run_status import RunStatusWriter

logger = logging.getLogger(__name__)


@dataclass
class AutoOptCandidate:
    kind: str
    run_dir: Path
    best_score: float | None
    rhat: float | None
    ess: float | None
    unique_fraction: float | None
    status: str
    quality: str
    warnings: list[str]
    diagnostics: dict[str, object]


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        config_path.parent / cfg.motif_store.catalog_root,
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
    lock_root = config_path.parent / cfg.motif_store.catalog_root
    lock_path = lock_root / "locks" / f"{config_path.stem}.lock.json"
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
        catalog_root=lock_root,
        expected_pwm_source=cfg.motif_store.pwm_source,
    )
    return lockfile.resolved


def _save_config(
    cfg: CruncherConfig,
    batch_dir: Path,
    config_path: Path,
    *,
    tfs: list[str],
    set_index: int,
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
    logger.info("Wrote config_used.yaml to %s", cfg_path.relative_to(batch_dir.parent))


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


def _derive_pt_beta(
    base_cfg: SampleConfig,
    auto_cfg: AutoOptimizeConfig,
    *,
    chains: int,
) -> tuple[list[float], list[str]]:
    notes: list[str] = []
    if auto_cfg.pt_beta is not None:
        if len(auto_cfg.pt_beta) != chains:
            raise ValueError(
                "auto_opt.pt_beta length must match sample.chains for the selected PT run "
                f"(expected {chains}, got {len(auto_cfg.pt_beta)})"
            )
        return list(auto_cfg.pt_beta), notes
    if base_cfg.optimiser.kind == "pt" and base_cfg.optimiser.cooling.kind == "geometric":
        base_beta = list(base_cfg.optimiser.cooling.beta)
        if len(base_beta) == chains:
            return base_beta, notes
        notes.append("derived PT beta ladder from base config range")
        return _default_beta_ladder(chains, min(base_beta), max(base_beta)), notes
    notes.append("derived PT beta ladder from auto_opt bounds")
    return _default_beta_ladder(chains, auto_cfg.pt_beta_min, auto_cfg.pt_beta_max), notes


def _boost_cooling(cooling: Any, factor: float) -> tuple[Any, list[str]]:
    if factor <= 1:
        return cooling, []
    notes = [f"boosted cooling by ×{factor:g} for stabilization"]
    if isinstance(cooling, CoolingLinear):
        beta = [float(cooling.beta[0]) * factor, float(cooling.beta[1]) * factor]
        return CoolingLinear(beta=beta), notes
    if isinstance(cooling, CoolingFixed):
        return CoolingFixed(beta=float(cooling.beta) * factor), notes
    if isinstance(cooling, CoolingGeometric):
        beta = [float(b) * factor for b in cooling.beta]
        return CoolingGeometric(beta=beta), notes
    return cooling, notes


def _derive_gibbs_cooling(base_cfg: SampleConfig, auto_cfg: AutoOptimizeConfig) -> tuple[Any, list[str]]:
    notes: list[str] = []
    if base_cfg.optimiser.kind == "gibbs" and base_cfg.optimiser.cooling.kind != "geometric":
        return base_cfg.optimiser.cooling, notes
    notes.append("using auto_opt.gibbs_cooling for Gibbs pilot")
    return auto_cfg.gibbs_cooling, notes


def _build_pilot_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    auto_cfg: AutoOptimizeConfig,
    draws_factor: float = 1.0,
    tune_factor: float = 1.0,
    cooling_boost: float = 1.0,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    pilot = base_cfg.model_copy(deep=True)
    pilot.draws = max(4, int(round(auto_cfg.pilot_draws * draws_factor)))
    pilot.tune = max(0, int(round(auto_cfg.pilot_tune * tune_factor)))
    if base_cfg.draws != pilot.draws or base_cfg.tune != pilot.tune:
        notes.append("overrode draws/tune for pilot sweeps")
    pilot.save_trace = True
    if not base_cfg.save_trace:
        notes.append("enabled save_trace for pilot diagnostics")
    pilot.save_sequences = True
    if not base_cfg.save_sequences:
        notes.append("enabled save_sequences for pilot diagnostics")
    pilot.record_tune = False
    if base_cfg.record_tune:
        notes.append("disabled record_tune for pilot sweeps")
    pilot.progress_bar = auto_cfg.pilot_progress_bar
    pilot.progress_every = auto_cfg.pilot_progress_every
    if base_cfg.progress_bar != pilot.progress_bar or base_cfg.progress_every != pilot.progress_every:
        notes.append("adjusted progress output for pilots")

    if kind == "gibbs":
        pilot.chains = auto_cfg.pilot_chains_gibbs
        if base_cfg.chains != pilot.chains:
            notes.append("overrode chains for Gibbs pilot")
        pilot.optimiser.kind = "gibbs"
        cooling, cooling_notes = _derive_gibbs_cooling(base_cfg, auto_cfg)
        cooling, boost_notes = _boost_cooling(cooling, cooling_boost)
        pilot.optimiser.cooling = cooling
        notes.extend(cooling_notes + boost_notes)
        return pilot, notes

    if kind == "pt":
        pilot.optimiser.kind = "pt"
        beta, beta_notes = _derive_pt_beta(base_cfg, auto_cfg, chains=auto_cfg.pilot_chains_pt)
        pilot.chains = len(beta)
        if base_cfg.chains != pilot.chains:
            notes.append("overrode chains for PT pilot")
        cooling, boost_notes = _boost_cooling(CoolingGeometric(beta=beta), cooling_boost)
        pilot.optimiser.cooling = cooling
        notes.extend(beta_notes + boost_notes)
        return pilot, notes

    raise ValueError(f"Unknown auto-opt candidate kind '{kind}'")


def _build_final_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    auto_cfg: AutoOptimizeConfig,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    final_cfg = base_cfg.model_copy(deep=True)

    if kind == "gibbs":
        final_cfg.optimiser.kind = "gibbs"
        if base_cfg.optimiser.kind != "gibbs":
            cooling, cooling_notes = _derive_gibbs_cooling(base_cfg, auto_cfg)
            final_cfg.optimiser.cooling = cooling
            notes.extend(cooling_notes)
        return final_cfg, notes

    if kind == "pt":
        if final_cfg.chains < 2:
            raise ValueError("PT requires sample.chains >= 2 for the final run.")
        final_cfg.optimiser.kind = "pt"
        beta, beta_notes = _derive_pt_beta(base_cfg, auto_cfg, chains=final_cfg.chains)
        final_cfg.optimiser.cooling = CoolingGeometric(beta=beta)
        notes.extend(beta_notes)
        return final_cfg, notes

    raise ValueError(f"Unknown auto-opt candidate kind '{kind}'")


def _assess_candidate_quality(candidate: AutoOptCandidate, auto_cfg: AutoOptimizeConfig) -> list[str]:
    notes: list[str] = []
    if candidate.status == "fail":
        candidate.quality = "fail"
        return notes
    if candidate.rhat is None or candidate.ess is None or candidate.unique_fraction is None:
        candidate.quality = "fail"
        notes.append("missing pilot metrics required for auto-opt thresholds")
        return notes
    quality = "ok"
    if candidate.rhat > auto_cfg.max_rhat:
        quality = "warn"
        notes.append(f"rhat={candidate.rhat:.3f} exceeds auto_opt.max_rhat={auto_cfg.max_rhat:.3f}")
    if candidate.ess < auto_cfg.min_ess:
        quality = "warn"
        notes.append(f"ess={candidate.ess:.1f} below auto_opt.min_ess={auto_cfg.min_ess:.1f}")
    if candidate.unique_fraction < auto_cfg.min_unique_fraction:
        quality = "warn"
        notes.append(
            (
                f"unique_fraction={candidate.unique_fraction:.2f} below "
                f"auto_opt.min_unique_fraction={auto_cfg.min_unique_fraction:.2f}"
            )
        )
    if auto_cfg.max_unique_fraction is not None and candidate.unique_fraction > auto_cfg.max_unique_fraction:
        quality = "warn"
        notes.append(
            (
                f"unique_fraction={candidate.unique_fraction:.2f} above "
                f"auto_opt.max_unique_fraction={auto_cfg.max_unique_fraction:.2f}"
            )
        )
    candidate.quality = quality
    return notes


def _run_auto_optimize_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    auto_cfg: AutoOptimizeConfig,
) -> Path:
    logger.info("Auto-optimize enabled: running pilot sweeps (gibbs + pt).")

    def _run_pilots(
        *,
        label: str,
        draws_factor: float,
        tune_factor: float,
        cooling_boost: float,
    ) -> list[AutoOptCandidate]:
        pilot_candidates: list[AutoOptCandidate] = []
        for kind in ("gibbs", "pt"):
            pilot_cfg, notes = _build_pilot_sample_cfg(
                sample_cfg,
                kind=kind,
                auto_cfg=auto_cfg,
                draws_factor=draws_factor,
                tune_factor=tune_factor,
                cooling_boost=cooling_boost,
            )
            if notes:
                logger.info("Auto-opt %s %s overrides: %s", label, kind, "; ".join(notes))
            pilot_meta = {
                "mode": "pilot",
                "attempt": label,
                "candidate": kind,
                "draws": pilot_cfg.draws,
                "tune": pilot_cfg.tune,
                "chains": pilot_cfg.chains,
                "cooling_boost": cooling_boost,
            }
            pilot_run_dir = _run_sample_for_set(
                cfg,
                config_path,
                set_index=set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=pilot_cfg,
                stage="pilot",
                run_kind=f"auto_opt_{label}_{kind}",
                auto_opt_meta=pilot_meta,
            )
            try:
                candidate = _evaluate_pilot_run(pilot_run_dir, kind)
            except Exception as exc:
                logger.warning("Auto-opt %s %s failed: %s", label, kind, exc)
                candidate = AutoOptCandidate(
                    kind=kind,
                    run_dir=pilot_run_dir,
                    best_score=None,
                    rhat=None,
                    ess=None,
                    unique_fraction=None,
                    status="fail",
                    quality="fail",
                    warnings=[str(exc)],
                    diagnostics={},
                )
            pilot_candidates.append(candidate)
        return pilot_candidates

    candidates = _run_pilots(label="pilot", draws_factor=1.0, tune_factor=1.0, cooling_boost=1.0)

    def _log_candidate(candidate: AutoOptCandidate, *, label: str) -> None:
        logger.info(
            "Auto-opt %s %s: status=%s quality=%s best_score=%s rhat=%s ess=%s unique_fraction=%s",
            label,
            candidate.kind,
            candidate.status,
            candidate.quality,
            f"{candidate.best_score:.3f}" if candidate.best_score is not None else "n/a",
            f"{candidate.rhat:.3f}" if candidate.rhat is not None else "n/a",
            f"{candidate.ess:.1f}" if candidate.ess is not None else "n/a",
            f"{candidate.unique_fraction:.2f}" if candidate.unique_fraction is not None else "n/a",
        )
        if candidate.warnings:
            logger.warning("Auto-opt %s %s warnings: %s", label, candidate.kind, "; ".join(candidate.warnings))

    for candidate in candidates:
        candidate_notes = _assess_candidate_quality(candidate, auto_cfg)
        if candidate_notes:
            candidate.warnings.extend(candidate_notes)
        _log_candidate(candidate, label="pilot")

    ok_candidates = [candidate for candidate in candidates if candidate.quality == "ok"]
    if not ok_candidates and auto_cfg.retry_on_warn:
        logger.info(
            "Auto-optimize: no pilot met thresholds; retrying with cooler schedule (boost=%sx).",
            auto_cfg.cooling_boost,
        )
        retry_candidates = _run_pilots(
            label="retry",
            draws_factor=auto_cfg.retry_draws_factor,
            tune_factor=auto_cfg.retry_tune_factor,
            cooling_boost=auto_cfg.cooling_boost,
        )
        for candidate in retry_candidates:
            candidate_notes = _assess_candidate_quality(candidate, auto_cfg)
            if candidate_notes:
                candidate.warnings.extend(candidate_notes)
            _log_candidate(candidate, label="retry")
        candidates.extend(retry_candidates)
        ok_candidates = [candidate for candidate in candidates if candidate.quality == "ok"]

    if not ok_candidates:
        summary = ", ".join(f"{c.kind}:{c.quality}" for c in candidates)
        raise ValueError(
            "Auto-optimize failed: no pilot candidate met quality thresholds. "
            f"Candidates={summary}. Adjust auto_opt thresholds or cooling."
        )

    winner = _select_auto_opt_candidate(ok_candidates)
    logger.info(
        "Auto-optimize selected %s (best_score=%s, rhat=%s, ess=%s, unique_fraction=%s).",
        winner.kind,
        f"{winner.best_score:.3f}" if winner.best_score is not None else "n/a",
        f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
        f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
        f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
    )

    final_cfg, notes = _build_final_sample_cfg(sample_cfg, kind=winner.kind, auto_cfg=auto_cfg)
    if notes:
        logger.info("Auto-opt final overrides: %s", "; ".join(notes))

    decision_payload = {
        "mode": "final",
        "selected": winner.kind,
        "thresholds": {
            "max_rhat": auto_cfg.max_rhat,
            "min_ess": auto_cfg.min_ess,
            "min_unique_fraction": auto_cfg.min_unique_fraction,
            "max_unique_fraction": auto_cfg.max_unique_fraction,
        },
        "candidates": [
            {
                "kind": candidate.kind,
                "run": candidate.run_dir.name,
                "status": candidate.status,
                "quality": candidate.quality,
                "best_score": candidate.best_score,
                "rhat": candidate.rhat,
                "ess": candidate.ess,
                "unique_fraction": candidate.unique_fraction,
            }
            for candidate in candidates
        ],
    }
    return _run_sample_for_set(
        cfg,
        config_path,
        set_index=set_index,
        tfs=tfs,
        lockmap=lockmap,
        sample_cfg=final_cfg,
        stage="sample",
        run_kind="auto_opt_final",
        auto_opt_meta=decision_payload,
    )


def _evaluate_pilot_run(run_dir: Path, kind: str) -> AutoOptCandidate:
    manifest = load_manifest(run_dir)
    status_payload = json.loads(status_path(run_dir).read_text()) if status_path(run_dir).exists() else {}
    best_score = status_payload.get("best_score")
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
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=trace_idata,
        sequences_df=seq_df,
        elites_df=elites_df,
        tf_names=tf_names,
        optimizer=manifest.get("optimizer", {}),
        optimizer_stats=manifest.get("optimizer_stats", {}),
        sample_meta=None,
    )
    metrics = diagnostics.get("metrics", {}) if isinstance(diagnostics, dict) else {}
    trace_metrics = metrics.get("trace", {})
    seq_metrics = metrics.get("sequences", {})
    rhat = trace_metrics.get("rhat")
    ess = trace_metrics.get("ess")
    unique_fraction = seq_metrics.get("unique_fraction")
    status = diagnostics.get("status", "warn") if isinstance(diagnostics, dict) else "warn"
    warnings = diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []

    missing = []
    if best_score is None:
        warnings = list(warnings)
        warnings.append("Missing pilot metric: best_score")
    if rhat is None:
        missing.append("rhat")
    if ess is None:
        missing.append("ess")
    if unique_fraction is None:
        missing.append("unique_fraction")
    if missing:
        warnings = list(warnings)
        warnings.append(f"Missing pilot metrics: {', '.join(missing)}")
        status = "fail"

    return AutoOptCandidate(
        kind=kind,
        run_dir=run_dir,
        best_score=best_score,
        rhat=rhat,
        ess=ess,
        unique_fraction=unique_fraction,
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
    )


def _select_auto_opt_candidate(candidates: list[AutoOptCandidate]) -> AutoOptCandidate:
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates")

    ranked: list[tuple[tuple[float, float, float, float, float], AutoOptCandidate]] = []
    status_rank = {"ok": 2, "warn": 1, "fail": 0}
    for candidate in candidates:
        score = candidate.best_score if candidate.best_score is not None else float("-inf")
        ess = candidate.ess if candidate.ess is not None else float("-inf")
        uniq = candidate.unique_fraction if candidate.unique_fraction is not None else float("-inf")
        rhat = candidate.rhat
        rhat_score = -abs(rhat - 1.0) if rhat is not None else float("-inf")
        rank = (
            float(status_rank.get(candidate.quality, status_rank.get(candidate.status, 0))),
            float(score),
            float(ess),
            float(uniq),
            float(rhat_score),
        )
        ranked.append((rank, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    winner = ranked[0][1]
    if winner.status == "fail":
        raise ValueError("Auto-optimize failed: all pilot candidates reported missing diagnostics.")
    return winner


def _run_sample_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
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
    )
    ensure_run_dirs(out_dir, meta=True, artifacts=True, live=sample_cfg.live_metrics)
    logger.info("=== RUN %s: %s ===", stage.upper(), out_dir)
    logger.debug("Full sample config: %s", sample_cfg.model_dump_json())

    metrics_path = live_metrics_path(out_dir) if sample_cfg.live_metrics else None
    status_writer = RunStatusWriter(
        path=status_path(out_dir),
        stage=stage,
        run_dir=out_dir,
        metrics_path=metrics_path,
        payload={
            "config_path": str(config_path.resolve()),
            "status_message": "initializing",
            "regulator_set": {"index": set_index, "tfs": tfs},
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
        draws=sample_cfg.draws,
        tune=sample_cfg.tune,
        chains=sample_cfg.chains,
        optimiser=sample_cfg.optimiser.kind,
    )

    # 1) LOAD all required PWMs
    store = _store(cfg, config_path)
    pwms: dict[str, PWM] = {}
    for tf in sorted(tfs):
        logger.debug("  Loading PWM for %s", tf)
        entry = lockmap.get(tf)
        if entry is None:
            raise ValueError(f"Missing lock entry for TF '{tf}'")
        ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
        pwms[tf] = store.get_pwm(ref)

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
    scale = sample_cfg.optimiser.scorer_scale
    logger.info("Using scorer_scale = %r", scale)
    if scale == "consensus-neglop-sum":
        logger.debug("Building Scorer with inner scale='logp' in order to feed SequenceEvaluator(consensus-neglop-sum)")
        scorer = Scorer(
            pwms,
            bidirectional=sample_cfg.bidirectional,
            scale="logp",  # produce pure neglogp per PWM
            background=(0.25, 0.25, 0.25, 0.25),
        )
        evaluator = SequenceEvaluator(
            pwms=pwms,
            scale="consensus-neglop-sum",
            combiner=lambda vs: sum(vs),
            bidirectional=sample_cfg.bidirectional,
            background=(0.25, 0.25, 0.25, 0.25),
        )
    else:
        logger.debug("Building Scorer and SequenceEvaluator with scale=%r", scale)
        scorer = Scorer(
            pwms,
            bidirectional=sample_cfg.bidirectional,
            scale=scale,
            background=(0.25, 0.25, 0.25, 0.25),
        )
        evaluator = SequenceEvaluator(
            pwms=pwms,
            scale=scale,
            combiner=None,  # defaults to min(...) if scale ∈ {"llr","logp"}
            bidirectional=sample_cfg.bidirectional,
            background=(0.25, 0.25, 0.25, 0.25),
        )

    logger.info("Scorer and SequenceEvaluator instantiated")
    logger.debug("  Scorer.scale = %r", scorer.scale)
    logger.debug("  SequenceEvaluator.scale = %r", evaluator._scale)
    logger.debug("  SequenceEvaluator.combiner = %r", evaluator._combiner)

    # 3) FLATTEN optimiser config for Gibbs/PT
    optblock = sample_cfg.optimiser
    opt_cfg: dict[str, object] = {
        "draws": sample_cfg.draws,
        "tune": sample_cfg.tune,
        "chains": sample_cfg.chains,
        "min_dist": sample_cfg.min_dist,
        "top_k": sample_cfg.top_k,
        "record_tune": sample_cfg.record_tune,
        "progress_bar": sample_cfg.progress_bar,
        "progress_every": sample_cfg.progress_every,
        **sample_cfg.moves.model_dump(),
        **optblock.cooling.model_dump(),
        "swap_prob": optblock.swap_prob,
    }

    if optblock.cooling.kind == "geometric":
        logger.info(
            "β-ladder (%d levels): %s",
            len(optblock.cooling.beta),
            ", ".join(f"{b:g}" for b in optblock.cooling.beta),
        )

    logger.debug("Optimizer config flattened: %s", opt_cfg)

    # 4) INSTANTIATE OPTIMIZER (Gibbs or PT), passing in init_cfg and pwms
    from dnadesign.cruncher.core.optimizers.registry import get_optimizer

    optimizer_factory = get_optimizer(sample_cfg.optimiser.kind)
    logger.info("Instantiating optimizer: %s", sample_cfg.optimiser.kind)
    rng = np.random.default_rng(sample_cfg.seed + set_index - 1)
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
    logger.info("Starting MCMC sampling …")
    try:
        optimizer.optimise()
    except Exception as exc:  # pragma: no cover - ensures run status is updated on failure
        status_writer.finish(status="failed", error=str(exc))
        raise
    if status_writer is not None and hasattr(optimizer, "best_score"):
        best_meta = getattr(optimizer, "best_meta", None)
        status_writer.update(
            best_score=getattr(optimizer, "best_score", None),
            best_chain=(best_meta[0] + 1) if best_meta else None,
            best_draw=(best_meta[1]) if best_meta else None,
        )
    logger.info("MCMC sampling complete.")
    status_writer.update(status_message="sampling complete")
    status_writer.update(status_message="saving_artifacts")

    # 6) SAVE enriched config
    _save_config(cfg, out_dir, config_path, tfs=tfs, set_index=set_index)

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
    if sample_cfg.save_trace and hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.utils.traces import save_trace

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
        logger.info("Saved MCMC trace → %s", trace_path(out_dir).relative_to(out_dir.parent))
    elif not sample_cfg.save_trace:
        logger.info("Skipping trace.nc (sample.save_trace=false)")

    # 8) SAVE sequences.parquet (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.save_sequences
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
                if draw_i < sample_cfg.tune:
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
        logger.info(
            "Saved all sequences with per-TF scores → %s",
            seq_parquet.relative_to(out_dir.parent),
        )

    # 9)  BUILD elites list — keep draws whose ∑ normalised ≥ pwm_sum_threshold
    #     + enforce a per-sequence Hamming-distance diversity filter
    thr_norm: float = sample_cfg.pwm_sum_threshold  # e.g. 1.50
    min_dist: int = sample_cfg.min_dist  # e.g. 1
    raw_elites: list[tuple[np.ndarray, int, int, float, dict[str, float]]] = []
    norm_sums: list[float] = []  # diagnostics only

    for (chain_id, draw_idx), seq_arr, per_tf_map in zip(
        optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
    ):
        # ---------- per-PWM normalisation --------------------------
        #   raw_llr  →   frac = (raw_llr – μ_null) / (llr_consensus – μ_null)
        #   • μ_null  = null_mean  (expected background)
        #   • frac < 0  ⇒ treat as 0  (below background contributes nothing)
        #   • frac may exceed 1 if a window scores better than “column max”
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
        logger.info("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        logger.info(
            "Threshold %.2f ⇒ ~%.0f%%-of-consensus on average per TF (%d regulators)",
            thr_norm,
            avg_thr_pct,
            n_tf,
        )
        logger.info(
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

    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int((a != b).sum())

    for tpl in raw_elites:
        seq_arr = tpl[0]
        if all(_hamming(seq_arr, s) >= min_dist for s in kept_seqs):
            kept_elites.append(tpl)
            kept_seqs.append(seq_arr)

    if min_dist > 0:
        logger.info(
            "Diversity filter (≥%d mismatches): kept %d / %d candidates",
            min_dist,
            len(kept_elites),
            len(raw_elites),
        )
    else:
        kept_elites = raw_elites  # no filtering

    # serialise elites
    elites: list[dict[str, object]] = []
    want_cons = bool(getattr(sample_cfg, "include_consensus_in_elites", False))

    for rank, (seq_arr, chain_id, draw_idx, total_norm, per_tf_map) in enumerate(kept_elites, 1):
        seq_str = SequenceState(seq_arr).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}

        for tf_name in scorer.tf_names:
            # best site in this *sequence*
            raw_llr, offset, strand = scorer.best_llr(seq_arr, tf_name)
            width = scorer.pwm_width(tf_name)
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

    logger.info(
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
            yield row

    _write_parquet_rows(parquet_path, _elite_rows(), chunk_size=2000)
    logger.info("Saved elites Parquet → %s", parquet_path.relative_to(out_dir.parent))

    json_path = elites_json_path(out_dir)
    json_path.write_text(json.dumps(elites, indent=2))
    logger.info("Saved elites JSON → %s", json_path.relative_to(out_dir.parent))

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
    yaml_path = elites_yaml_path(out_dir)
    with yaml_path.open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    logger.info("Saved metadata → %s", yaml_path.relative_to(out_dir.parent))

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
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    catalog = CatalogIndex.load(catalog_root)
    lock_root = catalog_root / "locks"
    lock_path = lock_root / f"{config_path.stem}.lock.json"
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
            "seed": sample_cfg.seed,
            "seed_effective": sample_cfg.seed + set_index - 1,
            "record_tune": sample_cfg.record_tune,
            "save_trace": sample_cfg.save_trace,
            "regulator_set": {"index": set_index, "tfs": tfs},
            "run_group": run_group_label(tfs, set_index),
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
            "optimizer": {
                "kind": sample_cfg.optimiser.kind,
                "scorer_scale": sample_cfg.optimiser.scorer_scale,
                "cooling": sample_cfg.optimiser.cooling.model_dump(),
                "swap_prob": sample_cfg.optimiser.swap_prob,
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
    logger.info("Wrote run manifest → %s", manifest_path.relative_to(out_dir.parent))
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
    ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
    lockmap = _lockmap_for(cfg, config_path)
    statuses = target_statuses(cfg=cfg, config_path=config_path)
    sample_cfg = cfg.sample
    auto_cfg = sample_cfg.auto_opt
    if auto_opt_override is not None:
        if auto_opt_override:
            if auto_cfg is None:
                auto_cfg = AutoOptimizeConfig()
            else:
                auto_cfg = auto_cfg.model_copy(update={"enabled": True})
        else:
            auto_cfg = None
    use_auto_opt = auto_cfg is not None and auto_cfg.enabled
    for set_index, group in enumerate(regulator_sets(cfg.regulator_sets), start=1):
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
            _run_auto_optimize_for_set(
                cfg,
                config_path,
                set_index=set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=sample_cfg,
                auto_cfg=auto_cfg,
            )
        else:
            _run_sample_for_set(
                cfg,
                config_path,
                set_index=set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=sample_cfg,
            )
