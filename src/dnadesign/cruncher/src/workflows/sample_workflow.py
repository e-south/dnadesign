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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
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
from dnadesign.cruncher.utils.labels import (
    build_run_name,
    format_regulator_slug,
    regulator_sets,
)
from dnadesign.cruncher.utils.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from dnadesign.cruncher.utils.run_status import RunStatusWriter

logger = logging.getLogger(__name__)


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        config_path.parent / cfg.motif_store.catalog_root,
        pwm_source=cfg.motif_store.pwm_source,
        site_kinds=cfg.motif_store.site_kinds,
        combine_sites=cfg.motif_store.combine_sites,
        site_window_lengths=cfg.motif_store.site_window_lengths,
        site_window_center=cfg.motif_store.site_window_center,
        min_sites_for_pwm=cfg.motif_store.min_sites_for_pwm,
        allow_low_sites=cfg.motif_store.allow_low_sites,
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
    Save the exact Pydantic-validated config into <batch_dir>/config_used.yaml,
    plus, for each TF:
      - alphabet: ["A","C","G","T"]
      - pwm_matrix: a list of [p_A, p_C, p_G, p_T] for each position
      - consensus: consensus sequence string
    """
    cfg_path = batch_dir / "config_used.yaml"
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


def _run_sample_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    tfs: list[str],
    lockmap: dict[str, object],
) -> None:
    """
    Run MCMC sampler, save enriched config, trace.nc, sequences.parquet (including per-TF scores),
    and elites.json. Each chain gets its own independent seed (random/consensus/consensus_mix).
    """
    if cfg.sample is None:
        raise ValueError("sample section is required for sample")
    sample_cfg = cfg.sample
    base_out = config_path.parent / Path(cfg.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out / build_run_name("sample", tfs, set_index=set_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== RUN SAMPLE: %s ===", out_dir)
    logger.debug("Full sample config: %s", sample_cfg.model_dump_json())

    metrics_path = out_dir / "live_metrics.jsonl" if sample_cfg.live_metrics else None
    status_writer = RunStatusWriter(
        path=out_dir / "run_status.json",
        stage="sample",
        run_dir=out_dir,
        metrics_path=metrics_path,
        payload={
            "config_path": str(config_path.resolve()),
            "status_message": "initializing",
            "regulator_set": {"index": set_index, "tfs": tfs},
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
    if cfg.sample.init.length < max_w:
        names = ", ".join(f"{tf}:{pwms[tf].length}" for tf in sorted(pwms))
        raise ValueError(
            f"init.length={cfg.sample.init.length} is shorter than the widest PWM (max={max_w}). "
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
    logger.info("MCMC sampling complete.")
    status_writer.update(status_message="sampling complete")
    status_writer.update(status_message="saving_artifacts")

    # 6) SAVE enriched config
    _save_config(cfg, out_dir, config_path, tfs=tfs, set_index=set_index)

    artifacts: list[dict[str, object]] = [
        artifact_entry(
            out_dir / "config_used.yaml",
            out_dir,
            kind="config",
            label="Resolved config (config_used.yaml)",
            stage="sample",
        )
    ]

    # 7) SAVE trace.nc
    if sample_cfg.save_trace and hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.utils.traces import save_trace

        save_trace(optimizer.trace_idata, out_dir / "trace.nc")
        artifacts.append(
            artifact_entry(
                out_dir / "trace.nc",
                out_dir,
                kind="trace",
                label="Trace (NetCDF)",
                stage="sample",
            )
        )
        logger.info("Saved MCMC trace → %s", (out_dir / "trace.nc").relative_to(out_dir.parent))
    elif not sample_cfg.save_trace:
        logger.info("Skipping trace.nc (sample.save_trace=false)")

    # 8) SAVE sequences.parquet (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.save_sequences
        and hasattr(optimizer, "all_samples")
        and hasattr(optimizer, "all_meta")
        and hasattr(optimizer, "all_scores")
    ):
        seq_parquet = out_dir / "sequences.parquet"
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
                stage="sample",
            )
        )
        logger.info(
            "Saved all sequences with per-TF scores → %s",
            seq_parquet.relative_to(out_dir.parent),
        )

    # 9)  BUILD elites list — keep draws whose ∑ normalised ≥ pwm_sum_threshold
    #     + enforce a per-sequence Hamming-distance diversity filter
    thr_norm: float = cfg.sample.pwm_sum_threshold  # e.g. 1.50
    min_dist: int = cfg.sample.min_dist  # e.g. 1
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
    want_cons = bool(getattr(cfg.sample, "include_consensus_in_elites", False))

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
    parquet_path = out_dir / "elites.parquet"

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

    json_path = out_dir / "elites.json"
    json_path.write_text(json.dumps(elites, indent=2))
    logger.info("Saved elites JSON → %s", json_path.relative_to(out_dir.parent))

    # 2)  .yaml run-metadata -----------------------------------------------
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "threshold_norm_sum": thr_norm,
        "min_hamming_dist": min_dist,
        "tf_label": tf_label,
        "sequence_length": cfg.sample.init.length,
        #  you can inline the full cfg if you prefer – this keeps it concise
        "config_file": str((out_dir / "config_used.yaml").resolve()),
        "regulator_set": {"index": set_index, "tfs": tfs},
    }
    yaml_path = out_dir / "elites.yaml"
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
                stage="sample",
            ),
            artifact_entry(
                json_path,
                out_dir,
                kind="json",
                label="Elite sequences (JSON)",
                stage="sample",
            ),
            artifact_entry(
                yaml_path,
                out_dir,
                kind="metadata",
                label="Elite metadata (YAML)",
                stage="sample",
            ),
        ]
    )

    # 10) RUN MANIFEST (for reporting + provenance)
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    catalog = CatalogIndex.load(catalog_root)
    lock_root = catalog_root / "locks"
    lock_path = lock_root / f"{config_path.stem}.lock.json"
    manifest = build_run_manifest(
        stage="sample",
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


def run_sample(cfg: CruncherConfig, config_path: Path) -> None:
    """
    Run MCMC sampler, save enriched config, trace.nc, sequences.parquet (including per-TF scores),
    and elites.json. Each regulator set is sampled independently for clear provenance.
    """
    if cfg.sample is None:
        raise ValueError("sample section is required for sample")
    ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
    lockmap = _lockmap_for(cfg, config_path)
    statuses = target_statuses(cfg=cfg, config_path=config_path)
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
        _run_sample_for_set(cfg, config_path, set_index=set_index, tfs=tfs, lockmap=lockmap)
