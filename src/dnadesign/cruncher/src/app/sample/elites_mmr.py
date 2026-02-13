"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/elites_mmr.py

MMR-based elite pool and metadata helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from dnadesign.cruncher.app.sample.diagnostics import _EliteCandidate, _norm_map_for_elites
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.selection.mmr import (
    MmrCandidate,
    select_mmr_elites,
    select_score_elites,
    tfbs_cores_from_hits,
)
from dnadesign.cruncher.core.sequence import canon_int
from dnadesign.cruncher.core.state import SequenceState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElitePoolResult:
    raw_elites: list[_EliteCandidate]
    norm_sums: list[float]
    min_norms: list[float]
    total_draws_seen: int


@dataclass(frozen=True)
class EliteSelectionResult:
    kept_elites: list[_EliteCandidate]
    mmr_meta_rows: list[dict[str, object]] | None
    mmr_summary: dict[str, object] | None
    kept_after_mmr: int


def build_elite_pool(
    *,
    optimizer: object,
    evaluator: SequenceEvaluator,
    scorer: Scorer,
    sample_cfg: SampleConfig,
    beta_softmin_final: float | None,
) -> ElitePoolResult:
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

        combined_score = evaluator.combined_from_scores(
            per_tf_map,
            beta=beta_softmin_final,
            length=seq_arr.size,
        )
        per_tf_hits: dict[str, dict[str, object]] = {}
        for tf_name in scorer.tf_names:
            hit = scorer.best_hit(seq_arr, tf_name)
            per_tf_hits[tf_name] = {
                **hit,
                "best_score_scaled": float(per_tf_map[tf_name]),
                "best_score_norm": float(norm_map.get(tf_name, 0.0)),
            }
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
                per_tf_hits=per_tf_hits,
            )
        )

    return ElitePoolResult(
        raw_elites=raw_elites,
        norm_sums=norm_sums,
        min_norms=min_norms,
        total_draws_seen=total_draws_seen,
    )


def select_elites_mmr(
    *,
    raw_elites: list[_EliteCandidate],
    elite_k: int,
    pool_size: int,
    scorer: Scorer,
    pwms: dict[str, PWM],
    dsdna_mode: bool,
    diversity: float,
    sample_sequence_length: int,
    cooling_config: object,
) -> EliteSelectionResult:
    kept_elites: list[_EliteCandidate] = []
    kept_after_mmr = 0
    mmr_meta_rows: list[dict[str, object]] | None = None
    mmr_summary: dict[str, object] | None = None

    if elite_k > 0 and raw_elites:
        pool_size = max(1, pool_size)
        requested_pool_size = int(pool_size)
        diversity_value = float(diversity)
        score_only = diversity_value <= 0.0
        score_weight = 1.0 if score_only else float(max(0.0, 1.0 - diversity_value))
        diversity_weight = 0.0 if score_only else float(diversity_value)
        metric = "none" if score_only else "hybrid"
        min_hamming_eff = None
        min_core_hamming_eff = None
        if not score_only:
            min_hamming_eff = int(round(diversity_value * max(0, int(sample_sequence_length) // 4)))
            core_width = sum(int(pwms[tf].length) for tf in scorer.tf_names)
            min_core_hamming_eff = int(round(diversity_value * max(0, core_width // 4)))

        def _candidate_relevance_value(cand: MmrCandidate) -> float:
            if score_only:
                return float(cand.combined_score)
            return float(cand.min_norm)

        def _cooling_stage_index(draw_idx: int) -> int:
            cfg = cooling_config if isinstance(cooling_config, dict) else {}
            if str(cfg.get("kind", "")).strip().lower() != "piecewise":
                return 0
            stages = cfg.get("stages")
            if not isinstance(stages, list) or not stages:
                return 0
            for idx, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    continue
                sweeps = stage.get("sweeps")
                if isinstance(sweeps, (int, float)) and draw_idx <= int(sweeps):
                    return idx
            return max(0, len(stages) - 1)

        def _build_pool(candidates: list[MmrCandidate], max_size: int) -> list[MmrCandidate]:
            max_size = max(1, int(max_size))
            ordered = sorted(
                candidates,
                key=lambda cand: (_candidate_relevance_value(cand), f"{cand.chain_id}:{cand.draw_idx}"),
                reverse=True,
            )
            if score_only:
                return ordered[:max_size]
            buckets: dict[tuple[int, int], list[MmrCandidate]] = {}
            for cand in ordered:
                key = (int(cand.chain_id), _cooling_stage_index(int(cand.draw_idx)))
                buckets.setdefault(key, []).append(cand)
            if not buckets:
                return []
            bucket_keys = sorted(buckets.keys())
            cursor = {key: 0 for key in bucket_keys}
            pooled: list[MmrCandidate] = []
            while len(pooled) < max_size:
                progressed = False
                for key in bucket_keys:
                    idx = cursor[key]
                    bucket = buckets[key]
                    if idx >= len(bucket):
                        continue
                    pooled.append(bucket[idx])
                    cursor[key] = idx + 1
                    progressed = True
                    if len(pooled) >= max_size:
                        break
                if not progressed:
                    break
            return pooled

        mmr_candidates = [
            MmrCandidate(
                seq_arr=cand.seq_arr,
                chain_id=cand.chain_id,
                draw_idx=cand.draw_idx,
                combined_score=cand.combined_score,
                min_norm=cand.min_norm,
                sum_norm=cand.sum_norm,
                per_tf_map=cand.per_tf_map,
                norm_map=cand.norm_map,
            )
            for cand in raw_elites
        ]
        raw_by_id = {f"{cand.chain_id}:{cand.draw_idx}": cand for cand in raw_elites}
        pool_size_active = int(pool_size)
        max_pool_size = min(len(mmr_candidates), 50_000)
        pool_expansions = 0
        result = None
        while True:
            mmr_pool = _build_pool(mmr_candidates, pool_size_active)
            if score_only:
                result = select_score_elites(
                    mmr_pool,
                    k=elite_k,
                    pool_size=pool_size_active,
                    dsdna=dsdna_mode,
                )
            else:
                core_maps = {
                    f"{cand.chain_id}:{cand.draw_idx}": tfbs_cores_from_hits(
                        cand.seq_arr,
                        per_tf_hits=raw_by_id[f"{cand.chain_id}:{cand.draw_idx}"].per_tf_hits,
                        tf_names=scorer.tf_names,
                    )
                    for cand in mmr_pool
                }
                result = select_mmr_elites(
                    mmr_pool,
                    k=elite_k,
                    pool_size=pool_size_active,
                    alpha=score_weight,
                    relevance="min_tf_score",
                    dsdna=dsdna_mode,
                    tf_names=scorer.tf_names,
                    pwms=pwms,
                    core_maps=core_maps,
                    distance_metric=metric,
                    min_hamming_bp=min_hamming_eff,
                    min_core_hamming_bp=min_core_hamming_eff,
                    constraint_policy="relax",
                    relax_step_bp=1,
                    relax_min_bp=0,
                )
            if len(result.selected) >= elite_k:
                break
            if pool_size_active >= max_pool_size:
                break
            pool_size_active = min(max_pool_size, pool_size_active * 2)
            pool_expansions += 1
        if result is None:
            raise RuntimeError("MMR selection did not produce a result.")
        kept_elites = [raw_by_id[f"{cand.chain_id}:{cand.draw_idx}"] for cand in result.selected]
        kept_after_mmr = len(kept_elites)
        mmr_meta_rows = result.meta
        mmr_summary = {
            "pool_size": pool_size_active,
            "pool_size_initial": requested_pool_size,
            "pool_expansions": pool_expansions,
            "k": elite_k,
            "diversity": diversity_value,
            "score_weight": score_weight,
            "diversity_weight": diversity_weight,
            "selection_policy": "score_topk" if score_only else "mmr",
            "relevance": "joint_score" if score_only else "min_tf_score",
            "distance_metric": metric,
            "constraint_policy": "disabled" if score_only else "relax",
            "min_hamming_bp_requested": min_hamming_eff,
            "min_hamming_bp_final": result.min_hamming_bp_final,
            "min_core_hamming_bp_requested": min_core_hamming_eff,
            "min_core_hamming_bp_final": result.min_core_hamming_bp_final,
            "relax_steps_used": result.relax_steps_used,
            "pool_strategy": "top_score" if score_only else "stratified",
            "median_relevance_raw": result.median_relevance_raw,
            "mean_pairwise_distance": result.mean_pairwise_distance,
            "min_pairwise_distance": result.min_pairwise_distance,
        }

    return EliteSelectionResult(
        kept_elites=kept_elites,
        mmr_meta_rows=mmr_meta_rows,
        mmr_summary=mmr_summary,
        kept_after_mmr=kept_after_mmr,
    )


def build_elite_entries(
    candidates: list[_EliteCandidate],
    *,
    scorer: Scorer,
    sample_cfg: SampleConfig,
    want_consensus: bool,
    want_canonical: bool,
    meta_source: str,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    adapt_sweeps = int(sample_cfg.budget.tune)
    record_tune = bool(sample_cfg.output.include_tune_in_sequences)
    for rank, cand in enumerate(candidates, 1):
        seq_arr = cand.seq_arr
        seq_str = SequenceState(seq_arr).to_string()
        canonical_seq = None
        if want_canonical:
            canonical_seq = SequenceState(canon_int(seq_arr)).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}

        for tf_name in scorer.tf_names:
            hit = cand.per_tf_hits.get(tf_name)
            if not isinstance(hit, dict):
                raise ValueError(f"Missing best-hit metadata for TF '{tf_name}'.")
            offset = hit.get("offset")
            width = hit.get("width")
            strand = hit.get("strand")
            if not isinstance(offset, int) or not isinstance(width, int) or not isinstance(strand, str):
                raise ValueError(f"Invalid best-hit metadata for TF '{tf_name}'.")
            start_pos = int(offset) + 1
            strand_label = f"{strand}1"
            motif_diag = f"{start_pos}_[{strand_label}]_{width}"
            if want_consensus:
                consensus = scorer.consensus_sequence(tf_name)
            else:
                consensus = None
            per_tf_details[tf_name] = {
                **hit,
                "motif_diagram": motif_diag,
            }
            if want_consensus:
                per_tf_details[tf_name]["consensus"] = consensus

        draw_in_phase = cand.draw_idx
        if record_tune and cand.draw_idx >= adapt_sweeps:
            draw_in_phase = cand.draw_idx - adapt_sweeps

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
            "draw_in_phase": draw_in_phase,
            "per_tf": per_tf_details,
            "meta_type": "mcmc-elite",
            "meta_source": meta_source,
            "meta_date": datetime.now(timezone.utc).isoformat(),
        }
        if want_canonical:
            entry["canonical_sequence"] = canonical_seq
        entries.append(entry)
    return entries
