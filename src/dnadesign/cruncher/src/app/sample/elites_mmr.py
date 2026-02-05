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

from dnadesign.cruncher.app.sample.diagnostics import _elite_filter_passes, _EliteCandidate, _norm_map_for_elites
from dnadesign.cruncher.config.schema_v2 import SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.selection.mmr import MmrCandidate, select_mmr_elites, tfbs_cores_from_scorer
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
    filters = sample_cfg.elites.filters
    min_per_tf_norm = filters.min_per_tf_norm
    require_all = filters.require_all_tfs_over_min_norm
    pwm_sum_min = filters.pwm_sum_min
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
    sample_cfg: SampleConfig,
    scorer: Scorer,
    pwms: dict[str, PWM],
    dsdna_mode: bool,
) -> EliteSelectionResult:
    selection_cfg = sample_cfg.elites.selection
    kept_elites: list[_EliteCandidate] = []
    kept_after_mmr = 0
    mmr_meta_rows: list[dict[str, object]] | None = None
    mmr_summary: dict[str, object] | None = None

    if elite_k > 0 and raw_elites:
        if selection_cfg.pool_size < elite_k:
            logger.warning(
                "MMR pool_size=%d < elites.k=%d; using pool_size=%d.",
                selection_cfg.pool_size,
                elite_k,
                elite_k,
            )
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
        pool_size = max(selection_cfg.pool_size, elite_k)
        if selection_cfg.relevance == "min_per_tf_norm":
            mmr_pool = sorted(
                mmr_candidates,
                key=lambda cand: (cand.min_norm, f"{cand.chain_id}:{cand.draw_idx}"),
                reverse=True,
            )[:pool_size]
        else:
            mmr_pool = sorted(
                mmr_candidates,
                key=lambda cand: (cand.combined_score, f"{cand.chain_id}:{cand.draw_idx}"),
                reverse=True,
            )[:pool_size]
        core_maps = {
            f"{cand.chain_id}:{cand.draw_idx}": tfbs_cores_from_scorer(
                cand.seq_arr,
                scorer=scorer,
                tf_names=scorer.tf_names,
            )
            for cand in mmr_pool
        }
        result = select_mmr_elites(
            mmr_pool,
            k=elite_k,
            pool_size=selection_cfg.pool_size,
            alpha=selection_cfg.alpha,
            relevance=selection_cfg.relevance,
            dsdna=dsdna_mode,
            tf_names=scorer.tf_names,
            pwms=pwms,
            core_maps=core_maps,
        )
        kept_elites = [raw_by_id[f"{cand.chain_id}:{cand.draw_idx}"] for cand in result.selected]
        kept_after_mmr = len(kept_elites)
        mmr_meta_rows = result.meta
        mmr_summary = {
            "pool_size": selection_cfg.pool_size,
            "k": elite_k,
            "alpha": selection_cfg.alpha,
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
    for rank, cand in enumerate(candidates, 1):
        seq_arr = cand.seq_arr
        seq_str = SequenceState(seq_arr).to_string()
        canonical_seq = None
        if want_canonical:
            canonical_seq = SequenceState(canon_int(seq_arr)).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}
        norm_map = cand.norm_map
        per_tf_map = cand.per_tf_map

        for tf_name in scorer.tf_names:
            raw_llr, offset, strand = scorer.best_llr(seq_arr, tf_name)
            width = scorer.pwm_width(tf_name)
            if strand == "-":
                offset = len(seq_arr) - width - offset
            start_pos = offset + 1
            strand_label = f"{strand}1"
            motif_diag = f"{start_pos}_[{strand_label}]_{width}"
            if want_consensus:
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
            if want_consensus:
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
            "meta_source": meta_source,
            "meta_date": datetime.now(timezone.utc).isoformat(),
        }
        if want_canonical:
            entry["canonical_sequence"] = canonical_seq
        entries.append(entry)
    return entries
