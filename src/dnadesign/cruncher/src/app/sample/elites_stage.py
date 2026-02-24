"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/elites_stage.py

Select, validate, and persist elite outputs for sample runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dnadesign.cruncher.app.sample.artifacts import (
    _elite_hits_parquet_schema,
    _elite_parquet_schema,
    _write_parquet_rows,
)
from dnadesign.cruncher.app.sample.diagnostics import _EliteCandidate
from dnadesign.cruncher.app.sample.elites_mmr import (
    build_elite_entries,
    build_elite_pool,
    hydrate_candidate_hits,
    select_elites_mmr,
)
from dnadesign.cruncher.app.sample.preflight import RunError, _resolve_elite_pool_size
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_hits_path,
    elites_json_path,
    elites_mmr_meta_path,
    elites_path,
    elites_yaml_path,
)
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.labels import format_regulator_slug
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.sequence import canon_int


def _log_candidate_percentiles(
    *,
    norm_sums: list[float],
    min_norms: list[float],
    scorer: Scorer,
    run_logger: Callable[..., None],
) -> None:
    if norm_sums:
        p50, p90 = np.percentile(norm_sums, [50, 90])
        n_tf = scorer.pwm_count
        run_logger("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        run_logger(
            "Typical draw: med %.2f (≈ %.0f%%/TF); top-10%% %.2f (≈ %.0f%%/TF)",
            p50,
            100 * p50 / n_tf if n_tf else 0.0,
            p90,
            100 * p90 / n_tf if n_tf else 0.0,
        )
    if min_norms:
        p50_min, p90_min = np.percentile(min_norms, [50, 90])
        run_logger("Normalised-min percentiles |  median %.2f   90%% %.2f", p50_min, p90_min)


def _require_elite_candidates(
    *,
    elite_k: int,
    raw_elites: list[object],
    finish_failed: Callable[[Exception], None],
) -> None:
    if elite_k <= 0 or raw_elites:
        return
    finish_failed(
        RunError(
            "No elite candidates were generated from sampled draws. "
            "Increase cruncher.sample.sequence_length or cruncher.sample.budget.draws."
        )
    )


def _select_elites(
    *,
    raw_elites: list[object],
    elite_k: int,
    scorer: Scorer,
    pwms: dict[str, object],
    sample_cfg: SampleConfig,
    diversity_value: float,
) -> tuple[list[dict[str, object]], int, int, list[dict[str, object]] | None, dict[str, object] | None]:
    pool_size = _resolve_elite_pool_size(
        pool_size_cfg=sample_cfg.elites.select.pool_size,
        elite_k=elite_k,
        candidate_count=len(raw_elites),
    )
    selection_result = select_elites_mmr(
        raw_elites=raw_elites,
        elite_k=elite_k,
        pool_size=pool_size,
        scorer=scorer,
        pwms=pwms,
        dsdna_mode=bool(sample_cfg.objective.bidirectional),
        diversity=diversity_value,
        sample_sequence_length=int(sample_cfg.sequence_length),
        cooling_config=sample_cfg.optimizer.cooling.model_dump(mode="json"),
    )
    mmr_summary = selection_result.mmr_summary
    if isinstance(mmr_summary, dict):
        summary_pool = mmr_summary.get("pool_size")
        if isinstance(summary_pool, (int, float)):
            pool_size = int(summary_pool)
    return (
        selection_result.kept_elites,
        int(selection_result.kept_after_mmr),
        pool_size,
        selection_result.mmr_meta_rows,
        mmr_summary,
    )


def _require_requested_elite_count(
    *,
    elite_k: int,
    kept_elites: list[dict[str, object]],
    finish_failed: Callable[[Exception], None],
) -> None:
    if elite_k <= 0 or len(kept_elites) >= elite_k:
        return
    finish_failed(
        RunError(
            f"Elite selection returned {len(kept_elites)} candidates, "
            f"fewer than cruncher.sample.elites.k={elite_k}. "
            "Increase cruncher.sample.sequence_length or cruncher.sample.budget.draws."
        )
    )


def _validate_elite_uniqueness(
    *,
    elites: list[dict[str, object]],
    dedupe_key: str,
    finish_failed: Callable[[Exception], None],
) -> None:
    seen_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    for entry in elites:
        key_val = entry.get(dedupe_key)
        if not isinstance(key_val, str) or not key_val:
            finish_failed(RunError(f"Elite entry missing required '{dedupe_key}' for uniqueness validation."))
        if key_val in seen_keys:
            duplicate_keys.add(key_val)
        seen_keys.add(key_val)
    if duplicate_keys:
        finish_failed(
            RunError(
                "Elite selection produced duplicate sequences after MMR selection "
                f"(key={dedupe_key}, duplicates={len(duplicate_keys)})."
            )
        )


def _elite_rows_from(entries: list[dict[str, object]]) -> Iterable[dict[str, object]]:
    for entry in entries:
        row = dict(entry)
        per_tf = row.pop("per_tf", None)
        if per_tf is not None:
            row["per_tf_json"] = json.dumps(per_tf, sort_keys=True)
            for tf_name, details in per_tf.items():
                row[f"score_{tf_name}"] = details.get("best_score_scaled")
                row[f"norm_{tf_name}"] = details.get("best_score_norm")
        yield row


def _elite_hits_rows(
    *,
    entries: list[dict[str, object]],
    pwms: dict[str, object],
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
) -> Iterable[dict[str, object]]:
    for entry in entries:
        elite_id = entry.get("id")
        rank = entry.get("rank")
        chain_id = entry.get("chain")
        draw_idx = entry.get("draw_idx")
        per_tf = entry.get("per_tf")
        if not isinstance(per_tf, dict):
            raise ValueError("Elite entry missing per_tf metadata.")
        for tf_name, details in per_tf.items():
            if not isinstance(details, dict):
                raise ValueError(f"Elite per_tf details missing for '{tf_name}'.")
            start = details.get("best_start")
            if start is None:
                start = details.get("offset")
            if not isinstance(start, int):
                raise ValueError(f"Elite hit missing best_start for '{tf_name}'.")
            strand = details.get("strand")
            if not isinstance(strand, str):
                raise ValueError(f"Elite hit missing strand for '{tf_name}'.")
            width = details.get("width")
            pwm = pwms.get(tf_name)
            if pwm is None:
                raise ValueError(f"Missing PWM for TF '{tf_name}'.")
            pwm_width = int(pwm.length)
            core_width = int(width) if isinstance(width, int) else pwm_width
            yield {
                "elite_id": elite_id,
                "tf": tf_name,
                "rank": rank,
                "chain": chain_id,
                "draw_idx": draw_idx,
                "best_start": int(start),
                "best_core_offset": int(start),
                "best_strand": strand,
                "best_window_seq": details.get("best_window_seq"),
                "best_core_seq": details.get("best_core_seq"),
                "best_score_raw": details.get("best_score_raw"),
                "best_score_scaled": details.get("best_score_scaled"),
                "best_score_norm": details.get("best_score_norm"),
                "tiebreak_rule": details.get("best_hit_tiebreak"),
                "pwm_ref": pwm_ref_by_tf.get(tf_name),
                "pwm_hash": pwm_hash_by_tf.get(tf_name),
                "pwm_width": pwm_width,
                "core_width": core_width,
                "core_def_hash": core_def_by_tf.get(tf_name),
            }


def _hit_window_map(
    *,
    per_tf_hits: dict[str, dict[str, object]] | None,
    scorer: Scorer,
) -> dict[str, tuple[int, int, str]]:
    if not isinstance(per_tf_hits, dict):
        raise ValueError("Elite candidate missing per_tf_hits for postprocessing.")
    windows: dict[str, tuple[int, int, str]] = {}
    for tf_name in scorer.tf_names:
        hit = per_tf_hits.get(tf_name)
        if not isinstance(hit, dict):
            raise ValueError(f"Elite candidate missing hit metadata for TF '{tf_name}'.")
        start = hit.get("best_start")
        if start is None:
            start = hit.get("offset")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(start, int) or not isinstance(width, int) or not isinstance(strand, str):
            raise ValueError(f"Invalid hit metadata for TF '{tf_name}' in elite postprocessing.")
        windows[tf_name] = (int(start), int(width), str(strand))
    return windows


def _windows_cover_ownership(
    *,
    windows: dict[str, tuple[int, int, str]],
    seq_length: int,
) -> list[tuple[int, str]]:
    owners: list[list[str]] = [[] for _ in range(int(seq_length))]
    for tf_name, (start, width, _strand) in windows.items():
        end = int(start) + int(width)
        if start < 0 or width < 1 or end > seq_length:
            raise ValueError(f"Elite hit window is out of bounds for TF '{tf_name}' during postprocessing.")
        for pos in range(int(start), int(end)):
            owners[pos].append(tf_name)
    return [(idx, names[0]) for idx, names in enumerate(owners) if len(names) == 1]


def _hit_raw_score(*, per_tf_hits: dict[str, dict[str, object]], tf_name: str) -> float:
    hit = per_tf_hits.get(tf_name)
    if not isinstance(hit, dict):
        raise ValueError(f"Postprocess payload missing hit data for TF '{tf_name}'.")
    raw_score = hit.get("best_score_raw")
    if not isinstance(raw_score, (int, float)):
        raise ValueError(f"Postprocess payload missing raw hit score for TF '{tf_name}'.")
    return float(raw_score)


def _hits_match_contract(
    *,
    per_tf_hits: dict[str, dict[str, object]],
    expected_windows: dict[str, tuple[int, int, str]],
    left_shift: int = 0,
) -> bool:
    for tf_name, (expected_start, expected_width, expected_strand) in expected_windows.items():
        hit = per_tf_hits.get(tf_name)
        if not isinstance(hit, dict):
            return False
        start = hit.get("best_start")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(start, int) or not isinstance(width, int) or not isinstance(strand, str):
            return False
        if int(start) != int(expected_start) - int(left_shift):
            return False
        if int(width) != int(expected_width):
            return False
        if str(strand) != str(expected_strand):
            return False
    return True


def _hits_match_polish_contract(
    *,
    per_tf_hits: dict[str, dict[str, object]],
    expected_windows: dict[str, tuple[int, int, str]],
    owner_tf: str,
    owner_position: int,
) -> bool:
    for tf_name, (expected_start, expected_width, expected_strand) in expected_windows.items():
        hit = per_tf_hits.get(tf_name)
        if not isinstance(hit, dict):
            return False
        start = hit.get("best_start")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(start, int) or not isinstance(width, int) or not isinstance(strand, str):
            return False
        if int(start) != int(expected_start):
            return False
        if int(width) != int(expected_width):
            return False
        if str(strand) != str(expected_strand):
            return False
        if tf_name != owner_tf:
            continue
        owner_end = int(expected_start) + int(expected_width)
        if int(expected_start) < 0 or owner_end <= int(expected_start):
            return False
        if int(owner_position) < int(expected_start) or int(owner_position) >= owner_end:
            return False
    return True


def _candidate_payload(
    *,
    seq_arr: np.ndarray,
    scorer: Scorer,
) -> tuple[dict[str, float], dict[str, dict[str, object]], dict[str, float], float, float]:
    per_tf_map, per_tf_hits = scorer.compute_all_per_pwm_and_hits(seq_arr, int(seq_arr.size))
    norm_map = scorer.normalized_llr_map(seq_arr)
    for tf_name in scorer.tf_names:
        hit = per_tf_hits.get(tf_name)
        if not isinstance(hit, dict):
            raise ValueError(f"Postprocess payload missing hit data for TF '{tf_name}'.")
        scaled_value = per_tf_map.get(tf_name)
        norm_value = norm_map.get(tf_name)
        if scaled_value is None or norm_value is None:
            raise ValueError(f"Postprocess payload missing score values for TF '{tf_name}'.")
        hit["best_score_scaled"] = float(scaled_value)
        hit["best_score_norm"] = float(norm_value)
    min_norm = float(min(norm_map.values())) if norm_map else 0.0
    sum_norm = float(sum(norm_map.values())) if norm_map else 0.0
    return per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm


def _remaining_single_owner_polish_improvements(
    *,
    seq_arr: np.ndarray,
    per_tf_hits: dict[str, dict[str, object]],
    scorer: Scorer,
    eps: float = 1.0e-12,
) -> list[dict[str, object]]:
    expected_windows = _hit_window_map(per_tf_hits=per_tf_hits, scorer=scorer)
    single_owner_positions = _windows_cover_ownership(windows=expected_windows, seq_length=int(seq_arr.size))
    source = np.asarray(seq_arr, dtype=np.int8)
    remaining: list[dict[str, object]] = []
    for pos, tf_owner in single_owner_positions:
        current_base = int(source[pos])
        owner_score = _hit_raw_score(per_tf_hits=per_tf_hits, tf_name=tf_owner)
        for base in (0, 1, 2, 3):
            if int(base) == current_base:
                continue
            trial = source.copy()
            trial[pos] = int(base)
            _, trial_hits, _, _, _ = _candidate_payload(
                seq_arr=trial,
                scorer=scorer,
            )
            if not _hits_match_polish_contract(
                per_tf_hits=trial_hits,
                expected_windows=expected_windows,
                owner_tf=tf_owner,
                owner_position=int(pos),
            ):
                continue
            trial_owner_score = _hit_raw_score(per_tf_hits=trial_hits, tf_name=tf_owner)
            if trial_owner_score > owner_score + float(eps):
                start, width, strand = expected_windows[tf_owner]
                remaining.append(
                    {
                        "tf": str(tf_owner),
                        "position": int(pos),
                        "strand": str(strand),
                        "start": int(start),
                        "width": int(width),
                        "base_before": int(current_base),
                        "base_after": int(base),
                        "owner_raw_before": float(owner_score),
                        "owner_raw_after": float(trial_owner_score),
                    }
                )
                break
    return remaining


def _apply_edge_trim(
    *,
    candidate: _EliteCandidate,
    scorer: Scorer,
    left_trim: int,
    right_trim: int,
) -> None:
    if left_trim <= 0 and right_trim <= 0:
        return
    source = np.asarray(candidate.seq_arr, dtype=np.int8)
    end_idx = int(source.size) - int(right_trim) if int(right_trim) > 0 else int(source.size)
    if end_idx <= int(left_trim):
        raise ValueError("Elite edge trim would remove the full sequence.")
    expected_windows = _hit_window_map(per_tf_hits=candidate.per_tf_hits, scorer=scorer)
    trimmed = source[int(left_trim) : end_idx].copy()
    per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm = _candidate_payload(seq_arr=trimmed, scorer=scorer)
    if not _hits_match_contract(
        per_tf_hits=per_tf_hits,
        expected_windows=expected_windows,
        left_shift=int(left_trim),
    ):
        raise ValueError("Elite edge trim violated hit-window validity contract.")
    candidate.seq_arr = trimmed
    candidate.per_tf_map = per_tf_map
    candidate.per_tf_hits = per_tf_hits
    candidate.norm_map = norm_map
    candidate.min_norm = float(min_norm)
    candidate.sum_norm = float(sum_norm)


def _dedupe_postprocessed_candidates(
    *,
    candidates: list[_EliteCandidate],
    dsdna_mode: bool,
) -> tuple[list[_EliteCandidate], int]:
    deduped: list[_EliteCandidate] = []
    seen: set[bytes] = set()
    dropped = 0
    for cand in candidates:
        seq_arr = np.asarray(cand.seq_arr, dtype=np.int8)
        key_arr = canon_int(seq_arr) if dsdna_mode else seq_arr
        key = np.asarray(key_arr, dtype=np.int8).tobytes()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped, dropped


def _postprocess_elite_candidates(
    *,
    candidates: list[_EliteCandidate],
    scorer: Scorer,
    dsdna_mode: bool,
) -> tuple[list[_EliteCandidate], dict[str, int]]:
    stats = {
        "polish_edits": 0,
        "trim_left": 0,
        "trim_right": 0,
        "dedup_dropped": 0,
    }
    if not candidates:
        return candidates, stats

    for cand in candidates:
        seq_arr = np.asarray(cand.seq_arr, dtype=np.int8).copy()
        per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm = _candidate_payload(seq_arr=seq_arr, scorer=scorer)

        while True:
            expected_windows = _hit_window_map(per_tf_hits=per_tf_hits, scorer=scorer)
            single_owner_positions = _windows_cover_ownership(windows=expected_windows, seq_length=int(seq_arr.size))
            changed = False
            for pos, tf_owner in single_owner_positions:
                current_base = int(seq_arr[pos])
                owner_score = _hit_raw_score(per_tf_hits=per_tf_hits, tf_name=tf_owner)
                best_update: (
                    tuple[
                        np.ndarray,
                        dict[str, float],
                        dict[str, dict[str, object]],
                        dict[str, float],
                        float,
                        float,
                    ]
                    | None
                ) = None
                best_owner_score = owner_score
                for base in (0, 1, 2, 3):
                    if int(base) == current_base:
                        continue
                    trial = seq_arr.copy()
                    trial[pos] = int(base)
                    trial_per_tf_map, trial_hits, trial_norm_map, trial_min_norm, trial_sum_norm = _candidate_payload(
                        seq_arr=trial,
                        scorer=scorer,
                    )
                    if not _hits_match_polish_contract(
                        per_tf_hits=trial_hits,
                        expected_windows=expected_windows,
                        owner_tf=tf_owner,
                        owner_position=int(pos),
                    ):
                        continue
                    trial_owner_score = _hit_raw_score(per_tf_hits=trial_hits, tf_name=tf_owner)
                    if trial_owner_score > best_owner_score + 1.0e-12:
                        best_owner_score = trial_owner_score
                        best_update = (
                            trial,
                            trial_per_tf_map,
                            trial_hits,
                            trial_norm_map,
                            float(trial_min_norm),
                            float(trial_sum_norm),
                        )
                if best_update is None:
                    continue
                seq_arr, per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm = best_update
                stats["polish_edits"] += 1
                changed = True
            if not changed:
                break

        cand.seq_arr = np.asarray(seq_arr, dtype=np.int8)
        cand.per_tf_map = per_tf_map
        cand.per_tf_hits = per_tf_hits
        cand.norm_map = norm_map
        cand.min_norm = float(min_norm)
        cand.sum_norm = float(sum_norm)
        expected_windows = _hit_window_map(per_tf_hits=per_tf_hits, scorer=scorer)
        starts = [int(start) for start, _width, _strand in expected_windows.values()]
        ends = [int(start) + int(width) for start, width, _strand in expected_windows.values()]
        trim_left = min(starts) if starts else 0
        trim_right = max(0, int(seq_arr.size) - max(ends)) if ends else 0
        if trim_left > 0 or trim_right > 0:
            _apply_edge_trim(
                candidate=cand,
                scorer=scorer,
                left_trim=int(trim_left),
                right_trim=int(trim_right),
            )
            stats["trim_left"] += int(trim_left)
            stats["trim_right"] += int(trim_right)
        remaining = _remaining_single_owner_polish_improvements(
            seq_arr=np.asarray(cand.seq_arr, dtype=np.int8),
            per_tf_hits=cand.per_tf_hits,
            scorer=scorer,
        )
        if remaining:
            first = remaining[0]
            raise ValueError(
                "Elite polish convergence failed for chain=%d draw=%d tf=%s pos=%d strand=%s."
                % (
                    int(cand.chain_id),
                    int(cand.draw_idx),
                    str(first["tf"]),
                    int(first["position"]),
                    str(first["strand"]),
                )
            )

    deduped, dropped = _dedupe_postprocessed_candidates(candidates=candidates, dsdna_mode=bool(dsdna_mode))
    stats["dedup_dropped"] = int(dropped)
    return deduped, stats


def _write_elite_tables(
    *,
    out_dir: Path,
    tfs: list[str],
    elites: list[dict[str, object]],
    pwms: dict[str, object],
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
    want_canonical: bool,
) -> tuple[Path, Path]:
    parquet_path = elites_path(out_dir)
    elite_schema = _elite_parquet_schema(tfs, include_canonical=want_canonical)
    _write_parquet_rows(parquet_path, _elite_rows_from(elites), chunk_size=2000, schema=elite_schema)

    hits_path = elites_hits_path(out_dir)
    hits_schema = _elite_hits_parquet_schema()
    _write_parquet_rows(
        hits_path,
        _elite_hits_rows(
            entries=elites,
            pwms=pwms,
            pwm_ref_by_tf=pwm_ref_by_tf,
            pwm_hash_by_tf=pwm_hash_by_tf,
            core_def_by_tf=core_def_by_tf,
        ),
        chunk_size=2000,
        schema=hits_schema,
    )
    return parquet_path, hits_path


def _selection_fields(
    *,
    mmr_summary: dict[str, object] | None,
    diversity_value: float,
) -> tuple[str, str, str, float, float]:
    selection_policy = "mmr"
    relevance_label = "min_tf_score"
    pool_strategy = "stratified"
    score_weight = 1.0 - diversity_value
    diversity_weight = diversity_value
    if isinstance(mmr_summary, dict):
        selection_policy = str(mmr_summary.get("selection_policy") or selection_policy)
        relevance_label = str(mmr_summary.get("relevance") or relevance_label)
        pool_strategy = str(mmr_summary.get("pool_strategy") or pool_strategy)
        score_weight_meta = mmr_summary.get("score_weight")
        diversity_weight_meta = mmr_summary.get("diversity_weight")
        if score_weight_meta is not None:
            score_weight = float(score_weight_meta)
        if diversity_weight_meta is not None:
            diversity_weight = float(diversity_weight_meta)
    return selection_policy, relevance_label, pool_strategy, score_weight, diversity_weight


def _write_mmr_meta(*, out_dir: Path, mmr_meta_rows: list[dict[str, object]] | None) -> Path | None:
    if not mmr_meta_rows:
        return None
    mmr_meta_path = elites_mmr_meta_path(out_dir)
    _write_parquet_rows(mmr_meta_path, mmr_meta_rows, chunk_size=2000)
    return mmr_meta_path


def _append_optimizer_stats(*, meta: dict[str, object], optimizer: object) -> None:
    if not hasattr(optimizer, "stats"):
        return
    optimizer_stats = optimizer.stats()
    if not isinstance(optimizer_stats, dict) or not optimizer_stats:
        return
    beta_base = optimizer_stats.get("beta_ladder_base")
    beta_final = optimizer_stats.get("beta_ladder_final")
    if isinstance(beta_base, (list, tuple)):
        meta["beta_ladder_base"] = [float(item) for item in beta_base]
    if isinstance(beta_final, (list, tuple)):
        meta["beta_ladder_final"] = [float(item) for item in beta_final]
    final_beta = optimizer_stats.get("final_mcmc_beta")
    meta["final_mcmc_beta"] = float(final_beta) if final_beta is not None else None
    cooling_payload = optimizer_stats.get("mcmc_cooling")
    if isinstance(cooling_payload, dict):
        meta["mcmc_cooling"] = cooling_payload


def _build_elites_metadata(
    *,
    sample_cfg: SampleConfig,
    tfs: list[str],
    out_dir: Path,
    elites: list[dict[str, object]],
    raw_elites: list[object],
    kept_after_mmr: int,
    total_draws_seen: int,
    combine_resolved: str,
    beta_softmin_final: float | None,
    pool_size: int,
    diversity_value: float,
    dsdna_mode: bool,
    mmr_summary: dict[str, object] | None,
    optimizer: object,
    postprocess_stats: dict[str, int] | None = None,
) -> dict[str, object]:
    tf_label = format_regulator_slug(tfs)
    selection_policy, relevance_label, pool_strategy, score_weight, diversity_weight = _selection_fields(
        mmr_summary=mmr_summary,
        diversity_value=diversity_value,
    )
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "selection_policy": selection_policy,
        "selection_relevance": relevance_label,
        "selection_score_weight": score_weight,
        "selection_diversity_weight": diversity_weight,
        "selection_diversity": diversity_value,
        "dsdna_canonicalize": dsdna_mode,
        "total_draws_seen": total_draws_seen,
        "candidate_count": len(raw_elites),
        "kept_after_mmr": kept_after_mmr,
        "objective_combine": combine_resolved,
        "softmin_final_beta_used": beta_softmin_final,
        "pool_size": pool_size,
        "pool_strategy": pool_strategy,
        "indexing_note": (
            "chain is a 0-based independent chain index; chain_1based is 1-based; "
            "draw_idx/sweep_idx are absolute sweeps; "
            "draw_in_phase is phase-relative"
        ),
        "tf_label": tf_label,
        "sequence_length": sample_cfg.sequence_length,
        "config_file": str(config_used_path(out_dir).resolve()),
    }
    _append_optimizer_stats(meta=meta, optimizer=optimizer)
    if mmr_summary is not None:
        meta["mmr_summary"] = mmr_summary
    if isinstance(postprocess_stats, dict):
        meta["postprocess"] = {
            "polish_edits": int(postprocess_stats.get("polish_edits", 0)),
            "trim_left": int(postprocess_stats.get("trim_left", 0)),
            "trim_right": int(postprocess_stats.get("trim_right", 0)),
            "dedup_dropped": int(postprocess_stats.get("dedup_dropped", 0)),
        }
    return meta


def _append_elite_artifacts(
    *,
    out_dir: Path,
    stage: str,
    artifacts: list[dict[str, object]],
    parquet_path: Path,
    json_path: Path,
    yaml_path: Path,
    hits_path: Path,
    mmr_meta_path: Path | None,
) -> None:
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
            artifact_entry(
                hits_path,
                out_dir,
                kind="table",
                label="Elite best-hit metadata (Parquet)",
                stage=stage,
            ),
        ]
    )
    if mmr_meta_path is None:
        return
    artifacts.append(
        artifact_entry(
            mmr_meta_path,
            out_dir,
            kind="table",
            label="Elite MMR selection metadata",
            stage=stage,
        )
    )


def select_and_persist_elites(
    *,
    optimizer: object,
    evaluator: object,
    scorer: Scorer,
    sample_cfg: SampleConfig,
    pwms: dict[str, object],
    tfs: list[str],
    out_dir: Path,
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
    beta_softmin_final: float | None,
    combine_resolved: str,
    stage: str,
    status_writer: object,
    run_logger: Callable[..., None],
    artifacts: list[dict[str, object]],
    finish_failed: Callable[[Exception], None],
) -> None:
    status_writer.update(status_message="building_elite_pool")
    pool_result = build_elite_pool(
        optimizer=optimizer,
        evaluator=evaluator,
        scorer=scorer,
        sample_cfg=sample_cfg,
        beta_softmin_final=beta_softmin_final,
    )
    raw_elites = pool_result.raw_elites
    norm_sums = pool_result.norm_sums
    min_norms = pool_result.min_norms
    total_draws_seen = int(pool_result.total_draws_seen)

    _log_candidate_percentiles(
        norm_sums=norm_sums,
        min_norms=min_norms,
        scorer=scorer,
        run_logger=run_logger,
    )

    elite_k = int(sample_cfg.elites.k or 0)
    _require_elite_candidates(
        elite_k=elite_k,
        raw_elites=raw_elites,
        finish_failed=finish_failed,
    )

    dsdna_mode = bool(sample_cfg.objective.bidirectional)
    diversity_value = float(sample_cfg.elites.select.diversity)

    status_writer.update(status_message="selecting_elites")
    kept_elites, kept_after_mmr, pool_size, mmr_meta_rows, mmr_summary = _select_elites(
        raw_elites=raw_elites,
        elite_k=elite_k,
        scorer=scorer,
        pwms=pwms,
        sample_cfg=sample_cfg,
        diversity_value=diversity_value,
    )
    _require_requested_elite_count(
        elite_k=elite_k,
        kept_elites=kept_elites,
        finish_failed=finish_failed,
    )

    status_writer.update(status_message="hydrating_elite_hits")
    hydrate_candidate_hits(kept_elites, scorer=scorer)
    kept_elites, postprocess_stats = _postprocess_elite_candidates(
        candidates=kept_elites,
        scorer=scorer,
        dsdna_mode=bool(dsdna_mode),
    )
    if postprocess_stats["polish_edits"] > 0:
        run_logger("Elite polish edits applied: %d", int(postprocess_stats["polish_edits"]))
    if postprocess_stats["trim_left"] > 0 or postprocess_stats["trim_right"] > 0:
        run_logger(
            "Elite edge trim applied across elites: left=%d right=%d",
            int(postprocess_stats["trim_left"]),
            int(postprocess_stats["trim_right"]),
        )
    if postprocess_stats["dedup_dropped"] > 0:
        run_logger("Elite dedup dropped %d postprocessed duplicates.", int(postprocess_stats["dedup_dropped"]))

    status_writer.update(status_message="serializing_elites")
    want_canonical = bool(dsdna_mode)
    elites = build_elite_entries(
        kept_elites,
        scorer=scorer,
        sample_cfg=sample_cfg,
        want_consensus=False,
        want_canonical=want_canonical,
        meta_source=out_dir.name,
    )
    dedupe_key = "canonical_sequence" if want_canonical else "sequence"
    _validate_elite_uniqueness(
        elites=elites,
        dedupe_key=dedupe_key,
        finish_failed=finish_failed,
    )
    run_logger("Final elite count: %d", len(elites))

    parquet_path, hits_path = _write_elite_tables(
        out_dir=out_dir,
        tfs=tfs,
        elites=elites,
        pwms=pwms,
        pwm_ref_by_tf=pwm_ref_by_tf,
        pwm_hash_by_tf=pwm_hash_by_tf,
        core_def_by_tf=core_def_by_tf,
        want_canonical=want_canonical,
    )
    json_path = elites_json_path(out_dir)
    atomic_write_json(json_path, elites)

    mmr_meta_path = _write_mmr_meta(out_dir=out_dir, mmr_meta_rows=mmr_meta_rows)

    meta = _build_elites_metadata(
        sample_cfg=sample_cfg,
        tfs=tfs,
        out_dir=out_dir,
        elites=elites,
        raw_elites=raw_elites,
        kept_after_mmr=kept_after_mmr,
        total_draws_seen=total_draws_seen,
        combine_resolved=combine_resolved,
        beta_softmin_final=beta_softmin_final,
        pool_size=pool_size,
        diversity_value=diversity_value,
        dsdna_mode=dsdna_mode,
        mmr_summary=mmr_summary,
        optimizer=optimizer,
        postprocess_stats=postprocess_stats,
    )
    yaml_path = elites_yaml_path(out_dir)
    atomic_write_yaml(yaml_path, meta, sort_keys=False)

    _append_elite_artifacts(
        out_dir=out_dir,
        stage=stage,
        artifacts=artifacts,
        parquet_path=parquet_path,
        json_path=json_path,
        yaml_path=yaml_path,
        hits_path=hits_path,
        mmr_meta_path=mmr_meta_path,
    )
