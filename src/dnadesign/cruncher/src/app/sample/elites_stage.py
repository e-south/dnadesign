"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/elites_stage.py

Select, validate, and persist elite outputs for sample runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from dnadesign.cruncher.app.sample.diagnostics import _EliteCandidate, _norm_map_for_elites
from dnadesign.cruncher.app.sample.elites_mmr import (
    build_elite_entries,
    build_elite_pool,
    hydrate_candidate_hits,
    select_elites_mmr,
)
from dnadesign.cruncher.app.sample.elites_persistence import (
    _append_elite_artifacts,
    _build_elites_metadata,
    _write_elite_tables,
    _write_mmr_meta,
)
from dnadesign.cruncher.app.sample.objective_sidecars import write_elite_objective_sidecars
from dnadesign.cruncher.app.sample.preflight import RunError, _resolve_elite_pool_size
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.layout import (
    elites_json_path,
    elites_yaml_path,
)
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.objectives.compiler import ObjectivePlanCompilation
from dnadesign.cruncher.core.objectives.models import ObjectiveResult, SelectedHit
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
    evaluator: object,
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
        evaluator=evaluator,
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


def _slot_hits_from_objective_results(
    objective_results: dict[str, ObjectiveResult] | None,
) -> dict[str, dict[str, object]]:
    if not isinstance(objective_results, dict):
        return {}
    slot_hits: dict[str, dict[str, object]] = {}
    for objective_id, result in objective_results.items():
        if not isinstance(result, ObjectiveResult):
            continue
        selected = tuple(result.selected_hits)
        if not selected:
            continue
        multi = len(selected) > 1
        for occurrence_rank, hit in enumerate(selected, start=1):
            slot_id = f"{objective_id}#{occurrence_rank}" if multi else str(objective_id)
            slot_hits[slot_id] = {
                "objective_id": str(objective_id),
                "occurrence_rank": int(occurrence_rank),
                "best_start": int(hit.start),
                "offset": int(hit.start),
                "width": int(hit.width),
                "strand": str(hit.strand),
                "best_score_raw": float(hit.raw_score),
                "best_score_scaled": float(hit.scaled_score),
                "best_score_norm": float(hit.normalized_score),
                "best_window_seq": hit.window_seq,
                "best_core_seq": hit.core_seq,
                "best_hit_tiebreak": hit.tiebreak_rule or "selected_hit",
            }
    return slot_hits


def _slot_hits_from_per_tf_hits(
    *,
    per_tf_hits: dict[str, dict[str, object]] | None,
    scorer: Scorer,
) -> dict[str, dict[str, object]]:
    if not isinstance(per_tf_hits, dict):
        raise ValueError("Elite candidate missing per_tf_hits for postprocessing.")
    slot_hits: dict[str, dict[str, object]] = {}
    for tf_name in scorer.tf_names:
        hit = per_tf_hits.get(tf_name)
        if not isinstance(hit, dict):
            raise ValueError(f"Elite candidate missing hit metadata for TF '{tf_name}'.")
        slot_hits[tf_name] = {
            "objective_id": str(tf_name),
            "occurrence_rank": 1,
            **hit,
        }
    return slot_hits


def _resolve_slot_hits(
    *,
    per_tf_hits: dict[str, dict[str, object]] | None,
    objective_results: dict[str, ObjectiveResult] | None,
    scorer: Scorer,
) -> dict[str, dict[str, object]]:
    slot_hits = _slot_hits_from_objective_results(objective_results)
    if slot_hits:
        return slot_hits
    return _slot_hits_from_per_tf_hits(per_tf_hits=per_tf_hits, scorer=scorer)


def _slot_window_map(
    *,
    slot_hits: dict[str, dict[str, object]],
) -> dict[str, tuple[int, int, str]]:
    windows: dict[str, tuple[int, int, str]] = {}
    for slot_id, hit in slot_hits.items():
        start = hit.get("best_start")
        if start is None:
            start = hit.get("offset")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(start, int) or not isinstance(width, int) or not isinstance(strand, str):
            raise ValueError(f"Invalid hit metadata for slot '{slot_id}' in elite postprocessing.")
        windows[str(slot_id)] = (int(start), int(width), str(strand))
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


def _slot_raw_score(*, slot_hits: dict[str, dict[str, object]], slot_id: str) -> float:
    hit = slot_hits.get(slot_id)
    if not isinstance(hit, dict):
        raise ValueError(f"Postprocess payload missing hit data for slot '{slot_id}'.")
    raw_score = hit.get("best_score_raw")
    if not isinstance(raw_score, (int, float)):
        raise ValueError(f"Postprocess payload missing raw hit score for slot '{slot_id}'.")
    return float(raw_score)


def _removed_bp_before(*, removed_segments: list[tuple[int, int]], position: int) -> int:
    removed = 0
    for start, end in removed_segments:
        if int(end) <= int(position):
            removed += int(end) - int(start)
    return int(removed)


def _hits_match_trim_contract(
    *,
    per_tf_hits: dict[str, dict[str, object]],
    expected_windows: dict[str, tuple[int, int, str]],
    removed_segments: list[tuple[int, int]],
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
        shifted_expected_start = int(expected_start) - _removed_bp_before(
            removed_segments=removed_segments,
            position=int(expected_start),
        )
        if int(start) != int(shifted_expected_start):
            return False
        if int(width) != int(expected_width):
            return False
        if str(strand) != str(expected_strand):
            return False
    return True


def _slot_hits_match_trim_contract(
    *,
    slot_hits: dict[str, dict[str, object]],
    expected_windows: dict[str, tuple[int, int, str]],
    removed_segments: list[tuple[int, int]],
) -> bool:
    for slot_id, (expected_start, expected_width, expected_strand) in expected_windows.items():
        hit = slot_hits.get(slot_id)
        if not isinstance(hit, dict):
            return False
        start = hit.get("best_start")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(start, int) or not isinstance(width, int) or not isinstance(strand, str):
            return False
        shifted_expected_start = int(expected_start) - _removed_bp_before(
            removed_segments=removed_segments,
            position=int(expected_start),
        )
        if int(start) != int(shifted_expected_start):
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


def _slot_hits_match_polish_contract(
    *,
    slot_hits: dict[str, dict[str, object]],
    expected_windows: dict[str, tuple[int, int, str]],
    owner_slot: str,
    owner_position: int,
) -> bool:
    for slot_id, (expected_start, expected_width, expected_strand) in expected_windows.items():
        hit = slot_hits.get(slot_id)
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
        if slot_id != owner_slot:
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
) -> tuple[
    dict[str, float], dict[str, dict[str, object]], dict[str, float], float, float, dict[str, ObjectiveResult] | None
]:
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
    return per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, None


def _candidate_payload_with_evaluator(
    *,
    seq_arr: np.ndarray,
    scorer: Scorer,
    evaluator: object,
) -> tuple[dict[str, float], dict[str, dict[str, object]], dict[str, float], float, float, dict[str, ObjectiveResult]]:
    objective_engine = getattr(evaluator, "objective_engine", None)
    if objective_engine is None:
        raise ValueError("Elite postprocess requires evaluator.objective_engine for occurrence-aware payloads.")
    evaluation = objective_engine.evaluate(seq_arr, seq_length=int(seq_arr.size))
    per_tf_map = {str(key): float(value) for key, value in evaluation.scalars.items()}
    norm_map = _norm_map_for_elites(
        seq_arr,
        per_tf_map,
        objective_results=evaluation.results,
        scorer=scorer,
        score_scale=scorer.scale,
    )
    per_tf_hits: dict[str, dict[str, object]] = {}
    for objective_id, result in evaluation.results.items():
        hit: SelectedHit | None = result.representative_hit
        if hit is None:
            raise ValueError(f"Postprocess payload missing representative hit for objective '{objective_id}'.")
        per_tf_hits[str(objective_id)] = {
            "best_score_raw": float(hit.raw_score),
            "offset": int(hit.start),
            "best_start": int(hit.start),
            "strand": str(hit.strand),
            "width": int(hit.width),
            "best_window_seq": hit.window_seq,
            "best_core_seq": hit.core_seq,
            "best_hit_tiebreak": hit.tiebreak_rule or "representative_hit",
            "best_score_scaled": float(per_tf_map[str(objective_id)]),
            "best_score_norm": float(norm_map.get(str(objective_id), 0.0)),
            "objective_kind": result.diagnostics.get("objective_kind"),
            "requested_copies": int(result.diagnostics.get("requested_copies", 1)),
            "selected_copies": int(result.diagnostics.get("selected_copies", 1)),
        }
    min_norm = float(min(norm_map.values())) if norm_map else 0.0
    sum_norm = float(sum(norm_map.values())) if norm_map else 0.0
    return per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, evaluation.results


def _uncovered_segments(
    *,
    windows: dict[str, tuple[int, int, str]],
    seq_length: int,
) -> list[tuple[int, int]]:
    coverage: list[bool] = [False for _ in range(int(seq_length))]
    for tf_name, (start, width, _strand) in windows.items():
        end = int(start) + int(width)
        if start < 0 or width < 1 or end > int(seq_length):
            raise ValueError(f"Elite hit window is out of bounds for TF '{tf_name}' during postprocessing.")
        for pos in range(int(start), int(end)):
            coverage[pos] = True
    segments: list[tuple[int, int]] = []
    idx = 0
    while idx < int(seq_length):
        if coverage[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < int(seq_length) and not coverage[end]:
            end += 1
        segments.append((int(idx), int(end)))
        idx = end
    return segments


def _normalize_removed_segments(
    *,
    removed_segments: list[tuple[int, int]],
    seq_length: int,
) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in removed_segments)
    normalized: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in ordered:
        if start < 0 or end > int(seq_length) or end <= start:
            raise ValueError("Elite trim segments must be valid non-empty in-bounds ranges.")
        if start < prev_end:
            raise ValueError("Elite trim segments must be disjoint and sorted.")
        normalized.append((int(start), int(end)))
        prev_end = int(end)
    return normalized


def _remaining_single_owner_polish_improvements(
    *,
    seq_arr: np.ndarray,
    per_tf_hits: dict[str, dict[str, object]],
    scorer: Scorer,
    objective_results: dict[str, ObjectiveResult] | None = None,
    evaluator: object | None = None,
    eps: float = 1.0e-12,
) -> list[dict[str, object]]:
    slot_hits = _resolve_slot_hits(per_tf_hits=per_tf_hits, objective_results=objective_results, scorer=scorer)
    expected_windows = _slot_window_map(slot_hits=slot_hits)
    single_owner_positions = _windows_cover_ownership(windows=expected_windows, seq_length=int(seq_arr.size))
    source = np.asarray(seq_arr, dtype=np.int8)
    remaining: list[dict[str, object]] = []
    for pos, slot_owner in single_owner_positions:
        current_base = int(source[pos])
        owner_score = _slot_raw_score(slot_hits=slot_hits, slot_id=slot_owner)
        for base in (0, 1, 2, 3):
            if int(base) == current_base:
                continue
            trial = source.copy()
            trial[pos] = int(base)
            if evaluator is not None:
                _, trial_hits, _, _, _, trial_objective_results = _candidate_payload_with_evaluator(
                    seq_arr=trial,
                    scorer=scorer,
                    evaluator=evaluator,
                )
            else:
                _, trial_hits, _, _, _, trial_objective_results = _candidate_payload(
                    seq_arr=trial,
                    scorer=scorer,
                )
            trial_slot_hits = _resolve_slot_hits(
                per_tf_hits=trial_hits,
                objective_results=trial_objective_results,
                scorer=scorer,
            )
            if not _slot_hits_match_polish_contract(
                slot_hits=trial_slot_hits,
                expected_windows=expected_windows,
                owner_slot=slot_owner,
                owner_position=int(pos),
            ):
                continue
            trial_owner_score = _slot_raw_score(slot_hits=trial_slot_hits, slot_id=slot_owner)
            if trial_owner_score > owner_score + float(eps):
                start, width, strand = expected_windows[slot_owner]
                objective_id = str(trial_slot_hits[slot_owner].get("objective_id") or slot_owner)
                occurrence_rank = int(trial_slot_hits[slot_owner].get("occurrence_rank") or 1)
                remaining.append(
                    {
                        "tf": objective_id,
                        "slot_id": str(slot_owner),
                        "occurrence_rank": occurrence_rank,
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


def _apply_uncovered_trim(
    *,
    candidate: _EliteCandidate,
    scorer: Scorer,
    removed_segments: list[tuple[int, int]],
    evaluator: object | None = None,
) -> None:
    if not removed_segments:
        return
    source = np.asarray(candidate.seq_arr, dtype=np.int8)
    normalized_segments = _normalize_removed_segments(
        removed_segments=removed_segments,
        seq_length=int(source.size),
    )
    slot_hits = _resolve_slot_hits(
        per_tf_hits=candidate.per_tf_hits,
        objective_results=candidate.objective_results,
        scorer=scorer,
    )
    expected_windows = _slot_window_map(slot_hits=slot_hits)
    for slot_id, (start, width, _strand) in expected_windows.items():
        end = int(start) + int(width)
        for seg_start, seg_end in normalized_segments:
            overlap_start = max(int(start), int(seg_start))
            overlap_end = min(int(end), int(seg_end))
            if overlap_start < overlap_end:
                raise ValueError(f"Elite trim segment overlaps hit window for slot '{slot_id}'.")
    kept_ranges: list[tuple[int, int]] = []
    cursor = 0
    for seg_start, seg_end in normalized_segments:
        if int(cursor) < int(seg_start):
            kept_ranges.append((int(cursor), int(seg_start)))
        cursor = int(seg_end)
    if int(cursor) < int(source.size):
        kept_ranges.append((int(cursor), int(source.size)))
    if not kept_ranges:
        raise ValueError("Elite uncovered trim would remove the full sequence.")
    trimmed = np.concatenate([source[int(start) : int(end)] for start, end in kept_ranges]).astype(np.int8, copy=False)
    if int(trimmed.size) < 1:
        raise ValueError("Elite uncovered trim produced empty sequence.")
    if evaluator is not None:
        per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, objective_results = _candidate_payload_with_evaluator(
            seq_arr=trimmed,
            scorer=scorer,
            evaluator=evaluator,
        )
    else:
        per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, objective_results = _candidate_payload(
            seq_arr=trimmed,
            scorer=scorer,
        )
    trimmed_slot_hits = _resolve_slot_hits(
        per_tf_hits=per_tf_hits,
        objective_results=objective_results,
        scorer=scorer,
    )
    if not _slot_hits_match_trim_contract(
        slot_hits=trimmed_slot_hits,
        expected_windows=expected_windows,
        removed_segments=normalized_segments,
    ):
        raise ValueError("Elite uncovered trim violated hit-window validity contract.")
    candidate.seq_arr = np.asarray(trimmed, dtype=np.int8)
    candidate.per_tf_map = per_tf_map
    candidate.per_tf_hits = per_tf_hits
    candidate.norm_map = norm_map
    candidate.min_norm = float(min_norm)
    candidate.sum_norm = float(sum_norm)
    candidate.objective_results = objective_results


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


def _polish_candidate_to_convergence(
    *,
    seq_arr: np.ndarray,
    scorer: Scorer,
    evaluator: object | None = None,
) -> tuple[
    np.ndarray,
    dict[str, float],
    dict[str, dict[str, object]],
    dict[str, float],
    float,
    float,
    int,
    dict[str, ObjectiveResult] | None,
]:
    polished_seq = np.asarray(seq_arr, dtype=np.int8).copy()
    if evaluator is not None:
        per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, objective_results = _candidate_payload_with_evaluator(
            seq_arr=polished_seq,
            scorer=scorer,
            evaluator=evaluator,
        )
    else:
        per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, objective_results = _candidate_payload(
            seq_arr=polished_seq,
            scorer=scorer,
        )
    edits = 0

    while True:
        slot_hits = _resolve_slot_hits(per_tf_hits=per_tf_hits, objective_results=objective_results, scorer=scorer)
        expected_windows = _slot_window_map(slot_hits=slot_hits)
        single_owner_positions = _windows_cover_ownership(windows=expected_windows, seq_length=int(polished_seq.size))
        changed = False
        for pos, slot_owner in single_owner_positions:
            current_base = int(polished_seq[pos])
            owner_score = _slot_raw_score(slot_hits=slot_hits, slot_id=slot_owner)
            best_update: (
                tuple[
                    np.ndarray,
                    dict[str, float],
                    dict[str, dict[str, object]],
                    dict[str, float],
                    float,
                    float,
                    dict[str, ObjectiveResult] | None,
                ]
                | None
            ) = None
            best_owner_score = owner_score
            for base in (0, 1, 2, 3):
                if int(base) == current_base:
                    continue
                trial = polished_seq.copy()
                trial[pos] = int(base)
                if evaluator is not None:
                    (
                        trial_per_tf_map,
                        trial_hits,
                        trial_norm_map,
                        trial_min_norm,
                        trial_sum_norm,
                        trial_objective_results,
                    ) = _candidate_payload_with_evaluator(
                        seq_arr=trial,
                        scorer=scorer,
                        evaluator=evaluator,
                    )
                else:
                    (
                        trial_per_tf_map,
                        trial_hits,
                        trial_norm_map,
                        trial_min_norm,
                        trial_sum_norm,
                        trial_objective_results,
                    ) = _candidate_payload(
                        seq_arr=trial,
                        scorer=scorer,
                    )
                trial_slot_hits = _resolve_slot_hits(
                    per_tf_hits=trial_hits,
                    objective_results=trial_objective_results,
                    scorer=scorer,
                )
                if not _slot_hits_match_polish_contract(
                    slot_hits=trial_slot_hits,
                    expected_windows=expected_windows,
                    owner_slot=slot_owner,
                    owner_position=int(pos),
                ):
                    continue
                trial_owner_score = _slot_raw_score(slot_hits=trial_slot_hits, slot_id=slot_owner)
                if trial_owner_score > best_owner_score + 1.0e-12:
                    best_owner_score = trial_owner_score
                    best_update = (
                        trial,
                        trial_per_tf_map,
                        trial_hits,
                        trial_norm_map,
                        float(trial_min_norm),
                        float(trial_sum_norm),
                        trial_objective_results,
                    )
            if best_update is None:
                continue
            polished_seq, per_tf_map, per_tf_hits, norm_map, min_norm, sum_norm, objective_results = best_update
            edits += 1
            changed = True
        if not changed:
            break

    return polished_seq, per_tf_map, per_tf_hits, norm_map, float(min_norm), float(sum_norm), edits, objective_results


def _postprocess_elite_candidates(
    *,
    candidates: list[_EliteCandidate],
    scorer: Scorer,
    dsdna_mode: bool,
    evaluator: object | None = None,
    trim_uncovered_internal: bool = True,
) -> tuple[list[_EliteCandidate], dict[str, int]]:
    stats = {
        "polish_edits": 0,
        "trim_left": 0,
        "trim_right": 0,
        "trim_internal_bp": 0,
        "trim_internal_segments": 0,
        "dedup_dropped": 0,
    }
    if not candidates:
        return candidates, stats

    for cand in candidates:
        (
            seq_arr,
            per_tf_map,
            per_tf_hits,
            norm_map,
            min_norm,
            sum_norm,
            polish_edits,
            objective_results,
        ) = _polish_candidate_to_convergence(
            seq_arr=np.asarray(cand.seq_arr, dtype=np.int8),
            scorer=scorer,
            evaluator=evaluator,
        )
        stats["polish_edits"] += int(polish_edits)

        cand.seq_arr = np.asarray(seq_arr, dtype=np.int8)
        cand.per_tf_map = per_tf_map
        cand.per_tf_hits = per_tf_hits
        cand.norm_map = norm_map
        cand.min_norm = float(min_norm)
        cand.sum_norm = float(sum_norm)
        cand.objective_results = objective_results
        slot_hits = _resolve_slot_hits(
            per_tf_hits=per_tf_hits,
            objective_results=objective_results,
            scorer=scorer,
        )
        expected_windows = _slot_window_map(slot_hits=slot_hits)
        removed_segments: list[tuple[int, int]] = []
        for seg_start, seg_end in _uncovered_segments(windows=expected_windows, seq_length=int(seq_arr.size)):
            segment_bp = int(seg_end) - int(seg_start)
            if int(seg_start) == 0:
                removed_segments.append((int(seg_start), int(seg_end)))
                stats["trim_left"] += int(segment_bp)
                continue
            if int(seg_end) == int(seq_arr.size):
                removed_segments.append((int(seg_start), int(seg_end)))
                stats["trim_right"] += int(segment_bp)
                continue
            if bool(trim_uncovered_internal):
                removed_segments.append((int(seg_start), int(seg_end)))
                stats["trim_internal_bp"] += int(segment_bp)
                stats["trim_internal_segments"] += 1
        if removed_segments:
            _apply_uncovered_trim(
                candidate=cand,
                scorer=scorer,
                removed_segments=removed_segments,
                evaluator=evaluator,
            )
            (
                trimmed_seq_arr,
                trimmed_per_tf_map,
                trimmed_per_tf_hits,
                trimmed_norm_map,
                trimmed_min_norm,
                trimmed_sum_norm,
                post_trim_polish_edits,
                trimmed_objective_results,
            ) = _polish_candidate_to_convergence(
                seq_arr=np.asarray(cand.seq_arr, dtype=np.int8),
                scorer=scorer,
                evaluator=evaluator,
            )
            cand.seq_arr = np.asarray(trimmed_seq_arr, dtype=np.int8)
            cand.per_tf_map = trimmed_per_tf_map
            cand.per_tf_hits = trimmed_per_tf_hits
            cand.norm_map = trimmed_norm_map
            cand.min_norm = float(trimmed_min_norm)
            cand.sum_norm = float(trimmed_sum_norm)
            cand.objective_results = trimmed_objective_results
            stats["polish_edits"] += int(post_trim_polish_edits)
        remaining = _remaining_single_owner_polish_improvements(
            seq_arr=np.asarray(cand.seq_arr, dtype=np.int8),
            per_tf_hits=cand.per_tf_hits,
            scorer=scorer,
            objective_results=cand.objective_results,
            evaluator=evaluator,
        )
        if remaining:
            first = remaining[0]
            raise ValueError(
                "Elite polish convergence failed for chain=%d draw=%d tf=%s slot=%s pos=%d strand=%s."
                % (
                    int(cand.chain_id),
                    int(cand.draw_idx),
                    str(first["tf"]),
                    str(first["slot_id"]),
                    int(first["position"]),
                    str(first["strand"]),
                )
            )

    deduped, dropped = _dedupe_postprocessed_candidates(candidates=candidates, dsdna_mode=bool(dsdna_mode))
    stats["dedup_dropped"] = int(dropped)
    return deduped, stats


def _refresh_candidate_combined_scores(
    *,
    candidates: list[_EliteCandidate],
    evaluator: object,
    beta_softmin_final: float | None,
) -> None:
    combine_from_scores = getattr(evaluator, "combined_from_scores", None)
    if not callable(combine_from_scores):
        raise ValueError("Evaluator must expose callable combined_from_scores for elite postprocess score refresh.")
    for cand in candidates:
        seq_arr = np.asarray(cand.seq_arr, dtype=np.int8)
        cand.combined_score = float(
            combine_from_scores(
                cand.per_tf_map,
                beta=beta_softmin_final,
                length=int(seq_arr.size),
            )
        )


def select_and_persist_elites(
    *,
    optimizer: object,
    evaluator: object,
    objective_plan: ObjectivePlanCompilation,
    scorer: Scorer,
    sample_cfg: SampleConfig,
    pwms: dict[str, object],
    tfs: list[str],
    out_dir: Path,
    workspace_slug: str,
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
        evaluator=evaluator,
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
    hydrate_candidate_hits(kept_elites, evaluator=evaluator)
    kept_elites, postprocess_stats = _postprocess_elite_candidates(
        candidates=kept_elites,
        scorer=scorer,
        evaluator=evaluator,
        dsdna_mode=bool(dsdna_mode),
        trim_uncovered_internal=bool(sample_cfg.elites.postprocess.trim_uncovered_internal),
    )
    _refresh_candidate_combined_scores(
        candidates=kept_elites,
        evaluator=evaluator,
        beta_softmin_final=beta_softmin_final,
    )
    if postprocess_stats["polish_edits"] > 0:
        run_logger("Elite polish edits applied: %d", int(postprocess_stats["polish_edits"]))
    if postprocess_stats["trim_left"] > 0 or postprocess_stats["trim_right"] > 0:
        run_logger(
            "Elite edge trim applied across elites: left=%d right=%d",
            int(postprocess_stats["trim_left"]),
            int(postprocess_stats["trim_right"]),
        )
    if postprocess_stats["trim_internal_segments"] > 0:
        run_logger(
            "Elite internal trim applied across elites: segments=%d bp=%d",
            int(postprocess_stats["trim_internal_segments"]),
            int(postprocess_stats["trim_internal_bp"]),
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
        workspace_slug=workspace_slug,
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
        write_representative_hits=bool(objective_plan.runtime.supports_representative_hit_artifact),
    )
    objective_scores_file: Path | None = None
    occurrences_file: Path | None = None
    if not objective_plan.runtime.supports_representative_hit_artifact:
        objective_scores_file, occurrences_file = write_elite_objective_sidecars(
            out_dir=out_dir,
            entries=elites,
            evaluator=evaluator,
            objective_plan=objective_plan,
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
        objective_scores_path=objective_scores_file,
        occurrences_path=occurrences_file,
        mmr_meta_path=mmr_meta_path,
    )
