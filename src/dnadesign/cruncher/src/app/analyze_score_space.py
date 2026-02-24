"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_score_space.py

Resolve score-space context and project trajectory/baseline/elites objective
views for analyze workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.consensus import compute_consensus_anchors
from dnadesign.cruncher.analysis.trajectory import add_raw_llr_objective, build_trajectory_points
from dnadesign.cruncher.app.analyze.score_space import _resolve_score_space_spec, _resolve_worst_second_tf_pair
from dnadesign.cruncher.core.scoring import Scorer

__all__ = [
    "_ScoreSpaceContext",
    "_objective_caption",
    "_project_trajectory_views_with_cleanup",
    "_resolve_objective_projection_inputs",
    "_resolve_score_space_context",
]


def _objective_caption(objective_cfg: dict[str, object]) -> str:
    combine = str(objective_cfg.get("combine") or "min").strip().lower()
    if combine == "sum":
        base = "Joint objective: maximize sum_TF(score_TF). Each score_TF is the best-window scan score."
    else:
        base = "Joint objective: maximize min_TF(score_TF). Each score_TF is the best-window scan score."
    softmin_cfg = objective_cfg.get("softmin")
    if isinstance(softmin_cfg, dict) and bool(softmin_cfg.get("enabled")):
        return f"{base} Soft-min shaping enabled."
    return base


def _objective_from_manifest(manifest: dict[str, object]) -> dict[str, object]:
    payload = manifest.get("objective")
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Run manifest field 'objective' must be an object when provided.")
    return dict(payload)


def _resolve_total_sweeps(manifest: dict[str, object], sample_meta: object) -> int:
    total_sweeps_raw = manifest.get("total_sweeps")
    if total_sweeps_raw is None:
        if sample_meta.tune is None or sample_meta.draws is None:
            raise ValueError("Run metadata missing total_sweeps and sample tune/draws required for objective replay.")
        total_sweeps_raw = int(sample_meta.tune) + int(sample_meta.draws)
    try:
        total_sweeps = int(total_sweeps_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Run manifest field 'total_sweeps' must be an integer.") from exc
    if total_sweeps < 1:
        raise ValueError("Run manifest field 'total_sweeps' must be >= 1.")
    return total_sweeps


def _resolve_projection_scoring_params(used_cfg: dict[str, object]) -> tuple[float, float | None]:
    sample_payload = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
    objective_used = sample_payload.get("objective") if isinstance(sample_payload, dict) else None
    scoring_used = objective_used.get("scoring") if isinstance(objective_used, dict) else None
    pseudocounts_raw = 0.10
    if isinstance(scoring_used, dict) and scoring_used.get("pwm_pseudocounts") is not None:
        pseudocounts_raw = float(scoring_used.get("pwm_pseudocounts"))
    log_odds_clip_raw = None
    if isinstance(scoring_used, dict) and scoring_used.get("log_odds_clip") is not None:
        log_odds_clip_raw = float(scoring_used.get("log_odds_clip"))
    return pseudocounts_raw, log_odds_clip_raw


def _resolve_retain_sequences(elites_df: pd.DataFrame, *, bidirectional: bool) -> set[str]:
    retain_sequences: set[str] = set()
    if "sequence" in elites_df.columns:
        retain_sequences.update(
            value.strip() for value in elites_df["sequence"].astype(str).tolist() if value and value.strip()
        )
    if bidirectional and "canonical_sequence" in elites_df.columns:
        retain_sequences.update(
            value.strip() for value in elites_df["canonical_sequence"].astype(str).tolist() if value and value.strip()
        )
    return retain_sequences


def _resolve_beta_ladder(optimizer_stats: dict[str, object] | None) -> list[float] | None:
    if not isinstance(optimizer_stats, dict):
        return None
    ladder_payload = optimizer_stats.get("beta_ladder_final")
    if ladder_payload is None:
        return None
    if not isinstance(ladder_payload, list):
        raise ValueError("optimizer_stats.beta_ladder_final must be a list when provided.")
    return [float(v) for v in ladder_payload]


def _resolve_objective_projection_inputs(
    *,
    manifest: dict[str, object],
    sample_meta: object,
    used_cfg: dict[str, object],
    elites_df: pd.DataFrame,
    optimizer_stats: dict[str, object] | None,
) -> tuple[dict[str, object], bool, float, float | None, set[str], list[float] | None]:
    objective_from_manifest = _objective_from_manifest(manifest)
    objective_from_manifest["total_sweeps"] = _resolve_total_sweeps(manifest, sample_meta)
    pseudocounts_raw, log_odds_clip_raw = _resolve_projection_scoring_params(used_cfg)
    bidirectional = bool(objective_from_manifest.get("bidirectional", True))
    retain_sequences = _resolve_retain_sequences(elites_df, bidirectional=bidirectional)
    beta_ladder = _resolve_beta_ladder(optimizer_stats)
    return objective_from_manifest, bidirectional, pseudocounts_raw, log_odds_clip_raw, retain_sequences, beta_ladder


@dataclass(frozen=True)
class _ScoreSpaceContext:
    mode: str
    pairs: list[tuple[str, str]]
    focus_pair: tuple[str, str] | None
    trajectory_tf_pair: tuple[str, str]
    trajectory_scale: str
    objective_caption: str
    consensus_anchors: list[dict[str, object]] | None
    consensus_anchors_by_pair: dict[str, list[dict[str, object]]] | None


def _resolve_default_trajectory_pair(
    *,
    tf_names: Sequence[str],
    score_space_mode: str,
    score_space_pairs: Sequence[tuple[str, str]],
) -> tuple[str, str]:
    if score_space_mode == "pair" and score_space_pairs:
        return score_space_pairs[0]
    if score_space_pairs:
        return score_space_pairs[0]
    if len(tf_names) >= 2:
        return str(tf_names[0]), str(tf_names[1])
    tf_name = str(tf_names[0])
    return tf_name, tf_name


def _resolve_sequence_length_for_anchors(manifest: dict[str, object], sequences_df: pd.DataFrame) -> int:
    sequence_length = manifest.get("sequence_length")
    if sequence_length is None and "sequence" in sequences_df.columns and not sequences_df.empty:
        sequence_length = int(sequences_df["sequence"].astype(str).str.len().iloc[0])
    if sequence_length is None:
        raise ValueError("Run manifest missing sequence_length required for trajectory consensus anchors.")
    return int(sequence_length)


def _resolve_consensus_anchor_payloads(
    *,
    score_space_mode: str,
    score_space_pairs: Sequence[tuple[str, str]],
    trajectory_tf_pair: tuple[str, str],
    pwms: dict[str, object],
    sequence_length: int,
    anchor_objective_cfg: dict[str, object],
) -> tuple[list[dict[str, object]] | None, dict[str, list[dict[str, object]]] | None]:
    if score_space_mode in {"pair", "worst_vs_second_worst"}:
        anchors = compute_consensus_anchors(
            pwms=pwms,
            tf_names=[str(trajectory_tf_pair[0]), str(trajectory_tf_pair[1])],
            sequence_length=sequence_length,
            objective_config=anchor_objective_cfg,
            x_metric=f"score_{trajectory_tf_pair[0]}",
            y_metric=f"score_{trajectory_tf_pair[1]}",
        )
        return anchors, None
    if score_space_mode == "all_pairs_grid":
        anchors_by_pair: dict[str, list[dict[str, object]]] = {}
        for pair in score_space_pairs:
            tf_a, tf_b = str(pair[0]), str(pair[1])
            pair_key = f"{tf_a}|{tf_b}"
            anchors_by_pair[pair_key] = compute_consensus_anchors(
                pwms=pwms,
                tf_names=[tf_a, tf_b],
                sequence_length=sequence_length,
                objective_config=anchor_objective_cfg,
                x_metric=f"score_{tf_a}",
                y_metric=f"score_{tf_b}",
            )
        return None, anchors_by_pair
    raise ValueError(f"Unsupported score-space mode: {score_space_mode}")


def _resolve_score_space_context(
    *,
    tf_names: list[str],
    analysis_cfg: object,
    elites_plot_df: pd.DataFrame,
    pwms: dict[str, object],
    sequences_df: pd.DataFrame,
    manifest: dict[str, object],
    objective_from_manifest: dict[str, object],
) -> _ScoreSpaceContext:
    score_space_spec = _resolve_score_space_spec(tf_names, analysis_cfg.pairwise)
    score_space_mode = str(score_space_spec["mode"])
    score_space_pairs = [tuple(pair) for pair in score_space_spec.get("pairs") or []]
    focus_pair = score_space_pairs[0] if score_space_mode == "pair" and score_space_pairs else None
    trajectory_tf_pair = _resolve_default_trajectory_pair(
        tf_names=tf_names,
        score_space_mode=score_space_mode,
        score_space_pairs=score_space_pairs,
    )
    if score_space_mode == "worst_vs_second_worst":
        if str(analysis_cfg.trajectory_scatter_scale).strip().lower() not in {"llr", "raw-llr", "raw_llr"}:
            raise ValueError(
                "analysis.pairwise=auto (multi-TF) requires analysis.trajectory_scatter_scale='llr' "
                "to render TF-specific worst/second-worst axes."
            )
        trajectory_tf_pair = _resolve_worst_second_tf_pair(
            elites_df=elites_plot_df,
            tf_names=tf_names,
            score_prefix="raw_llr_",
        )
        focus_pair = trajectory_tf_pair
    trajectory_scale = str(analysis_cfg.trajectory_scatter_scale)
    sequence_length = _resolve_sequence_length_for_anchors(manifest, sequences_df)
    anchor_objective_cfg = dict(objective_from_manifest)
    anchor_objective_cfg["score_scale"] = trajectory_scale
    objective_caption = _objective_caption(objective_from_manifest)
    consensus_anchors, consensus_anchors_by_pair = _resolve_consensus_anchor_payloads(
        score_space_mode=score_space_mode,
        score_space_pairs=score_space_pairs,
        trajectory_tf_pair=trajectory_tf_pair,
        pwms=pwms,
        sequence_length=sequence_length,
        anchor_objective_cfg=anchor_objective_cfg,
    )
    return _ScoreSpaceContext(
        mode=score_space_mode,
        pairs=score_space_pairs,
        focus_pair=focus_pair,
        trajectory_tf_pair=trajectory_tf_pair,
        trajectory_scale=trajectory_scale,
        objective_caption=objective_caption,
        consensus_anchors=consensus_anchors,
        consensus_anchors_by_pair=consensus_anchors_by_pair,
    )


def _project_score_views(
    *,
    trajectory_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, object],
    objective_config: dict[str, object],
    bidirectional: bool,
    pwm_pseudocounts: float,
    log_odds_clip: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_pwms = [str(tf) for tf in tf_names if str(tf) not in pwms]
    if missing_pwms:
        raise ValueError(f"Cannot compute raw-LLR objective; missing PWMs for TFs: {missing_pwms}")
    tf_pwms = {str(tf): pwms[str(tf)] for tf in tf_names}
    scorer_raw = Scorer(
        tf_pwms,
        bidirectional=bool(bidirectional),
        scale="llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )
    scorer_norm = Scorer(
        tf_pwms,
        bidirectional=bool(bidirectional),
        scale="normalized-llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )
    score_cache: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    trajectory_plot_df = add_raw_llr_objective(
        trajectory_df,
        tf_names,
        pwms=pwms,
        objective_config=objective_config,
        bidirectional=bidirectional,
        pwm_pseudocounts=pwm_pseudocounts,
        log_odds_clip=log_odds_clip,
        score_cache=score_cache,
        scorer_raw=scorer_raw,
        scorer_norm=scorer_norm,
    )
    baseline_plot_df = add_raw_llr_objective(
        baseline_df,
        tf_names,
        pwms=pwms,
        objective_config=objective_config,
        bidirectional=bidirectional,
        pwm_pseudocounts=pwm_pseudocounts,
        log_odds_clip=log_odds_clip,
        score_cache=score_cache,
        scorer_raw=scorer_raw,
        scorer_norm=scorer_norm,
    )
    elites_plot_df = add_raw_llr_objective(
        elites_df,
        tf_names,
        pwms=pwms,
        objective_config=objective_config,
        bidirectional=bidirectional,
        pwm_pseudocounts=pwm_pseudocounts,
        log_odds_clip=log_odds_clip,
        score_cache=score_cache,
        scorer_raw=scorer_raw,
        scorer_norm=scorer_norm,
    )
    return trajectory_plot_df, baseline_plot_df, elites_plot_df


def _project_trajectory_views_with_cleanup(
    *,
    tmp_root: Path,
    sequences_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, object],
    analysis_cfg: object,
    objective_from_manifest: dict[str, object],
    bidirectional: bool,
    pseudocounts_raw: float,
    log_odds_clip_raw: float | None,
    beta_ladder: list[float] | None,
    retain_sequences: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        trajectory_df = build_trajectory_points(
            sequences_df,
            tf_names,
            max_points=analysis_cfg.max_points,
            objective_config=objective_from_manifest,
            beta_ladder=beta_ladder,
            retain_sequences=retain_sequences,
        )
        return _project_score_views(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            elites_df=elites_df,
            tf_names=tf_names,
            pwms=pwms,
            objective_config=objective_from_manifest,
            bidirectional=bidirectional,
            pwm_pseudocounts=pseudocounts_raw,
            log_odds_clip=log_odds_clip_raw,
        )
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
