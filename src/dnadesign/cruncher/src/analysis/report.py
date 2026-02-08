"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/report.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dnadesign.cruncher.analysis.layout import (
    analysis_manifest_path,
    analysis_plot_path,
    analysis_table_filename,
    analysis_table_path,
    analysis_used_path,
    plot_manifest_path,
    report_json_path,
    report_md_path,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative_if_exists(base: Path, target: Path) -> str | None:
    if not target.exists():
        return None
    return str(target.relative_to(base))


def _table_path(base: Path, stem: str, fmt: str) -> str | None:
    return _relative_if_exists(base, analysis_table_path(base, stem, fmt))


def _plot_path(base: Path, stem: str, fmt: str) -> str | None:
    return _relative_if_exists(base, analysis_plot_path(base, stem, fmt))


def build_report_payload(
    *,
    analysis_root: Path,
    summary_payload: dict,
    diagnostics_payload: dict | None = None,
    objective_components: dict | None = None,
    overlap_summary: dict | None = None,
    analysis_used_payload: dict | None = None,
) -> dict[str, object]:
    diagnostics_payload = diagnostics_payload or summary_payload.get("diagnostics")
    objective_components = objective_components or summary_payload.get("objective_components")
    overlap_summary = overlap_summary or summary_payload.get("overlap_summary")

    def _first_non_none(*values: object) -> object | None:
        for value in values:
            if value is not None:
                return value
        return None

    analysis_cfg = {}
    if isinstance(analysis_used_payload, dict):
        analysis_cfg = analysis_used_payload.get("analysis") or analysis_used_payload.get("analysis_base") or {}
    table_format = "parquet"
    plot_format = "png"
    if isinstance(analysis_cfg, dict):
        table_format = str(analysis_cfg.get("table_format") or table_format)
        plot_format = str(analysis_cfg.get("plot_format") or plot_format)

    metrics = {}
    warnings = []
    status = None
    if isinstance(diagnostics_payload, dict):
        metrics = diagnostics_payload.get("metrics") or {}
        warnings = diagnostics_payload.get("warnings") or []
        status = diagnostics_payload.get("status")

    trace_metrics = metrics.get("trace") if isinstance(metrics, dict) else {}
    optimizer_metrics = metrics.get("optimizer") if isinstance(metrics, dict) else {}
    seq_metrics = metrics.get("sequences") if isinstance(metrics, dict) else {}
    elites_metrics = metrics.get("elites") if isinstance(metrics, dict) else {}
    sample_metrics = metrics.get("sample") if isinstance(metrics, dict) else {}

    tf_names = summary_payload.get("tf_names") if isinstance(summary_payload, dict) else None
    run_name = summary_payload.get("run") if isinstance(summary_payload, dict) else None
    analysis_id = summary_payload.get("analysis_id") if isinstance(summary_payload, dict) else None

    highlights_objective = {}
    if isinstance(objective_components, dict):
        highlights_objective = {
            "best_score_final": _safe_float(objective_components.get("best_score_final")),
            "top_k_median_final": _safe_float(objective_components.get("top_k_median_final")),
            "median_min_scaled_tf": _safe_float(objective_components.get("median_min_scaled_tf")),
            "worst_tf_frequency": objective_components.get("worst_tf_frequency"),
        }
    highlights_diversity = {
        "unique_fraction": _safe_float(
            _first_non_none(
                (objective_components or {}).get("unique_fraction_canonical"),
                (objective_components or {}).get("unique_fraction_raw"),
                (seq_metrics or {}).get("unique_fraction"),
            )
        ),
        "n_elites": _safe_int((elites_metrics or {}).get("n_elites")),
        "diversity_hamming": _safe_float((elites_metrics or {}).get("diversity_hamming")),
    }
    highlights_overlap = {
        "overlap_rate_median": _safe_float(
            _first_non_none(
                (overlap_summary or {}).get("overlap_rate_median"),
                (elites_metrics or {}).get("overlap_rate_median"),
            )
        ),
        "overlap_total_bp_median": _safe_float(
            _first_non_none(
                (overlap_summary or {}).get("overlap_total_bp_median"),
                (objective_components or {}).get("overlap_total_bp_median"),
            )
        ),
    }
    highlights_sampling = {
        "rhat": _safe_float((trace_metrics or {}).get("rhat")),
        "ess": _safe_float((trace_metrics or {}).get("ess")),
        "ess_ratio": _safe_float((trace_metrics or {}).get("ess_ratio")),
        "acceptance_rate_mh_tail": _safe_float((optimizer_metrics or {}).get("acceptance_rate_mh_tail")),
        "swap_acceptance_rate": _safe_float((optimizer_metrics or {}).get("swap_acceptance_rate")),
    }
    highlights_learning = {}
    if isinstance(objective_components, dict):
        learning_payload = objective_components.get("learning")
        if isinstance(learning_payload, dict):
            early_payload = learning_payload.get("early_stop") if isinstance(learning_payload, dict) else None
            early_payload = early_payload if isinstance(early_payload, dict) else {}
            highlights_learning = {
                "best_score_draw": _safe_int(learning_payload.get("best_score_draw")),
                "best_score_chain": _safe_int(learning_payload.get("best_score_chain")),
                "last_improvement_draw": _safe_int(learning_payload.get("last_improvement_draw")),
                "plateau_draws": _safe_int(learning_payload.get("plateau_draws")),
                "early_stop_earliest_draw": _safe_int(early_payload.get("earliest_draw")),
                "early_stop_stopped_chains": _safe_int(early_payload.get("stopped_chains")),
            }

    autopicks = None
    if isinstance(analysis_used_payload, dict):
        autopicks = analysis_used_payload.get("analysis_autopicks")

    run_block = {
        "run": run_name,
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tf_names": tf_names,
        "optimizer_kind": optimizer_metrics.get("kind") if isinstance(optimizer_metrics, dict) else None,
        "chains": _safe_int(
            _first_non_none(
                sample_metrics.get("chains") if isinstance(sample_metrics, dict) else None,
                trace_metrics.get("chains") if isinstance(trace_metrics, dict) else None,
            )
        ),
        "draws": _safe_int(
            _first_non_none(
                sample_metrics.get("draws") if isinstance(sample_metrics, dict) else None,
                trace_metrics.get("draws") if isinstance(trace_metrics, dict) else None,
            )
        ),
        "tune": _safe_int(sample_metrics.get("tune")) if isinstance(sample_metrics, dict) else None,
        "n_sequences": _safe_int(seq_metrics.get("n_sequences")) if isinstance(seq_metrics, dict) else None,
        "n_elites": _safe_int(elites_metrics.get("n_elites")) if isinstance(elites_metrics, dict) else None,
    }

    pointers = {
        "start_here_plot": _plot_path(analysis_root, "run_summary", plot_format),
        "diagnostics": _table_path(analysis_root, "diagnostics_summary", "json"),
        "objective_components": _table_path(analysis_root, "objective_components", "json"),
        "elites_mmr_summary": _table_path(analysis_root, "elites_mmr_summary", table_format),
        "overlap_summary": _table_path(analysis_root, "overlap_pair_summary", table_format),
        "elite_topk": _table_path(analysis_root, "elites_topk", table_format),
        "manifest": _relative_if_exists(analysis_root, analysis_manifest_path(analysis_root)),
        "plot_manifest": _relative_if_exists(analysis_root, plot_manifest_path(analysis_root)),
        "table_manifest": _relative_if_exists(analysis_root, table_manifest_path(analysis_root)),
    }

    payload: dict[str, object] = {
        "report_version": "v1",
        "run": run_block,
        "status": status,
        "warnings": warnings,
        "highlights": {
            "objective": highlights_objective,
            "diversity": highlights_diversity,
            "overlap": highlights_overlap,
            "sampling": highlights_sampling,
            "learning": highlights_learning,
        },
        "autopicks": autopicks,
        "paths": pointers,
    }
    return payload


def write_report_json(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, payload, allow_nan=False)


def write_report_md(
    report_path: Path,
    payload: dict[str, object],
    *,
    analysis_root: Path,
) -> None:
    highlights = payload.get("highlights") or {}
    objective = highlights.get("objective") or {}
    diversity = highlights.get("diversity") or {}
    overlap = highlights.get("overlap") or {}
    sampling = highlights.get("sampling") or {}
    learning = highlights.get("learning") or {}
    warnings = payload.get("warnings") or []
    status = payload.get("status") or "unknown"
    pointers = payload.get("paths") or {}
    table_format = "parquet"
    elite_topk_path = pointers.get("elite_topk")
    if isinstance(elite_topk_path, str) and "." in elite_topk_path:
        table_format = elite_topk_path.rsplit(".", 1)[-1]

    def _fmt(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    best_score_draw = _fmt(learning.get("best_score_draw"))
    best_score_chain = _fmt(learning.get("best_score_chain"))
    last_improvement_draw = _fmt(learning.get("last_improvement_draw"))
    plateau_draws = _fmt(learning.get("plateau_draws"))
    early_stop_earliest_draw = _fmt(learning.get("early_stop_earliest_draw"))
    early_stop_stopped_chains = _fmt(learning.get("early_stop_stopped_chains"))
    lines = [
        "# Cruncher Analysis Report",
        "",
        f"- Status: {status}",
        f"- Best score (final): {_fmt(objective.get('best_score_final'))}",
        f"- Top‑K median (final): {_fmt(objective.get('top_k_median_final'))}",
        f"- Median min‑TF score: {_fmt(objective.get('median_min_scaled_tf'))}",
        f"- Unique fraction: {_fmt(diversity.get('unique_fraction'))}",
        f"- Elites: {_fmt(diversity.get('n_elites'))}",
        f"- Overlap rate median: {_fmt(overlap.get('overlap_rate_median'))}",
        f"- Overlap bp median: {_fmt(overlap.get('overlap_total_bp_median'))}",
        f"- MH acceptance (tail): {_fmt(sampling.get('acceptance_rate_mh_tail'))}",
        f"- PT swap acceptance: {_fmt(sampling.get('swap_acceptance_rate'))}",
        "- Trace diagnostics: directional indicators only (not convergence proofs).",
    ]
    if any(
        learning.get(key) is not None
        for key in (
            "best_score_draw",
            "best_score_chain",
            "last_improvement_draw",
            "plateau_draws",
            "early_stop_earliest_draw",
            "early_stop_stopped_chains",
        )
    ):
        lines.extend(
            [
                f"- Best score draw: {best_score_draw} (chain {best_score_chain})",
                f"- Last improvement draw: {last_improvement_draw}",
                f"- Plateau draws: {plateau_draws}",
                f"- Early-stop earliest draw: {early_stop_earliest_draw} (stopped chains: {early_stop_stopped_chains})",
            ]
        )
    lines.extend(
        [
            "",
            "## Start here",
            "",
            f"- {pointers.get('start_here_plot') or 'plots/plot__run_summary.<ext>'}",
            f"- {pointers.get('diagnostics') or 'output/table__diagnostics_summary.json'}",
            f"- {pointers.get('objective_components') or 'output/table__objective_components.json'}",
        ]
    )
    elites_mmr_path = pointers.get("elites_mmr_summary")
    if elites_mmr_path:
        lines.append(f"- {elites_mmr_path}")
    overlap_path = pointers.get("overlap_summary") or (
        f"output/{analysis_table_filename('overlap_pair_summary', table_format)}"
    )
    elite_topk_path = pointers.get("elite_topk") or f"output/{analysis_table_filename('elites_topk', table_format)}"
    lines.extend([f"- {overlap_path}", f"- {elite_topk_path}"])

    autopicks = payload.get("autopicks")
    if isinstance(autopicks, dict) and autopicks:
        lines.extend(["", "## Autopicks", ""])
        for key, value in autopicks.items():
            lines.append(f"- {key}: {value}")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {item}" for item in warnings])

    artifact_index = [
        pointers.get("start_here_plot") or "plots/plot__run_summary.<ext>",
        pointers.get("diagnostics") or "output/table__diagnostics_summary.json",
        pointers.get("objective_components") or "output/table__objective_components.json",
        pointers.get("elites_mmr_summary") or None,
        overlap_path,
        elite_topk_path,
        pointers.get("manifest") or "output/manifest.json",
        pointers.get("plot_manifest") or "output/plot_manifest.json",
        pointers.get("table_manifest") or "output/table_manifest.json",
    ]
    artifact_index = [item for item in artifact_index if item]
    move_summary = _table_path(analysis_root, "move_stats_summary", table_format)
    if move_summary:
        artifact_index.append(move_summary)
    pt_swap = _table_path(analysis_root, "pt_swap_pairs", table_format)
    if pt_swap:
        artifact_index.append(pt_swap)
    lines.extend(["", "## Artifact index", ""] + [f"- {item}" for item in artifact_index])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(report_path, "\n".join(lines))


def ensure_report(
    *,
    analysis_root: Path,
    summary_payload: dict | None = None,
    diagnostics_payload: dict | None = None,
    objective_components: dict | None = None,
    overlap_summary: dict | None = None,
    analysis_used_payload: dict | None = None,
    refresh: bool = False,
) -> tuple[Path, Path]:
    report_json = report_json_path(analysis_root)
    report_md = report_md_path(analysis_root)
    if not refresh and report_json.exists() and report_md.exists():
        return report_json, report_md

    if summary_payload is None:
        loaded_summary = _load_json(summary_path(analysis_root))
        if loaded_summary is None:
            raise FileNotFoundError(f"Missing analysis summary JSON: {summary_path(analysis_root)}")
        summary_payload = loaded_summary
    if diagnostics_payload is None:
        diagnostics_payload = _load_json(analysis_table_path(analysis_root, "diagnostics_summary", "json"))
    if objective_components is None:
        objective_components = _load_json(analysis_table_path(analysis_root, "objective_components", "json"))
    if overlap_summary is None:
        overlap_summary = summary_payload.get("overlap_summary") if isinstance(summary_payload, dict) else None
    if analysis_used_payload is None:
        analysis_used_payload = _load_yaml(analysis_used_path(analysis_root))
    payload = build_report_payload(
        analysis_root=analysis_root,
        summary_payload=summary_payload,
        diagnostics_payload=diagnostics_payload,
        objective_components=objective_components,
        overlap_summary=overlap_summary,
        analysis_used_payload=analysis_used_payload,
    )
    write_report_json(report_json, payload)
    write_report_md(report_md, payload, analysis_root=analysis_root)
    return report_json, report_md
