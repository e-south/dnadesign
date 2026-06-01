"""Cross-root suite review for DenseGen OPAL probe runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ..core.constants import ACTIVE_LABEL_FAMILY_IDS, CAMPAIGNS, DEFAULT_SUITE_ID, DEFAULT_SUITE_SEEDS, ORACLES, SPLITS
from .suite_replicates import (
    numeric_mean_ci_summary,
    replicate_summary,
    write_replicate_ci_plot,
    write_replicate_summary_csv,
)


def build_probe_suite_review(
    run_roots: list[Path],
    *,
    out_dir: Path | None = None,
    expected_seeds: tuple[int, ...] = DEFAULT_SUITE_SEEDS,
) -> dict[str, Any]:
    root_rows = [_root_summary(Path(root).resolve()) for root in run_roots]
    observed_seeds = [row["seed"] for row in root_rows if row.get("seed") is not None]
    expected_seed_set = set(expected_seeds)
    problems: list[str] = []
    for seed in sorted(expected_seed_set - set(observed_seeds)):
        problems.append(f"expected_seed_missing:{seed}")
    for seed in sorted(set(observed_seeds) - expected_seed_set):
        problems.append(f"unexpected_seed:{seed}")
    for seed in sorted(seed for seed in set(observed_seeds) if observed_seeds.count(seed) > 1):
        problems.append(f"duplicate_seed:{seed}")
    for row in root_rows:
        problems.extend(_root_completion_problems(row))

    trajectory_pairs = [pair for row in root_rows for pair in row.get("trajectory_pairs", [])]
    null_attention_rows = [row for root in root_rows for row in root.get("null_attention_rows", [])]
    seed_replicates = replicate_summary(trajectory_pairs)
    payload = {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.suite_review.v1",
        "suite_id": DEFAULT_SUITE_ID,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "expected_seeds": list(expected_seeds),
        "status": "ok" if not problems else "attention",
        "problems": problems,
        "root_count": len(root_rows),
        "roots": root_rows,
        "trajectory_summary": _trajectory_summary(trajectory_pairs),
        "replicate_summary": seed_replicates,
        "null_attention": {
            "count": len(null_attention_rows),
            "rows": null_attention_rows,
        },
        "plot_quality": _plot_quality_summary(root_rows),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "suite_review.json"
        markdown_path = out_dir / "suite_review.md"
        replicate_csv_path = out_dir / "replicate_seed_mean_ci.csv"
        auc_plot_path = out_dir / "paired_auc_delta_mean_ci.png"
        final_plot_path = out_dir / "final_positive_minus_null_lift_mean_ci.png"
        write_replicate_summary_csv(seed_replicates, replicate_csv_path)
        write_replicate_ci_plot(
            seed_replicates,
            metric="paired_auc_delta",
            path=auc_plot_path,
            title="Paired AUC Delta Across Seed Replicates",
            ylabel="Mean positive - null AUC delta",
        )
        write_replicate_ci_plot(
            seed_replicates,
            metric="final_positive_minus_null_lift",
            path=final_plot_path,
            title="Final Lift Delta Across Seed Replicates",
            ylabel="Mean final positive - null lift",
        )
        payload["artifacts"] = {
            "suite_review": str(manifest_path),
            "suite_review_markdown": str(markdown_path),
            "replicate_seed_mean_ci_csv": str(replicate_csv_path),
            "paired_auc_delta_mean_ci_plot": str(auc_plot_path),
            "final_positive_minus_null_lift_mean_ci_plot": str(final_plot_path),
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(_render_suite_markdown(payload), encoding="utf-8")
    return payload


def _root_summary(root: Path) -> dict[str, Any]:
    status = _read_json(root / "reports" / "status.json")
    review = _read_json(root / "reports" / "review_manifest.json")
    metrics = _read_json(root / "reports" / "metrics.json")
    plan = _read_json(root / "probe_plan.json")
    plan_payload = plan.get("plan") if isinstance(plan.get("plan"), Mapping) else {}
    runs = metrics.get("runs") if isinstance(metrics.get("runs"), list) else []
    rounds = metrics.get("rounds") if isinstance(metrics.get("rounds"), list) else []
    seed = _seed_from_metrics(runs) or _seed_from_plan(plan)
    gate_coverage = review.get("gate_coverage") if isinstance(review.get("gate_coverage"), Mapping) else {}
    plot_quality = review.get("plot_quality") if isinstance(review.get("plot_quality"), Mapping) else {}
    campaign_reviews = (
        review.get("opal_campaign_reviews") if isinstance(review.get("opal_campaign_reviews"), list) else []
    )
    trajectory = review.get("trajectory_qa") if isinstance(review.get("trajectory_qa"), Mapping) else {}
    round_dynamics = review.get("round_dynamics") if isinstance(review.get("round_dynamics"), list) else []
    return {
        "run_root": str(root),
        "seed": seed,
        "status": status.get("status"),
        "decision": status.get("decision") or review.get("decision") or metrics.get("decision"),
        "problems": list(status.get("problems") or []) + list(review.get("problems") or []),
        "metrics_run_count": len(runs),
        "round_metric_count": len(rounds),
        "expected_run_count": _expected_run_count(plan_payload),
        "expected_round_metric_count": _expected_run_count(plan_payload) * _expected_round_count(plan_payload),
        "final_rounds": sorted(
            {
                int(row.get("as_of_round"))
                for row in runs
                if isinstance(row, Mapping) and row.get("as_of_round") is not None
            }
        ),
        "gate_coverage": gate_coverage,
        "plot_quality": plot_quality,
        "nested_warning_count": sum(
            len(row.get("warnings") or []) for row in campaign_reviews if isinstance(row, Mapping)
        ),
        "nested_stale_artifact_count": sum(
            len(row.get("stale_artifacts") or []) for row in campaign_reviews if isinstance(row, Mapping)
        ),
        "trajectory_pairs": list(trajectory.get("pairs") or []),
        "null_attention_rows": _null_attention_rows(seed=seed, rows=round_dynamics),
    }


def _root_completion_problems(row: Mapping[str, Any]) -> list[str]:
    prefix = f"root:{Path(str(row.get('run_root'))).name}"
    problems = [f"{prefix}:missing_status_ok"] if row.get("status") != "ok" else []
    decision = str(row.get("decision") or "")
    if not decision.startswith("PASS_"):
        problems.append(f"{prefix}:decision_not_pass:{decision or 'missing'}")
    if row.get("metrics_run_count") != row.get("expected_run_count"):
        problems.append(f"{prefix}:metrics_run_count:{row.get('metrics_run_count')}")
    if row.get("round_metric_count") != row.get("expected_round_metric_count"):
        problems.append(f"{prefix}:round_metric_count:{row.get('round_metric_count')}")
    if row.get("final_rounds") != [11]:
        problems.append(f"{prefix}:final_rounds:{row.get('final_rounds')}")
    coverage = row.get("gate_coverage") if isinstance(row.get("gate_coverage"), Mapping) else {}
    if coverage.get("positive_null_pairs_complete") is not True:
        problems.append(f"{prefix}:positive_null_pairs_incomplete")
    if coverage.get("omitted_scored_gates"):
        problems.append(f"{prefix}:omitted_scored_gates:{','.join(coverage.get('omitted_scored_gates') or [])}")
    plot_quality = row.get("plot_quality") if isinstance(row.get("plot_quality"), Mapping) else {}
    if plot_quality.get("status") != "ok":
        problems.append(f"{prefix}:plot_quality:{plot_quality.get('status', 'missing')}")
    if int(row.get("nested_warning_count") or 0):
        problems.append(f"{prefix}:nested_warnings:{row.get('nested_warning_count')}")
    if int(row.get("nested_stale_artifact_count") or 0):
        problems.append(f"{prefix}:nested_stale_artifacts:{row.get('nested_stale_artifact_count')}")
    problems.extend(f"{prefix}:{problem}" for problem in row.get("problems") or [])
    return problems


def _expected_run_count(plan: Mapping[str, Any]) -> int:
    planned_runs = _positive_int(plan.get("planned_runs"))
    if planned_runs is not None:
        return planned_runs
    active_families = plan.get("active_label_families")
    family_count = (
        len(active_families) if isinstance(active_families, list) and active_families else len(ACTIVE_LABEL_FAMILY_IDS)
    )
    return family_count * len(CAMPAIGNS) * len(ORACLES) * len(SPLITS)


def _expected_round_count(plan: Mapping[str, Any]) -> int:
    return _positive_int(plan.get("rounds")) or 12


def _positive_int(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def _trajectory_summary(pairs: list[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(pair["paired_auc_delta"]) for pair in pairs if pair.get("paired_auc_delta") is not None]
    final_deltas = [
        float(pair["final_positive_minus_null_lift"])
        for pair in pairs
        if pair.get("final_positive_minus_null_lift") is not None
    ]
    return {
        "pair_count": len(pairs),
        "paired_auc_delta": numeric_mean_ci_summary(deltas),
        "final_positive_minus_null_lift": numeric_mean_ci_summary(final_deltas),
    }


def _plot_quality_summary(root_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    qualities = [row.get("plot_quality") for row in root_rows if isinstance(row.get("plot_quality"), Mapping)]
    return {
        "root_count": len(qualities),
        "status": "ok" if qualities and all(row.get("status") == "ok" for row in qualities) else "attention",
        "plot_count": sum(int(row.get("plot_count") or 0) for row in qualities),
        "problem_count": sum(int(row.get("problem_count") or 0) for row in qualities),
    }


def _null_attention_rows(*, seed: int | None, rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("null_transient_spike") or row.get("null_final_threshold_exceeded"):
            out.append(
                {
                    "seed": seed,
                    "run_key": row.get("run_key"),
                    "campaign": row.get("campaign"),
                    "label_family_id": row.get("label_family_id"),
                    "split_id": row.get("split_id"),
                    "max_round": row.get("max_round"),
                    "max_lift": row.get("max_lift"),
                    "final_lift": row.get("final_lift"),
                    "null_transient_spike": row.get("null_transient_spike"),
                    "null_final_threshold_exceeded": row.get("null_final_threshold_exceeded"),
                }
            )
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def _seed_from_metrics(runs: Any) -> int | None:
    if not isinstance(runs, list):
        return None
    seeds = {int(row["seed"]) for row in runs if isinstance(row, Mapping) and row.get("seed") is not None}
    return next(iter(seeds)) if len(seeds) == 1 else None


def _seed_from_plan(plan_record: Mapping[str, Any]) -> int | None:
    plan = plan_record.get("plan") if isinstance(plan_record.get("plan"), Mapping) else {}
    seed = plan.get("seed")
    return int(seed) if seed is not None else None


def _render_suite_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# DenseGen OPAL Probe Suite Review",
        "",
        f"- status: `{payload.get('status')}`",
        f"- suite_id: `{payload.get('suite_id')}`",
        f"- expected_seeds: `{', '.join(map(str, payload.get('expected_seeds') or []))}`",
        f"- root_count: `{payload.get('root_count')}`",
        "",
        "## Problems",
    ]
    problems = payload.get("problems") or []
    lines.extend([f"- `{problem}`" for problem in problems] or ["- none"])
    lines.extend(["", "## Roots"])
    for row in payload.get("roots") or []:
        root_line = (
            "- seed `{seed}` decision `{decision}` status `{status}` runs `{runs}` rounds `{rounds}` plots `{plots}`"
        )
        lines.append(
            root_line.format(
                seed=row.get("seed"),
                decision=row.get("decision"),
                status=row.get("status"),
                runs=row.get("metrics_run_count"),
                rounds=row.get("round_metric_count"),
                plots=(row.get("plot_quality") or {}).get("plot_count"),
            )
        )
    lines.extend(["", "## Trajectory", f"```json\n{json.dumps(payload.get('trajectory_summary'), indent=2)}\n```"])
    replicate_summary = (
        payload.get("replicate_summary") if isinstance(payload.get("replicate_summary"), Mapping) else {}
    )
    lines.extend(
        [
            "",
            "## Seed Replicate Means",
            "",
            f"- replicate_unit: `{replicate_summary.get('replicate_unit', 'seed')}`",
            f"- interval: `{replicate_summary.get('interval_kind', 'student_t_mean_ci')}` "
            f"`{replicate_summary.get('confidence_level', 0.95)}`",
            f"- groups: `{replicate_summary.get('group_count', 0)}`",
        ]
    )
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), Mapping) else {}
    if artifacts:
        for key in (
            "replicate_seed_mean_ci_csv",
            "paired_auc_delta_mean_ci_plot",
            "final_positive_minus_null_lift_mean_ci_plot",
        ):
            if artifacts.get(key):
                lines.append(f"- {key}: `{artifacts[key]}`")
    lines.extend(["", "## Null Attention", f"- count: `{(payload.get('null_attention') or {}).get('count')}`"])
    return "\n".join(lines) + "\n"
