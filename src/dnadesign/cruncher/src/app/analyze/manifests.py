"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/manifests.py

Build analysis plot/table manifests and corresponding artifact entries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.analysis.layout import (
    analysis_manifest_path,
    plot_manifest_path,
    table_manifest_path,
)
from dnadesign.cruncher.analysis.plot_registry import PLOT_SPECS
from dnadesign.cruncher.artifacts.entries import artifact_entry


@dataclass
class AnalysisManifestBundle:
    plot_manifest: dict[str, object]
    plot_manifest_file: Path
    table_manifest: dict[str, object]
    table_manifest_file: Path
    analysis_manifest_payload: dict[str, object]
    analysis_manifest_file: Path
    analysis_manifest_entries: list[dict[str, object]]
    plot_artifacts: list[dict[str, object]]
    analysis_artifacts: list[dict[str, object]]


def build_analysis_manifests(
    *,
    analysis_id: str,
    created_at: str,
    analysis_root: Path,
    sample_dir: Path,
    analysis_used_file: Path,
    plot_format: str,
    plots: object,
    tier0_plot_keys: set[str],
    mcmc_plot_keys: set[str],
    extra_plots: bool,
    extra_tables: bool,
    mcmc_diagnostics: bool,
    per_pwm_path: Path | None,
    score_summary_path: Path,
    topk_path: Path,
    joint_metrics_path: Path,
    overlap_summary_path: Path,
    elite_overlap_path: Path,
    diagnostics_path: Path,
    objective_components_path: Path,
    elites_mmr_summary_path: Path | None,
    move_stats_summary_path: Path | None,
    move_stats_path: Path | None,
    pt_swap_pairs_path: Path | None,
    auto_opt_table_path: Path | None,
    auto_opt_plot_path: Path | None,
    table_ext: str,
) -> AnalysisManifestBundle:
    plot_manifest = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "plots": [],
    }
    plot_artifacts: list[dict[str, object]] = []
    for spec in PLOT_SPECS:
        enabled_flag = getattr(plots, spec.key, False)
        spec_outputs = [out.replace("{ext}", plot_format) for out in spec.outputs]
        outputs = []
        missing = []
        for output in spec_outputs:
            out_path = analysis_root / output
            exists = out_path.exists()
            outputs.append({"path": output, "exists": exists})
            if enabled_flag and not exists:
                missing.append(output)
            elif enabled_flag:
                kind = "plot" if out_path.suffix.lower() in {".png", ".pdf", ".svg"} else "text"
                plot_artifacts.append(
                    artifact_entry(
                        out_path,
                        sample_dir,
                        kind=kind,
                        label=f"{spec.label} ({out_path.name})",
                        stage="analysis",
                    )
                )
        plot_manifest["plots"].append(
            {
                "key": spec.key,
                "label": spec.label,
                "group": spec.group,
                "description": spec.description,
                "requires": list(spec.requires),
                "enabled": enabled_flag,
                "outputs": outputs,
                "missing_outputs": missing,
                "generated": enabled_flag and any(out["exists"] for out in outputs),
            }
        )
    if auto_opt_plot_path is not None:
        plot_manifest["plots"].append(
            {
                "key": "auto_opt_tradeoffs",
                "label": f"Auto-opt tradeoffs ({plot_format.upper()})",
                "group": "auto_opt",
                "description": "Balance vs auto-opt scorecard metric across auto-opt pilots.",
                "requires": [],
                "enabled": True,
                "outputs": [
                    {
                        "path": str(auto_opt_plot_path.relative_to(sample_dir)),
                        "exists": auto_opt_plot_path.exists(),
                    }
                ],
                "missing_outputs": []
                if auto_opt_plot_path.exists()
                else [str(auto_opt_plot_path.relative_to(sample_dir))],
                "generated": auto_opt_plot_path.exists(),
            }
        )
        plot_artifacts.append(
            artifact_entry(
                auto_opt_plot_path,
                sample_dir,
                kind="plot",
                label=f"Auto-opt tradeoffs ({plot_format.upper()})",
                stage="analysis",
            )
        )

    plot_manifest_file = plot_manifest_path(analysis_root)
    plot_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    plot_manifest_file.write_text(json.dumps(plot_manifest, indent=2))

    table_label_suffix = "Parquet" if table_ext == "parquet" else "CSV"
    tables_manifest_entries: list[dict[str, object]] = []
    if per_pwm_path is not None:
        tables_manifest_entries.append(
            {
                "key": "per_pwm",
                "label": f"Per-PWM scores ({table_label_suffix})",
                "path": str(per_pwm_path.relative_to(sample_dir)),
                "exists": per_pwm_path.exists(),
            }
        )
    tables_manifest_entries.extend(
        [
            {
                "key": "score_summary",
                "label": f"Per-TF summary ({table_label_suffix})",
                "path": str(score_summary_path.relative_to(sample_dir)),
                "exists": score_summary_path.exists(),
            },
            {
                "key": "elite_topk",
                "label": f"Elite top-K ({table_label_suffix})",
                "path": str(topk_path.relative_to(sample_dir)),
                "exists": topk_path.exists(),
            },
            {
                "key": "joint_metrics",
                "label": f"Joint score metrics ({table_label_suffix})",
                "path": str(joint_metrics_path.relative_to(sample_dir)),
                "exists": joint_metrics_path.exists(),
            },
            {
                "key": "overlap_summary",
                "label": f"Overlap summary ({table_label_suffix})",
                "path": str(overlap_summary_path.relative_to(sample_dir)),
                "exists": overlap_summary_path.exists(),
            },
            {
                "key": "elite_overlap",
                "label": f"Elite overlap details ({table_label_suffix})",
                "path": str(elite_overlap_path.relative_to(sample_dir)),
                "exists": elite_overlap_path.exists(),
            },
            {
                "key": "diagnostics",
                "label": "Diagnostics summary (JSON)",
                "path": str(diagnostics_path.relative_to(sample_dir)),
                "exists": diagnostics_path.exists(),
            },
            {
                "key": "objective_components",
                "label": "Objective components (JSON)",
                "path": str(objective_components_path.relative_to(sample_dir)),
                "exists": objective_components_path.exists(),
            },
        ]
    )
    if elites_mmr_summary_path is not None:
        tables_manifest_entries.append(
            {
                "key": "elites_mmr_summary",
                "label": f"Elites MMR summary ({table_label_suffix})",
                "path": str(elites_mmr_summary_path.relative_to(sample_dir)),
                "exists": elites_mmr_summary_path.exists(),
            }
        )
    table_manifest = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "tables": tables_manifest_entries,
    }
    if move_stats_summary_path is not None:
        table_manifest["tables"].append(
            {
                "key": "move_stats_summary",
                "label": f"Move stats summary ({table_label_suffix})",
                "path": str(move_stats_summary_path.relative_to(sample_dir)),
                "exists": move_stats_summary_path.exists(),
            }
        )
    if move_stats_path is not None:
        table_manifest["tables"].append(
            {
                "key": "move_stats",
                "label": f"Move stats ({table_label_suffix})",
                "path": str(move_stats_path.relative_to(sample_dir)),
                "exists": move_stats_path.exists(),
            }
        )
    if pt_swap_pairs_path is not None:
        table_manifest["tables"].append(
            {
                "key": "pt_swap_pairs",
                "label": f"PT swap by pair ({table_label_suffix})",
                "path": str(pt_swap_pairs_path.relative_to(sample_dir)),
                "exists": pt_swap_pairs_path.exists(),
            }
        )
    if auto_opt_table_path is not None:
        table_manifest["tables"].append(
            {
                "key": "auto_opt_pilots",
                "label": f"Auto-opt pilot scorecard ({table_label_suffix})",
                "path": str(auto_opt_table_path.relative_to(sample_dir)),
                "exists": auto_opt_table_path.exists(),
            }
        )
    table_manifest_file = table_manifest_path(analysis_root)
    table_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    table_manifest_file.write_text(json.dumps(table_manifest, indent=2))

    def _plot_reason(key: str) -> str:
        if key in tier0_plot_keys:
            return "default"
        if key in mcmc_plot_keys:
            return "mcmc_diagnostics"
        return "extra_plots"

    def _table_reason(key: str) -> str:
        if key in {"move_stats", "pt_swap_pairs"}:
            return "mcmc_diagnostics"
        if key in {"move_stats_summary"}:
            return "default"
        if key in {"auto_opt_pilots"}:
            return "extra_tables"
        if key in {"per_pwm"}:
            return "scatter_pwm"
        return "default"

    analysis_manifest_entries: list[dict[str, object]] = [
        {
            "path": str(analysis_used_file.relative_to(sample_dir)),
            "kind": "config",
            "label": "Analysis settings",
            "reason": "default",
            "exists": analysis_used_file.exists(),
        },
        {
            "path": str(plot_manifest_file.relative_to(sample_dir)),
            "kind": "manifest",
            "label": "Plot manifest",
            "reason": "default",
            "exists": plot_manifest_file.exists(),
        },
        {
            "path": str(table_manifest_file.relative_to(sample_dir)),
            "kind": "manifest",
            "label": "Table manifest",
            "reason": "default",
            "exists": table_manifest_file.exists(),
        },
    ]
    for table in table_manifest.get("tables", []):
        if not isinstance(table, dict):
            continue
        key = str(table.get("key") or "")
        path = table.get("path")
        analysis_manifest_entries.append(
            {
                "path": path,
                "kind": "table",
                "label": table.get("label"),
                "reason": _table_reason(key),
                "exists": bool(table.get("exists")),
                "key": key,
            }
        )
    for plot in plot_manifest.get("plots", []):
        if not isinstance(plot, dict):
            continue
        key = str(plot.get("key") or "")
        enabled = bool(plot.get("enabled"))
        for output in plot.get("outputs", []):
            if not isinstance(output, dict):
                continue
            if not enabled and not output.get("exists"):
                continue
            analysis_manifest_entries.append(
                {
                    "path": output.get("path"),
                    "kind": "plot",
                    "label": plot.get("label"),
                    "reason": _plot_reason(key),
                    "exists": bool(output.get("exists")),
                    "key": key,
                }
            )

    analysis_manifest_file = analysis_manifest_path(analysis_root)
    analysis_manifest_payload = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "extra_plots": extra_plots,
        "extra_tables": extra_tables,
        "mcmc_diagnostics": mcmc_diagnostics,
        "artifacts": analysis_manifest_entries,
    }
    analysis_manifest_file.write_text(json.dumps(analysis_manifest_payload, indent=2))

    analysis_artifacts: list[dict[str, object]] = [
        artifact_entry(
            analysis_used_file,
            sample_dir,
            kind="config",
            label="Analysis settings",
            stage="analysis",
        ),
    ]
    if per_pwm_path is not None:
        analysis_artifacts.append(
            artifact_entry(
                per_pwm_path,
                sample_dir,
                kind="table",
                label=f"Per-PWM scores ({table_label_suffix})",
                stage="analysis",
            )
        )
    analysis_artifacts.extend(
        [
            artifact_entry(
                score_summary_path,
                sample_dir,
                kind="table",
                label=f"Per-TF summary ({table_label_suffix})",
                stage="analysis",
            ),
            artifact_entry(
                topk_path,
                sample_dir,
                kind="table",
                label=f"Elite top-K ({table_label_suffix})",
                stage="analysis",
            ),
            artifact_entry(
                joint_metrics_path,
                sample_dir,
                kind="table",
                label=f"Joint score metrics ({table_label_suffix})",
                stage="analysis",
            ),
            artifact_entry(
                overlap_summary_path,
                sample_dir,
                kind="table",
                label=f"Overlap summary ({table_label_suffix})",
                stage="analysis",
            ),
            artifact_entry(
                elite_overlap_path,
                sample_dir,
                kind="table",
                label=f"Elite overlap details ({table_label_suffix})",
                stage="analysis",
            ),
            artifact_entry(
                diagnostics_path,
                sample_dir,
                kind="json",
                label="Diagnostics summary (JSON)",
                stage="analysis",
            ),
            artifact_entry(
                objective_components_path,
                sample_dir,
                kind="json",
                label="Objective components (JSON)",
                stage="analysis",
            ),
        ]
    )
    if elites_mmr_summary_path is not None:
        analysis_artifacts.append(
            artifact_entry(
                elites_mmr_summary_path,
                sample_dir,
                kind="table",
                label=f"Elites MMR summary ({table_label_suffix})",
                stage="analysis",
            )
        )
    if move_stats_summary_path is not None:
        analysis_artifacts.append(
            artifact_entry(
                move_stats_summary_path,
                sample_dir,
                kind="table",
                label=f"Move stats summary ({table_label_suffix})",
                stage="analysis",
            )
        )
    if move_stats_path is not None:
        analysis_artifacts.append(
            artifact_entry(
                move_stats_path,
                sample_dir,
                kind="table",
                label=f"Move stats ({table_label_suffix})",
                stage="analysis",
            )
        )
    if pt_swap_pairs_path is not None:
        analysis_artifacts.append(
            artifact_entry(
                pt_swap_pairs_path,
                sample_dir,
                kind="table",
                label=f"PT swap by pair ({table_label_suffix})",
                stage="analysis",
            )
        )
    if auto_opt_table_path is not None:
        analysis_artifacts.append(
            artifact_entry(
                auto_opt_table_path,
                sample_dir,
                kind="table",
                label=f"Auto-opt pilot scorecard ({table_label_suffix})",
                stage="analysis",
            )
        )
    analysis_artifacts.append(
        artifact_entry(
            plot_manifest_file,
            sample_dir,
            kind="json",
            label="Plot manifest",
            stage="analysis",
        )
    )
    analysis_artifacts.append(
        artifact_entry(
            table_manifest_file,
            sample_dir,
            kind="json",
            label="Table manifest",
            stage="analysis",
        )
    )
    analysis_artifacts.append(
        artifact_entry(
            analysis_manifest_file,
            sample_dir,
            kind="json",
            label="Analysis manifest",
            stage="analysis",
        )
    )
    analysis_artifacts.extend(plot_artifacts)

    return AnalysisManifestBundle(
        plot_manifest=plot_manifest,
        plot_manifest_file=plot_manifest_file,
        table_manifest=table_manifest,
        table_manifest_file=table_manifest_file,
        analysis_manifest_payload=analysis_manifest_payload,
        analysis_manifest_file=analysis_manifest_file,
        analysis_manifest_entries=analysis_manifest_entries,
        plot_artifacts=plot_artifacts,
        analysis_artifacts=analysis_artifacts,
    )
