"""
--------------------------------------------------------------------------------
cruncher
src/dnadesign/cruncher/src/app/portfolio_preflight.py

Preflight readiness checks and preparation command helpers for portfolio workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex

from dnadesign.cruncher.analysis.layout import analysis_root, summary_path
from dnadesign.cruncher.artifacts.layout import (
    elites_hits_path,
    elites_path,
    elites_yaml_path,
    manifest_path,
    random_baseline_hits_path,
    random_baseline_path,
    run_export_sequences_manifest_path,
    sequences_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource, PortfolioSpec


def _preflight_source_readiness(source: PortfolioSource) -> dict[str, object]:
    issues: list[str] = []
    run_dir = source.run_dir
    if not run_dir.exists():
        issues.append(f"missing run directory: {run_dir}")
        return {
            "source_id": str(source.id),
            "workspace_name": source.workspace.name,
            "workspace_path": str(source.workspace),
            "run_dir": str(run_dir),
            "ready": False,
            "issues": issues,
        }
    if not run_dir.is_dir():
        issues.append(f"run path is not a directory: {run_dir}")
        return {
            "source_id": str(source.id),
            "workspace_name": source.workspace.name,
            "workspace_path": str(source.workspace),
            "run_dir": str(run_dir),
            "ready": False,
            "issues": issues,
        }

    run_manifest_file = manifest_path(run_dir)
    if not run_manifest_file.exists():
        issues.append(f"missing metadata manifest: {run_manifest_file}")
    else:
        try:
            run_manifest = load_manifest(run_dir)
        except Exception as exc:
            issues.append(f"invalid metadata manifest: {run_manifest_file} ({exc})")
        else:
            run_stage = str(run_manifest.get("stage") or "").strip().lower()
            if run_stage != "sample":
                issues.append(f"run manifest stage must be 'sample' (found {run_stage!r})")
            top_k_raw = run_manifest.get("top_k")
            if not isinstance(top_k_raw, int) or top_k_raw < 1:
                issues.append("run manifest top_k must be a positive integer")

    analysis_summary_file = summary_path(analysis_root(run_dir))
    if not analysis_summary_file.exists():
        issues.append(f"missing analysis summary: {analysis_summary_file}")
        analyze_prereqs = (
            sequences_path(run_dir),
            elites_path(run_dir),
            elites_yaml_path(run_dir),
            elites_hits_path(run_dir),
            random_baseline_path(run_dir),
            random_baseline_hits_path(run_dir),
        )
        for prereq in analyze_prereqs:
            if not prereq.exists():
                issues.append(f"missing sample artifact required to recompute analysis summary: {prereq}")

    export_manifest_file = run_export_sequences_manifest_path(run_dir)
    if not export_manifest_file.exists():
        issues.append(f"missing export manifest: {export_manifest_file}")
    else:
        try:
            export_payload = json.loads(export_manifest_file.read_text())
        except Exception as exc:
            issues.append(f"invalid export manifest JSON: {export_manifest_file} ({exc})")
        else:
            if not isinstance(export_payload, dict):
                issues.append(f"export manifest must be a JSON object: {export_manifest_file}")
            else:
                files = export_payload.get("files")
                if not isinstance(files, dict):
                    issues.append(f"export manifest missing files mapping: {export_manifest_file}")
                else:
                    relative_elites = files.get("elites")
                    if not isinstance(relative_elites, str) or not relative_elites.strip():
                        issues.append(f"export manifest missing elites path: {export_manifest_file}")
                    else:
                        windows_path = (run_dir / relative_elites).resolve()
                        try:
                            windows_path.relative_to(run_dir.resolve())
                        except ValueError:
                            issues.append(f"export elites path escapes run directory: {windows_path}")
                        else:
                            if not windows_path.exists():
                                issues.append(f"missing export elites table: {windows_path}")

    return {
        "source_id": str(source.id),
        "workspace_name": source.workspace.name,
        "workspace_path": str(source.workspace),
        "run_dir": str(run_dir),
        "ready": not issues,
        "issues": issues,
    }


def _collect_source_readiness(spec: PortfolioSpec) -> dict[str, dict[str, object]]:
    readiness: dict[str, dict[str, object]] = {}
    for source in spec.sources:
        readiness[str(source.id)] = _preflight_source_readiness(source)
    return readiness


def _resolve_source_label(source: PortfolioSource) -> str:
    if source.label:
        return str(source.label)
    return source.workspace.name


def _render_prepare_runbook_path(source: PortfolioSource) -> str:
    if source.prepare is None:
        return ""
    runbook = source.prepare.runbook
    try:
        return str(runbook.relative_to(source.workspace))
    except ValueError:
        return str(runbook)


def _render_prepare_runbook_command(source: PortfolioSource, *, include_steps: bool) -> str:
    if source.prepare is None:
        raise ValueError(f"Portfolio source has no prepare runbook: id={source.id!r}")
    workspace_selector = shlex.quote(str(source.workspace.resolve()))
    runbook_selector = shlex.quote(_render_prepare_runbook_path(source))
    command = f"cruncher workspaces run --workspace {workspace_selector} --runbook {runbook_selector}"
    if include_steps and source.prepare.step_ids:
        step_args = " ".join(f"--step {shlex.quote(step_id)}" for step_id in source.prepare.step_ids)
        command = f"{command} {step_args}"
    return command


def _requires_full_runbook_prepare(issues: list[str]) -> bool:
    full_runbook_markers = (
        "missing run directory:",
        "run path is not a directory:",
        "missing metadata manifest:",
        "run manifest stage must be",
        "run manifest top_k must be",
        "missing sample artifact required to recompute analysis summary:",
    )
    return any(issue.startswith(full_runbook_markers) for issue in issues)


def _raise_aggregate_only_preflight(spec: PortfolioSpec, readiness: dict[str, dict[str, object]]) -> None:
    failing = [record for record in readiness.values() if not bool(record.get("ready"))]
    if not failing:
        return
    lines = [
        "Portfolio aggregate_only preflight failed: required source artifacts are missing or invalid.",
        "Each source must already have sample + analyze summary + export outputs.",
    ]
    for record in failing:
        source_id = str(record["source_id"])
        workspace_name = str(record["workspace_name"])
        run_dir = str(record["run_dir"])
        lines.append(f"- source={source_id} workspace={workspace_name} run_dir={run_dir}")
        for issue in list(record.get("issues", [])):
            lines.append(f"    * {issue}")

        source = next(item for item in spec.sources if str(item.id) == source_id)
        prepare = getattr(source, "prepare", None)
        if prepare is not None:
            lines.append(
                "    * nudge: switch this spec to execution.mode=prepare_then_aggregate "
                "or run source preparation before aggregate_only."
            )
            lines.append(f"    * nudge: {_render_prepare_runbook_command(source, include_steps=True)}")
            if _requires_full_runbook_prepare(list(record.get("issues", []))) and prepare.step_ids:
                lines.append(
                    "    * nudge: missing source run artifacts require a full runbook execution: "
                    f"{_render_prepare_runbook_command(source, include_steps=False)}"
                )
        else:
            lines.append(
                "    * nudge: run source pipeline first (`sample`, `analyze --summary`, `export sequences`) "
                "or add portfolio.sources[].prepare and use prepare_then_aggregate."
            )
    raise ValueError("\n".join(lines))
