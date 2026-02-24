"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/run_resolution.py

Resolve sample run selection and analyze eligibility for analysis workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.run_service import get_run, list_runs, load_run_status
from dnadesign.cruncher.artifacts.layout import manifest_path
from dnadesign.cruncher.config.schema_v3 import AnalysisConfig, CruncherConfig


def _status_text(status: object) -> str:
    return str(status or "").strip().lower()


def _run_status_detail(run_dir: Path) -> str | None:
    try:
        payload = load_run_status(run_dir)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    status_message = payload.get("status_message")
    if isinstance(status_message, str) and status_message.strip():
        return status_message.strip()
    return None


def _is_analyzable_sample_run(run: object) -> bool:
    status = _status_text(getattr(run, "status", None))
    if status in {"failed", "aborted", "running"}:
        return False
    run_dir = getattr(run, "run_dir", None)
    if not isinstance(run_dir, Path):
        return False
    return manifest_path(run_dir).exists()


def _latest_unavailable_reason(run: object) -> str:
    run_name = str(getattr(run, "name", "<unknown>"))
    status = _status_text(getattr(run, "status", None)) or "unknown"
    run_dir = getattr(run, "run_dir", None)
    has_manifest = isinstance(run_dir, Path) and manifest_path(run_dir).exists()
    detail = _run_status_detail(run_dir) if isinstance(run_dir, Path) else None
    if not has_manifest:
        if status in {"failed", "aborted"} and detail:
            return f"run '{run_name}' {status}: {detail}"
        return f"run '{run_name}' status={status} is missing run_manifest.json"
    if status in {"failed", "aborted"}:
        if detail:
            return f"run '{run_name}' {status}: {detail}"
        return f"run '{run_name}' status={status}"
    if status == "running":
        return f"run '{run_name}' is still running"
    return f"run '{run_name}' is not ready for analysis (status={status})"


def _resolve_run_names(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    analysis_cfg: AnalysisConfig,
    runs_override: list[str] | None,
    use_latest: bool,
) -> list[str]:
    if runs_override:
        return runs_override
    if analysis_cfg.run_selector == "explicit":
        if not analysis_cfg.runs:
            raise ValueError("analysis.run_selector=explicit requires analysis.runs to be non-empty")
        return list(analysis_cfg.runs)
    if use_latest or analysis_cfg.run_selector == "latest":
        runs = list_runs(cfg, config_path, stage="sample")
        if not runs:
            raise ValueError("No sample runs found for analysis.")
        for run in runs:
            if _is_analyzable_sample_run(run):
                return [run.name]
        latest_reason = _latest_unavailable_reason(runs[0])
        raise ValueError(
            "No completed sample runs found for analysis. "
            f"Latest sample run unavailable: {latest_reason}. "
            "Re-run sampling with `cruncher sample -c <CONFIG>`."
        )
    raise ValueError("analysis.run_selector must be 'latest' or 'explicit'")


def _resolve_run_dir(cfg: CruncherConfig, config_path: Path, run_name: str) -> Path:
    run = get_run(cfg, config_path, run_name)
    if run.stage != "sample":
        raise ValueError(f"Run '{run_name}' is not a sample run (stage={run.stage}).")
    return run.run_dir
