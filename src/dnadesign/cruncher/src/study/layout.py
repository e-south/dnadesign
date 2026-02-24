"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/layout.py

Path and filename layout helpers for Study artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

TABLE_FILE_PREFIX = "table__"
PLOT_FILE_PREFIX = "plot__"
STUDY_OUTPUT_ROOT = Path("outputs/studies")
STUDY_PLOT_NAMESPACE = "study"


def _ensure_inside_workspace(path: Path, workspace_root: Path) -> Path:
    resolved = path.resolve()
    workspace = workspace_root.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path must remain inside workspace: {resolved}") from exc
    return resolved


def _resolve_study_root(workspace_root: Path) -> Path:
    return _ensure_inside_workspace(workspace_root / STUDY_OUTPUT_ROOT, workspace_root)


def _study_context(study_run_dir: Path) -> tuple[Path, str, str]:
    resolved = study_run_dir.resolve()
    study_id = resolved.name
    study_name = resolved.parent.name
    studies_root = resolved.parent.parent
    outputs_root = studies_root.parent
    workspace_root = outputs_root.parent
    expected_studies_root = _resolve_study_root(workspace_root)
    if studies_root != expected_studies_root:
        raise ValueError(
            f"Study run directory must be under <workspace>/outputs/studies/<study_name>/<study_id>: {resolved}"
        )
    return workspace_root, study_name, study_id


def resolve_study_run_dir(
    workspace_root: Path,
    study_name: str,
    study_id: str,
) -> Path:
    base = _resolve_study_root(workspace_root)
    return _ensure_inside_workspace(base / study_name / study_id, workspace_root)


def study_meta_dir(study_run_dir: Path) -> Path:
    return study_run_dir / "study"


def study_logs_dir(study_run_dir: Path) -> Path:
    return study_meta_dir(study_run_dir) / "logs"


def study_trials_root(study_run_dir: Path) -> Path:
    return study_run_dir / "trials"


def study_tables_dir(study_run_dir: Path) -> Path:
    return study_run_dir / "tables"


def study_plots_dir(study_run_dir: Path) -> Path:
    workspace_root, _, _ = _study_context(study_run_dir)
    return workspace_root / "outputs" / "plots"


def study_manifests_dir(study_run_dir: Path) -> Path:
    return study_run_dir / "manifests"


def spec_frozen_path(study_run_dir: Path) -> Path:
    return study_meta_dir(study_run_dir) / "spec_frozen.yaml"


def study_manifest_path(study_run_dir: Path) -> Path:
    return study_meta_dir(study_run_dir) / "study_manifest.json"


def study_status_path(study_run_dir: Path) -> Path:
    return study_meta_dir(study_run_dir) / "study_status.json"


def study_log_path(study_run_dir: Path) -> Path:
    return study_logs_dir(study_run_dir) / "study.log"


def trial_seed_root(study_run_dir: Path, *, trial_id: str, seed: int) -> Path:
    return study_trials_root(study_run_dir) / trial_id / f"seed_{seed}"


def trial_run_pointer_path(study_run_dir: Path, *, trial_id: str, seed: int) -> Path:
    return trial_seed_root(study_run_dir, trial_id=trial_id, seed=seed) / "run_dir.txt"


def study_table_path(study_run_dir: Path, key: str, table_format: str = "parquet") -> Path:
    name = str(key).strip()
    if not name:
        raise ValueError("study table key must be non-empty")
    ext = str(table_format).strip().lstrip(".")
    if ext not in {"parquet", "csv"}:
        raise ValueError(f"Unsupported study table format: {table_format!r}")
    return study_tables_dir(study_run_dir) / f"{TABLE_FILE_PREFIX}{name}.{ext}"


def study_plot_path(study_run_dir: Path, key: str, plot_format: str = "pdf") -> Path:
    name = str(key).strip()
    if not name:
        raise ValueError("study plot key must be non-empty")
    ext = str(plot_format).strip().lstrip(".")
    if ext not in {"pdf", "png"}:
        raise ValueError(f"Unsupported study plot format: {plot_format!r}")
    _, study_name, study_id = _study_context(study_run_dir)
    filename = f"{STUDY_PLOT_NAMESPACE}__{study_name}__{study_id}__{PLOT_FILE_PREFIX}{name}.{ext}"
    return study_plots_dir(study_run_dir) / filename


def study_plot_glob(study_run_dir: Path) -> str:
    _, study_name, study_id = _study_context(study_run_dir)
    return f"{STUDY_PLOT_NAMESPACE}__{study_name}__{study_id}__{PLOT_FILE_PREFIX}*"
