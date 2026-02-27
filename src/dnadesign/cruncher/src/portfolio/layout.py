"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/portfolio/layout.py

Path and filename layout helpers for Portfolio aggregation artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

TABLE_FILE_PREFIX = "table__"
PLOT_FILE_PREFIX = "plot__"
PORTFOLIO_OUTPUT_ROOT = Path("outputs")
PORTFOLIO_META_DIR = "meta"
PORTFOLIO_TABLES_DIR = "tables"
PORTFOLIO_PLOTS_DIR = "plots"


def _ensure_inside_workspace(path: Path, workspace_root: Path) -> Path:
    resolved = path.resolve()
    workspace = workspace_root.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path must remain inside workspace: {resolved}") from exc
    return resolved


def _resolve_portfolio_root(workspace_root: Path) -> Path:
    return _ensure_inside_workspace(workspace_root / PORTFOLIO_OUTPUT_ROOT, workspace_root)


def _portfolio_context(portfolio_run_dir: Path) -> tuple[Path, str, str]:
    resolved = portfolio_run_dir.resolve()
    portfolio_id = resolved.name
    portfolio_name = resolved.parent.name
    outputs_root = resolved.parent.parent
    workspace_root = outputs_root.parent
    expected_root = _resolve_portfolio_root(workspace_root)
    if outputs_root != expected_root:
        raise ValueError(
            f"Portfolio run directory must be under <workspace>/outputs/<portfolio_name>/<portfolio_id>: {resolved}"
        )
    return workspace_root, portfolio_name, portfolio_id


def resolve_portfolio_run_dir(
    workspace_root: Path,
    portfolio_name: str,
    portfolio_id: str,
) -> Path:
    base = _resolve_portfolio_root(workspace_root)
    return _ensure_inside_workspace(base / portfolio_name / portfolio_id, workspace_root)


def portfolio_meta_dir(portfolio_run_dir: Path) -> Path:
    return portfolio_run_dir / PORTFOLIO_META_DIR


def portfolio_manifest_path(portfolio_run_dir: Path) -> Path:
    return portfolio_meta_dir(portfolio_run_dir) / "manifest.json"


def portfolio_status_path(portfolio_run_dir: Path) -> Path:
    return portfolio_meta_dir(portfolio_run_dir) / "status.json"


def portfolio_logs_dir(portfolio_run_dir: Path) -> Path:
    return portfolio_meta_dir(portfolio_run_dir) / "logs"


def portfolio_log_path(portfolio_run_dir: Path) -> Path:
    return portfolio_logs_dir(portfolio_run_dir) / "portfolio.log"


def portfolio_tables_dir(portfolio_run_dir: Path) -> Path:
    workspace_root, _, _ = _portfolio_context(portfolio_run_dir)
    return _ensure_inside_workspace(portfolio_run_dir.resolve() / PORTFOLIO_TABLES_DIR, workspace_root)


def portfolio_plots_dir(portfolio_run_dir: Path) -> Path:
    workspace_root, _, _ = _portfolio_context(portfolio_run_dir)
    return _ensure_inside_workspace(portfolio_run_dir.resolve() / PORTFOLIO_PLOTS_DIR, workspace_root)


def portfolio_table_path(portfolio_run_dir: Path, key: str, table_format: str = "parquet") -> Path:
    name = str(key).strip()
    if not name:
        raise ValueError("portfolio table key must be non-empty")
    ext = str(table_format).strip().lstrip(".")
    if ext not in {"parquet", "csv"}:
        raise ValueError(f"Unsupported portfolio table format: {table_format!r}")
    return portfolio_tables_dir(portfolio_run_dir) / f"{TABLE_FILE_PREFIX}{name}.{ext}"


def portfolio_plot_path(portfolio_run_dir: Path, key: str, plot_format: str = "pdf") -> Path:
    name = str(key).strip()
    if not name:
        raise ValueError("portfolio plot key must be non-empty")
    ext = str(plot_format).strip().lstrip(".")
    if ext not in {"pdf", "png"}:
        raise ValueError(f"Unsupported portfolio plot format: {plot_format!r}")
    filename = f"{PLOT_FILE_PREFIX}{name}.{ext}"
    return portfolio_plots_dir(portfolio_run_dir) / filename


def portfolio_plot_glob(portfolio_run_dir: Path) -> str:
    _portfolio_context(portfolio_run_dir)
    return f"{PLOT_FILE_PREFIX}*"
