"""Ledger loading helpers for dashboard views."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..facade import read_predictions, read_runs
from .diagnostics import Diagnostics


@dataclass(frozen=True)
class LedgerDiagnostics:
    status: str
    path: str | None
    rows: int
    error: str | None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    run_id: str | None = None
    as_of_round: int | None = None


def _ledger_paths(workdir: Path | None) -> dict[str, Path | None]:
    if workdir is None:
        return {"runs": None, "labels": None, "preds_dir": None}
    outputs = Path(workdir) / "outputs"
    return {
        "runs": outputs / "ledger.runs.parquet",
        "labels": outputs / "ledger.labels.parquet",
        "preds_dir": outputs / "ledger.predictions",
    }


def load_ledger_runs(workdir: Path | None) -> tuple[pl.DataFrame, LedgerDiagnostics]:
    paths = _ledger_paths(workdir)
    runs_path = paths["runs"]
    if runs_path is None:
        diag = LedgerDiagnostics(
            status="missing_workdir",
            path=None,
            rows=0,
            error=None,
            diagnostics=Diagnostics().add_warning("Workdir is not set; ledger runs unavailable."),
        )
        return pl.DataFrame(), diag
    diag = LedgerDiagnostics(
        status="missing",
        path=str(runs_path),
        rows=0,
        error=None,
        diagnostics=Diagnostics(),
    )
    if not runs_path.exists():
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing",
                "diagnostics": diag.diagnostics.add_warning("ledger.runs.parquet not found."),
            }
        )
        return pl.DataFrame(), diag
    try:
        df = pl.read_parquet(runs_path)
    except Exception as exc:
        err_msg = f"Failed to read ledger runs: {exc}"
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "error": str(exc),
                "diagnostics": diag.diagnostics.add_error(err_msg),
            }
        )
        return pl.DataFrame(), diag
    diag = LedgerDiagnostics(**{**diag.__dict__, "status": "ok", "rows": df.height})
    return df, diag


def load_ledger_labels(workdir: Path | None) -> tuple[pl.DataFrame, LedgerDiagnostics]:
    paths = _ledger_paths(workdir)
    labels_path = paths["labels"]
    if labels_path is None:
        diag = LedgerDiagnostics(
            status="missing_workdir",
            path=None,
            rows=0,
            error=None,
            diagnostics=Diagnostics().add_warning("Workdir is not set; ledger labels unavailable."),
        )
        return pl.DataFrame(), diag
    diag = LedgerDiagnostics(
        status="missing",
        path=str(labels_path),
        rows=0,
        error=None,
        diagnostics=Diagnostics(),
    )
    if not labels_path.exists():
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing",
                "diagnostics": diag.diagnostics.add_warning("ledger.labels.parquet not found."),
            }
        )
        return pl.DataFrame(), diag
    try:
        df = pl.read_parquet(labels_path)
    except Exception as exc:
        err_msg = f"Failed to read ledger labels: {exc}"
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "error": str(exc),
                "diagnostics": diag.diagnostics.add_error(err_msg),
            }
        )
        return pl.DataFrame(), diag
    diag = LedgerDiagnostics(**{**diag.__dict__, "status": "ok", "rows": df.height})
    return df, diag


def load_ledger_predictions(
    workdir: Path | None,
    *,
    run_id: str | None,
    as_of_round: int | None = None,
) -> tuple[pl.DataFrame, LedgerDiagnostics]:
    paths = _ledger_paths(workdir)
    preds_dir = paths["preds_dir"]
    if preds_dir is None:
        diag = LedgerDiagnostics(
            status="missing_workdir",
            path=None,
            rows=0,
            error=None,
            diagnostics=Diagnostics().add_warning("Workdir is not set; ledger predictions unavailable."),
            run_id=run_id,
            as_of_round=as_of_round,
        )
        return pl.DataFrame(), diag
    diag = LedgerDiagnostics(
        status="missing",
        path=str(preds_dir),
        rows=0,
        error=None,
        diagnostics=Diagnostics(),
        run_id=run_id,
        as_of_round=as_of_round,
    )
    if not preds_dir.exists():
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing",
                "diagnostics": diag.diagnostics.add_warning("ledger.predictions directory not found."),
            }
        )
        return pl.DataFrame(), diag
    if run_id is None:
        missing_msg = "run_id is required to read ledger predictions without ambiguity."
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing_run_id",
                "error": missing_msg,
                "diagnostics": diag.diagnostics.add_error(missing_msg),
            }
        )
        return pl.DataFrame(), diag
    runs_path = paths["runs"]
    if runs_path is None or not runs_path.exists():
        missing_msg = "ledger.runs.parquet is required to resolve run_id → as_of_round."
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing_runs",
                "error": missing_msg,
                "diagnostics": diag.diagnostics.add_error(missing_msg),
            }
        )
        return pl.DataFrame(), diag

    try:
        runs_df = read_runs(runs_path)
        df = read_predictions(
            preds_dir,
            round_selector=as_of_round,
            run_id=run_id,
            runs_df=runs_df,
            require_run_id=True,
        )
    except Exception as exc:
        err_msg = f"Failed to read ledger predictions: {exc}"
        diag = LedgerDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "error": str(exc),
                "diagnostics": diag.diagnostics.add_error(err_msg),
            }
        )
        return pl.DataFrame(), diag

    diag = LedgerDiagnostics(**{**diag.__dict__, "status": "ok", "rows": df.height})
    return df, diag
