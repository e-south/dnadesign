"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/reporter.py

Reporter - writes tabular outputs and dispatches visualisations.

• All visualisations live in dedicated modules under `dnadiesn.permuter.plots`.
  Each module must expose a function:

      plot(elite_df: pd.DataFrame,
           all_df:   pd.DataFrame,
           output_path: Path,
           job_name: str) -> None

  The reporter will import the module by name and call this function.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .logging_utils import init_logger

_LOG = init_logger(__name__)


def _as_dataframe(records: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_results(
    variants: List[Dict],
    elites: List[Dict],
    output_dir: Path,
    job_name: str,
    plot_names: List[str] | None = None,
) -> None:
    """
    Persist tidy CSV outputs and any requested plots.
    JSONL export removed as redundant with elites CSV.
    """
    plot_names = plot_names or []
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSVs
    _write_csv(_as_dataframe(variants), output_dir / f"{job_name}.csv")
    _write_csv(_as_dataframe(elites), output_dir / f"{job_name}_elites.csv")
    _LOG.info(f"[{job_name}] wrote CSVs to {output_dir}")

    # 2) visualisations
    for name in plot_names:
        try:
            mod = importlib.import_module(f"dnadesign.permuter.plots.{name}")
        except ModuleNotFoundError:
            _LOG.warning(f"[{job_name}] plot '{name}' not found; skipping")
            continue

        if not hasattr(mod, "plot"):
            _LOG.warning(f"[{job_name}] module '{name}' has no plot(); skipping")
            continue

        out_path = output_dir / f"{job_name}_{name}.png"
        try:
            mod.plot(
                elite_df=_as_dataframe(elites),
                all_df=_as_dataframe(variants),
                output_path=out_path,
                job_name=job_name,
            )
            _LOG.info(f"[{job_name}] saved plot '{name}' → {out_path}")
        except Exception as exc:
            _LOG.error(f"[{job_name}] failed plotting '{name}': {exc}")
