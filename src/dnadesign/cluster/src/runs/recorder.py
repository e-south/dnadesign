"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runs/recorder.py

Persist typed run artifacts and command records.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import AnalysisRun, ClusterRun, EmbeddingRun
from .index import add_or_update_index
from .store import (
    append_records_md,
    create_run_dir,
    umap_dir,
    write_analysis_meta,
    write_labels,
    write_run_meta,
    write_summary,
    write_umap_coords,
    write_umap_meta,
)


@dataclass(frozen=True, slots=True)
class CommandRecord:
    command: str
    subject: str
    job: str | None
    preset: str | None
    resolved: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "job": self.job,
            "preset": self.preset,
            "resolved": dict(self.resolved),
        }

    def markdown(self) -> str:
        return (
            f"## cluster {self.command} — {self.subject}\n\n"
            f"```json\n{json.dumps(self.payload(), indent=2, sort_keys=True)}\n```"
        )


def record_fit_run(
    *,
    root: Path,
    run: ClusterRun,
    labels_df: pd.DataFrame,
    summary: dict[str, Any],
    input_sig_hash: str,
) -> Path:
    run_dir = create_run_dir(root, run.slug)
    write_run_meta(run_dir, run.meta_payload())
    labels_path = write_labels(run_dir, labels_df)
    write_summary(run_dir, summary)
    fit_index_entry = run.index_entry(labels_path=labels_path, input_sig_hash=input_sig_hash)
    if fit_index_entry.input_sig_hash != input_sig_hash:
        raise RuntimeError("Cluster run input-signature hash drifted during fit bookkeeping.")
    add_or_update_index(fit_index_entry)
    return run_dir


def record_umap_run(
    *,
    root: Path,
    run_alias: str,
    run: EmbeddingRun,
    coords_df: pd.DataFrame,
) -> tuple[Path, Path]:
    run_dir = create_run_dir(root, run_alias)
    udir = umap_dir(run_dir)
    coords_path = write_umap_coords(udir, coords_df)
    write_umap_meta(udir, run.meta_payload())
    add_or_update_index(run.index_entry(coords_path=coords_path, plot_root=udir))
    return run_dir, udir


def record_analysis_run(
    *,
    out_dir: Path,
    run: AnalysisRun,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_analysis_meta(out_dir, run.meta_payload())


def append_command_record_entry(run_dir: Path, record: CommandRecord) -> Path:
    return append_records_md(run_dir, record.markdown())


__all__ = [
    "CommandRecord",
    "append_command_record_entry",
    "record_analysis_run",
    "record_fit_run",
    "record_umap_run",
]
