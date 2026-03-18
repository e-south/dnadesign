"""
--------------------------------------------------------------------------------
dnadesign
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

from .contracts import AnalysisRun, ClusterRun, EmbeddingRun, SweepRun
from .index import add_or_update_index
from .store import (
    append_records_md,
    fit_run_dir,
    write_analysis_meta,
    write_labels,
    write_run_meta,
    write_summary,
    write_sweep_meta,
    write_umap_coords,
    write_umap_meta,
)


@dataclass(frozen=True, slots=True)
class CommandRecord:
    command: str
    subject: str
    workspace: str | None
    preset: str | None
    resolved: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "workspace": self.workspace,
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
    method_sig_hash: str,
) -> Path:
    run_dir = fit_run_dir(root, run.alias, run.slug)
    write_run_meta(run_dir, run.meta_payload())
    labels_path = write_labels(run_dir, labels_df)
    write_summary(run_dir, summary)
    fit_index_entry = run.index_entry(
        labels_path=labels_path,
        method_sig_hash=method_sig_hash,
        input_sig_hash=input_sig_hash,
    )
    if fit_index_entry.input_sig_hash != input_sig_hash:
        raise RuntimeError("Cluster run input-signature hash drifted during fit bookkeeping.")
    if fit_index_entry.method_sig_hash != method_sig_hash:
        raise RuntimeError("Cluster run method-signature hash drifted during fit bookkeeping.")
    add_or_update_index(fit_index_entry, root=root)
    return run_dir


def record_umap_run(
    *,
    root: Path,
    artifact_dir: Path,
    run: EmbeddingRun,
    coords_df: pd.DataFrame,
    plot_root: Path | None,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    coords_path = write_umap_coords(artifact_dir, coords_df)
    write_umap_meta(artifact_dir, run.meta_payload())
    add_or_update_index(run.index_entry(coords_path=coords_path, plot_root=plot_root), root=root)
    return artifact_dir


def record_analysis_run(
    *,
    root: Path,
    out_dir: Path,
    run: AnalysisRun,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = write_analysis_meta(out_dir, run.meta_payload())
    add_or_update_index(run.index_entry(analysis_path=analysis_path), root=root)
    return analysis_path


def record_sweep_run(
    *,
    root: Path,
    out_dir: Path,
    run: SweepRun,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = write_sweep_meta(out_dir, run.meta_payload())
    add_or_update_index(run.index_entry(sweep_path=sweep_path), root=root)
    return sweep_path


def append_command_record_entry(run_dir: Path, record: CommandRecord) -> Path:
    return append_records_md(run_dir, record.markdown())


__all__ = [
    "CommandRecord",
    "append_command_record_entry",
    "record_analysis_run",
    "record_fit_run",
    "record_sweep_run",
    "record_umap_run",
]
