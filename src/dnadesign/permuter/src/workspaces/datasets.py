"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/workspaces/datasets.py

Workspace-backed dataset path resolution.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dnadesign.permuter.src.core.config import ScopeConfig
from dnadesign.permuter.src.core.paths import resolve, resolve_workspace_config_hint
from dnadesign.permuter.src.workspaces.loader import load_workspace


@dataclass(frozen=True)
class WorkspaceDatasetPath:
    records: Path
    config: ScopeConfig
    config_path: Path
    ref_name: str


def resolve_workspace_dataset_path(
    *,
    workspace_hint: str | Path,
    ref: str | None,
    out: Path | None,
) -> WorkspaceDatasetPath:
    """Resolve the configured records.parquet path for a workspace/ref pair."""

    config_path = resolve_workspace_config_hint(workspace_hint)
    workspace = load_workspace(config_path)
    cfg = workspace.config
    resolved_config_path = workspace.config_path
    pending_paths = resolve(
        config_yaml=resolved_config_path,
        refs=cfg.scope.input.refs,
        output_dir=cfg.scope.output.dir,
        ref_name="__PENDING__",
        out_override=out,
        layout=cfg.scope.output.layout,
        require_writable_output=False,
    )
    refs = pd.read_csv(pending_paths.refs_csv, dtype=str)
    ref_name, _sequence = pick_reference(
        refs,
        name_col=cfg.scope.input.name_col,
        seq_col=cfg.scope.input.seq_col,
        desired=ref or cfg.scope.input.reference_sequence,
    )
    paths = resolve(
        config_yaml=resolved_config_path,
        refs=cfg.scope.input.refs,
        output_dir=cfg.scope.output.dir,
        ref_name=ref_name,
        out_override=out,
        layout=cfg.scope.output.layout,
        require_writable_output=False,
    )
    return WorkspaceDatasetPath(
        records=paths.records_parquet,
        config=cfg,
        config_path=resolved_config_path,
        ref_name=ref_name,
    )


def pick_reference(
    refs: pd.DataFrame,
    *,
    name_col: str,
    seq_col: str,
    desired: str | None,
) -> tuple[str, str]:
    if name_col not in refs.columns:
        raise ValueError(f"Reference CSV missing name column {name_col!r}")
    if seq_col not in refs.columns:
        raise ValueError(f"Reference CSV missing sequence column {seq_col!r}")
    if desired:
        sub = refs[refs[name_col] == desired]
        if sub.empty:
            raise ValueError(f"Reference {desired!r} not found in {name_col!r}")
        if len(sub) > 1:
            raise ValueError(f"Reference {desired!r} not unique in CSV")
        row = sub.iloc[0]
        return str(row[name_col]), str(row[seq_col])
    if len(refs) == 1:
        row = refs.iloc[0]
        return str(row[name_col]), str(row[seq_col])
    raise ValueError("--ref is required because the refs CSV has multiple rows")
