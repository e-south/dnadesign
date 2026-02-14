"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/cli_actions.py

Command actions backing the CLI without depending on Typer presentation logic.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .api import run_job as run_job_public
from .api import validate_job as validate_job_public
from .config import (
    SequenceRowsJobV3,
    list_style_presets,
    resolve_preset_path,
)
from .core import BaseRenderError
from .workspace import Workspace, discover_workspaces, init_workspace, resolve_workspace_job_path


def resolve_job_spec(job: str | None, workspace: str | None, workspace_root: Path | None) -> str:
    if (job is None and workspace is None) or (job is not None and workspace is not None):
        raise BaseRenderError("Provide exactly one of <job> or --workspace")
    if workspace is not None:
        return str(resolve_workspace_job_path(workspace, root=workspace_root))
    assert job is not None
    return job


def validate_job_action(
    job: str | None,
    workspace: str | None,
    workspace_root: Path | None,
    *,
    caller_root: Path | None = None,
) -> SequenceRowsJobV3:
    return validate_job_public(
        resolve_job_spec(job, workspace, workspace_root),
        kind="sequence_rows_v3",
        caller_root=caller_root,
    )


def run_job_action(
    job: str | None,
    workspace: str | None,
    workspace_root: Path | None,
    *,
    caller_root: Path | None = None,
):
    return run_job_public(
        resolve_job_spec(job, workspace, workspace_root),
        kind="sequence_rows_v3",
        caller_root=caller_root,
    )


def _job_to_mapping(parsed: SequenceRowsJobV3) -> dict[str, Any]:
    return {
        "version": 3,
        "results_root": str(parsed.results_root),
        "input": {
            "kind": parsed.input.kind,
            "path": str(parsed.input.path),
            "adapter": {
                "kind": parsed.input.adapter.kind,
                "columns": dict(parsed.input.adapter.columns),
                "policies": dict(parsed.input.adapter.policies),
            },
            "alphabet": parsed.input.alphabet,
            "limit": parsed.input.limit,
            "sample": (
                None
                if parsed.input.sample is None
                else {
                    "mode": parsed.input.sample.mode,
                    "n": parsed.input.sample.n,
                    "seed": parsed.input.sample.seed,
                }
            ),
        },
        "selection": (
            None
            if parsed.selection is None
            else {
                "path": str(parsed.selection.path),
                "match_on": parsed.selection.match_on,
                "column": parsed.selection.column,
                "overlay_column": parsed.selection.overlay_column,
                "keep_order": parsed.selection.keep_order,
                "on_missing": parsed.selection.on_missing,
            }
        ),
        "pipeline": {
            "plugins": [
                (spec.name if not spec.params else {spec.name: dict(spec.params)}) for spec in parsed.pipeline.plugins
            ]
        },
        "render": {
            "renderer": parsed.render.renderer,
            "style": {
                "preset": parsed.render.style_preset,
                "overrides": dict(parsed.render.style_overrides),
            },
        },
        "outputs": [
            (
                {
                    "kind": "images",
                    "dir": str(cfg.dir),
                    "fmt": cfg.fmt,
                }
                if cfg.kind == "images"
                else {
                    "kind": "video",
                    "path": str(cfg.path),
                    "fmt": cfg.fmt,
                    "fps": cfg.fps,
                    "frames_per_record": cfg.frames_per_record,
                    "pauses": dict(cfg.pauses),
                    "width_px": cfg.width_px,
                    "height_px": cfg.height_px,
                    "aspect": cfg.aspect_ratio,
                    "total_duration": cfg.total_duration,
                }
            )
            for cfg in parsed.outputs
        ],
        "run": {
            "strict": parsed.run.strict,
            "fail_on_skips": parsed.run.fail_on_skips,
            "emit_report": parsed.run.emit_report,
            "report_path": str(parsed.run.report_path) if parsed.run.report_path else None,
        },
    }


def normalize_job_action(
    job: str | None,
    workspace: str | None,
    workspace_root: Path | None,
    *,
    out: Path,
    caller_root: Path | None = None,
) -> Path:
    parsed = validate_job_public(
        resolve_job_spec(job, workspace, workspace_root),
        kind="sequence_rows_v3",
        caller_root=caller_root,
    )
    data = _job_to_mapping(parsed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def list_style_presets_action() -> tuple[str, ...]:
    return list_style_presets()


def show_style_action(preset: str) -> dict[str, Any]:
    path = resolve_preset_path(preset)
    if path is None:
        raise BaseRenderError("Preset path is null")
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise BaseRenderError(f"Style preset must be a mapping: {path}")
    return loaded


def discover_workspaces_action(root: Path | None) -> tuple[Workspace, ...]:
    return discover_workspaces(root=root)


def init_workspace_action(name: str, root: Path | None) -> Workspace:
    return init_workspace(name, root=root)
