"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/notebook_generation.py

Shared helpers for OPAL notebook generation CLI contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...analysis.campaign import CampaignAnalysis
from ...analysis.notebook_scope import resolve_notebook_run_scope

NOTEBOOK_GENERATE_SCHEMA_VERSION = "opal.notebook_generate.v1"


def notebook_generate_payload(
    *,
    kind: str,
    out_path: Path,
    round_selector: str,
    run_id: str | None,
    validate: bool,
    force: bool,
    overwritten: bool,
    analyses: list[CampaignAnalysis],
    collection_manifest_path: Path | None = None,
    collection_visual_index_path: Path | None = None,
) -> dict[str, object]:
    config_paths = [str(analysis.config_path) for analysis in analyses]
    workdirs = [str(analysis.workspace.workdir) for analysis in analyses]
    return {
        "schema_version": NOTEBOOK_GENERATE_SCHEMA_VERSION,
        "ok": True,
        "kind": kind,
        "campaign_count": len(analyses),
        "notebook_path": str(out_path),
        "round_selector": round_selector,
        "run_id": run_id,
        "validate": bool(validate),
        "force": bool(force),
        "overwritten": bool(overwritten),
        "config_paths": config_paths,
        "collection_manifest_path": str(collection_manifest_path) if collection_manifest_path is not None else None,
        "collection_visual_index_path": (
            str(collection_visual_index_path) if collection_visual_index_path is not None else None
        ),
        "workdirs": workdirs,
        "next_commands": {
            "run": f"uv run opal notebook run -c {config_paths[0]} --path {out_path}",
            "marimo_check": f"uv run marimo check {out_path}",
        },
    }


def resolve_generation_run_scope(
    analysis: CampaignAnalysis,
    *,
    round_selector: str,
    run_id: str | None,
) -> tuple[str, str | None]:
    resolved_round, resolved_run_id = resolve_notebook_run_scope(
        analysis,
        round_selector=round_selector,
        run_id=run_id,
    )
    return str(resolved_round or round_selector), resolved_run_id
