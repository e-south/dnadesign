"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_show_present.py

Typed presentation surface for preserved-site Snapback readback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from dnadesign.cruncher.app.snapback_show_load import (
    SnapbackExplicitShowArtifacts,
    SnapbackShowArtifacts,
    SnapbackSolveShowArtifacts,
)
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel


def _required_text(value: object, *, error_message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_message)
    return value


def _existing_triptych_render(artifacts: SnapbackExplicitShowArtifacts) -> str | None:
    for fmt in ("png", "svg", "pdf"):
        candidate = artifacts.renders_dir_path / f"snapback_triptych.{fmt}"
        if candidate.exists():
            return str(candidate.resolve())
    return None


class SnapbackExplicitShowOutcome(StrictBaseModel):
    kind: Literal["explicit"] = "explicit"
    run_dir: str
    spec_name: str
    status: str
    status_message: str
    manifest_path: str
    status_path: str
    report_json: str
    report_md: str
    spec_snapshot: str
    catalog_snapshot: str
    views_manifest: str | None = None
    pre_nick_duplex_visual_contract: str | None = None
    post_nick_exposed_visual_contract: str | None = None
    post_nick_foldback_visual_contract: str | None = None
    snapback_triptych_visual_contracts: str | None = None
    pre_nick_duplex_view: str | None = None
    post_nick_exposed_view: str | None = None
    post_nick_foldback_view: str | None = None
    snapback_triptych_job: str | None = None
    baserender_jobs_dir: str | None = None
    plots_dir: str | None = None
    snapback_triptych_render: str | None = None
    artifacts: list[dict[str, object]]


class SnapbackSolveShowOutcome(StrictBaseModel):
    kind: Literal["solve"] = "solve"
    run_dir: str
    spec_name: str
    status: str
    status_message: str
    solve_report: str
    solve_report_md: str
    solve_manifest: str
    solve_status: str
    input_solve_spec: str
    resolved_catalog: str
    table__hits: str
    table__frontier: str
    materialized_hits_dir: str


def _build_explicit_show_payload(artifacts: SnapbackExplicitShowArtifacts) -> dict[str, object]:
    return SnapbackExplicitShowOutcome(
        run_dir=str(artifacts.run_dir.resolve()),
        spec_name=_required_text(
            artifacts.manifest.get("spec_name"),
            error_message="Snapback manifest spec_name drift detected.",
        ),
        status=_required_text(
            artifacts.status.get("status"),
            error_message="Snapback status status drift detected.",
        ),
        status_message=_required_text(
            artifacts.status.get("status_message"),
            error_message="Snapback status status_message drift detected.",
        ),
        manifest_path=str(artifacts.manifest_path.resolve()),
        status_path=str(artifacts.status_path.resolve()),
        report_json=str(artifacts.report_path.resolve()),
        report_md=str(artifacts.report_markdown_path.resolve()),
        spec_snapshot=str(artifacts.spec_snapshot_path.resolve()),
        catalog_snapshot=str(artifacts.catalog_snapshot_path.resolve()),
        views_manifest=(
            str(artifacts.views_manifest_path.resolve()) if artifacts.views_manifest_path.exists() else None
        ),
        pre_nick_duplex_visual_contract=(
            str(artifacts.pre_nick_visual_contract_path.resolve())
            if artifacts.pre_nick_visual_contract_path.exists()
            else None
        ),
        post_nick_exposed_visual_contract=(
            str(artifacts.post_nick_exposed_visual_contract_path.resolve())
            if artifacts.post_nick_exposed_visual_contract_path.exists()
            else None
        ),
        post_nick_foldback_visual_contract=(
            str(artifacts.post_nick_foldback_visual_contract_path.resolve())
            if artifacts.post_nick_foldback_visual_contract_path.exists()
            else None
        ),
        snapback_triptych_visual_contracts=(
            str(artifacts.triptych_visual_contracts_path.resolve())
            if artifacts.triptych_visual_contracts_path.exists()
            else None
        ),
        pre_nick_duplex_view=(
            str(artifacts.pre_nick_view_path.resolve()) if artifacts.pre_nick_view_path.exists() else None
        ),
        post_nick_exposed_view=(
            str(artifacts.post_nick_exposed_view_path.resolve())
            if artifacts.post_nick_exposed_view_path.exists()
            else None
        ),
        post_nick_foldback_view=(
            str(artifacts.post_nick_foldback_view_path.resolve())
            if artifacts.post_nick_foldback_view_path.exists()
            else None
        ),
        snapback_triptych_job=(
            str(artifacts.triptych_job_path.resolve()) if artifacts.triptych_job_path.exists() else None
        ),
        baserender_jobs_dir=(
            str(artifacts.baserender_jobs_dir_path.resolve()) if artifacts.baserender_jobs_dir_path.exists() else None
        ),
        plots_dir=str(artifacts.renders_dir_path.resolve()) if artifacts.renders_dir_path.exists() else None,
        snapback_triptych_render=_existing_triptych_render(artifacts),
        artifacts=artifacts.manifest.get("artifacts", []),
    ).model_dump(mode="json")


def _build_solve_show_payload(artifacts: SnapbackSolveShowArtifacts) -> dict[str, object]:
    return SnapbackSolveShowOutcome(
        run_dir=str(artifacts.run_dir.resolve()),
        spec_name=_required_text(
            artifacts.manifest.get("spec_name"),
            error_message="Snapback solve manifest spec_name drift detected.",
        ),
        status=_required_text(
            artifacts.status.get("status"),
            error_message="Snapback solve status status drift detected.",
        ),
        status_message=_required_text(
            artifacts.status.get("status_message"),
            error_message="Snapback solve status status_message drift detected.",
        ),
        solve_report=str(artifacts.report_path.resolve()),
        solve_report_md=str(artifacts.report_markdown_path.resolve()),
        solve_manifest=str(artifacts.manifest_path.resolve()),
        solve_status=str(artifacts.status_path.resolve()),
        input_solve_spec=str(artifacts.input_spec_path.resolve()),
        resolved_catalog=str(artifacts.resolved_catalog_path.resolve()),
        table__hits=str(artifacts.hits_table_path.resolve()),
        table__frontier=str(artifacts.frontier_table_path.resolve()),
        materialized_hits_dir=str(artifacts.materialized_hits_dir_path.resolve()),
    ).model_dump(mode="json")


def build_snapback_show_payload(artifacts: SnapbackShowArtifacts) -> dict[str, object]:
    if isinstance(artifacts, SnapbackExplicitShowArtifacts):
        return _build_explicit_show_payload(artifacts)
    return _build_solve_show_payload(artifacts)


__all__ = [
    "SnapbackExplicitShowOutcome",
    "SnapbackSolveShowOutcome",
    "build_snapback_show_payload",
]
