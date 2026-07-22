"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_show_load.py

Load preserved-site Snapback bundle artifacts for readback validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from dnadesign.cruncher.snapback.artifacts import (
    baserender_jobs_dir,
    candidate_table_path,
    catalog_snapshot_path,
    load_manifest,
    load_solve_manifest,
    load_solve_status,
    load_status,
    materialized_hits_dir,
    post_nick_exposed_view_path,
    post_nick_exposed_visual_contract_path,
    post_nick_foldback_view_path,
    post_nick_foldback_visual_contract_path,
    pre_nick_duplex_view_path,
    pre_nick_duplex_visual_contract_path,
    renders_dir,
    report_json_path,
    report_md_path,
    snapback_manifest_path,
    snapback_status_path,
    snapback_triptych_job_path,
    snapback_triptych_visual_contracts_path,
    solve_frontier_table_path,
    solve_hits_table_path,
    solve_input_spec_path,
    solve_manifest_path,
    solve_report_json_path,
    solve_report_md_path,
    solve_resolved_catalog_path,
    solve_status_path,
    spec_snapshot_path,
    views_manifest_path,
)

ShowPayload = dict[str, Any]


def _load_json_value(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(f"Snapback {label} JSON is invalid.") from exc


def _load_json_mapping(path: Path, *, label: str) -> ShowPayload:
    payload = _load_json_value(path, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"Snapback {label} must be a JSON object.")
    return payload


def _load_optional_json_mapping(path: Path, *, label: str) -> ShowPayload | None:
    if not path.exists():
        return None
    return _load_json_mapping(path, label=label)


def _load_optional_triptych_contracts(path: Path) -> list[ShowPayload] | None:
    if not path.exists():
        return None
    records: list[ShowPayload] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except JSONDecodeError as exc:
            raise ValueError("Snapback triptych visual contract JSON is invalid.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Snapback triptych visual contract record must be a JSON object.")
        records.append(payload)
    return records


@dataclass(frozen=True)
class SnapbackExplicitShowArtifacts:
    run_dir: Path
    manifest: ShowPayload
    status: ShowPayload
    report_payload: ShowPayload
    views_manifest_payload: ShowPayload | None
    pre_nick_visual_payload: ShowPayload | None
    post_nick_exposed_visual_payload: ShowPayload | None
    post_nick_foldback_visual_payload: ShowPayload | None
    triptych_visual_contracts: list[ShowPayload] | None

    @property
    def manifest_path(self) -> Path:
        return snapback_manifest_path(self.run_dir)

    @property
    def status_path(self) -> Path:
        return snapback_status_path(self.run_dir)

    @property
    def report_path(self) -> Path:
        return report_json_path(self.run_dir)

    @property
    def report_markdown_path(self) -> Path:
        return report_md_path(self.run_dir)

    @property
    def spec_snapshot_path(self) -> Path:
        return spec_snapshot_path(self.run_dir)

    @property
    def catalog_snapshot_path(self) -> Path:
        return catalog_snapshot_path(self.run_dir)

    @property
    def candidate_table_path(self) -> Path:
        return candidate_table_path(self.run_dir)

    @property
    def views_manifest_path(self) -> Path:
        return views_manifest_path(self.run_dir)

    @property
    def pre_nick_view_path(self) -> Path:
        return pre_nick_duplex_view_path(self.run_dir)

    @property
    def post_nick_exposed_view_path(self) -> Path:
        return post_nick_exposed_view_path(self.run_dir)

    @property
    def post_nick_foldback_view_path(self) -> Path:
        return post_nick_foldback_view_path(self.run_dir)

    @property
    def pre_nick_visual_contract_path(self) -> Path:
        return pre_nick_duplex_visual_contract_path(self.run_dir)

    @property
    def post_nick_exposed_visual_contract_path(self) -> Path:
        return post_nick_exposed_visual_contract_path(self.run_dir)

    @property
    def post_nick_foldback_visual_contract_path(self) -> Path:
        return post_nick_foldback_visual_contract_path(self.run_dir)

    @property
    def triptych_visual_contracts_path(self) -> Path:
        return snapback_triptych_visual_contracts_path(self.run_dir)

    @property
    def triptych_job_path(self) -> Path:
        return snapback_triptych_job_path(self.run_dir)

    @property
    def baserender_jobs_dir_path(self) -> Path:
        return baserender_jobs_dir(self.run_dir)

    @property
    def renders_dir_path(self) -> Path:
        return renders_dir(self.run_dir)

    def required_artifact_paths(self) -> tuple[Path, ...]:
        return (
            self.report_path,
            self.report_markdown_path,
            self.spec_snapshot_path,
            self.catalog_snapshot_path,
            self.candidate_table_path,
        )


@dataclass(frozen=True)
class SnapbackSolveShowArtifacts:
    run_dir: Path
    manifest: ShowPayload
    status: ShowPayload
    report_payload: ShowPayload

    @property
    def manifest_path(self) -> Path:
        return solve_manifest_path(self.run_dir)

    @property
    def status_path(self) -> Path:
        return solve_status_path(self.run_dir)

    @property
    def report_path(self) -> Path:
        return solve_report_json_path(self.run_dir)

    @property
    def report_markdown_path(self) -> Path:
        return solve_report_md_path(self.run_dir)

    @property
    def input_spec_path(self) -> Path:
        return solve_input_spec_path(self.run_dir)

    @property
    def resolved_catalog_path(self) -> Path:
        return solve_resolved_catalog_path(self.run_dir)

    @property
    def hits_table_path(self) -> Path:
        return solve_hits_table_path(self.run_dir)

    @property
    def frontier_table_path(self) -> Path:
        return solve_frontier_table_path(self.run_dir)

    @property
    def materialized_hits_dir_path(self) -> Path:
        return materialized_hits_dir(self.run_dir)

    def required_artifact_paths(self) -> tuple[Path, ...]:
        return (
            self.report_path,
            self.report_markdown_path,
            self.manifest_path,
            self.status_path,
            self.input_spec_path,
            self.resolved_catalog_path,
            self.hits_table_path,
            self.frontier_table_path,
            self.materialized_hits_dir_path,
        )


SnapbackShowArtifacts = SnapbackExplicitShowArtifacts | SnapbackSolveShowArtifacts


def load_snapback_explicit_show_artifacts(run_dir: str | Path) -> SnapbackExplicitShowArtifacts:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    return SnapbackExplicitShowArtifacts(
        run_dir=resolved_run_dir,
        manifest=load_manifest(resolved_run_dir),
        status=load_status(resolved_run_dir),
        report_payload=_load_json_mapping(report_json_path(resolved_run_dir), label="report"),
        views_manifest_payload=_load_optional_json_mapping(
            views_manifest_path(resolved_run_dir),
            label="views manifest",
        ),
        pre_nick_visual_payload=_load_optional_json_mapping(
            pre_nick_duplex_visual_contract_path(resolved_run_dir),
            label="pre-nick visual contract",
        ),
        post_nick_exposed_visual_payload=_load_optional_json_mapping(
            post_nick_exposed_visual_contract_path(resolved_run_dir),
            label="post-nick exposed visual contract",
        ),
        post_nick_foldback_visual_payload=_load_optional_json_mapping(
            post_nick_foldback_visual_contract_path(resolved_run_dir),
            label="post-nick foldback visual contract",
        ),
        triptych_visual_contracts=_load_optional_triptych_contracts(
            snapback_triptych_visual_contracts_path(resolved_run_dir)
        ),
    )


def load_snapback_solve_show_artifacts(run_dir: str | Path) -> SnapbackSolveShowArtifacts:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    return SnapbackSolveShowArtifacts(
        run_dir=resolved_run_dir,
        manifest=load_solve_manifest(resolved_run_dir),
        status=load_solve_status(resolved_run_dir),
        report_payload=_load_json_mapping(solve_report_json_path(resolved_run_dir), label="solve report"),
    )


def load_snapback_show_artifacts(run_dir: str | Path) -> SnapbackShowArtifacts:
    resolved = Path(run_dir).expanduser().resolve()
    explicit_manifest_exists = snapback_manifest_path(resolved).exists()
    solve_manifest_exists = solve_manifest_path(resolved).exists()
    if explicit_manifest_exists and solve_manifest_exists:
        raise ValueError(f"Ambiguous snapback run directory contains explicit and solve manifests: {resolved}")
    if explicit_manifest_exists:
        return load_snapback_explicit_show_artifacts(resolved)
    if solve_manifest_exists:
        return load_snapback_solve_show_artifacts(resolved)
    raise FileNotFoundError(f"Unsupported snapback run directory: {resolved}")


__all__ = [
    "SnapbackExplicitShowArtifacts",
    "SnapbackShowArtifacts",
    "SnapbackSolveShowArtifacts",
    "load_snapback_explicit_show_artifacts",
    "load_snapback_show_artifacts",
    "load_snapback_solve_show_artifacts",
]
