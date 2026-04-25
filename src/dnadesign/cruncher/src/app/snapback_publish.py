"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_publish.py

Publish snapback QA views, renderer-facing visual contracts, and optional render
jobs for explicit and materialized runs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dnadesign.cruncher.snapback.artifacts import (
    snapback_triptych_job_path,
    write_baserender_job,
    write_view_bundle,
)
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport
from dnadesign.cruncher.snapback.public_visuals import (
    build_post_nick_exposed_snapback_visual,
    build_post_nick_foldback_snapback_visual,
    build_pre_nick_snapback_visual,
)
from dnadesign.cruncher.snapback.view_contracts import (
    build_post_nick_exposed_view,
    build_post_nick_foldback_view,
    build_pre_nick_duplex_view,
)
from dnadesign.cruncher.snapback.view_models import SnapbackViewsManifestV1

__all__ = [
    "SnapbackPublicationBundle",
    "build_publication_bundle",
    "build_triptych_job",
    "build_views_manifest",
    "write_publication_bundle",
]

_STATE_ORDER = ("pre_nick_duplex", "post_nick_exposed", "post_nick_foldback")
_DEFAULT_STATE_TITLES = {
    "pre_nick_duplex": "pre-nick duplex",
    "post_nick_exposed": "post-nick exposed",
    "post_nick_foldback": "post-nick foldback",
}
_TRIPTYCH_VISUAL_FILENAME = "snapback_triptych.snapback_visual.v1.jsonl"
_TRIPTYCH_JOB_RELATIVE_PATH = "../../baserender_jobs/snapback_triptych.job.yaml"
_VIEW_RELATIVE_PATHS = {
    "pre_nick_duplex": "analysis/views/pre_nick_duplex.v1.json",
    "post_nick_exposed": "analysis/views/post_nick_exposed.v1.json",
    "post_nick_foldback": "analysis/views/post_nick_foldback.v1.json",
}
_VISUAL_RELATIVE_PATHS = {
    "pre_nick_duplex": "analysis/views/pre_nick_duplex.snapback_visual.v1.json",
    "post_nick_exposed": "analysis/views/post_nick_exposed.snapback_visual.v1.json",
    "post_nick_foldback": "analysis/views/post_nick_foldback.snapback_visual.v1.json",
}


@dataclass(frozen=True)
class SnapbackPublicationBundle:
    pre_nick_duplex: dict[str, Any]
    post_nick_exposed: dict[str, Any]
    post_nick_foldback: dict[str, Any]
    pre_nick_duplex_visual_contract: dict[str, Any]
    post_nick_exposed_visual_contract: dict[str, Any]
    post_nick_foldback_visual_contract: dict[str, Any]
    triptych_visual_contracts: list[dict[str, Any]]
    manifest: dict[str, Any]
    baserender_job: dict[str, Any] | None


def _state_titles(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    titles = dict(_DEFAULT_STATE_TITLES)
    if overrides is not None:
        titles.update({key: str(value) for key, value in overrides.items()})
    missing = [state for state in _STATE_ORDER if state not in titles]
    if missing:
        raise ValueError(f"Missing snapback publication titles for states: {', '.join(missing)}")
    return titles


def build_views_manifest(*, solution_id: str, include_jobs: bool) -> dict[str, Any]:
    payload = {
        "version": 1,
        "kind": "snapback_views_manifest_v1",
        "solution_id": solution_id,
        "views": [
            {
                "name": "pre_nick_duplex_qa",
                "path": _VIEW_RELATIVE_PATHS["pre_nick_duplex"],
                "contract_kind": "snapback_pre_nick_duplex_v1",
            },
            {
                "name": "post_nick_exposed_qa",
                "path": _VIEW_RELATIVE_PATHS["post_nick_exposed"],
                "contract_kind": "snapback_post_nick_exposed_v1",
            },
            {
                "name": "post_nick_foldback_qa",
                "path": _VIEW_RELATIVE_PATHS["post_nick_foldback"],
                "contract_kind": "snapback_post_nick_foldback_v1",
            },
            {
                "name": "pre_nick_duplex_visual_contract",
                "path": _VISUAL_RELATIVE_PATHS["pre_nick_duplex"],
                "contract_kind": "snapback_visual_v1",
            },
            {
                "name": "post_nick_exposed_visual_contract",
                "path": _VISUAL_RELATIVE_PATHS["post_nick_exposed"],
                "contract_kind": "snapback_visual_v1",
            },
            {
                "name": "post_nick_foldback_visual_contract",
                "path": _VISUAL_RELATIVE_PATHS["post_nick_foldback"],
                "contract_kind": "snapback_visual_v1",
            },
            {
                "name": "snapback_triptych_visual_contracts",
                "path": f"analysis/views/{_TRIPTYCH_VISUAL_FILENAME}",
                "contract_kind": "snapback_visual_v1",
            },
        ],
        "recommended_jobs": [],
    }
    if include_jobs:
        payload["recommended_jobs"] = [{"name": "snapback_triptych", "path": _TRIPTYCH_JOB_RELATIVE_PATH}]
    return SnapbackViewsManifestV1.model_validate(payload).model_dump(mode="json")


def build_triptych_job(*, output_format: str) -> dict[str, object]:
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "jsonl",
            "path": f"../analysis/views/{_TRIPTYCH_VISUAL_FILENAME}",
            "adapter": {"kind": "snapback_visual_v1"},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "snapback_map",
            "style": {
                "preset": "presentation_default",
                "overrides": {
                    "legend": False,
                    "figure_scale": 1.05,
                    "font_size_seq": 13,
                    "font_size_label": 10,
                    "padding_x": 32.0,
                    "padding_y": 22.0,
                },
            },
        },
        "outputs": [
            {
                "kind": "images",
                "path": f"../plots/snapback_triptych.{output_format}",
                "fmt": output_format,
            }
        ],
        "run": {
            "strict": True,
            "fail_on_skips": True,
            "emit_report": False,
        },
    }


def build_publication_bundle(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    include_jobs: bool,
    render_format: str = "png",
    title_overrides: Mapping[str, str] | None = None,
) -> SnapbackPublicationBundle:
    titles = _state_titles(title_overrides)
    pre_nick_duplex_visual_contract = build_pre_nick_snapback_visual(
        report=report,
        solution_id=solution_id,
        title=titles["pre_nick_duplex"],
    )
    post_nick_exposed_visual_contract = build_post_nick_exposed_snapback_visual(
        report=report,
        solution_id=solution_id,
        title=titles["post_nick_exposed"],
    )
    post_nick_foldback_visual_contract = build_post_nick_foldback_snapback_visual(
        report=report,
        solution_id=solution_id,
        title=titles["post_nick_foldback"],
    )
    return SnapbackPublicationBundle(
        pre_nick_duplex=build_pre_nick_duplex_view(
            report=report,
            solution_id=solution_id,
            title=titles["pre_nick_duplex"],
        ),
        post_nick_exposed=build_post_nick_exposed_view(
            report=report,
            solution_id=solution_id,
            title=titles["post_nick_exposed"],
        ),
        post_nick_foldback=build_post_nick_foldback_view(
            report=report,
            solution_id=solution_id,
            title=titles["post_nick_foldback"],
        ),
        pre_nick_duplex_visual_contract=pre_nick_duplex_visual_contract,
        post_nick_exposed_visual_contract=post_nick_exposed_visual_contract,
        post_nick_foldback_visual_contract=post_nick_foldback_visual_contract,
        triptych_visual_contracts=[
            pre_nick_duplex_visual_contract,
            post_nick_exposed_visual_contract,
            post_nick_foldback_visual_contract,
        ],
        manifest=build_views_manifest(solution_id=solution_id, include_jobs=include_jobs),
        baserender_job=build_triptych_job(output_format=render_format) if include_jobs else None,
    )


def write_publication_bundle(run_dir: Path, *, bundle: SnapbackPublicationBundle) -> None:
    write_view_bundle(
        run_dir,
        pre_nick_duplex=bundle.pre_nick_duplex,
        post_nick_exposed=bundle.post_nick_exposed,
        post_nick_foldback=bundle.post_nick_foldback,
        pre_nick_duplex_visual_contract=bundle.pre_nick_duplex_visual_contract,
        post_nick_exposed_visual_contract=bundle.post_nick_exposed_visual_contract,
        post_nick_foldback_visual_contract=bundle.post_nick_foldback_visual_contract,
        triptych_visual_contracts=bundle.triptych_visual_contracts,
        manifest=bundle.manifest,
    )
    if bundle.baserender_job is None:
        return
    write_baserender_job(snapback_triptych_job_path(run_dir), bundle.baserender_job)
