"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/visual_publication.py

Publication and drift checks for scar_nick visual artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.cruncher.scar_nick.artifacts import (
    build_materialized_candidate_manifest_payload,
    candidate_json_path,
    candidate_manifest_path,
    ensure_visual_run_dirs,
    materialized_candidate_dir,
    post_terminal_nick_view_path,
    post_terminal_nick_visual_contract_path,
    scar_nick_terminal_nick_job_path,
    scar_nick_terminal_nick_visual_contracts_path,
    views_manifest_path,
    write_materialized_candidate_manifest,
    write_visual_bundle,
)
from dnadesign.cruncher.scar_nick.models import ScarNickCandidate, ScarNickEvaluationReport, ScarNickSpecDocument
from dnadesign.cruncher.scar_nick.ranking import unique_sequence_candidates
from dnadesign.cruncher.scar_nick.view_contracts import (
    build_candidate_visual_bundle,
    build_terminal_nick_visual_contract,
)


def publish_scar_nick_visuals(
    *,
    run_dir: Path,
    report: ScarNickEvaluationReport,
    spec: ScarNickSpecDocument,
) -> None:
    candidates = _top_visual_candidates(report, spec)
    if not candidates:
        return

    root_candidate = candidates[0]
    root_bundle = build_candidate_visual_bundle(
        candidate=root_candidate,
        solution_id=_visual_solution_id(report, root_candidate),
        visual_contracts=_terminal_nick_visual_records(report, candidates),
    )
    write_visual_bundle(
        run_dir,
        terminal_nick_view=root_bundle.terminal_nick_view,
        terminal_nick_visual_contract=root_bundle.terminal_nick_visual_contract,
        terminal_nick_visual_contracts=root_bundle.terminal_nick_visual_contracts,
        views_manifest=root_bundle.views_manifest,
        baserender_job=root_bundle.baserender_job,
    )

    for candidate in candidates:
        candidate_dir = materialized_candidate_dir(run_dir, rank=int(candidate.rank or 0))
        ensure_visual_run_dirs(candidate_dir)
        bundle = build_candidate_visual_bundle(
            candidate=candidate,
            solution_id=_visual_solution_id(report, candidate),
        )
        write_visual_bundle(
            candidate_dir,
            terminal_nick_view=bundle.terminal_nick_view,
            terminal_nick_visual_contract=bundle.terminal_nick_visual_contract,
            terminal_nick_visual_contracts=bundle.terminal_nick_visual_contracts,
            views_manifest=bundle.views_manifest,
            baserender_job=bundle.baserender_job,
        )
        write_materialized_candidate_manifest(
            candidate_dir,
            candidate_payload=candidate.model_dump(mode="json"),
            views_manifest=bundle.views_manifest,
        )


def assert_visual_publication_current(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    expected_count = report.metadata.materialized_candidate_count
    if expected_count == 0:
        return
    candidates = report.candidates[:expected_count]
    if len(candidates) != expected_count:
        raise ValueError("Scar-nick materialized candidate count drift detected.")
    root_records = _terminal_nick_visual_records(report, candidates)
    _assert_visual_bundle_current(
        run_dir=run_dir,
        report=report,
        candidate=candidates[0],
        terminal_nick_visual_contracts=root_records,
    )
    for candidate in candidates:
        candidate_dir = materialized_candidate_dir(run_dir, rank=int(candidate.rank or 0))
        _assert_visual_bundle_current(run_dir=candidate_dir, report=report, candidate=candidate)
        _assert_materialized_candidate_manifest_current(candidate_dir, candidate)


def _visual_solution_id(report: ScarNickEvaluationReport, candidate: ScarNickCandidate) -> str:
    rank = candidate.rank if candidate.rank is not None else 0
    return f"{report.spec_name}.candidate_{rank:02d}"


def _top_visual_candidates(report: ScarNickEvaluationReport, spec: ScarNickSpecDocument) -> list[ScarNickCandidate]:
    return unique_sequence_candidates(report.candidates, limit=spec.search.materialize_top_k)


def _terminal_nick_visual_records(
    report: ScarNickEvaluationReport,
    candidates: list[ScarNickCandidate],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in candidates:
        solution_id = _visual_solution_id(report, candidate)
        records.append(
            build_terminal_nick_visual_contract(
                candidate=candidate,
                solution_id=solution_id,
                state_kind="pre_post_terminal_nick",
            )
        )
    return records


def _read_json(path: Path, *, visual: bool) -> object:
    if not path.exists():
        label = "visual artifact" if visual else "artifact"
        raise FileNotFoundError(f"Missing scar-nick {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scar-nick visual artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_yaml(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing scar-nick visual artifact: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Scar-nick {label} drift detected.")


def _assert_visual_bundle_current(
    *,
    run_dir: Path,
    report: ScarNickEvaluationReport,
    candidate: ScarNickCandidate,
    terminal_nick_visual_contracts: list[dict[str, object]] | None = None,
) -> None:
    bundle = build_candidate_visual_bundle(
        candidate=candidate,
        solution_id=_visual_solution_id(report, candidate),
        visual_contracts=terminal_nick_visual_contracts,
    )
    _assert_equal(
        _read_json(post_terminal_nick_view_path(run_dir), visual=True),
        bundle.terminal_nick_view,
        label="terminal nick view",
    )
    _assert_equal(
        _read_json(post_terminal_nick_visual_contract_path(run_dir), visual=True),
        bundle.terminal_nick_visual_contract,
        label="terminal nick visual contract",
    )
    _assert_equal(
        _read_jsonl(scar_nick_terminal_nick_visual_contracts_path(run_dir)),
        bundle.terminal_nick_visual_contracts,
        label="terminal nick visual contract inventory",
    )
    _assert_equal(
        _read_json(views_manifest_path(run_dir), visual=True),
        bundle.views_manifest,
        label="views manifest",
    )
    _assert_equal(
        _read_yaml(scar_nick_terminal_nick_job_path(run_dir)),
        bundle.baserender_job,
        label="BaseRender job",
    )


def _assert_materialized_candidate_manifest_current(run_dir: Path, candidate: ScarNickCandidate) -> None:
    views_manifest = _read_json(views_manifest_path(run_dir), visual=True)
    _assert_equal(
        _read_json(candidate_json_path(run_dir), visual=False),
        candidate.model_dump(mode="json"),
        label="materialized candidate payload",
    )
    expected = build_materialized_candidate_manifest_payload(
        candidate_payload=candidate.model_dump(mode="json"),
        views_manifest=views_manifest if isinstance(views_manifest, dict) else {},
    )
    _assert_equal(
        _read_json(candidate_manifest_path(run_dir), visual=True),
        expected,
        label="materialized candidate manifest",
    )


__all__ = [
    "assert_visual_publication_current",
    "publish_scar_nick_visuals",
]
