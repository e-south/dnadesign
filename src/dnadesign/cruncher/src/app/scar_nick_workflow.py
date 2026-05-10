"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/scar_nick_workflow.py

Application orchestration for scar-nick validation, design, and show.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from dnadesign.cruncher.nickases.catalog import (
    dump_nickase_catalog_yaml,
    load_merged_nickase_catalog,
)
from dnadesign.cruncher.release_enzymes.catalog import (
    dump_release_enzyme_catalog_yaml,
    load_merged_release_enzyme_catalog,
)
from dnadesign.cruncher.scar_nick.artifacts import (
    assert_manifest_artifacts_present,
    assert_manifest_hashes_current,
    assert_provenance_current,
    build_manifest,
    build_materialized_candidate_manifest_payload,
    build_run_dir,
    candidate_json_path,
    candidate_manifest_path,
    candidate_pair_call_table_path,
    candidate_profiles_path,
    candidate_table_path,
    ensure_run_dirs,
    ensure_visual_run_dirs,
    load_manifest,
    load_status,
    manifest_path,
    materialized_candidate_dir,
    nickase_catalog_snapshot_path,
    nickase_geometry_audit_path,
    nickase_geometry_audit_table_path,
    post_terminal_nick_view_path,
    post_terminal_nick_visual_contract_path,
    release_catalog_snapshot_path,
    report_json_path,
    report_md_path,
    scar_nick_terminal_nick_job_path,
    scar_nick_terminal_nick_visual_contracts_path,
    snapshot_inputs,
    spec_snapshot_path,
    status_path,
    views_manifest_path,
    write_candidate_pair_call_table,
    write_candidate_table,
    write_manifest,
    write_materialized_candidate_manifest,
    write_nickase_geometry_audit_table,
    write_report,
    write_status,
    write_visual_bundle,
)
from dnadesign.cruncher.scar_nick.load import load_scar_nick_spec
from dnadesign.cruncher.scar_nick.models import ScarNickCandidate, ScarNickEvaluationReport, ScarNickSpecDocument
from dnadesign.cruncher.scar_nick.planner import (
    build_scar_nick_report,
    render_markdown_report,
)
from dnadesign.cruncher.scar_nick.ranking import unique_sequence_candidates
from dnadesign.cruncher.scar_nick.view_contracts import (
    build_candidate_visual_bundle,
    build_terminal_nick_visual_contract,
)


def _load_catalogs(spec: ScarNickSpecDocument, *, workspace_root: Path):
    release_ref = spec.processing.release.catalog
    nick_ref = spec.processing.nick.catalog
    release_catalog, _release_paths = load_merged_release_enzyme_catalog(
        preset_id=release_ref.preset,
        additional_preset_ids=release_ref.additional_presets,
        additional_paths=release_ref.additional_paths,
        workspace_root=workspace_root,
    )
    nickase_catalog, _nick_paths = load_merged_nickase_catalog(
        preset_id=nick_ref.preset,
        additional_preset_ids=nick_ref.additional_presets,
        additional_paths=nick_ref.additional_paths,
        workspace_root=workspace_root,
    )
    return release_catalog, nickase_catalog


def validate_scar_nick_spec(path: str | Path) -> ScarNickEvaluationReport:
    spec, spec_path, workspace_root = load_scar_nick_spec(path)
    release_catalog, nickase_catalog = _load_catalogs(spec, workspace_root=workspace_root)
    return build_scar_nick_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        release_catalog=release_catalog,
        nickase_catalog=nickase_catalog,
    )


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


def _publish_scar_nick_visuals(
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


def _assert_visual_publication_current(run_dir: Path, report: ScarNickEvaluationReport) -> None:
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


def run_scar_nick_design(path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, ScarNickEvaluationReport]:
    spec, spec_path, workspace_root = load_scar_nick_spec(path)
    release_catalog, nickase_catalog = _load_catalogs(spec, workspace_root=workspace_root)
    report = build_scar_nick_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        release_catalog=release_catalog,
        nickase_catalog=nickase_catalog,
    )
    run_dir = build_run_dir(workspace_root=workspace_root, run_dir=spec.output.run_dir)
    if report.status != "satisfied":
        issue_codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        raise ValueError(f"Scar-nick design is unsatisfied; run validate for details. issues={issue_codes}")
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"Scar-nick run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(run_dir)
    ensure_run_dirs(run_dir)
    snapshot_inputs(
        run_dir,
        spec_path=spec_path,
        release_catalog_yaml=dump_release_enzyme_catalog_yaml(release_catalog),
        nickase_catalog_yaml=dump_nickase_catalog_yaml(nickase_catalog),
    )
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(run_dir, report)
    write_candidate_pair_call_table(run_dir, report)
    write_nickase_geometry_audit_table(run_dir, report)
    _publish_scar_nick_visuals(run_dir=run_dir, report=report, spec=spec)
    manifest = build_manifest(run_dir=run_dir, workspace_root=workspace_root, spec_path=spec_path, report=report)
    write_manifest(run_dir, manifest)
    write_status(
        run_dir,
        report=report,
        status_message=(
            f"scar-nick design {report.status} (accepted={len(report.candidates)}, issues={len(report.issues)})"
        ),
    )
    return run_dir, report


def scar_nick_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(resolved)
    status = load_status(resolved)
    report_file = report_json_path(resolved)
    if not report_file.exists():
        raise FileNotFoundError(f"Missing scar-nick report: {report_file}")
    assert_provenance_current(resolved, manifest)
    assert_manifest_artifacts_present(resolved, manifest)
    assert_manifest_hashes_current(resolved, manifest)
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    report = ScarNickEvaluationReport.model_validate(report_payload)
    if status.get("status") != "completed" or report.status != "satisfied":
        raise ValueError("Scar-nick show requires a satisfied completed run.")
    _assert_visual_publication_current(resolved, report)
    return {
        "run_dir": str(resolved),
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "candidate_count": len(report.candidates),
        "manifest_path": str(manifest_path(resolved).resolve()),
        "status_path": str(status_path(resolved).resolve()),
        "report_json": str(report_file.resolve()),
        "report_md": str(report_md_path(resolved).resolve()),
        "candidate_profiles": str(candidate_profiles_path(resolved).resolve()),
        "nickase_geometry_audit": str(nickase_geometry_audit_path(resolved).resolve()),
        "candidate_table": str(candidate_table_path(resolved).resolve()),
        "candidate_pair_call_table": str(candidate_pair_call_table_path(resolved).resolve()),
        "nickase_geometry_audit_table": str(nickase_geometry_audit_table_path(resolved).resolve()),
        "views_manifest": (
            str(views_manifest_path(resolved).resolve()) if views_manifest_path(resolved).exists() else None
        ),
        "terminal_nick_visual_contract": (
            str(post_terminal_nick_visual_contract_path(resolved).resolve())
            if post_terminal_nick_visual_contract_path(resolved).exists()
            else None
        ),
        "scar_nick_terminal_nick_visual_contracts": (
            str(scar_nick_terminal_nick_visual_contracts_path(resolved).resolve())
            if scar_nick_terminal_nick_visual_contracts_path(resolved).exists()
            else None
        ),
        "baserender_job": (
            str(scar_nick_terminal_nick_job_path(resolved).resolve())
            if scar_nick_terminal_nick_job_path(resolved).exists()
            else None
        ),
        "spec_snapshot": str(spec_snapshot_path(resolved).resolve()),
        "nickase_catalog": str(nickase_catalog_snapshot_path(resolved).resolve()),
        "release_catalog": str(release_catalog_snapshot_path(resolved).resolve()),
        "artifacts": manifest.get("artifacts", []),
    }


__all__ = ["run_scar_nick_design", "scar_nick_show_payload", "validate_scar_nick_spec"]
