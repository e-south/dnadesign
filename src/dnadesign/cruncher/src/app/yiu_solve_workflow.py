"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_solve_workflow.py

Application orchestration for YIU solve/search workflows.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.app.yiu_workflow import (
    _annotation_rows,
    _build_yiu_report,
    _catalog_bytes,
    _fragment_rows,
    _parts_rows,
    _publish_views,
    _v2_region_lookup,
)
from dnadesign.cruncher.bio import iupac_bases_for_symbol
from dnadesign.cruncher.yiu.artifacts import (
    annotations_path,
    build_solve_run_dir,
    catalog_fingerprint,
    design_id,
    fragments_path,
    input_fingerprint,
    parts_path,
    prepare_run_dir,
    prepare_solve_run_dir,
    resolve_code_revision,
    solve_accepted_hits_path,
    solve_hits_csv_path,
    solve_id,
    solve_manifest_path,
    solve_report_path,
    solve_status_path,
    write_csv,
    write_manifest,
    write_report,
    write_status,
    write_trace,
    write_trace_manifest,
)
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs, load_yiu_catalogs
from dnadesign.cruncher.yiu.load import (
    load_yiu_solve_spec,
    load_yiu_spec,
    resolve_base_spec_path_for_yiu_solve_spec,
)
from dnadesign.cruncher.yiu.models import (
    RegionSpecV2,
    YiuProcessSpecV2,
    YiuSolveHit,
    YiuSolveIssue,
    YiuSolveReport,
    YiuSolveReportMetadata,
    YiuSolveSourceWindowSpec,
    YiuValidationReport,
)


def _issue(code: str, message: str, **details: object) -> YiuSolveIssue:
    return YiuSolveIssue(code=code, message=message, details=dict(details))


_WARNING_MESSAGES = {
    "MAX_SEARCH_NODES_REACHED": "search.max_search_nodes reached before exhausting the solve search tree.",
    "MAX_ENUMERATED_CANDIDATES_REACHED": (
        "search.max_enumerated_candidates reached before exhausting the solve search space."
    ),
}


def _solve_name(spec_path: Path) -> str:
    suffix = ".yiu.solve.yaml"
    if spec_path.name.endswith(suffix):
        return spec_path.name[: -len(suffix)]
    return spec_path.stem


def _hit_id(rank: int) -> str:
    return f"hit_{rank:04d}"


def _expand_pattern(pattern: str) -> list[str]:
    alphabets: list[list[str]] = []
    for symbol in pattern:
        bases = sorted(set(iupac_bases_for_symbol(symbol)))
        if not bases:
            raise ValueError(f"unsupported solve pattern symbol: {symbol!r}")
        alphabets.append(bases)
    return ["".join(parts) for parts in product(*alphabets)]


def _window_candidates(window: YiuSolveSourceWindowSpec) -> list[str]:
    if window.pattern is not None:
        return _expand_pattern(window.pattern)
    values: list[str] = []
    for pattern in window.allowed_patterns:
        values.extend(_expand_pattern(pattern))
    return sorted(set(values))


def _resolve_variable_regions(
    base_spec: YiuProcessSpecV2,
    *,
    source_windows: list[YiuSolveSourceWindowSpec],
) -> dict[str, RegionSpecV2]:
    regions = _v2_region_lookup(base_spec)
    resolved: dict[str, RegionSpecV2] = {}
    intervals: list[tuple[int, int, str]] = []
    for window in source_windows:
        try:
            region = regions[window.span_ref]
        except KeyError as exc:
            raise ValueError(
                f"yiu_solve.variables.source_windows references unknown span_ref {window.span_ref!r}"
            ) from exc
        candidates = _window_candidates(window)
        region_length = region.end - region.start
        if any(len(candidate) != region_length for candidate in candidates):
            raise ValueError(
                f"solve variable {window.id} length must match span_ref {window.span_ref!r} length {region_length}"
            )
        for other_start, other_end, other_id in intervals:
            if not (region.end <= other_start or other_end <= region.start):
                raise ValueError(f"solve variable {window.id} overlaps solve variable {other_id}")
        intervals.append((region.start, region.end, window.id))
        resolved[window.id] = region
    return resolved


def _candidate_sequence(
    base_sequence: str,
    *,
    windows: list[YiuSolveSourceWindowSpec],
    regions_by_id: dict[str, RegionSpecV2],
    assignments: dict[str, str],
) -> str:
    sequence_chars = list(base_sequence)
    for window in windows:
        region = regions_by_id[window.id]
        replacement = assignments[window.id]
        sequence_chars[region.start : region.end] = list(replacement)
    return "".join(sequence_chars)


def _candidate_payload(
    *,
    base_spec: YiuProcessSpecV2,
    candidate_sequence: str,
    publish_contract_version: int,
    emit_view_contracts: bool,
    emit_baserender_jobs: bool,
) -> dict[str, object]:
    payload = {"yiu": base_spec.model_dump(mode="json", by_alias=True)}
    source_oligo = payload["yiu"]["source_oligo"]
    source_oligo["sequence"] = candidate_sequence
    source_oligo["authored_sequence"] = candidate_sequence
    source_oligo["part_instances"] = []
    payload["yiu"]["output"]["publish_contract_version"] = publish_contract_version
    payload["yiu"]["output"]["emit_view_contracts"] = emit_view_contracts
    payload["yiu"]["output"]["emit_baserender_jobs"] = emit_baserender_jobs
    return payload


def _collect_hard_invariant_results(report: YiuValidationReport) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for state in report.states:
        raw = state.metadata.get("hard_invariants")
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            invariant_id = str(item.get("id") or "").strip()
            if not invariant_id:
                continue
            results[invariant_id] = item
    return results


def _extra_enzyme_site_count(base_spec: YiuProcessSpecV2, *, source_sequence: str) -> int:
    annotations = base_spec.source_oligo.annotations
    expected_counts: dict[str, int] = {}
    for site in [*annotations.restriction_sites, *annotations.nickase_sites]:
        expected_counts[site.recognition_sequence] = expected_counts.get(site.recognition_sequence, 0) + 1
    unexpected = 0
    for recognition_sequence, expected_count in expected_counts.items():
        observed = source_sequence.count(recognition_sequence)
        unexpected += max(0, observed - expected_count)
    return unexpected


def _gc_deviation(assignments: dict[str, str]) -> float:
    if not assignments:
        return 0.0
    sequence = "".join(assignments.values())
    if not sequence:
        return 0.0
    gc_count = sum(1 for base in sequence if base in {"G", "C"})
    return abs((gc_count / len(sequence)) - 0.5)


def _homopolymer_penalty(source_sequence: str) -> int:
    longest = 0
    current = 0
    last = ""
    for base in source_sequence:
        if base == last:
            current += 1
        else:
            current = 1
            last = base
        longest = max(longest, current)
    return max(0, longest - 4)


def _fragmentation_slack(base_spec: YiuProcessSpecV2, report: YiuValidationReport) -> int:
    threshold: int | None = None
    for invariant in base_spec.hard_invariants:
        if invariant.class_ == "sacrificial_fragmentation":
            raw = invariant.params.get("max_fragment_nt")
            threshold = int(raw) if raw is not None else None
            break
    if threshold is None:
        return 0
    fragment_lengths: list[int] = []
    for state in report.states:
        raw_lengths = state.metadata.get("fragment_lengths")
        if isinstance(raw_lengths, list):
            fragment_lengths.extend(int(value) for value in raw_lengths)
    if not fragment_lengths:
        return 0
    return threshold - max(fragment_lengths)


def _score_tuple(
    *,
    base_spec: YiuProcessSpecV2,
    source_sequence: str,
    assignments: dict[str, str],
    report: YiuValidationReport,
    invariant_results: dict[str, dict[str, Any]],
) -> list[float | int | str]:
    hard_margin_penalty_total = 0
    for result in invariant_results.values():
        observed = result.get("observed")
        if isinstance(observed, dict) and "margin_penalty" in observed:
            hard_margin_penalty_total += int(observed["margin_penalty"])
    return [
        hard_margin_penalty_total,
        _extra_enzyme_site_count(base_spec, source_sequence=source_sequence),
        -_fragmentation_slack(base_spec, report),
        _gc_deviation(assignments),
        _homopolymer_penalty(source_sequence),
        source_sequence,
    ]


def _materialize_hit_bundle(
    *,
    run_dir: Path,
    rank: int,
    candidate_payload: dict[str, object],
    report: YiuValidationReport,
    workspace_root: Path,
    catalog_paths: list[Path],
    catalog_bytes: bytes,
    code_revision: str | None,
    emit_view_contracts: bool,
    emit_baserender_jobs: bool,
) -> tuple[Path, str]:
    hit_dir = run_dir / "hits" / _hit_id(rank)
    prepare_run_dir(
        hit_dir,
        force_overwrite=False,
        emit_view_contracts=emit_view_contracts,
        emit_baserender_jobs=emit_baserender_jobs,
    )
    resolved_candidate_path = hit_dir / "resolved_candidate.yiu.yaml"
    resolved_candidate_path.write_text(yaml.safe_dump(candidate_payload, sort_keys=False), encoding="utf-8")
    report = report.model_copy(update={"run_dir": str(hit_dir.resolve())})
    write_report(hit_dir, report)
    write_status(
        hit_dir,
        report,
        input_fingerprint_value=input_fingerprint(
            spec_bytes=resolved_candidate_path.read_bytes(),
            catalog_bytes=catalog_bytes,
        ),
        catalog_fingerprint_value=catalog_fingerprint(catalog_bytes=catalog_bytes),
        code_revision=code_revision,
    )
    write_trace(hit_dir, report.states)
    write_trace_manifest(hit_dir, report)
    write_csv(
        parts_path(hit_dir),
        fieldnames=["state_id", "part_id", "role", "sequence"],
        rows=_parts_rows(report),
    )
    candidate_spec = YiuProcessSpecV2.model_validate(candidate_payload["yiu"])
    write_csv(
        annotations_path(hit_dir),
        fieldnames=["category", "id", "start", "end", "label"],
        rows=_annotation_rows(candidate_spec),
    )
    write_csv(
        fragments_path(hit_dir),
        fieldnames=["state_id", "fragment_id", "length_nt"],
        rows=_fragment_rows(report),
    )
    if emit_view_contracts:
        _publish_views(hit_dir, report, emit_baserender_jobs=emit_baserender_jobs)
    write_manifest(
        hit_dir,
        workspace_root=workspace_root,
        spec_path=resolved_candidate_path,
        report=report,
        input_fingerprint_value=input_fingerprint(
            spec_bytes=resolved_candidate_path.read_bytes(),
            catalog_bytes=catalog_bytes,
        ),
        catalog_fingerprint_value=catalog_fingerprint(catalog_bytes=catalog_bytes),
        code_revision=code_revision,
        catalog_paths=catalog_paths,
    )
    explicit_design_id = design_id(spec_bytes=resolved_candidate_path.read_bytes(), catalog_bytes=catalog_bytes)
    return hit_dir, explicit_design_id


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _append_warning(*, code: str, warnings: list[str], warning_codes: list[str]) -> None:
    if code in warning_codes:
        return
    warning_codes.append(code)
    warnings.append(_WARNING_MESSAGES[code])


def _score_key(hit: YiuSolveHit) -> tuple[float | int | str, ...]:
    return tuple(hit.score)


def _retain_top_hits(
    *,
    retained_hits: list[YiuSolveHit],
    candidate_hit: YiuSolveHit,
    max_hits: int,
) -> None:
    if len(retained_hits) < max_hits:
        retained_hits.append(candidate_hit)
        retained_hits.sort(key=_score_key)
        return
    worst_index, worst_hit = max(enumerate(retained_hits), key=lambda item: _score_key(item[1]))
    if _score_key(candidate_hit) < _score_key(worst_hit):
        retained_hits[worst_index] = candidate_hit
        retained_hits.sort(key=_score_key)


def solve_yiu_spec(
    path: str | Path,
    *,
    max_hits: int | None = None,
    materialize_top_k: int | None = None,
) -> tuple[YiuSolveReport, Path, YiuProcessSpecV2, LoadedYiuCatalogs, Path]:
    solve_spec, resolved_solve_spec_path, workspace_root = load_yiu_solve_spec(path)
    base_spec_path = resolve_base_spec_path_for_yiu_solve_spec(solve_spec, workspace_root=workspace_root)
    base_spec, _resolved_base_spec_path, _base_workspace_root = load_yiu_spec(base_spec_path)
    if not isinstance(base_spec, YiuProcessSpecV2):
        raise ValueError("YIU solve currently supports schema_version: 2 base specs only.")
    catalogs = load_yiu_catalogs(base_spec, workspace_root=workspace_root)
    search = solve_spec.search.model_copy(
        update={
            "max_hits": max_hits if max_hits is not None else solve_spec.search.max_hits,
            "materialize_top_k": (
                materialize_top_k if materialize_top_k is not None else solve_spec.search.materialize_top_k
            ),
        }
    )
    if search.materialize_top_k > search.max_hits:
        raise ValueError("search.materialize_top_k must be <= search.max_hits")
    regions_by_window_id = _resolve_variable_regions(base_spec, source_windows=solve_spec.variables.source_windows)
    base_sequence = base_spec.source_oligo.sequence or ""
    required_invariant_ids = {invariant.id for invariant in base_spec.hard_invariants}

    candidate_lists = [(_window.id, _window_candidates(_window)) for _window in solve_spec.variables.source_windows]
    warnings: list[str] = []
    warning_codes: list[str] = []
    search_node_count = 0
    enumerated_candidate_count = 0
    accepted_candidate_count = 0
    retained_hits: list[YiuSolveHit] = []
    for option_values in product(*(values for _id, values in candidate_lists)):
        if search_node_count >= search.max_search_nodes:
            _append_warning(
                code="MAX_SEARCH_NODES_REACHED",
                warnings=warnings,
                warning_codes=warning_codes,
            )
            break
        if enumerated_candidate_count >= search.max_enumerated_candidates:
            _append_warning(
                code="MAX_ENUMERATED_CANDIDATES_REACHED",
                warnings=warnings,
                warning_codes=warning_codes,
            )
            break
        search_node_count += 1
        enumerated_candidate_count += 1
        assignments = {
            window_id: value for (window_id, _values), value in zip(candidate_lists, option_values, strict=True)
        }
        candidate_source_sequence = _candidate_sequence(
            base_sequence,
            windows=solve_spec.variables.source_windows,
            regions_by_id=regions_by_window_id,
            assignments=assignments,
        )
        candidate_payload = _candidate_payload(
            base_spec=base_spec,
            candidate_sequence=candidate_source_sequence,
            publish_contract_version=solve_spec.output.publish_contract_version,
            emit_view_contracts=solve_spec.output.emit_view_contracts,
            emit_baserender_jobs=solve_spec.output.emit_baserender_jobs,
        )
        candidate_spec = YiuProcessSpecV2.model_validate(candidate_payload["yiu"])
        candidate_report = _build_yiu_report(candidate_spec, catalogs=catalogs)
        invariant_results = _collect_hard_invariant_results(candidate_report)
        if solve_spec.candidate_policy.require_guaranteed_hard_invariants:
            if required_invariant_ids - set(invariant_results):
                continue
            if any(str(result.get("status")) != "guaranteed" for result in invariant_results.values()):
                continue
        if solve_spec.candidate_policy.forbid_possible_hits and any(
            str(result.get("status")) == "possible" for result in invariant_results.values()
        ):
            continue
        if candidate_report.status != "satisfied":
            continue
        accepted_candidate_count += 1
        _retain_top_hits(
            retained_hits=retained_hits,
            candidate_hit=YiuSolveHit(
                rank=1,
                hit_id="candidate",
                score=_score_tuple(
                    base_spec=base_spec,
                    source_sequence=candidate_source_sequence,
                    assignments=assignments,
                    report=candidate_report,
                    invariant_results=invariant_results,
                ),
                source_sequence=candidate_source_sequence,
                variable_assignments=assignments,
            ),
            max_hits=search.max_hits,
        )

    normalized_hits: list[YiuSolveHit] = []
    for index, hit in enumerate(retained_hits, start=1):
        normalized_hits.append(
            hit.model_copy(
                update={
                    "rank": index,
                    "hit_id": _hit_id(index),
                }
            )
        )
    metadata = YiuSolveReportMetadata(
        max_hits=search.max_hits,
        materialize_top_k=search.materialize_top_k,
        warnings=warnings,
        warning_codes=warning_codes,
        search_node_count=search_node_count,
        enumerated_candidate_count=enumerated_candidate_count,
        accepted_candidate_count=accepted_candidate_count,
        returned_hit_count=len(normalized_hits),
        materialized_hit_count=0,
        search_truncated=bool(warning_codes),
        accepted_pool_truncated=accepted_candidate_count > len(normalized_hits),
    )
    report = YiuSolveReport(
        status="solved" if normalized_hits else "no_hits",
        spec_path=str(resolved_solve_spec_path.resolve()),
        base_spec_path=str(base_spec_path.resolve()),
        metadata=metadata,
        hits=normalized_hits,
        issues=[],
    )
    return report, workspace_root, base_spec, catalogs, resolved_solve_spec_path


def _solve_level_job_payload(*, view_filename: str, contract_kind: str, alphabet: str) -> dict[str, Any]:
    adapter_kind = {
        "yiu_linear_state_v1": "yiu_linear_state_v1",
        "yiu_hairpin_topology_v1": "yiu_hairpin_topology_v1",
        "yiu_topology_cartoon_v1": "yiu_topology_cartoon_v1",
    }[contract_kind]
    renderer_name = {
        "yiu_linear_state_v1": "sequence_rows",
        "yiu_hairpin_topology_v1": "hairpin_cartoon",
        "yiu_topology_cartoon_v1": "topology_cartoon",
    }[contract_kind]
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "json",
            "path": f"../views/{view_filename}",
            "adapter": {"kind": adapter_kind},
            "alphabet": "IUPAC_DNA" if alphabet.upper() == "IUPAC_DNA" else "DNA",
        },
        "render": {"renderer": renderer_name, "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "path": f"../renders/{Path(view_filename).stem}.pdf", "fmt": "pdf"}],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }


def run_yiu_solve(
    path: str | Path,
    *,
    force_overwrite: bool = False,
    max_hits: int | None = None,
    materialize_top_k: int | None = None,
) -> tuple[Path | None, YiuSolveReport]:
    report, workspace_root, base_spec, catalogs, resolved_solve_spec_path = solve_yiu_spec(
        path,
        max_hits=max_hits,
        materialize_top_k=materialize_top_k,
    )
    solve_spec, _resolved_spec_path, _workspace_root = load_yiu_solve_spec(path)
    catalog_paths = list(catalogs.paths)
    solve_spec_bytes = resolved_solve_spec_path.read_bytes()
    base_spec_bytes = resolve_base_spec_path_for_yiu_solve_spec(solve_spec, workspace_root=workspace_root).read_bytes()
    catalog_bytes = _catalog_bytes(catalog_paths)
    solve_run_id = solve_id(spec_bytes=solve_spec_bytes, base_spec_bytes=base_spec_bytes, catalog_bytes=catalog_bytes)
    run_dir = build_solve_run_dir(
        workspace_root=workspace_root,
        run_root=solve_spec.output.run_dir,
        solve_name=_solve_name(resolved_solve_spec_path),
        run_id=solve_run_id,
    )
    prepare_solve_run_dir(
        run_dir,
        force_overwrite=force_overwrite,
        emit_view_contracts=solve_spec.output.emit_view_contracts,
        emit_baserender_jobs=solve_spec.output.emit_baserender_jobs,
    )
    code_revision = resolve_code_revision(workspace_root)
    report = report.model_copy(update={"solve_id": solve_run_id, "run_dir": str(run_dir.resolve())})

    materialized_hits: list[dict[str, object]] = []
    visual_manifest_entries: list[dict[str, object]] = []
    top_k = min(
        solve_spec.search.materialize_top_k if materialize_top_k is None else materialize_top_k,
        len(report.hits),
    )
    for hit in report.hits[:top_k]:
        candidate_payload = _candidate_payload(
            base_spec=base_spec,
            candidate_sequence=hit.source_sequence,
            publish_contract_version=solve_spec.output.publish_contract_version,
            emit_view_contracts=solve_spec.output.emit_view_contracts,
            emit_baserender_jobs=solve_spec.output.emit_baserender_jobs,
        )
        candidate_spec = YiuProcessSpecV2.model_validate(candidate_payload["yiu"])
        candidate_report = _build_yiu_report(candidate_spec, catalogs=catalogs)
        hit_dir, explicit_design_id = _materialize_hit_bundle(
            run_dir=run_dir,
            rank=hit.rank,
            candidate_payload=candidate_payload,
            report=candidate_report,
            workspace_root=workspace_root,
            catalog_paths=catalog_paths,
            catalog_bytes=catalog_bytes,
            code_revision=code_revision,
            emit_view_contracts=solve_spec.output.emit_view_contracts,
            emit_baserender_jobs=solve_spec.output.emit_baserender_jobs,
        )
        final_state = candidate_report.states[-1]
        solve_view_path: Path | None = None
        job_path: Path | None = None
        if solve_spec.output.emit_view_contracts:
            final_view_path = hit_dir / "published" / "views" / f"{final_state.state_id}.json"
            solve_view_filename = f"{_hit_id(hit.rank)}__{final_state.state_id}.json"
            solve_view_path = run_dir / "published" / "views" / solve_view_filename
            solve_view_path.write_text(final_view_path.read_text(encoding="utf-8"), encoding="utf-8")
            view_payload = json.loads(solve_view_path.read_text(encoding="utf-8"))
            visual_manifest_entry = {
                "rank": hit.rank,
                "hit_id": _hit_id(hit.rank),
                "state_id": final_state.state_id,
                "path": f"published/views/{solve_view_filename}",
                "source_hit_path": f"hits/{_hit_id(hit.rank)}",
                "contract_kind": view_payload["contract_kind"],
            }
            if solve_spec.output.emit_baserender_jobs:
                job_path = (
                    run_dir / "published" / "baserender_jobs" / f"{_hit_id(hit.rank)}__{final_state.state_id}.job.yaml"
                )
                job_path.write_text(
                    yaml.safe_dump(
                        _solve_level_job_payload(
                            view_filename=solve_view_filename,
                            contract_kind=str(view_payload["contract_kind"]),
                            alphabet=str(view_payload.get("alphabet") or "DNA"),
                        ),
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                visual_manifest_entry["job_path"] = f"published/baserender_jobs/{job_path.name}"
            visual_manifest_entries.append(visual_manifest_entry)
        materialized_hits.append(
            hit.model_copy(
                update={
                    "materialized_run_dir": str(hit_dir.resolve()),
                    "explicit_design_id": explicit_design_id,
                    "final_state_id": final_state.state_id,
                    "final_state_view_path": str(solve_view_path.resolve()) if solve_view_path is not None else None,
                    "final_state_job_path": str(job_path.resolve()) if job_path is not None else None,
                }
            ).model_dump(mode="json")
        )

    report = report.model_copy(
        update={
            "hits": [YiuSolveHit.model_validate(hit_payload) for hit_payload in materialized_hits]
            + report.hits[top_k:],
            "metadata": report.metadata.model_copy(update={"materialized_hit_count": top_k}),
        }
    )

    _write_json(
        solve_report_path(run_dir),
        report.model_dump(mode="json"),
    )
    _write_jsonl(
        solve_accepted_hits_path(run_dir),
        [hit.model_dump(mode="json") for hit in report.hits],
    )
    write_csv(
        solve_hits_csv_path(run_dir),
        fieldnames=[
            "rank",
            "hit_id",
            "score",
            "source_sequence",
            "materialized_run_dir",
            "final_state_id",
        ],
        rows=[
            {
                "rank": hit.rank,
                "hit_id": hit.hit_id,
                "score": json.dumps(hit.score),
                "source_sequence": hit.source_sequence,
                "materialized_run_dir": hit.materialized_run_dir,
                "final_state_id": hit.final_state_id,
            }
            for hit in report.hits
        ],
    )
    visual_manifest_path: str | None = None
    if solve_spec.output.emit_view_contracts:
        manifest_path = run_dir / "published" / "visual_manifest.json"
        _write_json(
            manifest_path,
            {
                "contract_version": solve_spec.output.publish_contract_version,
                "family": "yiu",
                "workflow": "yiu_solve",
                "protocol_template": base_spec.protocol_template,
                "view_count": len(visual_manifest_entries),
                "job_count": sum(1 for entry in visual_manifest_entries if "job_path" in entry),
                "render_count": 0,
                "views": visual_manifest_entries,
            },
        )
        visual_manifest_path = str(manifest_path.resolve())
    materialized_hit_bundle_roots = [f"hits/{hit.hit_id}" for hit in report.hits[:top_k]]
    copied_top_hit_view_paths = [entry["path"] for entry in visual_manifest_entries]
    copied_top_hit_job_paths = [entry["job_path"] for entry in visual_manifest_entries if "job_path" in entry]
    _write_json(
        solve_status_path(run_dir),
        {
            "stage": "yiu_solve",
            "family": "yiu",
            "status": report.status,
            "solve_id": report.solve_id,
            "run_dir": str(run_dir.resolve()),
            "hit_count": len(report.hits),
            "accepted_candidate_count": report.metadata.accepted_candidate_count,
            "returned_hit_count": report.metadata.returned_hit_count,
            "materialized_hit_count": report.metadata.materialized_hit_count,
            "warning_codes": report.metadata.warning_codes,
            "warnings": report.metadata.warnings,
            "search_truncated": report.metadata.search_truncated,
            "accepted_pool_truncated": report.metadata.accepted_pool_truncated,
            "accepted_hits_path": str(solve_accepted_hits_path(run_dir).resolve()),
            "visual_manifest_path": visual_manifest_path,
            "first_hit_path": str((run_dir / "hits" / "hit_0001").resolve()) if top_k > 0 else None,
            "top_hit_bundle_paths": [
                str((run_dir / relative_path).resolve()) for relative_path in materialized_hit_bundle_roots
            ],
        },
    )
    artifacts = [
        {"name": "report", "path": solve_report_path(run_dir).name},
        {"name": "status", "path": solve_status_path(run_dir).name},
        {"name": "manifest", "path": solve_manifest_path(run_dir).name},
        {"name": "accepted_hits", "path": solve_accepted_hits_path(run_dir).name},
        {"name": "hits_csv", "path": solve_hits_csv_path(run_dir).name},
        {"name": "hits", "path": "hits"},
    ]
    if solve_spec.output.emit_view_contracts:
        artifacts.extend(
            [
                {"name": "visual_manifest", "path": "published/visual_manifest.json"},
                {"name": "published_views", "path": "published/views"},
            ]
        )
    if solve_spec.output.emit_baserender_jobs:
        artifacts.extend(
            [
                {"name": "published_jobs", "path": "published/baserender_jobs"},
                {"name": "published_renders", "path": "published/renders"},
            ]
        )
    published_artifacts: dict[str, str] = {}
    if (run_dir / "published" / "views").exists():
        published_artifacts["views_dir"] = "published/views"
    if (run_dir / "published" / "visual_manifest.json").exists():
        published_artifacts["visual_manifest"] = "published/visual_manifest.json"
    if (run_dir / "published" / "baserender_jobs").exists():
        published_artifacts["baserender_jobs_dir"] = "published/baserender_jobs"
    if (run_dir / "published" / "renders").exists():
        published_artifacts["renders_dir"] = "published/renders"
    _write_json(
        solve_manifest_path(run_dir),
        {
            "stage": "yiu_solve",
            "family": "yiu",
            "workflow": "yiu_solve",
            "solve_id": report.solve_id,
            "status": report.status,
            "spec_path": str(resolved_solve_spec_path.resolve()),
            "base_spec_path": report.base_spec_path,
            "run_dir": str(run_dir.resolve()),
            "published_artifacts": published_artifacts,
            "hit_bundle_root": "hits",
            "top_hit_ids": [hit.hit_id for hit in report.hits[:top_k]],
            "materialized_hit_bundle_roots": materialized_hit_bundle_roots,
            "copied_top_hit_artifacts": {
                "view_paths": copied_top_hit_view_paths,
                "job_paths": copied_top_hit_job_paths,
            },
            "hits_csv": "hits.csv",
            "accepted_hits_stream": "accepted_hits.jsonl",
            "artifacts": artifacts,
        },
    )
    return run_dir, report
