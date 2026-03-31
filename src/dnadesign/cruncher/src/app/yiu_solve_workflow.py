"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_solve_workflow.py

Canonical YIU v4 solve orchestration.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any

from dnadesign.cruncher.app.yiu_workflow.bundle import (
    _catalog_bytes,
    _write_csv,
    _write_explicit_bundle_from_report,
    _write_json,
)
from dnadesign.cruncher.app.yiu_workflow.report import _build_yiu_report
from dnadesign.cruncher.bio import iupac_bases_for_symbol
from dnadesign.cruncher.yiu.catalog import load_yiu_catalogs
from dnadesign.cruncher.yiu.load import (
    load_yiu_solve_spec,
    load_yiu_spec,
    resolve_base_spec_path_for_yiu_solve_spec,
)
from dnadesign.cruncher.yiu.models import (
    YiuProcessSpecV4,
    YiuSolveIssue,
    YiuSolveReport,
    YiuSolveReportMetadata,
    YiuSolveScaffoldWindowSpec,
    YiuSolveSpec,
)

_WARNING_MESSAGES = {
    "MAX_SEARCH_NODES_REACHED": "search.max_search_nodes reached before exhausting the solve search tree.",
    "MAX_ENUMERATED_CANDIDATES_REACHED": (
        "search.max_enumerated_candidates reached before exhausting the solve search space."
    ),
}


def _issue(code: str, message: str, **details: object) -> YiuSolveIssue:
    return YiuSolveIssue(code=code, message=message, details=dict(details))


def _solve_name(spec_path: Path) -> str:
    suffix = ".yiu.solve.yaml"
    if spec_path.name.endswith(suffix):
        return spec_path.name[: -len(suffix)]
    return spec_path.stem


def _run_id(*, solve_spec_bytes: bytes, base_spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256(solve_spec_bytes + b"\n" + base_spec_bytes + b"\n" + catalog_bytes).hexdigest()[:12]


def _expand_pattern(pattern: str) -> list[str]:
    alphabets: list[list[str]] = []
    for symbol in pattern:
        bases = sorted(set(iupac_bases_for_symbol(symbol)))
        if not bases:
            raise ValueError(f"unsupported solve pattern symbol: {symbol!r}")
        alphabets.append(bases)
    return ["".join(parts) for parts in product(*alphabets)]


def _unique_ordered(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _payload_candidates(solve_spec: YiuSolveSpec) -> list[str]:
    target = solve_spec.target
    if target.payload_sequence is not None:
        return [target.payload_sequence]
    assert target.payload_pattern is not None
    return _expand_pattern(target.payload_pattern)


def _window_candidates(window: YiuSolveScaffoldWindowSpec) -> list[str]:
    expanded: list[str] = []
    for pattern in window.allowed_patterns:
        expanded.extend(_expand_pattern(pattern))
    return _unique_ordered(expanded)


def _owner_ranges(spec: YiuProcessSpecV4) -> dict[str, tuple[int, int]]:
    return {owner.id: (owner.start, owner.end) for owner in spec.source_oligo.structural_owners}


def _resolve_window_ranges(
    spec: YiuProcessSpecV4,
    *,
    windows: list[YiuSolveScaffoldWindowSpec],
) -> dict[str, tuple[int, int]]:
    owner_ranges = _owner_ranges(spec)
    resolved: dict[str, tuple[int, int]] = {}
    for window in windows:
        owner_start, owner_end = owner_ranges[window.owner_id]
        owner_length = owner_end - owner_start
        if window.relative_end > owner_length:
            raise ValueError(f"scaffold window {window.id} exceeds owner {window.owner_id!r} length {owner_length}")
        resolved[window.id] = (owner_start + window.relative_start, owner_start + window.relative_end)
    return resolved


def _replace_interval(sequence: str, *, start: int, end: int, replacement: str) -> str:
    return f"{sequence[:start]}{replacement}{sequence[end:]}"


def _candidate_spec(
    *,
    base_spec: YiuProcessSpecV4,
    payload_sequence: str,
    bulge_mask: list[int],
    window_assignments: dict[str, str],
    window_ranges: dict[str, tuple[int, int]],
    windows: list[YiuSolveScaffoldWindowSpec],
) -> YiuProcessSpecV4:
    payload_left_length = len(base_spec.owner_sequence("payload_left_half"))
    payload_right_length = len(base_spec.owner_sequence("payload_right_half"))
    if len(payload_sequence) != payload_left_length + payload_right_length:
        raise ValueError(
            "solve target payload length must match payload_left_half + payload_right_half lengths "
            f"({payload_left_length + payload_right_length})"
        )

    payload_left = payload_sequence[:payload_left_length]
    payload_right = payload_sequence[payload_left_length:]

    owner_ranges = _owner_ranges(base_spec)
    source_sequence = base_spec.source_oligo.authored_sequence
    source_sequence = _replace_interval(
        source_sequence,
        start=owner_ranges["payload_left_half"][0],
        end=owner_ranges["payload_left_half"][1],
        replacement=payload_left,
    )
    source_sequence = _replace_interval(
        source_sequence,
        start=owner_ranges["payload_right_half"][0],
        end=owner_ranges["payload_right_half"][1],
        replacement=payload_right,
    )
    for window in windows:
        start, end = window_ranges[window.id]
        source_sequence = _replace_interval(
            source_sequence,
            start=start,
            end=end,
            replacement=window_assignments[window.id],
        )

    payload = base_spec.model_dump(mode="json", by_alias=True)
    payload["source_oligo"]["authored_sequence"] = source_sequence
    payload["payload"]["target_sequence"] = payload_sequence
    payload["payload"]["bulge_mask"] = list(bulge_mask)
    return YiuProcessSpecV4.model_validate(payload)


def _append_warning(*, code: str, warnings: list[str], warning_codes: list[str]) -> None:
    if code in warning_codes:
        return
    warning_codes.append(code)
    warnings.append(_WARNING_MESSAGES[code])


def _inventory_for_solution(run_dir: Path, *, solution_dir: Path) -> dict[str, Any]:
    solution_inventory = json.loads((solution_dir / "visual_inventory.json").read_text(encoding="utf-8"))
    views = [
        {
            **view,
            "view_contract_path": f"solution/{view['view_contract_path']}",
            "render_artifact_path": f"solution/{view['render_artifact_path']}",
            "render_job_path": (None if view.get("render_job_path") is None else f"solution/{view['render_job_path']}"),
        }
        for view in solution_inventory.get("views", [])
        if isinstance(view, dict)
    ]
    inventory = {
        "bundle_kind": "solve",
        "protocol_template": solution_inventory.get("protocol_template"),
        "renderer_kind": solution_inventory.get("renderer_kind"),
        "view_count": len(views),
        "render_count": 0,
        "render_status": "not_requested",
        "last_rendered_at": None,
        "views": views,
    }
    _write_json(run_dir / "visual_inventory.json", inventory)
    return inventory


def _write_solve_bundle(
    run_dir: Path,
    *,
    solve_spec: YiuSolveSpec,
    resolved_solve_spec_path: Path,
    base_spec_path: Path,
    report: YiuSolveReport,
    comparison_rows: list[dict[str, Any]],
    inventory: dict[str, Any] | None,
) -> None:
    _write_json(run_dir / "solve_report.json", report.model_dump(mode="json"))
    _write_json(
        run_dir / "solve_status.json",
        {
            "status": report.status,
            "schema_version": solve_spec.output.publish_contract_version,
            "protocol_template": "yiu_circularized_payload_v1",
            "satisfying_solution_count": report.satisfying_solution_count,
            "comparison_solution_count": report.comparison_solution_count,
            "selected_solution_path": report.selected_solution_path,
        },
    )
    _write_json(
        run_dir / "solve_manifest.json",
        {
            "run_dir": str(run_dir.resolve()),
            "solve_spec_path": str(resolved_solve_spec_path.resolve()),
            "base_spec_path": str(base_spec_path.resolve()),
            "selected_solution_path": report.selected_solution_path,
            "visual_inventory_path": (
                str((run_dir / "visual_inventory.json").resolve()) if inventory is not None else None
            ),
        },
    )
    if comparison_rows:
        _write_csv(
            run_dir / "comparison" / "solutions.csv",
            ["rank", "source_sequence", "solution_path"],
            comparison_rows,
        )


def run_yiu_solve(
    path: str | Path,
    *,
    force_overwrite: bool = False,
) -> tuple[Path, YiuSolveReport]:
    solve_spec, resolved_solve_spec_path, workspace_root = load_yiu_solve_spec(path)
    base_spec_path = resolve_base_spec_path_for_yiu_solve_spec(solve_spec, workspace_root=workspace_root)
    base_spec, _resolved_base_spec_path, _base_workspace_root = load_yiu_spec(base_spec_path)
    catalogs = load_yiu_catalogs(base_spec, workspace_root=workspace_root)

    solve_spec_bytes = resolved_solve_spec_path.read_bytes()
    base_spec_bytes = base_spec_path.read_bytes()
    catalog_bytes = _catalog_bytes(list(catalogs.paths))
    run_id = _run_id(
        solve_spec_bytes=solve_spec_bytes,
        base_spec_bytes=base_spec_bytes,
        catalog_bytes=catalog_bytes,
    )
    run_dir = workspace_root / solve_spec.output.run_dir / _solve_name(resolved_solve_spec_path) / run_id
    if run_dir.exists() and not force_overwrite:
        raise ValueError(f"YIU solve directory already exists: {run_dir}. Use --force-overwrite to replace it.")
    if run_dir.exists():
        for child in sorted(run_dir.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
        for child in sorted(run_dir.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
    run_dir.mkdir(parents=True, exist_ok=True)

    payload_candidates = _payload_candidates(solve_spec)
    window_ranges = _resolve_window_ranges(base_spec, windows=solve_spec.scaffold_windows)
    window_option_sets = [(_window.id, _window_candidates(_window)) for _window in solve_spec.scaffold_windows]

    warnings: list[str] = []
    warning_codes: list[str] = []
    exhaustive_search = True
    search_node_count = 0
    enumerated_candidate_count = 0
    satisfying_candidates: list[tuple[str, YiuProcessSpecV4]] = []

    for payload_sequence in payload_candidates:
        for option_values in product(*(values for _window_id, values in window_option_sets)):
            if search_node_count >= solve_spec.search.max_search_nodes:
                exhaustive_search = False
                _append_warning(
                    code="MAX_SEARCH_NODES_REACHED",
                    warnings=warnings,
                    warning_codes=warning_codes,
                )
                break
            if enumerated_candidate_count >= solve_spec.search.max_enumerated_candidates:
                exhaustive_search = False
                _append_warning(
                    code="MAX_ENUMERATED_CANDIDATES_REACHED",
                    warnings=warnings,
                    warning_codes=warning_codes,
                )
                break
            search_node_count += 1
            enumerated_candidate_count += 1
            assignments = {
                window_id: value for (window_id, _values), value in zip(window_option_sets, option_values, strict=True)
            }
            candidate_spec = _candidate_spec(
                base_spec=base_spec,
                payload_sequence=payload_sequence,
                bulge_mask=solve_spec.target.bulge_mask,
                window_assignments=assignments,
                window_ranges=window_ranges,
                windows=solve_spec.scaffold_windows,
            )
            candidate_report = _build_yiu_report(candidate_spec, catalogs=catalogs)
            if candidate_report.status != "satisfied":
                continue
            satisfying_candidates.append((candidate_spec.source_oligo.authored_sequence, candidate_spec))
        if not exhaustive_search:
            break

    satisfying_candidates = sorted(satisfying_candidates, key=lambda item: item[0])
    selected_solution_path: str | None = None
    selected_source_sequence: str | None = None
    comparison_solution_count = 0
    comparison_rows: list[dict[str, Any]] = []
    inventory: dict[str, Any] | None = None

    if not exhaustive_search:
        status = "incomplete_search"
    elif not satisfying_candidates:
        status = "unsatisfied"
    else:
        status = "solved"
        selected_source_sequence, selected_spec = satisfying_candidates[0]
        solution_dir = run_dir / "solution"
        selected_report = _build_yiu_report(selected_spec, catalogs=catalogs)
        _write_explicit_bundle_from_report(
            solution_dir,
            spec=selected_spec,
            resolved_spec_path=base_spec_path,
            report=selected_report,
            catalog_paths=list(catalogs.paths),
        )
        selected_solution_path = str(solution_dir.resolve())
        inventory = _inventory_for_solution(run_dir, solution_dir=solution_dir)

        if solve_spec.solve.compare_solutions:
            materialized = satisfying_candidates[: solve_spec.solve.max_solutions]
            comparison_solution_count = max(0, len(materialized) - 1)
            for index, (source_sequence, candidate_spec) in enumerate(materialized[1:], start=2):
                alternative_dir = run_dir / "alternatives" / f"solution_{index:04d}"
                candidate_report = _build_yiu_report(candidate_spec, catalogs=catalogs)
                _write_explicit_bundle_from_report(
                    alternative_dir,
                    spec=candidate_spec,
                    resolved_spec_path=base_spec_path,
                    report=candidate_report,
                    catalog_paths=list(catalogs.paths),
                )
                comparison_rows.append(
                    {
                        "rank": index,
                        "source_sequence": source_sequence,
                        "solution_path": str(alternative_dir.resolve()),
                    }
                )

    metadata = YiuSolveReportMetadata(
        search_node_count=search_node_count,
        enumerated_candidate_count=enumerated_candidate_count,
        satisfying_solution_count=len(satisfying_candidates),
        exhaustive_search=exhaustive_search,
        warning_codes=warning_codes,
        warnings=warnings,
    )
    report = YiuSolveReport(
        status=status,  # type: ignore[arg-type]
        solve_id=run_id,
        spec_path=str(resolved_solve_spec_path.resolve()),
        base_spec_path=str(base_spec_path.resolve()),
        run_dir=str(run_dir.resolve()),
        satisfying_solution_count=len(satisfying_candidates),
        comparison_solution_count=comparison_solution_count,
        selected_solution_path=selected_solution_path,
        selected_source_sequence=selected_source_sequence,
        metadata=metadata,
        issues=[],
    )
    _write_solve_bundle(
        run_dir,
        solve_spec=solve_spec,
        resolved_solve_spec_path=resolved_solve_spec_path,
        base_spec_path=base_spec_path,
        report=report,
        comparison_rows=comparison_rows,
        inventory=inventory,
    )
    return run_dir, report
