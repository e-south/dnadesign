"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/portfolio/load.py

Load and validate Portfolio spec files.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from dnadesign.cruncher.portfolio.schema import parse_portfolio_root
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource, PortfolioSpec


def _format_spec_path_error(path: Path) -> str:
    if path.is_dir():
        lines = [
            f"Portfolio spec path must be a file, not a directory: {path}",
            "Pass --spec <workspace>/configs/<name>.portfolio.yaml.",
        ]
        candidates = sorted(path.glob("*.portfolio.yaml"))
        if len(candidates) == 1:
            lines.append(f"Nudge: did you mean --spec {candidates[0]} ?")
        elif candidates:
            lines.append("Available portfolio specs:")
            lines.extend(f"- {candidate}" for candidate in candidates)
        return "\n".join(lines)
    return f"Portfolio spec path must be a file: {path}"


def _ensure_within_workspace(*, run_dir: Path, workspace: Path) -> None:
    try:
        run_dir.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(
            f"portfolio source run_dir must be inside workspace: run_dir={run_dir} workspace={workspace}"
        ) from exc


def _portfolio_workspace_root(spec_path: Path) -> Path:
    if spec_path.parent.name == "configs":
        return spec_path.parent.parent.resolve()
    return spec_path.parent.resolve()


def _format_schema_validation_error(exc: ValidationError) -> str:
    lines = ["Portfolio schema validation failed:"]
    for error in exc.errors():
        loc = error.get("loc", ())
        loc_path = ".".join(str(item) for item in loc) if isinstance(loc, tuple | list) else str(loc)
        if not loc_path:
            loc_path = "<root>"
        message = str(error.get("msg", "invalid value")).strip()
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
        lines.append(f"- {loc_path}: {message}")
    return "\n".join(lines)


def _resolve_source_workspace(*, workspace: Path, workspace_root: Path, source_id: str) -> Path:
    workspace_path = workspace.resolve() if workspace.is_absolute() else (workspace_root / workspace).resolve()
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Portfolio source workspace does not exist: id={source_id!r} workspace={workspace_path}"
        )
    if not workspace_path.is_dir():
        raise ValueError(f"Portfolio source workspace must be a directory: {workspace_path}")
    return workspace_path


def _resolve_source_run_dir(
    *,
    run_dir: Path,
    workspace_path: Path,
    source_id: str,
    require_exists: bool,
) -> Path:
    run_dir_path = run_dir.resolve() if run_dir.is_absolute() else (workspace_path / run_dir).resolve()
    _ensure_within_workspace(run_dir=run_dir_path, workspace=workspace_path)
    if require_exists:
        if not run_dir_path.exists():
            raise FileNotFoundError(f"Portfolio source run_dir does not exist: id={source_id!r} run_dir={run_dir_path}")
        if not run_dir_path.is_dir():
            raise ValueError(f"Portfolio source run_dir must be a directory: {run_dir_path}")
    return run_dir_path


def _resolve_source_prepare(source: PortfolioSource, *, workspace_path: Path) -> dict[str, object] | None:
    prepare = source.prepare
    if prepare is None:
        return None
    runbook = prepare.runbook
    runbook_path = runbook.resolve() if runbook.is_absolute() else (workspace_path / runbook).resolve()
    try:
        runbook_path.relative_to(workspace_path)
    except ValueError as exc:
        raise ValueError(
            "Portfolio source prepare.runbook must be inside source workspace: "
            f"id={source.id!r} runbook={runbook_path} workspace={workspace_path}"
        ) from exc
    if not runbook_path.exists():
        raise FileNotFoundError(f"Portfolio source prepare.runbook not found: id={source.id!r} runbook={runbook_path}")
    if not runbook_path.is_file():
        raise ValueError(f"Portfolio source prepare.runbook must be a file: id={source.id!r} runbook={runbook_path}")
    return {
        "runbook": runbook_path,
        "step_ids": list(prepare.step_ids),
    }


def _resolve_source_study_spec(source: PortfolioSource, *, workspace_path: Path) -> Path | None:
    study_spec = source.study_spec
    if study_spec is None:
        return None
    study_spec_path = study_spec.resolve() if study_spec.is_absolute() else (workspace_path / study_spec).resolve()
    try:
        study_spec_path.relative_to(workspace_path)
    except ValueError as exc:
        raise ValueError(
            "Portfolio source study_spec must be inside source workspace: "
            f"id={source.id!r} study_spec={study_spec_path} workspace={workspace_path}"
        ) from exc
    if not study_spec_path.exists():
        raise FileNotFoundError(f"Portfolio source study_spec not found: id={source.id!r} study_spec={study_spec_path}")
    if not study_spec_path.is_file():
        raise ValueError(f"Portfolio source study_spec must be a file: id={source.id!r} study_spec={study_spec_path}")
    return study_spec_path


def _resolve_source_workspace_file(
    *,
    workspace_path: Path,
    source_id: str,
    relative_path: Path,
    label: str,
) -> Path:
    candidate = relative_path.resolve() if relative_path.is_absolute() else (workspace_path / relative_path).resolve()
    try:
        candidate.relative_to(workspace_path)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside source workspace: id={source_id!r} path={candidate} workspace={workspace_path}"
        ) from exc
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: id={source_id!r} path={candidate}")
    if not candidate.is_file():
        raise ValueError(f"{label} must be a file: id={source_id!r} path={candidate}")
    return candidate


def _validate_global_study_specs(spec: PortfolioSpec, *, source_payloads: list[dict[str, object]]) -> None:
    required_specs = list(spec.studies.ensure_specs)
    table_cfg = spec.studies.sequence_length_table
    if table_cfg.enabled:
        required_specs.append(table_cfg.study_spec)

    seen_required: set[str] = set()
    deduped_required: list[Path] = []
    for path in required_specs:
        token = str(path)
        if token in seen_required:
            continue
        seen_required.add(token)
        deduped_required.append(path)

    if not deduped_required:
        return

    for source in source_payloads:
        source_id = str(source["id"])
        workspace_path = Path(source["workspace"])
        for spec_path in deduped_required:
            _resolve_source_workspace_file(
                workspace_path=workspace_path,
                source_id=source_id,
                relative_path=spec_path,
                label="portfolio.studies.ensure_specs entry",
            )


def _normalize_source(
    source: PortfolioSource,
    *,
    workspace_root: Path,
    require_run_dir_exists: bool,
) -> dict[str, object]:
    workspace_path = _resolve_source_workspace(
        workspace=source.workspace,
        workspace_root=workspace_root,
        source_id=str(source.id),
    )
    run_dir_path = _resolve_source_run_dir(
        run_dir=source.run_dir,
        workspace_path=workspace_path,
        source_id=str(source.id),
        require_exists=require_run_dir_exists,
    )

    source_payload = source.model_dump(mode="python")
    source_payload["workspace"] = workspace_path
    source_payload["run_dir"] = run_dir_path
    source_payload["prepare"] = _resolve_source_prepare(source, workspace_path=workspace_path)
    source_payload["study_spec"] = _resolve_source_study_spec(source, workspace_path=workspace_path)
    return source_payload


def load_portfolio_spec(path: Path) -> PortfolioSpec:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio spec not found: {path}")
    if not path.is_file():
        raise ValueError(_format_spec_path_error(path))

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Portfolio schema required (missing root key: portfolio)")
    if "portfolio" not in raw:
        if "runbook" in raw:
            raise ValueError(
                "Portfolio schema required (missing root key: portfolio). "
                "This file looks like a workspace runbook; use "
                "`cruncher workspaces run --runbook <path>`."
            )
        raise ValueError("Portfolio schema required (missing root key: portfolio)")
    payload = raw.get("portfolio")
    if not isinstance(payload, dict):
        raise ValueError("Portfolio schema required (portfolio must be a mapping)")

    try:
        spec = parse_portfolio_root(raw).portfolio
    except ValidationError as exc:
        raise ValueError(_format_schema_validation_error(exc)) from exc
    normalized = spec.model_dump(mode="python")
    workspace_root = _portfolio_workspace_root(path.resolve())
    require_run_dir_exists = spec.execution.mode != "prepare_then_aggregate"

    source_payloads: list[dict[str, object]] = []
    source_errors: list[tuple[str, Exception]] = []
    for source in spec.sources:
        try:
            source_payloads.append(
                _normalize_source(
                    source,
                    workspace_root=workspace_root,
                    require_run_dir_exists=require_run_dir_exists,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            source_errors.append((f"- source={source.id}: {exc}", exc))

    if source_errors:
        if len(source_errors) == 1:
            raise source_errors[0][1]
        lines = [
            "Portfolio spec has invalid source paths or artifacts:",
            *(item[0] for item in source_errors),
            "Fix the listed source workspace/run_dir/prepare paths and rerun.",
        ]
        raise ValueError("\n".join(lines))

    normalized["sources"] = source_payloads

    try:
        validated = PortfolioSpec.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(_format_schema_validation_error(exc)) from exc
    _validate_global_study_specs(validated, source_payloads=source_payloads)
    return validated
