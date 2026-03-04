"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli.py

CLI for rendering deterministic batch orchestration plans from machine runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from pydantic import ValidationError

from .orchestrator.execute import execute_batch_plan
from .orchestrator.plan import build_batch_plan
from .orchestrator.state import discover_active_job_ids_for_runbook
from .runbooks.path_policy import (
    REPO_TRANSIENT_OPERATIONAL_DIR_NAMES,
    WORKSPACE_AUDIT_RELATIVE_DIR,
    WORKSPACE_RUNBOOKS_RELATIVE_DIR,
    WORKSPACE_SGE_STDOUT_RELATIVE_DIR,
)
from .runbooks.schema import load_orchestration_runbook

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Cross-tool orchestration commands for deterministic HPC batch plans.",
)

runbook_app = typer.Typer(help="Runbook contract commands.")
app.add_typer(runbook_app, name="runbook")


def _load_runbook_or_exit(runbook_path: Path):
    try:
        return load_orchestration_runbook(runbook_path.expanduser())
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _workspace_runbook_path_hint() -> str:
    return f"<workspace-root>/{WORKSPACE_RUNBOOKS_RELATIVE_DIR.as_posix()}/<runbook-id>.yaml"


def _contract_path(path: Path, *, runbook_parent: Path) -> str:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        return str(expanded)
    resolved = expanded.resolve()
    try:
        return str(resolved.relative_to(runbook_parent.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_workspace_root_for_init(workspace_root: Path, *, repo_base: Path) -> Path:
    expanded = workspace_root.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (repo_base / expanded).resolve()


def _resolve_repo_base(repo_root: Path | None) -> Path:
    if repo_root is None:
        return Path.cwd().resolve()
    return repo_root.expanduser().resolve()


def _validate_runbook_output_path_for_init(*, runbook_path: Path, repo_base: Path) -> None:
    resolved_repo_base = repo_base.resolve()
    resolved_runbook = runbook_path.resolve()
    try:
        relative_to_repo = resolved_runbook.relative_to(resolved_repo_base)
    except ValueError:
        return
    if relative_to_repo.parent == Path("."):
        raise ValueError(f"runbook path must not be at repository root; use {_workspace_runbook_path_hint()}")
    for segment in REPO_TRANSIENT_OPERATIONAL_DIR_NAMES:
        if segment in relative_to_repo.parts:
            raise ValueError(f"runbook path must not use '{segment}'; use {_workspace_runbook_path_hint()}")


def _discover_repo_base_for_path(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    anchor = resolved if resolved.is_dir() else resolved.parent
    for parent in (anchor, *anchor.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "dnadesign").exists():
            return parent.resolve()
    return None


def _validate_runbook_input_path_for_runtime(*, runbook_path: Path, repo_base: Path) -> None:
    resolved_runbook = runbook_path.expanduser().resolve()
    discovered_repo_base = _discover_repo_base_for_path(resolved_runbook)
    resolved_repo_base = discovered_repo_base if discovered_repo_base is not None else repo_base
    _validate_runbook_output_path_for_init(runbook_path=resolved_runbook, repo_base=resolved_repo_base)


def _validate_audit_json_path_for_execute(*, audit_json_path: Path, workspace_root: Path) -> Path:
    resolved_audit_json = audit_json_path.expanduser().resolve()
    expected_audit_dir = (workspace_root / WORKSPACE_AUDIT_RELATIVE_DIR).resolve()
    if resolved_audit_json.parent != expected_audit_dir:
        raise ValueError(
            f"audit-json path must be exactly <workspace-root>/{WORKSPACE_AUDIT_RELATIVE_DIR.as_posix()}/<file>.json"
        )
    if resolved_audit_json.suffix.lower() != ".json":
        raise ValueError("audit-json file extension must be .json")
    return resolved_audit_json


def _build_init_payload(
    *,
    workflow: Literal["densegen", "infer"],
    with_notify: bool,
    runbook_id: str,
    project: str,
    workspace_root: Path,
    runbook_parent: Path,
    cuda_module: str,
    gcc_module: str,
    pe_omp: int | None,
    h_rt: str | None,
    mem_per_core: str | None,
    notify_qsub_template: str,
    densegen_qsub_template: str,
    densegen_post_run_qsub_template: str,
    infer_qsub_template: str,
) -> dict[str, object]:
    workspace_contract = Path(_contract_path(workspace_root, runbook_parent=runbook_parent))
    if workflow == "densegen":
        workflow_id = "densegen_batch_with_notify_slack" if with_notify else "densegen_batch_submit"
    else:
        workflow_id = "infer_batch_with_notify_slack" if with_notify else "infer_batch_submit"
    payload: dict[str, object] = {
        "runbook": {
            "schema_version": 1,
            "id": runbook_id,
            "workflow_id": workflow_id,
            "project": project,
            "workspace_root": str(workspace_contract),
            "logging": {
                "stdout_dir": str(workspace_contract / WORKSPACE_SGE_STDOUT_RELATIVE_DIR / runbook_id),
                "retention": {
                    "keep_last": 20,
                    "max_age_days": 14,
                },
            },
            "mode_policy": {
                "default": "auto",
                "on_active_job": "hold_jid",
            },
        }
    }
    if with_notify:
        notify_tool = "densegen" if workflow == "densegen" else "infer"
        notify_policy = "densegen" if workflow == "densegen" else "infer_evo2"
        payload["runbook"]["notify"] = {
            "tool": notify_tool,
            "policy": notify_policy,
            "profile": str(workspace_contract / "outputs" / "notify" / notify_tool / "profile.json"),
            "cursor": str(workspace_contract / "outputs" / "notify" / notify_tool / "cursor"),
            "spool_dir": str(workspace_contract / "outputs" / "notify" / notify_tool / "spool"),
            "webhook_env": "NOTIFY_WEBHOOK",
            "orchestration_events": True,
            "qsub_template": notify_qsub_template,
            "smoke": "dry",
        }
    if workflow == "densegen":
        payload["runbook"]["densegen"] = {
            "config": str(workspace_contract / "config.yaml"),
            "qsub_template": densegen_qsub_template,
            "run_args": {
                "fresh": "--fresh --no-plot",
                "resume": "--resume --no-plot",
            },
            "post_run": {
                "qsub_template": densegen_post_run_qsub_template,
            },
            "overlay_guard": {
                "max_projected_overlay_parts": 10000,
                "max_existing_overlay_parts": 1000,
                "auto_compact_existing_overlay_parts": True,
                "overlay_namespace": "densegen",
            },
            "records_part_guard": {
                "max_projected_records_parts": 10000,
                "max_existing_records_parts": 1000,
                "max_existing_records_part_age_days": 14,
                "auto_compact_existing_records_parts": True,
            },
            "archived_overlay_guard": {
                "max_archived_entries": 1000,
                "max_archived_bytes": 2147483648,
            },
        }
        payload["runbook"]["resources"] = {
            "pe_omp": pe_omp if pe_omp is not None else 16,
            "h_rt": h_rt or "08:00:00",
            "mem_per_core": mem_per_core or "8G",
        }
    else:
        payload["runbook"]["infer"] = {
            "config": str(workspace_contract / "config.yaml"),
            "qsub_template": infer_qsub_template,
            "cuda_module": cuda_module,
            "gcc_module": gcc_module,
        }
        payload["runbook"]["resources"] = {
            "pe_omp": pe_omp if pe_omp is not None else 4,
            "h_rt": h_rt or "04:00:00",
            "mem_per_core": mem_per_core or "8G",
            "gpus": 1,
            "gpu_capability": "8.9",
        }
    return payload


def _resolve_active_job_ids(
    *,
    runbook,
    active_job_ids: list[str],
    discover_active_jobs: bool,
    max_discovery_jobs: int,
) -> tuple[str, ...]:
    resolved_job_ids = [str(job_id).strip() for job_id in active_job_ids if str(job_id).strip()]
    if not discover_active_jobs:
        return tuple(dict.fromkeys(resolved_job_ids))

    try:
        discovered_job_ids = discover_active_job_ids_for_runbook(runbook, max_jobs=max_discovery_jobs)
    except RuntimeError as exc:
        typer.echo(f"Active-job discovery warning: {exc}", err=True)
        discovered_job_ids = ()

    for discovered in discovered_job_ids:
        if discovered not in resolved_job_ids:
            resolved_job_ids.append(discovered)
    return tuple(resolved_job_ids)


def _packaged_precedent_paths() -> list[Path]:
    preset_dir = Path(__file__).resolve().parent / "runbooks" / "presets"
    if not preset_dir.exists():
        return []
    return sorted(path.resolve() for path in preset_dir.glob("*.yaml"))


@runbook_app.command("init")
def runbook_init(
    runbook: Annotated[Path, typer.Option("--runbook", help="Output path for orchestration runbook yaml.")],
    workflow: Annotated[
        Literal["densegen", "infer"],
        typer.Option("--workflow", help="Workflow family for scaffolded runbook."),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Workspace root path used to derive config and notify paths."),
    ],
    project: Annotated[str, typer.Option("--project", help="Scheduler project/account id.")] = "dunlop",
    runbook_id: Annotated[str, typer.Option("--id", help="Runbook id slug.")] = "batch_demo",
    cuda_module: Annotated[
        str,
        typer.Option("--cuda-module", help="Infer workflow CUDA module name."),
    ] = "cuda/12.4",
    gcc_module: Annotated[
        str,
        typer.Option("--gcc-module", help="Infer workflow GCC module name."),
    ] = "gcc/13.2.0",
    pe_omp: Annotated[
        int | None,
        typer.Option("--pe-omp", help="Override resources.pe_omp in the scaffolded runbook."),
    ] = None,
    h_rt: Annotated[
        str | None,
        typer.Option("--h-rt", help="Override resources.h_rt in HH:MM:SS format."),
    ] = None,
    mem_per_core: Annotated[
        str | None,
        typer.Option("--mem-per-core", help="Override resources.mem_per_core."),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root used to resolve default qsub template paths."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite runbook path when it already exists."),
    ] = False,
    with_notify: Annotated[
        bool,
        typer.Option(
            "--with-notify/--no-notify",
            help="Include notify smoke and watcher submit contracts in the scaffold (default: on).",
        ),
    ] = True,
) -> None:
    runbook_path = runbook.expanduser()
    repo_base = _resolve_repo_base(repo_root)
    if pe_omp is not None and pe_omp <= 0:
        typer.echo("Runbook contract error: --pe-omp must be > 0", err=True)
        raise typer.Exit(code=2)
    try:
        _validate_runbook_output_path_for_init(runbook_path=runbook_path, repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if runbook_path.exists() and not force:
        typer.echo(f"Runbook contract error: file exists: {runbook_path}", err=True)
        raise typer.Exit(code=2)

    def _template_or_default(relative_path: str) -> Path:
        candidate = repo_base / relative_path
        if candidate.exists():
            return candidate
        return Path(relative_path)

    notify_template = _template_or_default("docs/bu-scc/jobs/notify-watch.qsub")
    densegen_template = _template_or_default("docs/bu-scc/jobs/densegen-cpu.qsub")
    densegen_post_run_template = _template_or_default("docs/bu-scc/jobs/densegen-analysis.qsub")
    infer_template = _template_or_default("docs/bu-scc/jobs/evo2-gpu-infer.qsub")
    resolved_workspace_root = _resolve_workspace_root_for_init(workspace_root, repo_base=repo_base)
    payload = _build_init_payload(
        workflow=workflow,
        with_notify=with_notify,
        runbook_id=runbook_id,
        project=project,
        workspace_root=resolved_workspace_root,
        runbook_parent=runbook_path.parent,
        cuda_module=cuda_module,
        gcc_module=gcc_module,
        pe_omp=pe_omp,
        h_rt=h_rt,
        mem_per_core=mem_per_core,
        notify_qsub_template=_contract_path(notify_template, runbook_parent=runbook_path.parent),
        densegen_qsub_template=_contract_path(densegen_template, runbook_parent=runbook_path.parent),
        densegen_post_run_qsub_template=_contract_path(
            densegen_post_run_template,
            runbook_parent=runbook_path.parent,
        ),
        infer_qsub_template=_contract_path(infer_template, runbook_parent=runbook_path.parent),
    )
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    typer.echo(str(runbook_path.resolve()))


@runbook_app.command("precedents")
def runbook_precedents() -> None:
    precedents = [{"name": path.stem, "path": str(path)} for path in _packaged_precedent_paths()]
    typer.echo(json.dumps({"precedents": precedents}, indent=2, sort_keys=True))


@runbook_app.command("plan")
def runbook_plan(
    runbook: Annotated[Path, typer.Option("--runbook", help="Path to orchestration runbook yaml.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for runtime path contract checks when invoking outside the repository.",
        ),
    ] = None,
    mode: Annotated[
        Literal["auto", "fresh", "resume"] | None,
        typer.Option("--mode", help="Run mode policy override."),
    ] = None,
    smoke: Annotated[
        Literal["dry", "live"] | None,
        typer.Option("--notify-smoke", help="Notify smoke override."),
    ] = None,
    active_job_id: Annotated[
        list[str],
        typer.Option("--active-job-id", help="Existing active job id for hold_jid policy decisions."),
    ] = [],
    discover_active_jobs: Annotated[
        bool,
        typer.Option(
            "--discover-active-jobs/--no-discover-active-jobs",
            help="Auto-discover active matching jobs from qstat/qstat -j and merge into hold_jid decisions.",
        ),
    ] = True,
    max_discovery_jobs: Annotated[
        int,
        typer.Option("--max-discovery-jobs", help="Maximum qstat jobs inspected during active-job discovery."),
    ] = 24,
    allow_fresh_reset: Annotated[
        bool,
        typer.Option(
            "--allow-fresh-reset/--no-allow-fresh-reset",
            help="Allow --mode fresh when resume artifacts already exist in the workspace.",
        ),
    ] = False,
) -> None:
    if max_discovery_jobs <= 0:
        typer.echo("Runbook contract error: --max-discovery-jobs must be > 0", err=True)
        raise typer.Exit(code=2)
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    loaded = _load_runbook_or_exit(runbook)
    resolved_active_job_ids = _resolve_active_job_ids(
        runbook=loaded,
        active_job_ids=active_job_id,
        discover_active_jobs=discover_active_jobs,
        max_discovery_jobs=max_discovery_jobs,
    )
    try:
        plan = build_batch_plan(
            runbook=loaded,
            requested_mode=mode,
            requested_smoke=smoke,
            active_job_ids=resolved_active_job_ids,
            allow_fresh_reset=allow_fresh_reset,
        )
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(plan.as_dict(), indent=2, sort_keys=True))


@runbook_app.command("active-jobs")
def runbook_active_jobs(
    runbook: Annotated[Path, typer.Option("--runbook", help="Path to orchestration runbook yaml.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for runtime path contract checks when invoking outside the repository.",
        ),
    ] = None,
    max_discovery_jobs: Annotated[
        int,
        typer.Option("--max-discovery-jobs", help="Maximum qstat jobs inspected during active-job discovery."),
    ] = 24,
) -> None:
    if max_discovery_jobs <= 0:
        typer.echo("Runbook contract error: --max-discovery-jobs must be > 0", err=True)
        raise typer.Exit(code=2)
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    loaded = _load_runbook_or_exit(runbook)
    try:
        active_job_ids = discover_active_job_ids_for_runbook(loaded, max_jobs=max_discovery_jobs)
    except RuntimeError as exc:
        typer.echo(f"Runbook contract error: active-job discovery failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    payload = {
        "runbook_id": loaded.id,
        "workflow_id": loaded.workflow_id,
        "active_job_ids": list(active_job_ids),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@runbook_app.command("execute")
def runbook_execute(
    runbook: Annotated[Path, typer.Option("--runbook", help="Path to orchestration runbook yaml.")],
    audit_json: Annotated[Path, typer.Option("--audit-json", help="Output path for audit artifact json.")],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for runtime path contract checks when invoking outside the repository.",
        ),
    ] = None,
    mode: Annotated[
        Literal["auto", "fresh", "resume"] | None,
        typer.Option("--mode", help="Run mode policy override."),
    ] = None,
    smoke: Annotated[
        Literal["dry", "live"] | None,
        typer.Option("--notify-smoke", help="Notify smoke override."),
    ] = None,
    active_job_id: Annotated[
        list[str],
        typer.Option("--active-job-id", help="Existing active job id for hold_jid policy decisions."),
    ] = [],
    discover_active_jobs: Annotated[
        bool,
        typer.Option(
            "--discover-active-jobs/--no-discover-active-jobs",
            help="Auto-discover active matching jobs from qstat/qstat -j and merge into hold_jid decisions.",
        ),
    ] = True,
    max_discovery_jobs: Annotated[
        int,
        typer.Option("--max-discovery-jobs", help="Maximum qstat jobs inspected during active-job discovery."),
    ] = 24,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit/--no-submit",
            help="Run submit-phase qsub commands after preflight/smoke pass. Default is no-submit.",
        ),
    ] = False,
    command_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--command-timeout-seconds",
            help="Per-command timeout in seconds for execute phases.",
        ),
    ] = 300.0,
    allow_fresh_reset: Annotated[
        bool,
        typer.Option(
            "--allow-fresh-reset/--no-allow-fresh-reset",
            help="Allow --mode fresh when resume artifacts already exist in the workspace.",
        ),
    ] = False,
) -> None:
    if command_timeout_seconds is not None and command_timeout_seconds <= 0:
        typer.echo("Runbook contract error: --command-timeout-seconds must be > 0", err=True)
        raise typer.Exit(code=2)
    if max_discovery_jobs <= 0:
        typer.echo("Runbook contract error: --max-discovery-jobs must be > 0", err=True)
        raise typer.Exit(code=2)
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    loaded = _load_runbook_or_exit(runbook)
    try:
        resolved_audit_json = _validate_audit_json_path_for_execute(
            audit_json_path=audit_json,
            workspace_root=loaded.workspace_root,
        )
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    resolved_active_job_ids = _resolve_active_job_ids(
        runbook=loaded,
        active_job_ids=active_job_id,
        discover_active_jobs=discover_active_jobs,
        max_discovery_jobs=max_discovery_jobs,
    )
    try:
        plan = build_batch_plan(
            runbook=loaded,
            requested_mode=mode,
            requested_smoke=smoke,
            active_job_ids=resolved_active_job_ids,
            allow_fresh_reset=allow_fresh_reset,
        )
    except ValueError as exc:
        typer.echo(f"Runbook contract error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    result = execute_batch_plan(
        plan=plan,
        audit_json_path=resolved_audit_json,
        submit=submit,
        command_timeout_seconds=command_timeout_seconds,
    )
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    if not result.ok:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
