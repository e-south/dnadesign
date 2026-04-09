"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/commands/runbook.py

Direct runbook command implementation for the OPS control plane.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Sequence

import typer
import yaml
from typer.main import get_command

from dnadesign.ops.cli.common import emit_stderr, raise_contract_error

app = typer.Typer(help="Control-plane runbook contract commands.")
diagnostics_app = typer.Typer(help="Supported scheduler diagnostics under the main ops CLI surface.")
app.add_typer(diagnostics_app, name="diagnostics")


def get_click_command():
    return get_command(app)


def _load_runbook_or_exit(runbook_path: Path):
    from pydantic import ValidationError

    from dnadesign.ops.runbooks.schema import load_orchestration_runbook

    try:
        return load_orchestration_runbook(runbook_path.expanduser())
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        raise_contract_error(f"Runbook contract error: {exc}")


def _workspace_runbook_path_hint() -> str:
    from dnadesign.ops.runbooks.path_policy import WORKSPACE_RUNBOOKS_RELATIVE_DIR

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


def _render_notify_contract_warning(*, workspace_root: Path, notify_tool: str) -> str:
    profile_path = (workspace_root / "outputs" / "notify" / notify_tool / "profile.json").resolve()
    return (
        "Notify contract required before planning.\n"
        "Set NOTIFY_WEBHOOK_FILE to a readable file path, or configure "
        f"{profile_path} with webhook.source=secret_ref and a file:// secret reference."
    )


def _validate_runbook_output_path_for_init(*, runbook_path: Path, repo_base: Path) -> None:
    from dnadesign.ops.runbooks.path_policy import REPO_TRANSIENT_OPERATIONAL_DIR_NAMES

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
    from dnadesign.ops.runbooks.path_policy import WORKSPACE_AUDIT_RELATIVE_DIR

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
    from dnadesign.ops.runbooks.path_policy import WORKSPACE_SGE_STDOUT_RELATIVE_DIR
    from dnadesign.ops.runbooks.workflow_metadata import resolve_workflow_id, resolve_workflow_tool

    workspace_contract = Path(_contract_path(workspace_root, runbook_parent=runbook_parent))
    workflow_id = resolve_workflow_id(tool=workflow, with_notify=with_notify)
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
        notify_tool = resolve_workflow_tool(workflow_id)
        notify_policy = "infer" if notify_tool == "infer" else "generic"
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
            "pe_omp": pe_omp if pe_omp is not None else 12,
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
            "gpu_memory_gib": 45.0,
        }
    return payload


def _split_active_job_id_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for item in str(value).split(","):
            token = item.strip()
            if token:
                tokens.append(token)
    return tokens


def _resolve_active_job_resolution(
    *,
    runbook,
    active_job_ids: list[str],
    discover_active_jobs: bool,
    max_discovery_jobs: int,
) -> object:
    from dnadesign.ops import api as ops_api

    return ops_api.resolve_active_job_resolution(
        runbook=runbook,
        explicit_job_ids=_split_active_job_id_tokens(active_job_ids),
        discover_active_jobs=discover_active_jobs,
        max_jobs=max_discovery_jobs,
    )


def _render_active_job_hints(*, runbook_path: Path, active_job_ids: Sequence[str]) -> dict[str, object]:
    deduped_job_ids = tuple(dict.fromkeys(_split_active_job_id_tokens(active_job_ids)))
    csv_value = ",".join(deduped_job_ids)
    repeat_args = " ".join(f"--active-job-id {shlex.quote(job_id)}" for job_id in deduped_job_ids)
    runbook_arg = shlex.quote(str(runbook_path.expanduser()))
    if repeat_args:
        plan_hint = f"uv run ops runbook plan --runbook {runbook_arg} --no-discover-active-jobs {repeat_args}"
    else:
        plan_hint = f"uv run ops runbook plan --runbook {runbook_arg}"
    return {
        "active_job_count": len(deduped_job_ids),
        "active_job_ids_csv": csv_value,
        "active_job_id_args": repeat_args,
        "plan_command_hint": plan_hint,
    }


@dataclass(frozen=True)
class RunbookInitPreset:
    name: str
    description: str
    project: str
    templates: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "project": self.project,
            "templates": dict(sorted(self.templates.items())),
        }


def _runbook_init_preset_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runbooks" / "init_presets.yaml"


def _load_runbook_init_presets() -> list[RunbookInitPreset]:
    manifest_path = _runbook_init_preset_manifest_path()
    if not manifest_path.exists():
        return []
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    presets: list[RunbookInitPreset] = []
    for entry in payload.get("presets") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        project = str(entry.get("project") or "").strip()
        if not name or not project:
            continue
        templates_payload = entry.get("templates") or {}
        presets.append(
            RunbookInitPreset(
                name=name,
                description=str(entry.get("description") or "").strip(),
                project=project,
                templates={
                    str(key).strip(): str(value).strip()
                    for key, value in dict(templates_payload).items()
                    if str(key).strip() and str(value).strip()
                },
            )
        )
    return presets


def _resolve_runbook_init_preset(name: str) -> RunbookInitPreset:
    normalized_name = str(name or "").strip()
    for preset in _load_runbook_init_presets():
        if preset.name == normalized_name:
            return preset
    raise ValueError(f"unknown init preset: {normalized_name}")


def _packaged_preset_paths() -> list[Path]:
    preset_dir = Path(__file__).resolve().parents[2] / "runbooks" / "presets"
    if not preset_dir.exists():
        return []
    return sorted(path.resolve() for path in preset_dir.glob("*.yaml"))


def _emit_packaged_runbook_presets() -> None:
    presets = [{"name": path.stem, "path": str(path)} for path in _packaged_preset_paths()]
    typer.echo(
        json.dumps(
            {
                "init_presets": [preset.as_dict() for preset in _load_runbook_init_presets()],
                "presets": presets,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_supported_gate_command(args: list[str]) -> None:
    from dnadesign.ops.orchestrator import gates as gates_module

    exit_code = gates_module.main(args)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@diagnostics_app.command("session-counts")
def runbook_diagnostics_session_counts(
    qstat_file: Annotated[
        Path | None,
        typer.Option("--qstat-file", help="Read qstat-like output from file for deterministic fixture mode."),
    ] = None,
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help="Emit an explicit degraded queue-probe record instead of failing when qstat is unavailable.",
        ),
    ] = False,
) -> None:
    args = ["session-counts"]
    if qstat_file is not None:
        args.extend(["--qstat-file", str(qstat_file)])
    if allow_missing_qstat:
        args.append("--allow-missing-qstat")
    _run_supported_gate_command(args)


@diagnostics_app.command("submit-shape-advisor")
def runbook_diagnostics_submit_shape_advisor(
    planned_submits: Annotated[int, typer.Option("--planned-submits", help="Number of planned submit commands.")],
    warn_over_running: Annotated[
        int,
        typer.Option("--warn-over-running", help="Warn when running job count exceeds this threshold."),
    ] = 3,
    requires_order: Annotated[
        bool,
        typer.Option("--requires-order/--no-requires-order", help="Mark the plan as an ordered pipeline."),
    ] = False,
    qstat_file: Annotated[
        Path | None,
        typer.Option("--qstat-file", help="Read qstat-like output from file for deterministic fixture mode."),
    ] = None,
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help="Emit an explicit degraded advisory record instead of failing when qstat is unavailable.",
        ),
    ] = False,
) -> None:
    args = [
        "submit-shape-advisor",
        "--planned-submits",
        str(planned_submits),
        "--warn-over-running",
        str(warn_over_running),
    ]
    if requires_order:
        args.append("--requires-order")
    if qstat_file is not None:
        args.extend(["--qstat-file", str(qstat_file)])
    if allow_missing_qstat:
        args.append("--allow-missing-qstat")
    _run_supported_gate_command(args)


@diagnostics_app.command("operator-brief")
def runbook_diagnostics_operator_brief(
    planned_submits: Annotated[int, typer.Option("--planned-submits", help="Number of planned submit commands.")],
    warn_over_running: Annotated[
        int,
        typer.Option("--warn-over-running", help="Warn when running job count exceeds this threshold."),
    ] = 3,
    requires_order: Annotated[
        bool,
        typer.Option("--requires-order/--no-requires-order", help="Mark the plan as an ordered pipeline."),
    ] = False,
    qstat_file: Annotated[
        Path | None,
        typer.Option("--qstat-file", help="Read qstat-like output from file for deterministic fixture mode."),
    ] = None,
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help="Emit an explicit degraded readiness record instead of failing when qstat is unavailable.",
        ),
    ] = False,
) -> None:
    args = [
        "operator-brief",
        "--planned-submits",
        str(planned_submits),
        "--warn-over-running",
        str(warn_over_running),
    ]
    if requires_order:
        args.append("--requires-order")
    if qstat_file is not None:
        args.extend(["--qstat-file", str(qstat_file)])
    if allow_missing_qstat:
        args.append("--allow-missing-qstat")
    _run_supported_gate_command(args)


@app.command("init")
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
    project: Annotated[
        str | None,
        typer.Option("--project", help="Explicit scheduler project/account id."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help=(
                "Explicit init preset that supplies site-local project/template defaults; "
                "use `ops runbook presets` to list presets."
            ),
        ),
    ] = None,
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
    try:
        resolved_preset = _resolve_runbook_init_preset(preset) if preset is not None else None
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    if (project is None) == (resolved_preset is None):
        raise_contract_error("Runbook contract error: provide exactly one of --project or --preset")
    selected_project = resolved_preset.project if resolved_preset is not None else str(project or "").strip()
    if not selected_project:
        raise_contract_error("Runbook contract error: project must be non-empty")
    preset_templates = resolved_preset.templates if resolved_preset is not None else {}
    if pe_omp is not None and pe_omp <= 0:
        raise_contract_error("Runbook contract error: --pe-omp must be > 0")
    try:
        _validate_runbook_output_path_for_init(runbook_path=runbook_path, repo_base=repo_base)
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    if runbook_path.exists() and not force:
        raise_contract_error(f"Runbook contract error: file exists: {runbook_path}")

    def _template_or_default(relative_path: str) -> Path:
        candidate = repo_base / relative_path
        if candidate.exists():
            return candidate
        return Path(relative_path)

    notify_template = _template_or_default(preset_templates.get("notify", "docs/bu-scc/jobs/notify-watch.qsub"))
    densegen_template = _template_or_default(preset_templates.get("densegen", "docs/bu-scc/jobs/densegen-cpu.qsub"))
    densegen_post_run_template = _template_or_default(
        preset_templates.get("densegen_post_run", "docs/bu-scc/jobs/densegen-analysis.qsub")
    )
    infer_template = _template_or_default(preset_templates.get("infer", "docs/bu-scc/jobs/evo2-gpu-infer.qsub"))
    resolved_workspace_root = _resolve_workspace_root_for_init(workspace_root, repo_base=repo_base)
    payload = _build_init_payload(
        workflow=workflow,
        with_notify=with_notify,
        runbook_id=runbook_id,
        project=selected_project,
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
    if with_notify:
        from dnadesign.ops.runbooks.workflow_metadata import resolve_workflow_tool

        emit_stderr(
            _render_notify_contract_warning(
                workspace_root=resolved_workspace_root,
                notify_tool=resolve_workflow_tool(workflow_id=payload["runbook"]["workflow_id"]),
            )
        )


@app.command("presets")
def runbook_presets() -> None:
    _emit_packaged_runbook_presets()


@app.command("plan")
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
        list[str] | None,
        typer.Option(
            "--active-job-id",
            help=(
                "Existing active job id(s) for hold_jid policy decisions; repeat option or pass a comma-delimited list."
            ),
        ),
    ] = None,
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
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help=(
                "Render preflight gate commands with explicit degraded queue-probe mode when `qstat` is unavailable. "
                "Useful for workstation dry-run demos."
            ),
        ),
    ] = False,
) -> None:
    from dnadesign.ops import api as ops_api

    if max_discovery_jobs <= 0:
        raise_contract_error("Runbook contract error: --max-discovery-jobs must be > 0")
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    loaded = _load_runbook_or_exit(runbook)
    active_job_resolution = _resolve_active_job_resolution(
        runbook=loaded,
        active_job_ids=list(active_job_id or ()),
        discover_active_jobs=discover_active_jobs,
        max_discovery_jobs=max_discovery_jobs,
    )
    try:
        plan = ops_api.build_batch_plan(
            runbook=loaded,
            requested_mode=mode,
            requested_smoke=smoke,
            active_job_ids=active_job_resolution.effective_job_ids,
            runtime_visibility=active_job_resolution.runtime_visibility,
            allow_fresh_reset=allow_fresh_reset,
            allow_missing_qstat=allow_missing_qstat,
        )
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    typer.echo(json.dumps(plan.as_dict(), indent=2, sort_keys=True))


@app.command("active-jobs")
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
    from dnadesign.ops import api as ops_api
    from dnadesign.ops.orchestrator.state import ActiveJobResolutionState, SchedulerProbeState

    if max_discovery_jobs <= 0:
        raise_contract_error("Runbook contract error: --max-discovery-jobs must be > 0")
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    loaded = _load_runbook_or_exit(runbook)
    resolution = ops_api.probe_active_jobs_for_runbook(loaded, max_jobs=max_discovery_jobs)
    runtime_visibility = resolution.runtime_visibility
    if (
        runtime_visibility.scheduler_probe_state != SchedulerProbeState.OK
        or runtime_visibility.active_job_resolution_state == ActiveJobResolutionState.UNKNOWN
    ):
        reasons = "; ".join(runtime_visibility.degraded_reasons) or "active-job visibility is unavailable"
        raise_contract_error(f"Runbook contract error: active-job discovery failed: {reasons}")
    active_job_ids = resolution.discovered_job_ids
    hints = _render_active_job_hints(runbook_path=runbook, active_job_ids=active_job_ids)
    payload = {
        "runbook_id": loaded.id,
        "workflow_id": loaded.workflow_id,
        "active_job_ids": list(active_job_ids),
        "runtime_visibility": runtime_visibility.as_dict(),
        **hints,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("execute")
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
        list[str] | None,
        typer.Option(
            "--active-job-id",
            help=(
                "Existing active job id(s) for hold_jid policy decisions; repeat option or pass a comma-delimited list."
            ),
        ),
    ] = None,
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
    allow_unknown_active_jobs: Annotated[
        bool,
        typer.Option(
            "--allow-unknown-active-jobs/--no-allow-unknown-active-jobs",
            help="Allow submit despite degraded active-job visibility; audit JSON records the override.",
        ),
    ] = False,
    allow_missing_qstat: Annotated[
        bool,
        typer.Option(
            "--allow-missing-qstat/--no-allow-missing-qstat",
            help=(
                "Allow qstat-dependent preflight gates to emit explicit degraded advisory records instead of failing "
                "when `qstat` is unavailable. Intended for workstation dry-run demos."
            ),
        ),
    ] = False,
) -> None:
    from dnadesign.ops import api as ops_api

    if command_timeout_seconds is not None and command_timeout_seconds <= 0:
        raise_contract_error("Runbook contract error: --command-timeout-seconds must be > 0")
    if max_discovery_jobs <= 0:
        raise_contract_error("Runbook contract error: --max-discovery-jobs must be > 0")
    if submit and allow_missing_qstat:
        raise_contract_error(
            "Runbook contract error: --allow-missing-qstat is only allowed with --no-submit dry-run demos."
        )
    repo_base = _resolve_repo_base(repo_root)
    try:
        _validate_runbook_input_path_for_runtime(runbook_path=runbook.expanduser(), repo_base=repo_base)
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    loaded = _load_runbook_or_exit(runbook)
    try:
        resolved_audit_json = _validate_audit_json_path_for_execute(
            audit_json_path=audit_json,
            workspace_root=loaded.workspace_root,
        )
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    active_job_resolution = _resolve_active_job_resolution(
        runbook=loaded,
        active_job_ids=list(active_job_id or ()),
        discover_active_jobs=discover_active_jobs,
        max_discovery_jobs=max_discovery_jobs,
    )
    try:
        plan = ops_api.build_batch_plan(
            runbook=loaded,
            requested_mode=mode,
            requested_smoke=smoke,
            active_job_ids=active_job_resolution.effective_job_ids,
            runtime_visibility=active_job_resolution.runtime_visibility,
            allow_fresh_reset=allow_fresh_reset,
            allow_missing_qstat=allow_missing_qstat,
            allow_unknown_active_jobs=allow_unknown_active_jobs,
        )
    except ValueError as exc:
        raise_contract_error(f"Runbook contract error: {exc}")
    if submit and plan.runtime_visibility.scheduler_probe_state == "host_denied":
        raise_contract_error(
            "Runbook contract error: current host is not a submit host for SCC batch submission; "
            "use a submit-capable SCC shell or OnDemand app shell."
        )
    if submit and not allow_unknown_active_jobs and plan.runtime_visibility.active_job_resolution_state == "unknown":
        raise_contract_error(
            "Runbook contract error: active-job visibility is unavailable; "
            "re-run with --allow-unknown-active-jobs only if degraded submit is intentional."
        )
    result = ops_api.execute_batch_plan(
        plan=plan,
        audit_json_path=resolved_audit_json,
        submit=submit,
        command_timeout_seconds=command_timeout_seconds,
    )
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    if not result.ok:
        raise typer.Exit(code=1)


__all__ = ["app", "get_click_command"]
