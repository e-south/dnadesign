"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/progress.py

Read-only progress adapters for registered runbooks and explicit campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from .catalog import (
    RunbookCatalog,
    discover_repo_root,
    load_catalog_procedure_owner_boundary,
    load_catalog_related_registry_ids,
    resolve_catalog_procedure_entry,
)

_PROGRESS_STATES = frozenset({"ok", "attention", "missing"})


@dataclass(frozen=True)
class ProgressInputs:
    repo_root: Path | None = None
    audit_json: Path | None = None
    sync_audit_json: Path | None = None
    usr_root: Path | None = None
    dataset: str | None = None
    study_dir: Path | None = None
    cluster_results_root: Path | None = None
    opal_config: Path | None = None
    opal_workdir: Path | None = None


@dataclass(frozen=True)
class ProcedureProgress:
    registry_id: str
    title: str
    doc_path: str
    owner_boundary: str
    progress_kind: str
    label: str | None
    state: str
    summary: str
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "registry_id": self.registry_id,
            "title": self.title,
            "doc_path": self.doc_path,
            "owner_boundary": self.owner_boundary,
            "progress_kind": self.progress_kind,
            "label": self.label,
            "state": self.state,
            "summary": self.summary,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ProgressFieldSpec:
    manifest_key: str
    cli_flag: str
    placeholder: str
    summary: str

    def as_dict(self) -> dict[str, str]:
        return {
            "manifest_key": self.manifest_key,
            "cli_flag": self.cli_flag,
            "placeholder": self.placeholder,
            "summary": self.summary,
        }


ProgressAdapter = Callable[[ProgressInputs], tuple[str, str, dict[str, object]]]


@dataclass(frozen=True)
class ProgressKindSpec:
    progress_kind: str
    required_inputs: tuple[ProgressFieldSpec, ...]
    adapter: ProgressAdapter


@dataclass(frozen=True)
class CommandExecution:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class CampaignScaffoldStep:
    registry_id: str
    title: str
    doc_path: str
    owner_boundary: str
    progress_kind: str
    label: str
    required_inputs: tuple[ProgressFieldSpec, ...]

    def manifest_step(self) -> dict[str, str]:
        payload = {
            "label": self.label,
            "registry_id": self.registry_id,
        }
        for field in self.required_inputs:
            payload[field.manifest_key] = field.placeholder
        return payload

    def as_dict(self) -> dict[str, object]:
        return {
            "registry_id": self.registry_id,
            "title": self.title,
            "doc_path": self.doc_path,
            "owner_boundary": self.owner_boundary,
            "progress_kind": self.progress_kind,
            "label": self.label,
            "required_inputs": [field.as_dict() for field in self.required_inputs],
            "manifest_step": self.manifest_step(),
        }


@dataclass(frozen=True)
class CampaignScaffold:
    campaign_id: str
    steps: tuple[CampaignScaffoldStep, ...]

    def as_manifest_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "steps": [step.manifest_step() for step in self.steps],
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "manifest": self.as_manifest_dict(),
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class CampaignProgress:
    manifest_path: Path
    campaign_id: str
    steps: tuple[ProcedureProgress, ...]

    def counts(self) -> dict[str, int]:
        state_counts = Counter(step.state for step in self.steps)
        return {state: int(state_counts.get(state, 0)) for state in ("ok", "attention", "missing")}

    def overall_state(self) -> str:
        counts = self.counts()
        if counts["attention"] > 0:
            return "attention"
        if counts["missing"] > 0:
            return "missing"
        return "ok"

    def as_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "manifest_path": str(self.manifest_path),
            "overall_state": self.overall_state(),
            "counts": self.counts(),
            "steps": [step.as_dict() for step in self.steps],
        }


def build_procedure_progress(
    catalog: RunbookCatalog,
    registry_id: str,
    *,
    inputs: ProgressInputs,
) -> ProcedureProgress:
    entry = resolve_catalog_procedure_entry(catalog, registry_id)
    spec = _load_progress_kind_spec(entry.progress_kind)
    state, summary, evidence = spec.adapter(inputs)
    if state not in _PROGRESS_STATES:
        raise ValueError(f"invalid progress state: {state}")
    return ProcedureProgress(
        registry_id=entry.registry_id,
        title=entry.title,
        doc_path=entry.doc_path,
        owner_boundary=load_catalog_procedure_owner_boundary(catalog, entry),
        progress_kind=spec.progress_kind,
        label=None,
        state=state,
        summary=summary,
        evidence=evidence,
    )


def load_campaign_progress(catalog: RunbookCatalog, *, manifest_path: Path) -> CampaignProgress:
    resolved_manifest = manifest_path.expanduser().resolve()
    manifest_dir = resolved_manifest.parent
    if not resolved_manifest.exists():
        raise ValueError(f"campaign manifest not found: {resolved_manifest}")
    payload = yaml.safe_load(resolved_manifest.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("campaign manifest must be a mapping with 'campaign_id' and 'steps'")
    campaign_id = str(payload.get("campaign_id") or resolved_manifest.stem).strip()
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("campaign manifest must define a non-empty 'steps' list")

    steps: list[ProcedureProgress] = []
    for index, step_payload in enumerate(steps_payload, start=1):
        if not isinstance(step_payload, dict):
            raise ValueError(f"campaign manifest step {index} must be a mapping")
        registry_id = str(step_payload.get("registry_id") or "").strip()
        if not registry_id:
            raise ValueError(f"campaign manifest step {index} missing 'registry_id'")
        if catalog.find_procedure(registry_id) is None:
            raise ValueError(f"unknown registry id: {registry_id}")
        try:
            step = build_procedure_progress(
                catalog,
                registry_id,
                inputs=ProgressInputs(
                    repo_root=catalog.repo_root,
                    audit_json=_path_or_none(step_payload.get("audit_json"), base_dir=manifest_dir),
                    sync_audit_json=_path_or_none(step_payload.get("sync_audit_json"), base_dir=manifest_dir),
                    usr_root=_path_or_none(step_payload.get("usr_root"), base_dir=manifest_dir),
                    dataset=_string_or_none(step_payload.get("dataset")),
                    study_dir=_path_or_none(step_payload.get("study_dir"), base_dir=manifest_dir),
                    cluster_results_root=_path_or_none(step_payload.get("cluster_results_root"), base_dir=manifest_dir),
                    opal_config=_path_or_none(step_payload.get("opal_config"), base_dir=manifest_dir),
                    opal_workdir=_path_or_none(step_payload.get("opal_workdir"), base_dir=manifest_dir),
                ),
            )
        except FileNotFoundError as exc:
            missing_path = exc.filename or str(exc)
            raise ValueError(
                f"campaign manifest step {index} ({registry_id}) references a missing file: {missing_path}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"campaign manifest step {index} ({registry_id}): {exc}") from exc
        label = str(step_payload.get("label") or "").strip()
        if label:
            step = ProcedureProgress(
                registry_id=step.registry_id,
                title=step.title,
                doc_path=step.doc_path,
                owner_boundary=step.owner_boundary,
                progress_kind=step.progress_kind,
                label=label,
                state=step.state,
                summary=step.summary,
                evidence=dict(step.evidence),
            )
        steps.append(step)
    return CampaignProgress(
        manifest_path=resolved_manifest,
        campaign_id=campaign_id or resolved_manifest.stem,
        steps=tuple(steps),
    )


def build_campaign_scaffold(
    catalog: RunbookCatalog,
    *,
    registry_ids: Sequence[str],
    campaign_id: str | None = None,
    related_to: str | None = None,
) -> CampaignScaffold:
    normalized_registry_ids = _resolve_campaign_scaffold_registry_ids(
        catalog,
        registry_ids=registry_ids,
        related_to=related_to,
    )

    resolved_campaign_id = str(campaign_id or "progress_campaign").strip() or "progress_campaign"
    used_labels: Counter[str] = Counter()
    steps: list[CampaignScaffoldStep] = []
    for registry_id in normalized_registry_ids:
        entry = resolve_catalog_procedure_entry(catalog, registry_id)
        spec = _load_progress_kind_spec(entry.progress_kind)
        label = _suggest_scaffold_label(entry.registry_id, used_labels)
        steps.append(
            CampaignScaffoldStep(
                registry_id=entry.registry_id,
                title=entry.title,
                doc_path=entry.doc_path,
                owner_boundary=load_catalog_procedure_owner_boundary(catalog, entry),
                progress_kind=spec.progress_kind,
                label=label,
                required_inputs=spec.required_inputs,
            )
        )
    return CampaignScaffold(campaign_id=resolved_campaign_id, steps=tuple(steps))


def load_progress_required_inputs(progress_kind: str) -> tuple[ProgressFieldSpec, ...]:
    return _load_progress_kind_spec(progress_kind).required_inputs


def _resolve_campaign_scaffold_registry_ids(
    catalog: RunbookCatalog,
    *,
    registry_ids: Sequence[str],
    related_to: str | None,
) -> tuple[str, ...]:
    ordered_registry_ids: list[str] = []
    seen_registry_ids: set[str] = set()

    normalized_related_to = str(related_to or "").strip()
    if normalized_related_to:
        for registry_id in load_catalog_related_registry_ids(catalog, normalized_related_to, include_self=True):
            if registry_id in seen_registry_ids:
                continue
            ordered_registry_ids.append(registry_id)
            seen_registry_ids.add(registry_id)

    for registry_id in registry_ids:
        normalized_registry_id = registry_id.strip()
        if not normalized_registry_id or normalized_registry_id in seen_registry_ids:
            continue
        ordered_registry_ids.append(normalized_registry_id)
        seen_registry_ids.add(normalized_registry_id)

    if not ordered_registry_ids:
        raise ValueError("progress scaffold requires at least one registry id or --related-to")
    return tuple(ordered_registry_ids)


def _ops_audit_progress(audit_json: Path | None) -> tuple[str, str, dict[str, object]]:
    resolved_audit = _required_path(audit_json, flag_name="--audit-json", progress_kind="ops-audit-json")
    if not resolved_audit.exists():
        return (
            "missing",
            "audit artifact not found",
            {"audit_json": str(resolved_audit)},
        )
    payload = json.loads(resolved_audit.read_text(encoding="utf-8"))
    execution = dict(payload.get("execution") or {})
    plan = dict(payload.get("plan") or {})
    commands = list(execution.get("commands") or [])
    phase_counts = Counter(str(command.get("phase") or "unknown") for command in commands)
    ok = bool(execution.get("ok", False))
    failed_phase = execution.get("failed_phase")
    queue_probe = _extract_queue_probe_evidence(commands)
    if ok and queue_probe is not None and queue_probe["status"] == "degraded":
        summary = "latest orchestration audit passed with degraded queue probe"
    elif ok:
        summary = "latest orchestration audit passed"
    else:
        summary = f"latest orchestration audit failed at {failed_phase or 'unknown'}"
    return (
        "attention" if (not ok or (queue_probe is not None and queue_probe["status"] == "degraded")) else "ok",
        summary,
        {
            "audit_json": str(resolved_audit),
            "workflow_id": plan.get("workflow_id"),
            "project": plan.get("project"),
            "runbook_id": plan.get("runbook_id"),
            "workspace_root": plan.get("workspace_root"),
            "execution_ok": ok,
            "failed_phase": failed_phase,
            "command_count": len(commands),
            "phase_counts": dict(sorted(phase_counts.items())),
            "queue_probe": queue_probe,
        },
    )


def _extract_queue_probe_evidence(commands: list[object]) -> dict[str, object] | None:
    queue_probe_commands: list[dict[str, object]] = []
    status = "ok"
    for command in commands:
        if not isinstance(command, dict):
            continue
        fields = _parse_record_fields(command.get("stdout"))
        queue_probe = fields.get("queue_probe")
        if queue_probe is None:
            continue
        if queue_probe != "ok":
            status = "degraded"
        queue_probe_commands.append(
            {
                "phase": command.get("phase"),
                "command": command.get("command"),
                "queue_probe": queue_probe,
                "next_action": fields.get("next_action"),
                "submit_gate": fields.get("submit_gate"),
                "advisor": fields.get("advisor"),
                "stderr": str(command.get("stderr") or "").strip() or None,
            }
        )
    if not queue_probe_commands:
        return None
    return {
        "status": status,
        "commands": queue_probe_commands,
    }


def _parse_record_fields(raw_text: object) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in str(raw_text or "").splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", maxsplit=1)
            if key:
                fields[key] = value
    return fields


def _usr_sync_audit_progress(sync_audit_json: Path | None) -> tuple[str, str, dict[str, object]]:
    resolved_audit = _required_path(
        sync_audit_json,
        flag_name="--sync-audit-json",
        progress_kind="usr-sync-audit",
    )
    if not resolved_audit.exists():
        return (
            "missing",
            "sync audit artifact not found",
            {"sync_audit_json": str(resolved_audit)},
        )
    payload = json.loads(resolved_audit.read_text(encoding="utf-8"))
    transfer_state = str(payload.get("transfer_state") or "UNKNOWN")
    changed_flags = {
        "primary": bool((payload.get("primary") or {}).get("changed")),
        "meta": bool((payload.get("meta") or {}).get("changed")),
        "_snapshots": bool((payload.get("_snapshots") or {}).get("changed")),
        "_derived": bool((payload.get("_derived") or {}).get("changed")),
        "_auxiliary": bool((payload.get("_auxiliary") or {}).get("changed")),
    }
    has_pending_drift = any(changed_flags.values())
    is_ok = transfer_state in {"NO-OP", "TRANSFERRED"} and not has_pending_drift
    summary = f"{payload.get('dataset', '<unknown>')}: {transfer_state}"
    if has_pending_drift:
        summary += " with remaining drift"
    return (
        "ok" if is_ok else "attention",
        summary,
        {
            "sync_audit_json": str(resolved_audit),
            "action": payload.get("action"),
            "dataset": payload.get("dataset"),
            "transfer_state": transfer_state,
            "verify": dict(payload.get("verify") or {}),
            "changed_flags": changed_flags,
            "events_log": dict(payload.get(".events.log") or {}),
        },
    )


def _usr_dataset_state_progress(
    *,
    usr_root: Path | None,
    dataset: str | None,
) -> tuple[str, str, dict[str, object]]:
    resolved_root = _required_path(usr_root, flag_name="--usr-root", progress_kind="usr-dataset-state")
    dataset_id = _required_text(dataset, flag_name="--dataset", progress_kind="usr-dataset-state")
    dataset_dir = (resolved_root / dataset_id).resolve()
    records_path = dataset_dir / "records.parquet"
    if not records_path.exists():
        return (
            "missing",
            f"USR dataset not found: {dataset_id}",
            {
                "usr_root": str(resolved_root),
                "dataset": dataset_id,
                "records_path": str(records_path),
            },
        )

    parquet_file = pq.ParquetFile(str(records_path))
    schema = parquet_file.schema_arrow
    columns = list(schema.names)
    namespace_counts = _namespace_column_counts(columns)
    overlay_namespaces = _overlay_namespace_names(dataset_dir)
    events_log_path = dataset_dir / ".events.log"
    snapshots_dir = dataset_dir / "_snapshots"
    events_count = _line_count(events_log_path) if events_log_path.exists() else 0
    snapshots_count = _file_count(snapshots_dir) if snapshots_dir.exists() else 0
    rows = int(parquet_file.metadata.num_rows)
    cols = int(parquet_file.metadata.num_columns)
    infer_columns = int(namespace_counts.get("infer", 0))
    summary = f"{dataset_id}: {rows} rows, {cols} columns"
    if infer_columns:
        summary += f", {infer_columns} infer-derived columns"
    return (
        "ok" if rows > 0 else "attention",
        summary,
        {
            "usr_root": str(resolved_root),
            "dataset": dataset_id,
            "dataset_dir": str(dataset_dir),
            "records_path": str(records_path),
            "rows": rows,
            "columns": cols,
            "namespace_column_counts": dict(sorted(namespace_counts.items())),
            "overlay_namespaces": overlay_namespaces,
            "overlay_namespace_count": len(overlay_namespaces),
            "events_count": events_count,
            "snapshots_count": snapshots_count,
        },
    )


def _promoter_study_record_progress(
    study_dir: Path | None,
    *,
    repo_root: Path | None = None,
) -> tuple[str, str, dict[str, object]]:
    context = _load_promoter_study_context(
        study_dir,
        repo_root=repo_root,
        progress_kind="promoter-study-record",
    )
    evidence = dict(context["evidence"])
    if not bool(context["study_dir_exists"]):
        return ("missing", "promoter study directory not found", evidence)
    missing_required_files = list(context["missing_required_files"])
    missing_declared_present = list(context["missing_declared_present"])
    missing_execution_surfaces = list(context["missing_execution_surfaces"])
    current_phase = _string_or_none(context["current_phase"])
    current_phase_is_known = bool(context["current_phase_is_known"])
    densegen_dataset_id = _string_or_none(context["densegen_dataset_id"])
    densegen_rows = _optional_positive_int(context["densegen_rows"])
    row_target = _optional_positive_int(context["densegen_row_target"])
    densegen_row_gap = _optional_positive_int(context["densegen_row_gap"])
    next_ready_phase = context["next_ready_phase"]
    next_in_progress_phase = context["next_in_progress_phase"]
    next_planned_phase = context["next_planned_phase"]
    blocked_phases = list(context["blocked_phases"])
    dataset_states = list(context["dataset_states"])
    present_but_planned = list(context["present_but_planned"])
    resolved_study_dir = Path(str(context["resolved_study_dir"]))

    if missing_required_files:
        summary = "study record missing required files: " + ", ".join(missing_required_files)
        return ("missing", summary, evidence)
    if missing_declared_present:
        summary = "study record references declared-present datasets that are missing: " + ", ".join(
            missing_declared_present
        )
        return ("missing", summary, evidence)
    if missing_execution_surfaces:
        summary = "study record references missing execution surfaces: " + ", ".join(missing_execution_surfaces)
        return ("missing", summary, evidence)

    summary_parts = [f"{resolved_study_dir.name}: phase {current_phase or 'unknown'}"]
    if densegen_dataset_id is not None and densegen_rows is not None and row_target is not None:
        summary_parts.append(f"{densegen_dataset_id} {densegen_rows}/{row_target} rows")
    elif densegen_dataset_id is not None and densegen_rows is not None:
        summary_parts.append(f"{densegen_dataset_id} {densegen_rows} rows")

    pending_shared_datasets = [
        state["dataset"] for state in dataset_states if state["declared_status"] != "present" and not state["exists"]
    ]
    if pending_shared_datasets:
        summary_parts.append("pending " + ", ".join(pending_shared_datasets))
    if next_ready_phase is not None:
        summary_parts.append(f"next ready {next_ready_phase['id']}")
    elif next_in_progress_phase is not None:
        summary_parts.append(f"next in_progress {next_in_progress_phase['id']}")
    elif next_planned_phase is not None:
        summary_parts.append(f"next planned {next_planned_phase['id']}")

    attention_reasons: list[str] = []
    if current_phase is not None and not current_phase_is_known:
        attention_reasons.append("current_phase does not match any declared phase id")
    if present_but_planned:
        attention_reasons.append("datasets.yaml is stale for newly materialized outputs")
    if densegen_row_gap not in (None, 0):
        attention_reasons.append("DenseGen anchor target not met")
    if blocked_phases:
        attention_reasons.append("GPU-only infer lanes remain blocked")
    if any(phase["status"] in {"ready", "planned", "in_progress", "blocked_gpu"} for phase in context["phase_states"]):
        attention_reasons.append("study is not complete")

    if attention_reasons:
        evidence["attention_reasons"] = attention_reasons
        return ("attention", "; ".join(summary_parts), evidence)
    return ("ok", "; ".join(summary_parts), evidence)


def _promoter_study_preflight_progress(
    study_dir: Path | None,
    *,
    repo_root: Path | None = None,
) -> tuple[str, str, dict[str, object]]:
    context = _load_promoter_study_context(
        study_dir,
        repo_root=repo_root,
        progress_kind="promoter-study-preflight",
    )
    evidence = dict(context["evidence"])
    if not bool(context["study_dir_exists"]):
        return ("missing", "promoter study directory not found", evidence)
    missing_required_files = list(context["missing_required_files"])
    missing_declared_present = list(context["missing_declared_present"])
    missing_execution_surfaces = list(context["missing_execution_surfaces"])
    if missing_required_files:
        summary = "study record missing required files: " + ", ".join(missing_required_files)
        return ("missing", summary, evidence)
    if missing_declared_present:
        summary = "study record references declared-present datasets that are missing: " + ", ".join(
            missing_declared_present
        )
        return ("missing", summary, evidence)
    if missing_execution_surfaces:
        summary = "study record references missing execution surfaces: " + ", ".join(missing_execution_surfaces)
        return ("missing", summary, evidence)

    study_repo_root = Path(str(context["study_repo_root"]))
    study_pipeline = dict(context["study_pipeline"])
    execution_surface_index: dict[str, Path] = dict(context["execution_surface_index"])
    dataset_index: dict[str, dict[str, object]] = dict(context["dataset_index"])
    checks: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    notify_env_state = {
        "NOTIFY_WEBHOOK": bool(str(os.environ.get("NOTIFY_WEBHOOK") or "").strip()),
        "NOTIFY_WEBHOOK_FILE": bool(str(os.environ.get("NOTIFY_WEBHOOK_FILE") or "").strip()),
        "SSL_CERT_FILE": bool(str(os.environ.get("SSL_CERT_FILE") or "").strip()),
    }

    def add_check(check: dict[str, object]) -> None:
        checks.append(check)
        state = str(check.get("state") or "attention")
        counts[state] += 1

    densegen_batch_runbook = execution_surface_index.get("densegen_batch_with_notify")
    if densegen_batch_runbook is not None:
        densegen_runbook = _load_orchestration_runbook_payload(densegen_batch_runbook)
        densegen_config_text = _string_or_none(((densegen_runbook.get("densegen") or {}).get("config")))
        densegen_resources = dict(densegen_runbook.get("resources") or {})
        add_check(
            _preflight_state_check(
                check_id="densegen.batch.resources",
                phase="densegen",
                state="ok",
                summary=(
                    "densegen batch resources declared"
                    if densegen_resources
                    else "densegen batch resources missing from runbook"
                ),
                details={
                    "runbook": str(densegen_batch_runbook),
                    "resources": densegen_resources,
                },
            )
        )
        if densegen_config_text is not None:
            densegen_config_path = _resolve_input_path(
                Path(densegen_config_text), base_dir=densegen_batch_runbook.parent
            )
            densegen_probe = _run_progress_command(
                ("uv", "run", "dense", "validate-config", "--probe-solver", "-c", str(densegen_config_path)),
                cwd=study_repo_root,
            )
            add_check(
                _preflight_command_check(
                    check_id="densegen.config.probe_solver",
                    phase="densegen",
                    summary=_choose_command_summary(densegen_probe, fallback="densegen config probe completed"),
                    execution=densegen_probe,
                    details={"config": str(densegen_config_path)},
                )
            )
        densegen_plan = _run_progress_command(
            (
                "uv",
                "run",
                "ops",
                "runbook",
                "plan",
                "--runbook",
                str(densegen_batch_runbook),
                "--repo-root",
                str(study_repo_root),
            ),
            cwd=study_repo_root,
        )
        densegen_plan_payload = _safe_json_loads(densegen_plan.stdout) if densegen_plan.returncode == 0 else None
        densegen_plan_details: dict[str, object] = {"runbook": str(densegen_batch_runbook)}
        if isinstance(densegen_plan_payload, dict):
            densegen_plan_details.update(
                {
                    "selected_mode": densegen_plan_payload.get("selected_mode"),
                    "workflow_id": densegen_plan_payload.get("workflow_id"),
                    "notify_secret_ref": dict(densegen_plan_payload.get("orchestration_notify") or {}).get(
                        "secret_ref"
                    ),
                }
            )
        add_check(
            _preflight_command_check(
                check_id="densegen.batch.plan",
                phase="densegen",
                summary=_choose_command_summary(densegen_plan, fallback="densegen batch plan completed"),
                execution=densegen_plan,
                details=densegen_plan_details,
            )
        )

    construct_workspace_path = execution_surface_index.get("construct_workspace")
    if construct_workspace_path is not None:
        construct_doctor = _run_progress_command(
            ("uv", "run", "construct", "workspace", "doctor", "--workspace", str(construct_workspace_path)),
            cwd=study_repo_root,
        )
        add_check(
            _preflight_command_check(
                check_id="construct.workspace.doctor",
                phase="construct",
                summary=_choose_command_summary(construct_doctor, fallback="construct workspace doctor completed"),
                execution=construct_doctor,
                details={"workspace": str(construct_workspace_path)},
            )
        )

        merged_anchor_dataset = _string_or_none(((study_pipeline.get("datasets") or {}).get("merged_anchor_dataset")))
        merged_anchor_state = dataset_index.get(merged_anchor_dataset or "") if merged_anchor_dataset else None
        construct_workspace_projects = list(((study_pipeline.get("construct") or {}).get("workspace_projects")) or [])
        for project_payload in construct_workspace_projects:
            if not isinstance(project_payload, dict):
                continue
            project_id = _string_or_none(project_payload.get("id"))
            if project_id is None:
                continue
            check_id = f"construct.runtime.{project_id}"
            if merged_anchor_state is None or not bool(merged_anchor_state.get("exists")):
                add_check(
                    _preflight_state_check(
                        check_id=check_id,
                        phase="construct",
                        state="missing",
                        summary=(
                            f"requires merged anchor dataset {merged_anchor_dataset} before runtime validation"
                            if merged_anchor_dataset is not None
                            else "requires merged anchor dataset before runtime validation"
                        ),
                        details={
                            "dataset": merged_anchor_dataset,
                            "records_path": merged_anchor_state.get("records_path") if merged_anchor_state else None,
                            "workspace": str(construct_workspace_path),
                            "project": project_id,
                        },
                    )
                )
                continue
            construct_runtime = _run_progress_command(
                (
                    "uv",
                    "run",
                    "construct",
                    "workspace",
                    "validate-project",
                    "--workspace",
                    str(construct_workspace_path),
                    "--project",
                    project_id,
                    "--runtime",
                ),
                cwd=study_repo_root,
            )
            add_check(
                _preflight_command_check(
                    check_id=check_id,
                    phase="construct",
                    summary=_choose_command_summary(
                        construct_runtime, fallback="construct runtime validation completed"
                    ),
                    execution=construct_runtime,
                    details={"workspace": str(construct_workspace_path), "project": project_id},
                )
            )

    infer_payload = dict(study_pipeline.get("infer") or {})
    infer_config_paths = _resolve_named_path_mapping(
        infer_payload.get("configs"),
        repo_root=study_repo_root,
        label="infer configs",
        progress_kind="promoter-study-preflight",
    )
    infer_notify_profile_paths = _resolve_named_path_mapping(
        infer_payload.get("notify_profiles"),
        repo_root=study_repo_root,
        label="infer notify profiles",
        progress_kind="promoter-study-preflight",
    )
    for config_label, config_path in sorted(infer_config_paths.items()):
        infer_validate = _run_progress_command(
            ("uv", "run", "infer", "validate", "config", "--config", str(config_path)),
            cwd=study_repo_root,
        )
        add_check(
            _preflight_command_check(
                check_id=f"infer.validate.{config_label}",
                phase="infer",
                summary=_choose_command_summary(infer_validate, fallback="infer config validation completed"),
                execution=infer_validate,
                details={"config": str(config_path)},
            )
        )

    runtime_config_labels = ("anchor_only_7b", "anchor_plus_template_7b")
    for runtime_label in runtime_config_labels:
        config_path = infer_config_paths.get(runtime_label)
        if config_path is None:
            continue
        profile_path = infer_notify_profile_paths.get(runtime_label)
        if profile_path is None:
            add_check(
                _preflight_state_check(
                    check_id=f"notify.profile.{runtime_label}",
                    phase="notify",
                    state="attention",
                    summary="infer notify profile path is not recorded in study pipeline",
                    details={"config": str(config_path)},
                )
            )
        elif not profile_path.is_file():
            add_check(
                _preflight_state_check(
                    check_id=f"notify.profile.{runtime_label}",
                    phase="notify",
                    state="attention",
                    summary="infer notify profile is not materialized yet",
                    details={
                        "config": str(config_path),
                        "profile": str(profile_path),
                        "setup_command": _build_infer_notify_setup_command(
                            config_path=config_path,
                            profile_path=profile_path,
                        ),
                        "tls_note": "Export SSL_CERT_FILE before `notify profile doctor` or live delivery.",
                    },
                )
            )
        else:
            notify_profile_doctor = _run_progress_command(
                ("uv", "run", "notify", "profile", "doctor", "--profile", str(profile_path), "--json"),
                cwd=study_repo_root,
            )
            add_check(
                _preflight_command_check(
                    check_id=f"notify.profile.{runtime_label}",
                    phase="notify",
                    summary=_choose_command_summary(
                        notify_profile_doctor,
                        fallback="infer notify profile doctor completed",
                    ),
                    execution=notify_profile_doctor,
                    details={
                        "config": str(config_path),
                        "profile": str(profile_path),
                    },
                )
            )
        usr_inputs = _infer_usr_dataset_requirements(config_path)
        missing_usr_inputs = [entry for entry in usr_inputs if not bool(entry.get("exists"))]
        if missing_usr_inputs:
            add_check(
                _preflight_state_check(
                    check_id=f"infer.dry_run.{runtime_label}",
                    phase="infer",
                    state="missing",
                    summary="requires study-owned USR datasets before infer dry-run",
                    details={
                        "config": str(config_path),
                        "missing_usr_inputs": missing_usr_inputs,
                    },
                )
            )
        else:
            infer_dry_run = _run_progress_command(
                ("uv", "run", "infer", "run", "--config", str(config_path), "--dry-run"),
                cwd=study_repo_root,
            )
            add_check(
                _preflight_command_check(
                    check_id=f"infer.dry_run.{runtime_label}",
                    phase="infer",
                    summary=_choose_command_summary(infer_dry_run, fallback="infer dry-run completed"),
                    execution=infer_dry_run,
                    details={"config": str(config_path)},
                )
            )

        resolve_events = _run_progress_command(
            (
                "uv",
                "run",
                "notify",
                "setup",
                "resolve-events",
                "--tool",
                "infer",
                "--config",
                str(config_path),
                "--json",
            ),
            cwd=study_repo_root,
        )
        resolve_payload = _safe_json_loads(resolve_events.stdout) if resolve_events.returncode == 0 else None
        resolve_details: dict[str, object] = {"config": str(config_path)}
        resolve_state = "ok" if resolve_events.returncode == 0 else "attention"
        resolve_summary = _choose_command_summary(resolve_events, fallback="notify event resolution completed")
        if isinstance(resolve_payload, dict):
            events_path = Path(str(resolve_payload.get("events") or "")).expanduser()
            resolve_details.update(
                {
                    "events": str(events_path),
                    "events_exists": events_path.exists(),
                    "policy": resolve_payload.get("policy"),
                }
            )
            if not events_path.exists():
                resolve_state = "missing"
                resolve_summary = f"resolved events path is not materialized yet: {events_path}"
        add_check(
            _preflight_command_check(
                check_id=f"notify.resolve_events.{runtime_label}",
                phase="notify",
                summary=resolve_summary,
                execution=resolve_events,
                details=resolve_details,
                override_state=resolve_state,
            )
        )

    infer_batch_surface_labels = (
        "infer_batch_7b_with_notify.anchor_only",
        "infer_batch_7b_with_notify.anchor_plus_template",
        "infer_batch_20b_with_notify.anchor_only",
        "infer_batch_20b_with_notify.anchor_plus_template",
    )
    for surface_label in infer_batch_surface_labels:
        runbook_path = execution_surface_index.get(surface_label)
        if runbook_path is None:
            continue
        runbook_plan = _run_progress_command(
            (
                "uv",
                "run",
                "ops",
                "runbook",
                "plan",
                "--runbook",
                str(runbook_path),
                "--repo-root",
                str(study_repo_root),
            ),
            cwd=study_repo_root,
        )
        runbook_plan_payload = _safe_json_loads(runbook_plan.stdout) if runbook_plan.returncode == 0 else None
        runbook_details: dict[str, object] = {
            "runbook": str(runbook_path),
            "notify_env": notify_env_state,
        }
        if isinstance(runbook_plan_payload, dict):
            runbook_details.update(
                {
                    "selected_mode": runbook_plan_payload.get("selected_mode"),
                    "workflow_id": runbook_plan_payload.get("workflow_id"),
                    "notify_secret_ref": dict(runbook_plan_payload.get("orchestration_notify") or {}).get("secret_ref"),
                }
            )
        add_check(
            _preflight_command_check(
                check_id=f"ops.runbook_plan.{surface_label}",
                phase="ops",
                summary=_choose_command_summary(runbook_plan, fallback="ops runbook plan completed"),
                execution=runbook_plan,
                details=runbook_details,
            )
        )

    evidence.update(
        {
            "notify_environment": notify_env_state,
            "checks": checks,
            "counts": {state: int(counts.get(state, 0)) for state in ("ok", "attention", "missing")},
        }
    )

    current_phase = _string_or_none(context["current_phase"])
    study_id = _string_or_none(context["study_id"]) or Path(str(context["resolved_study_dir"])).name
    summary_parts = [f"{study_id}: preflight phase {current_phase or 'unknown'}"]
    if counts.get("ok"):
        summary_parts.append(f"{counts['ok']} ok")
    if counts.get("attention"):
        summary_parts.append(f"{counts['attention']} attention")
    if counts.get("missing"):
        summary_parts.append(f"{counts['missing']} missing")
    failing_checks = [check for check in checks if check["state"] != "ok"]
    if failing_checks:
        summary_parts.append("first blockers: " + ", ".join(check["id"] for check in failing_checks[:3]))

    if counts.get("missing"):
        return ("missing", "; ".join(summary_parts), evidence)
    if counts.get("attention") or counts.get("missing"):
        return ("attention", "; ".join(summary_parts), evidence)
    return ("ok", "; ".join(summary_parts), evidence)


def _load_promoter_study_context(
    study_dir: Path | None,
    *,
    repo_root: Path | None = None,
    progress_kind: str,
) -> dict[str, object]:
    resolved_input_repo_root = repo_root.expanduser().resolve() if repo_root is not None else None
    selection_source = "explicit"
    requested_study_dir = str(study_dir) if study_dir is not None else None
    if study_dir is None:
        resolved_study_dir, registry_path, active_registry_study = _discover_active_promoter_study_dir(
            repo_root=resolved_input_repo_root,
            progress_kind=progress_kind,
        )
        selection_source = "active_registry"
    else:
        resolved_study_dir = _required_path(
            study_dir,
            flag_name="--study-dir",
            progress_kind=progress_kind,
            base_dir=resolved_input_repo_root,
        )
        registry_path = resolved_study_dir.parent / "index.yaml"
        active_registry_study = None
    if not resolved_study_dir.exists():
        return {
            "study_dir_exists": False,
            "resolved_study_dir": resolved_study_dir,
            "study_repo_root": resolved_input_repo_root,
            "study_id": resolved_study_dir.name,
            "missing_required_files": [],
            "missing_declared_present": [],
            "present_but_planned": [],
            "missing_execution_surfaces": [],
            "phase_states": [],
            "blocked_phases": [],
            "current_phase": None,
            "current_phase_is_known": False,
            "next_ready_phase": None,
            "next_in_progress_phase": None,
            "next_planned_phase": None,
            "densegen_dataset_id": None,
            "densegen_rows": None,
            "densegen_row_target": None,
            "densegen_row_gap": None,
            "dataset_states": [],
            "dataset_index": {},
            "execution_surface_states": [],
            "execution_surface_index": {},
            "study_pipeline": {},
            "evidence": {
                "requested_study_dir": requested_study_dir,
                "study_dir": str(resolved_study_dir),
                "study_selection_source": selection_source,
            },
        }
    if not resolved_study_dir.is_dir():
        raise ValueError(f"study_dir must be a directory: {resolved_study_dir}")

    study_repo_root = discover_repo_root(resolved_study_dir)
    if study_repo_root is None:
        raise ValueError(f"study_dir must live inside a dnadesign repository checkout: {resolved_study_dir}")

    required_paths = {
        "campaign.yaml": resolved_study_dir / "campaign.yaml",
        "datasets.yaml": resolved_study_dir / "datasets.yaml",
        "status.md": resolved_study_dir / "status.md",
    }
    missing_required_files = [name for name, path in required_paths.items() if not path.exists()]
    pipeline_path = resolved_study_dir / "pipeline.yaml"
    evidence: dict[str, object] = {
        "requested_study_dir": requested_study_dir,
        "study_dir": str(resolved_study_dir),
        "repo_root": str(study_repo_root),
        "study_id": resolved_study_dir.name,
        "study_selection_source": selection_source,
        "active_study_registry_path": str(registry_path),
        "required_files": {name: str(path) for name, path in required_paths.items()},
        "pipeline_path": str(pipeline_path),
        "pipeline_present": pipeline_path.exists(),
        "missing_required_files": missing_required_files,
    }
    if missing_required_files:
        return {
            "study_dir_exists": True,
            "requested_study_dir": requested_study_dir,
            "resolved_study_dir": resolved_study_dir,
            "study_repo_root": study_repo_root,
            "study_id": resolved_study_dir.name,
            "selection_source": selection_source,
            "registry_path": registry_path,
            "active_study": active_registry_study,
            "required_paths": required_paths,
            "missing_required_files": missing_required_files,
            "pipeline_path": pipeline_path,
            "pipeline_present": pipeline_path.exists(),
            "datasets_entries": [],
            "study_pipeline": {},
            "canonical_usr_root_path": None,
            "dataset_states": [],
            "dataset_index": {},
            "missing_declared_present": [],
            "present_but_planned": [],
            "execution_surface_states": [],
            "execution_surface_index": {},
            "missing_execution_surfaces": [],
            "phase_states": [],
            "current_phase": None,
            "current_phase_is_known": False,
            "next_ready_phase": None,
            "next_in_progress_phase": None,
            "next_planned_phase": None,
            "blocked_phases": [],
            "densegen_dataset_id": None,
            "densegen_rows": None,
            "densegen_row_target": None,
            "densegen_row_gap": None,
            "evidence": evidence,
        }

    datasets_payload = _load_yaml_mapping(required_paths["datasets.yaml"], label="datasets.yaml")
    datasets_entries = datasets_payload.get("datasets") or []
    if not isinstance(datasets_entries, list):
        raise ValueError(f"datasets.yaml must define a 'datasets' list: {required_paths['datasets.yaml']}")

    pipeline_payload = _load_yaml_mapping(pipeline_path, label="pipeline.yaml") if pipeline_path.exists() else {}
    study_pipeline = pipeline_payload.get("study_pipeline") or {}
    if study_pipeline and not isinstance(study_pipeline, dict):
        raise ValueError(f"pipeline.yaml must define a 'study_pipeline' mapping: {pipeline_path}")

    promoter_index_path = registry_path
    active_study = None
    if promoter_index_path.exists():
        promoter_index = _load_yaml_mapping(promoter_index_path, label="promoter index")
        active_study = _string_or_none(promoter_index.get("active_study"))
    if active_study is None:
        active_study = active_registry_study

    canonical_usr_root_text = _string_or_none(study_pipeline.get("canonical_usr_root"))
    if canonical_usr_root_text is None:
        for entry in datasets_entries:
            if isinstance(entry, dict):
                canonical_usr_root_text = _string_or_none(entry.get("usr_root"))
                if canonical_usr_root_text is not None:
                    break
    canonical_usr_root_path = (
        _resolve_repo_relative_path(
            repo_root=study_repo_root, raw_path=canonical_usr_root_text, progress_kind=progress_kind
        )
        if canonical_usr_root_text is not None
        else None
    )

    dataset_states: list[dict[str, object]] = []
    missing_declared_present: list[str] = []
    present_but_planned: list[str] = []
    dataset_index: dict[str, dict[str, object]] = {}
    for entry in datasets_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry must be a mapping: {required_paths['datasets.yaml']}")
        dataset_id = _required_metadata_text(
            entry.get("dataset"), label="dataset id", source=required_paths["datasets.yaml"]
        )
        role = _string_or_none(entry.get("role")) or dataset_id
        declared_status = _string_or_none(entry.get("status")) or "unknown"
        entry_usr_root_text = _string_or_none(entry.get("usr_root")) or canonical_usr_root_text
        entry_usr_root = _resolve_repo_relative_path(
            repo_root=study_repo_root,
            raw_path=entry_usr_root_text,
            progress_kind=progress_kind,
        )
        records_path = (entry_usr_root / dataset_id / "records.parquet").resolve()
        exists = records_path.exists()
        rows = _parquet_row_count(records_path) if exists else None
        dataset_state = {
            "role": role,
            "dataset": dataset_id,
            "declared_status": declared_status,
            "usr_root": str(entry_usr_root),
            "records_path": str(records_path),
            "exists": exists,
            "rows": rows,
        }
        dataset_states.append(dataset_state)
        dataset_index[dataset_id] = dataset_state
        if declared_status == "present" and not exists:
            missing_declared_present.append(dataset_id)
        if declared_status == "planned" and exists:
            present_but_planned.append(dataset_id)

    execution_surface_index = _resolve_named_path_mapping(
        study_pipeline.get("execution_surfaces"),
        repo_root=study_repo_root,
        label="execution_surfaces",
        progress_kind=progress_kind,
    )
    execution_surface_states: list[dict[str, object]] = []
    missing_execution_surfaces: list[str] = []
    for label, resolved_path in execution_surface_index.items():
        exists = resolved_path.exists()
        execution_surface_states.append({"label": label, "path": str(resolved_path), "exists": exists})
        if not exists:
            missing_execution_surfaces.append(label)

    phases_payload = study_pipeline.get("phases") or []
    if phases_payload and not isinstance(phases_payload, list):
        raise ValueError(f"phases must be a list: {pipeline_path}")
    phase_states: list[dict[str, object]] = []
    phase_index: dict[str, dict[str, object]] = {}
    for phase in phases_payload:
        if not isinstance(phase, dict):
            raise ValueError(f"phase entry must be a mapping: {pipeline_path}")
        phase_id = _required_metadata_text(phase.get("id"), label="phase id", source=pipeline_path)
        phase_state = {
            "id": phase_id,
            "status": _string_or_none(phase.get("status")) or "unknown",
            "next_surface": _string_or_none(phase.get("next_surface")),
            "blocker": _string_or_none(phase.get("blocker")),
            "output_dataset": _string_or_none(phase.get("output_dataset")),
            "primary_dataset": _string_or_none(phase.get("primary_dataset")),
        }
        phase_states.append(phase_state)
        phase_index[phase_id] = phase_state

    current_phase = _string_or_none(study_pipeline.get("current_phase"))
    current_phase_is_known = current_phase in phase_index if current_phase is not None else False
    next_ready_phase = _first_phase_by_status(phase_states, status="ready")
    next_in_progress_phase = _first_phase_by_status(phase_states, status="in_progress")
    next_planned_phase = _first_phase_by_status(phase_states, status="planned")
    blocked_phases = [phase for phase in phase_states if phase["status"] == "blocked_gpu"]

    densegen_dataset_id = _string_or_none((study_pipeline.get("datasets") or {}).get("densegen_anchor_source"))
    densegen_dataset_state = dataset_index.get(densegen_dataset_id or "") if densegen_dataset_id else None
    row_target = _optional_positive_int(
        ((study_pipeline.get("row_targets") or {}).get("densegen_anchor_minimum_before_first_full_lane_infer"))
    )
    densegen_rows = densegen_dataset_state.get("rows") if densegen_dataset_state is not None else None
    densegen_row_gap = (
        max(int(row_target) - int(densegen_rows), 0) if row_target is not None and densegen_rows is not None else None
    )

    evidence.update(
        {
            "active_study": active_study,
            "is_active_study": active_study == resolved_study_dir.name if active_study is not None else None,
            "canonical_usr_root": str(canonical_usr_root_path) if canonical_usr_root_path is not None else None,
            "datasets": dataset_states,
            "missing_declared_present": missing_declared_present,
            "present_but_planned": present_but_planned,
            "execution_surfaces": execution_surface_states,
            "missing_execution_surfaces": missing_execution_surfaces,
            "current_phase": current_phase,
            "current_phase_is_known": current_phase_is_known,
            "phase_states": phase_states,
            "next_ready_phase": next_ready_phase,
            "next_in_progress_phase": next_in_progress_phase,
            "next_planned_phase": next_planned_phase,
            "blocked_phases": blocked_phases,
            "densegen_dataset": densegen_dataset_id,
            "densegen_rows": densegen_rows,
            "densegen_row_target": row_target,
            "densegen_row_gap": densegen_row_gap,
        }
    )

    return {
        "study_dir_exists": True,
        "requested_study_dir": requested_study_dir,
        "resolved_study_dir": resolved_study_dir,
        "study_repo_root": study_repo_root,
        "study_id": resolved_study_dir.name,
        "selection_source": selection_source,
        "registry_path": registry_path,
        "active_study": active_study,
        "required_paths": required_paths,
        "missing_required_files": missing_required_files,
        "pipeline_path": pipeline_path,
        "pipeline_present": pipeline_path.exists(),
        "datasets_entries": datasets_entries,
        "study_pipeline": study_pipeline,
        "canonical_usr_root_path": canonical_usr_root_path,
        "dataset_states": dataset_states,
        "dataset_index": dataset_index,
        "missing_declared_present": missing_declared_present,
        "present_but_planned": present_but_planned,
        "execution_surface_states": execution_surface_states,
        "execution_surface_index": execution_surface_index,
        "missing_execution_surfaces": missing_execution_surfaces,
        "phase_states": phase_states,
        "current_phase": current_phase,
        "current_phase_is_known": current_phase_is_known,
        "next_ready_phase": next_ready_phase,
        "next_in_progress_phase": next_in_progress_phase,
        "next_planned_phase": next_planned_phase,
        "blocked_phases": blocked_phases,
        "densegen_dataset_id": densegen_dataset_id,
        "densegen_rows": densegen_rows,
        "densegen_row_target": row_target,
        "densegen_row_gap": densegen_row_gap,
        "evidence": evidence,
    }


def _cluster_run_index_progress(cluster_results_root: Path | None) -> tuple[str, str, dict[str, object]]:
    resolved_root = _required_path(
        cluster_results_root,
        flag_name="--cluster-results-root",
        progress_kind="cluster-run-index",
    )
    index_path = resolved_root / "index.parquet"
    if not index_path.exists():
        return (
            "missing",
            "cluster run index not found",
            {"cluster_results_root": str(resolved_root), "index_path": str(index_path)},
        )

    parquet_file = pq.ParquetFile(str(index_path))
    entry_count = int(parquet_file.metadata.num_rows)
    if entry_count == 0:
        return (
            "attention",
            "cluster run index is present but empty",
            {
                "cluster_results_root": str(resolved_root),
                "index_path": str(index_path),
                "entry_count": 0,
            },
        )

    table = parquet_file.read(columns=["kind", "run_slug", "created_utc", "status", "alias"])
    kind_values = [str(value or "unknown") for value in table.column("kind").to_pylist()]
    status_values = [str(value or "unknown") for value in table.column("status").to_pylist()]
    created_values = [str(value or "") for value in table.column("created_utc").to_pylist()]
    slug_values = [str(value or "<unknown>") for value in table.column("run_slug").to_pylist()]
    alias_values = table.column("alias").to_pylist()

    kind_counts = Counter(kind_values)
    status_counts = Counter(status_values)
    latest_index = max(range(entry_count), key=lambda index: (created_values[index], slug_values[index]))
    latest_kind = kind_values[latest_index]
    latest_slug = slug_values[latest_index]
    latest_status = status_values[latest_index]
    summary = f"{entry_count} cluster run-index entries; latest {latest_kind} {latest_slug} is {latest_status}"
    all_complete = set(status_counts.keys()) <= {"complete"}
    return (
        "ok" if all_complete else "attention",
        summary,
        {
            "cluster_results_root": str(resolved_root),
            "index_path": str(index_path),
            "entry_count": entry_count,
            "kind_counts": dict(sorted(kind_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "latest_entry": {
                "kind": latest_kind,
                "run_slug": latest_slug,
                "status": latest_status,
                "created_utc": created_values[latest_index] or None,
                "alias": alias_values[latest_index],
            },
        },
    )


def _opal_campaign_state_progress(
    *,
    opal_config: Path | None,
    opal_workdir: Path | None,
) -> tuple[str, str, dict[str, object]]:
    if opal_workdir is None:
        resolved_config = _required_path(
            opal_config,
            flag_name="--opal-config or --opal-workdir",
            progress_kind="opal-campaign-state",
        )
        if not resolved_config.exists():
            inferred_workdir = _resolve_opal_campaign_root(resolved_config)
            return (
                "missing",
                "OPAL config not found",
                {
                    "opal_workdir": str(inferred_workdir),
                    "opal_config": str(resolved_config),
                    "state_path": str(inferred_workdir / "state.json"),
                    "ledger_runs_path": str(inferred_workdir / "outputs" / "ledger" / "runs.parquet"),
                },
            )
    workdir, config_path = _resolve_opal_workdir(opal_config=opal_config, opal_workdir=opal_workdir)
    state_path = workdir / "state.json"
    ledger_runs_path = workdir / "outputs" / "ledger" / "runs.parquet"
    if not state_path.exists():
        return (
            "missing",
            "OPAL state.json not found",
            {
                "opal_workdir": str(workdir),
                "opal_config": str(config_path) if config_path is not None else None,
                "state_path": str(state_path),
                "ledger_runs_path": str(ledger_runs_path),
            },
        )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    rounds = sorted(
        list(payload.get("rounds") or []),
        key=lambda round_payload: int(round_payload.get("round_index", -1)),
    )
    latest_round = rounds[-1] if rounds else None
    num_rounds = len(rounds)
    campaign_slug = str(payload.get("campaign_slug") or payload.get("slug") or "")
    if num_rounds == 0:
        summary = "OPAL campaign initialized with no completed rounds yet"
        state = "attention"
    else:
        summary = f"OPAL campaign has {num_rounds} recorded rounds; latest round {latest_round.get('round_index')}"
        state = "ok"
    return (
        state,
        summary,
        {
            "opal_workdir": str(workdir),
            "opal_config": str(config_path) if config_path is not None else None,
            "state_path": str(state_path),
            "ledger_runs_path": str(ledger_runs_path),
            "ledger_runs_present": ledger_runs_path.exists(),
            "campaign_slug": campaign_slug,
            "campaign_name": payload.get("campaign_name") or payload.get("name"),
            "x_column_name": payload.get("x_column_name"),
            "y_column_name": payload.get("y_column_name"),
            "num_rounds": num_rounds,
            "latest_round": {
                "round_index": latest_round.get("round_index"),
                "run_id": latest_round.get("run_id"),
                "round_dir": latest_round.get("round_dir"),
                "selection_top_k_requested": latest_round.get("selection_top_k_requested"),
                "selection_top_k_effective_after_ties": latest_round.get("selection_top_k_effective_after_ties"),
            }
            if latest_round is not None
            else None,
        },
    )


def _resolve_opal_workdir(*, opal_config: Path | None, opal_workdir: Path | None) -> tuple[Path, Path | None]:
    if opal_workdir is not None:
        resolved_config = opal_config.expanduser().resolve() if opal_config else None
        return opal_workdir.expanduser().resolve(), resolved_config
    resolved_config = _required_path(
        opal_config,
        flag_name="--opal-config or --opal-workdir",
        progress_kind="opal-campaign-state",
    )
    if not resolved_config.exists():
        raise ValueError(f"OPAL config not found: {resolved_config}")
    payload = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"OPAL config must be a mapping: {resolved_config}")
    campaign_payload = payload.get("campaign")
    if not isinstance(campaign_payload, dict):
        raise ValueError(f"OPAL config missing 'campaign' mapping: {resolved_config}")
    workdir = str(campaign_payload.get("workdir") or "").strip()
    if not workdir:
        raise ValueError(f"OPAL config missing campaign.workdir: {resolved_config}")
    return _resolve_opal_config_workdir(config_path=resolved_config, workdir=workdir), resolved_config


def _resolve_opal_config_workdir(*, config_path: Path, workdir: str) -> Path:
    workdir_path = Path(workdir).expanduser()
    if workdir_path.is_absolute():
        return workdir_path.resolve()
    campaign_root = _resolve_opal_campaign_root(config_path)
    return (campaign_root / workdir_path).resolve()


def _resolve_opal_campaign_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _ops_audit_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _ops_audit_progress(inputs.audit_json)


def _usr_sync_audit_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _usr_sync_audit_progress(inputs.sync_audit_json)


def _usr_dataset_state_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _usr_dataset_state_progress(usr_root=inputs.usr_root, dataset=inputs.dataset)


def _cluster_run_index_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _cluster_run_index_progress(inputs.cluster_results_root)


def _promoter_study_record_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _promoter_study_record_progress(inputs.study_dir, repo_root=inputs.repo_root)


def _promoter_study_preflight_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _promoter_study_preflight_progress(inputs.study_dir, repo_root=inputs.repo_root)


def _opal_campaign_state_adapter(inputs: ProgressInputs) -> tuple[str, str, dict[str, object]]:
    return _opal_campaign_state_progress(opal_config=inputs.opal_config, opal_workdir=inputs.opal_workdir)


_PROGRESS_KIND_SPECS: dict[str, ProgressKindSpec] = {
    "ops-audit-json": ProgressKindSpec(
        progress_kind="ops-audit-json",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="audit_json",
                cli_flag="--audit-json",
                placeholder="<workspace-root>/outputs/logs/ops/audit/latest.json",
                summary="Workspace-scoped orchestration audit JSON emitted by ops runbook execute.",
            ),
        ),
        adapter=_ops_audit_adapter,
    ),
    "usr-sync-audit": ProgressKindSpec(
        progress_kind="usr-sync-audit",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="sync_audit_json",
                cli_flag="--sync-audit-json",
                placeholder="<audit-dir>/pull.json",
                summary="Machine-readable USR sync audit JSON from usr diff, pull, or push.",
            ),
        ),
        adapter=_usr_sync_audit_adapter,
    ),
    "usr-dataset-state": ProgressKindSpec(
        progress_kind="usr-dataset-state",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="usr_root",
                cli_flag="--usr-root",
                placeholder="<usr-root>",
                summary="USR root containing the target dataset directory.",
            ),
            ProgressFieldSpec(
                manifest_key="dataset",
                cli_flag="--dataset",
                placeholder="<dataset>",
                summary="USR dataset id to summarize.",
            ),
        ),
        adapter=_usr_dataset_state_adapter,
    ),
    "promoter-study-record": ProgressKindSpec(
        progress_kind="promoter-study-record",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="study_dir",
                cli_flag="--study-dir",
                placeholder="docs/studies/promoter/<study-id>",
                summary="Checked-in promoter-study directory containing campaign.yaml, datasets.yaml, and status.md.",
            ),
        ),
        adapter=_promoter_study_record_adapter,
    ),
    "promoter-study-preflight": ProgressKindSpec(
        progress_kind="promoter-study-preflight",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="study_dir",
                cli_flag="--study-dir",
                placeholder="docs/studies/promoter/<study-id>",
                summary="Checked-in promoter-study directory containing campaign.yaml, datasets.yaml, and status.md.",
            ),
        ),
        adapter=_promoter_study_preflight_adapter,
    ),
    "cluster-run-index": ProgressKindSpec(
        progress_kind="cluster-run-index",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="cluster_results_root",
                cli_flag="--cluster-results-root",
                placeholder="<cluster-results-root>",
                summary="Cluster results root containing index.parquet and run outputs.",
            ),
        ),
        adapter=_cluster_run_index_adapter,
    ),
    "opal-campaign-state": ProgressKindSpec(
        progress_kind="opal-campaign-state",
        required_inputs=(
            ProgressFieldSpec(
                manifest_key="opal_config",
                cli_flag="--opal-config",
                placeholder="<opal-workdir>/configs/campaign.yaml",
                summary="Canonical OPAL campaign config used to resolve campaign.workdir.",
            ),
        ),
        adapter=_opal_campaign_state_adapter,
    ),
}


def _load_progress_kind_spec(progress_kind: str) -> ProgressKindSpec:
    try:
        return _PROGRESS_KIND_SPECS[progress_kind]
    except KeyError as exc:
        raise ValueError(
            f"unsupported progress kind: {progress_kind}. Add an explicit read-only adapter before using this surface."
        ) from exc


def _suggest_scaffold_label(registry_id: str, used_labels: Counter[str]) -> str:
    base = registry_id.split(".")[-1].strip() or "step"
    used_labels[base] += 1
    if used_labels[base] == 1:
        return base
    return f"{base}-{used_labels[base]}"


def _namespace_column_counts(columns: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for column in columns:
        if "__" not in column:
            continue
        namespace = column.split("__", 1)[0].strip()
        if namespace:
            counts[namespace] += 1
    return counts


def _overlay_namespace_names(dataset_dir: Path) -> list[str]:
    derived_dir = dataset_dir / "_derived"
    if not derived_dir.exists() or not derived_dir.is_dir():
        return []
    namespaces: list[str] = []
    for entry in sorted(derived_dir.iterdir(), key=lambda item: item.name):
        if entry.is_file() and entry.suffix == ".parquet":
            namespaces.append(entry.stem)
            continue
        if entry.is_dir() and any(entry.glob("part-*.parquet")):
            namespaces.append(entry.name)
    return namespaces


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _file_count(path: Path) -> int:
    return sum(1 for candidate in path.rglob("*") if candidate.is_file())


def _required_path(
    path: Path | None,
    *,
    flag_name: str,
    progress_kind: str,
    base_dir: Path | None = None,
) -> Path:
    if path is None:
        raise ValueError(f"progress kind '{progress_kind}' requires {flag_name}")
    return _resolve_input_path(path, base_dir=base_dir)


def _resolve_input_path(path: Path, *, base_dir: Path | None = None) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    if base_dir is not None:
        return (base_dir / expanded).resolve()
    return expanded.resolve()


def _required_text(value: str | None, *, flag_name: str, progress_kind: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"progress kind '{progress_kind}' requires {flag_name}")
    return str(value).strip()


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping: {path}")
    return payload


def _discover_active_promoter_study_dir(
    *,
    repo_root: Path | None,
    progress_kind: str = "promoter-study-record",
) -> tuple[Path, Path, str]:
    resolved_repo_root = repo_root
    if resolved_repo_root is None:
        resolved_repo_root = discover_repo_root(Path.cwd())
    if resolved_repo_root is None:
        raise ValueError(
            f"progress kind '{progress_kind}' requires --study-dir or a dnadesign repository checkout "
            "with docs/studies/promoter/index.yaml"
        )

    promoter_index_path = resolved_repo_root / "docs" / "studies" / "promoter" / "index.yaml"
    if not promoter_index_path.exists():
        raise ValueError(f"promoter-study registry not found: {promoter_index_path}")

    promoter_index = _load_yaml_mapping(promoter_index_path, label="promoter index")
    active_study = _string_or_none(promoter_index.get("active_study"))
    if active_study is None:
        raise ValueError(f"promoter-study registry does not declare active_study: {promoter_index_path}")

    studies_payload = promoter_index.get("studies") or []
    if not isinstance(studies_payload, list):
        raise ValueError(f"promoter-study registry must define a 'studies' list: {promoter_index_path}")

    matching_entries = [
        entry
        for entry in studies_payload
        if isinstance(entry, dict) and _string_or_none(entry.get("study_id")) == active_study
    ]
    if not matching_entries:
        raise ValueError(f"active_study '{active_study}' is not declared under 'studies' in {promoter_index_path}")
    if len(matching_entries) > 1:
        raise ValueError(f"active_study '{active_study}' is declared more than once in {promoter_index_path}")

    raw_path = _required_metadata_text(
        matching_entries[0].get("path"),
        label="study path",
        source=promoter_index_path,
    )
    resolved_study_dir = _resolve_repo_relative_path(
        repo_root=resolved_repo_root,
        raw_path=raw_path,
        progress_kind=progress_kind,
    )
    return resolved_study_dir, promoter_index_path, active_study


def _resolve_repo_relative_path(
    *,
    repo_root: Path,
    raw_path: str | None,
    progress_kind: str = "promoter-study-record",
) -> Path:
    normalized = _required_text(raw_path, flag_name="<repo-relative-path>", progress_kind=progress_kind)
    path = Path(normalized).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _resolve_named_path_mapping(
    payload: object,
    *,
    repo_root: Path,
    label: str,
    progress_kind: str,
) -> dict[str, Path]:
    if payload and not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    resolved: dict[str, Path] = {}
    for name, raw_path in _flatten_named_paths(payload or {}):
        resolved[name] = _resolve_repo_relative_path(
            repo_root=repo_root,
            raw_path=raw_path,
            progress_kind=progress_kind,
        )
    return resolved


def _build_infer_notify_setup_command(*, config_path: Path, profile_path: Path) -> str:
    return _render_argv(
        (
            "uv",
            "run",
            "notify",
            "setup",
            "slack",
            "--profile",
            str(profile_path),
            "--tool",
            "infer",
            "--config",
            str(config_path),
            "--secret-source",
            "file",
            "--secret-ref",
            "file://$NOTIFY_WEBHOOK_FILE",
        )
    )


def _run_progress_command(argv: Sequence[str], *, cwd: Path, timeout_seconds: int = 180) -> CommandExecution:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandExecution(
            argv=tuple(str(token) for token in argv),
            cwd=str(cwd),
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            timed_out=True,
        )
    return CommandExecution(
        argv=tuple(str(token) for token in argv),
        cwd=str(cwd),
        returncode=int(completed.returncode),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
        timed_out=False,
    )


def _preflight_command_check(
    *,
    check_id: str,
    phase: str,
    summary: str,
    execution: CommandExecution,
    details: dict[str, object] | None = None,
    override_state: str | None = None,
) -> dict[str, object]:
    state = override_state
    if state is None:
        state = "attention" if execution.returncode != 0 or execution.timed_out else "ok"
    if execution.timed_out:
        summary = f"timed out: {summary}"
    return {
        "id": check_id,
        "phase": phase,
        "state": state,
        "summary": summary,
        "command": _render_argv(execution.argv),
        "cwd": execution.cwd,
        "returncode": execution.returncode,
        "timed_out": execution.timed_out,
        "stdout_tail": _trim_command_output(execution.stdout),
        "stderr_tail": _trim_command_output(execution.stderr),
        "details": details or {},
    }


def _preflight_state_check(
    *,
    check_id: str,
    phase: str,
    state: str,
    summary: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": check_id,
        "phase": phase,
        "state": state,
        "summary": summary,
        "command": None,
        "cwd": None,
        "returncode": None,
        "timed_out": False,
        "stdout_tail": None,
        "stderr_tail": None,
        "details": details or {},
    }


def _render_argv(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in argv)


def _trim_command_output(text: str | None, *, max_lines: int = 8, max_chars: int = 1200) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    lines = raw.splitlines()
    if len(lines) > max_lines:
        raw = "\n".join(lines[-max_lines:])
    if len(raw) > max_chars:
        raw = raw[-max_chars:]
    return raw


def _first_nonempty_line(text: str | None) -> str | None:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _choose_command_summary(execution: CommandExecution, *, fallback: str) -> str:
    candidate_texts = (
        (execution.stderr, execution.stdout) if execution.returncode != 0 else (execution.stdout, execution.stderr)
    )
    for text in candidate_texts:
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in {"{", "}", "[", "]"}:
                continue
            if stripped.startswith("WARNING:") or stripped.startswith("W0000 "):
                continue
            return stripped
    return fallback


def _safe_json_loads(text: str | None) -> dict[str, object] | None:
    payload = str(text or "").strip()
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _load_orchestration_runbook_payload(path: Path) -> dict[str, object]:
    payload = _load_yaml_mapping(path, label="runbook yaml")
    runbook = payload.get("runbook") or {}
    if not isinstance(runbook, dict):
        raise ValueError(f"runbook yaml must define a 'runbook' mapping: {path}")
    return runbook


def _infer_usr_dataset_requirements(config_path: Path) -> list[dict[str, object]]:
    payload = _load_yaml_mapping(config_path, label="infer config")
    jobs_payload = payload.get("jobs") or []
    if jobs_payload and not isinstance(jobs_payload, list):
        raise ValueError(f"infer config must define a 'jobs' list: {config_path}")
    requirements: list[dict[str, object]] = []
    for job_payload in jobs_payload:
        if not isinstance(job_payload, dict):
            raise ValueError(f"infer config job entry must be a mapping: {config_path}")
        ingest_payload = job_payload.get("ingest") or {}
        if not isinstance(ingest_payload, dict):
            raise ValueError(f"infer config job ingest must be a mapping: {config_path}")
        if _string_or_none(ingest_payload.get("source")) != "usr":
            continue
        dataset_id = _required_metadata_text(ingest_payload.get("dataset"), label="ingest dataset", source=config_path)
        raw_root = _required_metadata_text(ingest_payload.get("root"), label="ingest root", source=config_path)
        root_path = _resolve_input_path(Path(raw_root), base_dir=config_path.parent)
        records_path = (root_path / dataset_id / "records.parquet").resolve()
        requirements.append(
            {
                "job_id": _string_or_none(job_payload.get("id")),
                "dataset": dataset_id,
                "usr_root": str(root_path),
                "records_path": str(records_path),
                "exists": records_path.exists(),
            }
        )
    return requirements


def _parquet_row_count(records_path: Path) -> int:
    return int(pq.ParquetFile(str(records_path)).metadata.num_rows)


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"expected integer value, received: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"expected non-negative integer value, received: {value!r}")
    return parsed


def _first_phase_by_status(phases: Sequence[dict[str, object]], *, status: str) -> dict[str, object] | None:
    for phase in phases:
        if phase.get("status") == status:
            return dict(phase)
    return None


def _flatten_named_paths(payload: object, *, prefix: str = "") -> tuple[tuple[str, str], ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        return (((prefix or "path"), payload),)
    if not isinstance(payload, dict):
        raise ValueError("execution_surfaces entries must be strings or nested mappings")
    flattened: list[tuple[str, str]] = []
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("execution_surfaces keys must be non-empty strings")
        next_prefix = f"{prefix}.{key}" if prefix else key
        flattened.extend(_flatten_named_paths(value, prefix=next_prefix))
    return tuple(flattened)


def _required_metadata_text(value: object, *, label: str, source: Path) -> str:
    text = _string_or_none(value)
    if text is None:
        raise ValueError(f"{label} is required in {source}")
    return text


def _path_or_none(value: object, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if base_dir is not None and not path.is_absolute():
        return base_dir / path
    return path


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
