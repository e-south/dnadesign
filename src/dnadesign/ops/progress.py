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
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from .catalog import (
    RunbookCatalog,
    load_catalog_procedure_owner_boundary,
    load_catalog_related_registry_ids,
    resolve_catalog_procedure_entry,
)

_PROGRESS_STATES = frozenset({"ok", "attention", "missing"})


@dataclass(frozen=True)
class ProgressInputs:
    audit_json: Path | None = None
    sync_audit_json: Path | None = None
    usr_root: Path | None = None
    dataset: str | None = None
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
                    audit_json=_path_or_none(step_payload.get("audit_json"), base_dir=manifest_dir),
                    sync_audit_json=_path_or_none(step_payload.get("sync_audit_json"), base_dir=manifest_dir),
                    usr_root=_path_or_none(step_payload.get("usr_root"), base_dir=manifest_dir),
                    dataset=_string_or_none(step_payload.get("dataset")),
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


def _required_path(path: Path | None, *, flag_name: str, progress_kind: str) -> Path:
    if path is None:
        raise ValueError(f"progress kind '{progress_kind}' requires {flag_name}")
    return path.expanduser().resolve()


def _required_text(value: str | None, *, flag_name: str, progress_kind: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"progress kind '{progress_kind}' requires {flag_name}")
    return str(value).strip()


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
