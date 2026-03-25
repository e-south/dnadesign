"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/promoter/preflight_infer.py

Study-owned infer and notify preflight builders for the
promoter family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.preflight import (
    CommandCheckDependencies,
    CommandCheckTarget,
    CommandExecution,
    PreflightCheck,
    build_command_checks,
    build_state_check,
)

from .infer_runtime import PromoterStudyInferRuntimeResolvedContext


@dataclass(frozen=True)
class PromoterPreflightInferDependencies:
    inspect_local_gpu_inventory: Callable[[], dict[str, object]]
    infer_usr_dataset_requirements: Callable[[Path], list[dict[str, object]]]
    build_infer_notify_setup_command: Callable[[Path], str]
    run_preflight_command: Callable[..., CommandExecution]
    choose_command_summary: Callable[..., str]
    validate_infer_config_contract: Callable[[Path], object] | None = None
    validate_infer_dry_run_contract: Callable[[Path], object] | None = None
    resolve_infer_usr_output_contract: Callable[[Path], object] | None = None


@dataclass(frozen=True)
class PromoterPreflightInferChecksResult:
    checks: tuple[PreflightCheck, ...]
    evidence_updates: dict[str, object]


def build_promoter_preflight_infer_checks(
    *,
    study_repo_root: Path,
    infer_runtime: PromoterStudyInferRuntimeResolvedContext,
    infer_preparation_phase_id: str,
    enabled_groups: Collection[str],
    dependencies: PromoterPreflightInferDependencies,
) -> PromoterPreflightInferChecksResult:
    include_infer_checks = "infer" in enabled_groups
    include_notify_checks = "notify" in enabled_groups
    local_gpu_inventory = (
        dependencies.inspect_local_gpu_inventory()
        if include_infer_checks
        else {"count": 0, "devices": [], "probe_error": None}
    )
    runtime_model_summaries = {summary.label: summary for summary in infer_runtime.runtime_model_summaries}

    checks: list[PreflightCheck] = []
    evidence_updates = {
        "preferred_infer_model_family": infer_runtime.preferred_model_family,
        "supported_model_families": list(infer_runtime.supported_model_families),
        "infer_local_gpu_inventory": local_gpu_inventory,
        "infer_notify_profiles": {label: str(path) for label, path in infer_runtime.infer_notify_profile_paths.items()},
        "infer_notify_profile_errors": dict(infer_runtime.infer_notify_profile_errors),
    }

    validate_infer_config_contract = dependencies.validate_infer_config_contract
    validate_infer_dry_run_contract = dependencies.validate_infer_dry_run_contract
    resolve_infer_usr_output_contract = dependencies.resolve_infer_usr_output_contract

    if include_infer_checks and validate_infer_config_contract is None:
        raise ValueError("validate_infer_config_contract dependency is required when include_infer_checks=true")
    if include_infer_checks and validate_infer_dry_run_contract is None:
        raise ValueError("validate_infer_dry_run_contract dependency is required when include_infer_checks=true")
    if include_notify_checks and resolve_infer_usr_output_contract is None:
        raise ValueError("resolve_infer_usr_output_contract dependency is required when include_notify_checks=true")

    for config_label, config_path in sorted(infer_runtime.infer_config_paths.items()):
        if not include_infer_checks:
            continue
        try:
            config_contract = validate_infer_config_contract(config_path)
            checks.append(
                build_state_check(
                    check_id=f"infer.validate.{config_label}",
                    check_group="infer",
                    phase="infer",
                    phase_id=infer_runtime.config_phase_ids.get(config_label, infer_preparation_phase_id),
                    state="ok",
                    summary="infer config validation completed",
                    details={
                        "config": str(config_path),
                        "model_id": getattr(config_contract, "model_id", None),
                        "device": getattr(config_contract, "device", None),
                        "job_ids": list(getattr(config_contract, "job_ids", ())),
                        "usr_datasets": list(getattr(config_contract, "usr_datasets", ())),
                    },
                )
            )
        except Exception as exc:
            checks.append(
                build_state_check(
                    check_id=f"infer.validate.{config_label}",
                    check_group="infer",
                    phase="infer",
                    phase_id=infer_runtime.config_phase_ids.get(config_label, infer_preparation_phase_id),
                    state="attention",
                    summary=str(exc),
                    details={"config": str(config_path)},
                )
            )

    for runtime_lane in infer_runtime.runtime_lane_contracts:
        runtime_label = str(getattr(runtime_lane, "runtime_label"))
        config_path = Path(getattr(runtime_lane, "config_path"))
        runtime_phase_id = infer_runtime.runtime_phase_ids.get(runtime_label, infer_preparation_phase_id)
        model_summary = runtime_model_summaries.get(runtime_label)
        if model_summary is None:
            raise ValueError(f"missing infer runtime model summary for {runtime_label}")
        if include_infer_checks:
            checks.extend(
                _build_infer_runtime_checks(
                    runtime_label=runtime_label,
                    config_path=config_path,
                    runtime_phase_id=runtime_phase_id,
                    model_summary=model_summary.as_dict(),
                    local_gpu_inventory=local_gpu_inventory,
                    validate_infer_dry_run_contract=validate_infer_dry_run_contract,
                    infer_usr_dataset_requirements=dependencies.infer_usr_dataset_requirements,
                )
            )
        if include_notify_checks:
            checks.extend(
                _build_notify_runtime_checks(
                    study_repo_root=study_repo_root,
                    runtime_label=runtime_label,
                    config_path=config_path,
                    runtime_phase_id=runtime_phase_id,
                    infer_notify_profile_paths=infer_runtime.infer_notify_profile_paths,
                    runtime_infer_notify_profile_errors=infer_runtime.infer_notify_profile_errors,
                    resolve_infer_usr_output_contract=resolve_infer_usr_output_contract,
                    build_infer_notify_setup_command=dependencies.build_infer_notify_setup_command,
                    run_preflight_command=dependencies.run_preflight_command,
                    choose_command_summary=dependencies.choose_command_summary,
                )
            )

    return PromoterPreflightInferChecksResult(
        checks=tuple(checks),
        evidence_updates=evidence_updates,
    )


def _build_infer_runtime_checks(
    *,
    runtime_label: str,
    config_path: Path,
    runtime_phase_id: str,
    model_summary: Mapping[str, object],
    local_gpu_inventory: Mapping[str, object],
    validate_infer_dry_run_contract: Callable[[Path], object],
    infer_usr_dataset_requirements: Callable[[Path], list[dict[str, object]]],
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    requires_gpu = str(model_summary.get("device") or "").startswith("cuda")
    local_gpu_count = int(local_gpu_inventory.get("count") or 0)
    if requires_gpu and local_gpu_count == 0:
        checks.append(
            build_state_check(
                check_id=f"infer.local_runtime.{runtime_label}",
                check_group="infer",
                phase="infer",
                phase_id=runtime_phase_id,
                state="attention",
                summary="requires GPU host for direct infer execution; current host has no local GPUs",
                details={
                    "config": str(config_path),
                    "model_id": model_summary.get("model_id"),
                    "device": model_summary.get("device"),
                    "local_gpu_inventory": dict(local_gpu_inventory),
                },
            )
        )
    else:
        checks.append(
            build_state_check(
                check_id=f"infer.local_runtime.{runtime_label}",
                check_group="infer",
                phase="infer",
                phase_id=runtime_phase_id,
                state="ok",
                summary=(
                    "local GPU inventory detected for direct infer execution"
                    if requires_gpu
                    else "config does not require a GPU host for direct execution"
                ),
                details={
                    "config": str(config_path),
                    "model_id": model_summary.get("model_id"),
                    "device": model_summary.get("device"),
                    "local_gpu_inventory": dict(local_gpu_inventory),
                },
            )
        )

    usr_inputs = infer_usr_dataset_requirements(config_path)
    missing_usr_inputs = [entry for entry in usr_inputs if not bool(entry.get("exists"))]
    if missing_usr_inputs:
        checks.append(
            build_state_check(
                check_id=f"infer.dry_run.{runtime_label}",
                check_group="infer",
                phase="infer",
                phase_id=runtime_phase_id,
                state="missing",
                summary="requires study-owned USR datasets before infer dry-run",
                details={
                    "config": str(config_path),
                    "missing_usr_inputs": missing_usr_inputs,
                },
            )
        )
        return tuple(checks)

    try:
        dry_run_contract = validate_infer_dry_run_contract(config_path)
        checks.append(
            build_state_check(
                check_id=f"infer.dry_run.{runtime_label}",
                check_group="infer",
                phase="infer",
                phase_id=runtime_phase_id,
                state="ok",
                summary="infer dry-run contract completed",
                details={
                    "config": str(config_path),
                    "model_id": getattr(dry_run_contract, "model_id", None),
                    "device": getattr(dry_run_contract, "device", None),
                    "job_ids": list(getattr(dry_run_contract, "job_ids", ())),
                },
            )
        )
    except Exception as exc:
        checks.append(
            build_state_check(
                check_id=f"infer.dry_run.{runtime_label}",
                check_group="infer",
                phase="infer",
                phase_id=runtime_phase_id,
                state="attention",
                summary=str(exc),
                details={"config": str(config_path)},
            )
        )
    return tuple(checks)


def _build_notify_runtime_checks(
    *,
    study_repo_root: Path,
    runtime_label: str,
    config_path: Path,
    runtime_phase_id: str,
    infer_notify_profile_paths: Mapping[str, Path],
    runtime_infer_notify_profile_errors: Mapping[str, str],
    resolve_infer_usr_output_contract: Callable[[Path], object],
    build_infer_notify_setup_command: Callable[[Path], str],
    run_preflight_command: Callable[..., CommandExecution],
    choose_command_summary: Callable[..., str],
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    profile_path = infer_notify_profile_paths.get(runtime_label)
    if runtime_label in runtime_infer_notify_profile_errors:
        checks.append(
            build_state_check(
                check_id=f"notify.profile.{runtime_label}",
                check_group="notify",
                phase="notify",
                phase_id=runtime_phase_id,
                state="attention",
                summary=runtime_infer_notify_profile_errors[runtime_label],
                details={"config": str(config_path)},
            )
        )
    elif profile_path is None:
        checks.append(
            build_state_check(
                check_id=f"notify.profile.{runtime_label}",
                check_group="notify",
                phase="notify",
                phase_id=runtime_phase_id,
                state="attention",
                summary="infer notify profile path could not be derived from config",
                details={"config": str(config_path)},
            )
        )
    elif not profile_path.is_file():
        checks.append(
            build_state_check(
                check_id=f"notify.profile.{runtime_label}",
                check_group="notify",
                phase="notify",
                phase_id=runtime_phase_id,
                state="attention",
                summary="infer notify profile is not materialized yet",
                details={
                    "config": str(config_path),
                    "profile": str(profile_path),
                    "setup_command": build_infer_notify_setup_command(config_path),
                    "tls_note": "Export SSL_CERT_FILE before `notify profile doctor` or live delivery.",
                },
            )
        )
    else:
        checks.extend(
            build_command_checks(
                targets=(
                    CommandCheckTarget(
                        check_id=f"notify.profile.{runtime_label}",
                        check_group="notify",
                        phase="notify",
                        phase_id=runtime_phase_id,
                        argv=("uv", "run", "notify", "profile", "doctor", "--profile", str(profile_path), "--json"),
                        cwd=study_repo_root,
                        fallback_summary="infer notify profile doctor completed",
                        details={
                            "config": str(config_path),
                            "profile": str(profile_path),
                        },
                    ),
                ),
                dependencies=CommandCheckDependencies(
                    run_preflight_command=run_preflight_command,
                    choose_command_summary=choose_command_summary,
                ),
            )
        )

    try:
        resolved_output = resolve_infer_usr_output_contract(config_path)
        usr_root = Path(str(getattr(resolved_output, "usr_root"))).resolve()
        usr_dataset = str(getattr(resolved_output, "usr_dataset"))
        events_path = (usr_root / usr_dataset / ".events.log").resolve()
        resolve_state = "ok" if events_path.exists() else "missing"
        resolve_summary = (
            f"resolved infer events path: {events_path}"
            if events_path.exists()
            else f"resolved events path is not materialized yet: {events_path}"
        )
        checks.append(
            build_state_check(
                check_id=f"notify.resolve_events.{runtime_label}",
                check_group="notify",
                phase="notify",
                phase_id=runtime_phase_id,
                state=resolve_state,
                summary=resolve_summary,
                details={
                    "config": str(config_path),
                    "events": str(events_path),
                    "events_exists": events_path.exists(),
                    "policy": "infer",
                },
            )
        )
    except Exception as exc:
        checks.append(
            build_state_check(
                check_id=f"notify.resolve_events.{runtime_label}",
                check_group="notify",
                phase="notify",
                phase_id=runtime_phase_id,
                state="attention",
                summary=str(exc),
                details={"config": str(config_path)},
            )
        )
    return tuple(checks)


__all__ = [
    "PromoterPreflightInferChecksResult",
    "PromoterPreflightInferDependencies",
    "build_promoter_preflight_infer_checks",
]
