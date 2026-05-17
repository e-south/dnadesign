"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/status_adapters/promoter_status/adapter.py

Promoter study status adapter for OPS status and preflight surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.preflight import (
    choose_command_summary,
    execute_runbook_plan,
    run_preflight_command,
    safe_json_loads,
)
from dnadesign.ops.status import (
    load_yaml_mapping,
    optional_positive_int,
    parquet_row_count,
    required_metadata_text,
    required_path,
    resolve_named_path_mapping,
    resolve_repo_relative_path,
    string_list_or_empty,
    string_or_none,
)
from dnadesign.studies.core.models import StudyStatusAdapter, StudyStatusContext
from dnadesign.studies.core.record_locator import discover_active_study_selection
from dnadesign.usr import Dataset, SequenceViewContractExpectation, validate_sequence_view_contract

from .analysis_surfaces import inspect_promoter_exploratory_analysis
from .downstream_surfaces import inspect_promoter_downstream_surfaces
from .infer_runtime import PromoterStudyInferRuntimeDependencies
from .latentdna_readiness import inspect_promoter_latentdna_readiness
from .preflight import (
    PromoterPreflightContextDependencies,
    PromoterPreflightCoordinatorDependencies,
    build_promoter_preflight_progress,
    resolve_promoter_preflight_context,
)
from .record_normalizer import PromoterStudyContextDependencies, PromoterStudyResolvedContext
from .record_normalizer import resolve_promoter_study_context as resolve_checked_in_promoter_study_context
from .snapshot import (
    PromoterStudyStatusDependencies,
    build_promoter_study_status,
    resolve_promoter_study_status_context,
)


@dataclass(frozen=True)
class PromoterStatusAdapterContext:
    study_context: PromoterStudyResolvedContext


class PromoterStudyStatusAdapter(StudyStatusAdapter):
    status_kind = "promoter-study-status"
    preflight_kind = "promoter-study-preflight"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_checked_in_promoter_study_context(
            study_root,
            repo_root=repo_root,
            status_kind="promoter-study-status",
            dependencies=PromoterStudyContextDependencies(
                discover_active_study_dir=discover_active_study_dir,
                required_path=required_path,
                discover_repo_root=discover_repo_root,
                load_yaml_mapping=load_yaml_mapping,
                resolve_repo_relative_path=resolve_repo_relative_path,
                resolve_named_path_mapping=resolve_named_path_mapping,
                string_or_none=string_or_none,
                optional_positive_int=optional_positive_int,
                required_metadata_text=required_metadata_text,
                parquet_row_count=parquet_row_count,
            ),
        )
        contract = study_context.ops_contract
        if contract is None:
            raise ValueError(
                f"study record missing ops.study.yaml: {study_context.resolved_study_dir / 'ops.study.yaml'}"
            )
        if contract.status_kind != self.status_kind:
            raise ValueError(
                f"ops.study.yaml ops_surfaces.status_kind mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.status_kind}, found {contract.status_kind}"
            )
        if contract.preflight_kind != self.preflight_kind:
            raise ValueError(
                f"ops.study.yaml ops_surfaces.preflight_kind mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.preflight_kind}, found {contract.preflight_kind}"
            )
        if contract.study_id != study_context.study_id:
            raise ValueError(
                f"ops.study.yaml study_id mismatch for {study_context.resolved_study_dir}: "
                f"expected {study_context.study_id}, found {contract.study_id}"
            )
        if study_context.study_repo_root is None:
            raise ValueError("promoter study context requires a resolved study_repo_root")
        return StudyStatusContext(
            repo_root=study_context.study_repo_root,
            study_root=study_context.resolved_study_dir,
            contract=contract,
            adapter_context=PromoterStatusAdapterContext(study_context=study_context),
        )

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]:
        study_context = _study_adapter_context(context).study_context
        evidence = dict(study_context.evidence)
        evidence["ops_study_contract"] = dict(context.contract.raw_payload)
        missing_result = _missing_promoter_study_result(context=study_context, evidence=evidence)
        if missing_result is not None:
            return missing_result

        status_dependencies = PromoterStudyStatusDependencies(
            infer_runtime=build_promoter_study_infer_runtime_dependencies(),
            phase_matches_infer_model_family=phase_matches_infer_model_family,
            inspect_semantic_completeness=inspect_promoter_study_semantic_completeness,
            inspect_sequence_view_contracts=inspect_promoter_study_sequence_view_contracts,
            inspect_latentdna_readiness=inspect_promoter_latentdna_readiness,
            inspect_additional_downstream_surfaces=inspect_promoter_downstream_surfaces,
            inspect_exploratory_analysis=inspect_promoter_exploratory_analysis,
        )
        status_context = resolve_promoter_study_status_context(
            study_context=study_context,
            status_kind="promoter-study-status",
            dependencies=status_dependencies,
        )
        return build_promoter_study_status(
            study_context=study_context,
            status_context=status_context,
            dependencies=status_dependencies,
            summary_scope=context.contract.snapshot_summary_scope,
        )

    def build_preflight(
        self,
        context: StudyStatusContext,
        *,
        scope: str | None,
        command_timeout_seconds: object | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        study_context = _study_adapter_context(context).study_context
        evidence = dict(study_context.evidence)
        evidence["ops_study_contract"] = dict(context.contract.raw_payload)
        resolved_command_timeout_seconds = _resolve_preflight_command_timeout_seconds(command_timeout_seconds)
        if resolved_command_timeout_seconds is not None:
            evidence["command_timeout_seconds"] = resolved_command_timeout_seconds
        missing_result = _missing_promoter_study_result(context=study_context, evidence=evidence)
        if missing_result is not None:
            return missing_result

        resolved_context = resolve_promoter_preflight_context(
            study_context=study_context,
            scope=scope,
            status_kind="promoter-study-preflight",
            contract=context.contract,
            dependencies=PromoterPreflightContextDependencies(
                infer_runtime=build_promoter_study_infer_runtime_dependencies(),
                environ=os.environ,
            ),
        )

        return build_promoter_preflight_progress(
            context=resolved_context,
            evidence=evidence,
            dependencies=PromoterPreflightCoordinatorDependencies(
                run_preflight_command=_build_preflight_command_runner(resolved_command_timeout_seconds),
                execute_runbook_plan=execute_runbook_plan,
                safe_json_loads=safe_json_loads,
                choose_command_summary=choose_command_summary,
                inspect_local_gpu_inventory=inspect_local_infer_gpu_inventory,
                environ=os.environ,
            ),
        )


STUDY_STATUS_ADAPTER = PromoterStudyStatusAdapter()


def _resolve_preflight_command_timeout_seconds(value: object | None) -> int | None:
    resolved = optional_positive_int(value)
    if resolved is None:
        return None
    if resolved <= 0:
        raise ValueError("promoter-study-preflight command_timeout_seconds must be greater than zero")
    return resolved


def _build_preflight_command_runner(command_timeout_seconds: int | None):
    def _run_preflight_command(
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: int | float = 180,
    ):
        return run_preflight_command(
            argv,
            cwd=cwd,
            timeout_seconds=command_timeout_seconds if command_timeout_seconds is not None else timeout_seconds,
        )

    return _run_preflight_command


def discover_active_study_dir(
    *,
    repo_root: Path | None,
    status_kind: str = "promoter-study-status",
) -> tuple[Path, Path, str]:
    selection = discover_active_study_selection(
        repo_root=repo_root,
        status_kind=status_kind,
    )
    return selection.study_root, selection.index_path, selection.active_study_id


def build_promoter_study_infer_runtime_dependencies() -> PromoterStudyInferRuntimeDependencies:
    from dnadesign.infer import resolve_infer_runtime_lane_contracts

    return PromoterStudyInferRuntimeDependencies(
        resolve_named_path_mapping=resolve_named_path_mapping,
        resolve_infer_runtime_lane_contracts=resolve_infer_runtime_lane_contracts,
        derive_infer_notify_profile_paths=derive_infer_notify_profile_paths,
        load_infer_model_summary=load_infer_model_summary,
        string_or_none=string_or_none,
        string_list_or_empty=string_list_or_empty,
    )


def inspect_local_infer_gpu_inventory() -> dict[str, object]:
    try:
        from dnadesign.infer import inspect_local_gpu_inventory

        payload = inspect_local_gpu_inventory()
    except Exception as exc:
        return {"count": 0, "devices": [], "probe_error": str(exc)}
    if not isinstance(payload, dict):
        return {"count": 0, "devices": [], "probe_error": "infer.inspect_local_gpu_inventory returned invalid data"}
    devices = payload.get("devices")
    resolved_devices = list(devices) if isinstance(devices, list) else []
    return {
        "count": int(payload.get("count") or len(resolved_devices)),
        "devices": resolved_devices,
        "probe_error": string_or_none(payload.get("probe_error")),
    }


def derive_infer_notify_profile_paths(
    infer_config_paths: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, str]]:
    if not infer_config_paths:
        return {}, {}
    from dnadesign.infer.contracts import resolve_infer_notify_profile_path

    derived_paths: dict[str, Path] = {}
    derivation_errors: dict[str, str] = {}
    for config_label, config_path in infer_config_paths.items():
        try:
            derived_paths[config_label] = resolve_infer_notify_profile_path(config_path)
        except Exception as exc:
            derivation_errors[config_label] = str(exc)
    return derived_paths, derivation_errors


def load_infer_model_summary(config_path: Path) -> dict[str, object]:
    payload = load_yaml_mapping(config_path, label="infer config")
    model_payload = payload.get("model") or {}
    if not isinstance(model_payload, dict):
        raise ValueError(f"infer config must define a model mapping: {config_path}")
    return {
        "model_id": string_or_none(model_payload.get("id")),
        "device": string_or_none(model_payload.get("device")) or "unknown",
    }


def phase_matches_infer_model_family(*, phase_id: str, model_family: str | None) -> bool:
    from dnadesign.infer import infer_model_family_suffix

    suffix = infer_model_family_suffix(model_family)
    return suffix is not None and suffix in phase_id


def inspect_promoter_study_semantic_completeness(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, object] | None:
    root = study_context.canonical_usr_root_path
    expected_rows = study_context.densegen_rows
    if root is None or expected_rows is None:
        return None
    if study_context.densegen_dataset_id is None:
        return None

    try:
        source_overlay = _overlay_guardrail_state(
            root=root,
            dataset_id=study_context.densegen_dataset_id,
            namespace="densegen",
        )
        dataset_checks = []
        if study_context.merged_anchor_dataset_id is not None:
            dataset_checks.append(
                _densegen_metadata_projection_state(
                    root=root,
                    dataset_id=study_context.merged_anchor_dataset_id,
                    expected_rows=int(expected_rows),
                    label="anchor",
                )
            )
        if study_context.construct_context_dataset_id is not None:
            dataset_checks.append(
                _densegen_metadata_projection_state(
                    root=root,
                    dataset_id=study_context.construct_context_dataset_id,
                    expected_rows=int(expected_rows),
                    label="construct",
                )
            )
    except Exception as exc:
        return {
            "state": "attention",
            "drives_top_level_attention": True,
            "summary": f"semantic completeness probe failed: {exc}",
            "probe_error": str(exc),
        }

    attention = bool(source_overlay.get("state") == "attention") or any(
        str(check.get("state") or "") == "attention" for check in dataset_checks
    )
    summary_parts = [str(source_overlay["summary"]), *[str(check["summary"]) for check in dataset_checks]]
    return {
        "state": "attention" if attention else "ok",
        "drives_top_level_attention": attention,
        "source_overlay_state": source_overlay,
        "dataset_checks": dataset_checks,
        "summary": "; ".join(summary_parts),
    }


def inspect_promoter_study_sequence_view_contracts(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, object] | None:
    contract = study_context.ops_contract
    root = study_context.canonical_usr_root_path
    if contract is None or root is None:
        return None
    specs = _preflight_specs_by_kind(contract=contract, kind="sequence_view_contract")
    if not specs:
        return None

    checks: list[dict[str, object]] = []
    product_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    pooling_counts: Counter[str] = Counter()
    stale_or_incomplete: list[str] = []
    for spec in specs:
        check_id = str(spec.get("check_id") or "").strip()
        artifact_id = str(spec.get("artifact") or "").strip()
        required = bool(spec.get("required", True))
        dataset_id = string_or_none((contract.artifacts.get(artifact_id) or {}).get("dataset_id"))
        base_check = {
            "check_id": check_id,
            "artifact": artifact_id,
            "dataset": dataset_id,
            "required": required,
        }
        if dataset_id is None:
            checks.append(
                {
                    **base_check,
                    "state": "attention",
                    "summary": f"sequence-view contract check {check_id} references unknown dataset artifact",
                    "errors": [f"artifact {artifact_id!r} does not define dataset_id"],
                    "generated_artifact_freshness": "stale_or_incomplete",
                }
            )
            continue
        try:
            report = validate_sequence_view_contract(
                Dataset(root, dataset_id),
                expectation=_sequence_view_expectation_from_payload(spec.get("expected")),
                raise_on_error=False,
            )
            product_counts.update(report.counts_by_product_kind)
            orientation_counts.update(report.counts_by_orientation)
            pooling_counts.update(report.counts_by_recommended_pooling)
            state = "ok" if report.ok else "attention"
            if not report.ok:
                stale_or_incomplete.append(dataset_id)
            checks.append(
                {
                    **base_check,
                    "state": state,
                    "total_records": report.total_records,
                    "total_views": report.total_views,
                    "counts_by_product_kind": report.counts_by_product_kind,
                    "counts_by_orientation": report.counts_by_orientation,
                    "counts_by_context_kind": report.counts_by_context_kind,
                    "counts_by_recommended_pooling": report.counts_by_recommended_pooling,
                    "invalid_bounds": report.invalid_bounds,
                    "errors": list(report.errors),
                    "generated_artifact_freshness": "current" if report.ok else "stale_or_incomplete",
                    "summary": (
                        f"sequence-view contract ready {dataset_id}"
                        if report.ok
                        else f"sequence-view contract attention {dataset_id}: {len(report.errors)} error(s)"
                    ),
                }
            )
        except Exception as exc:
            stale_or_incomplete.append(dataset_id)
            checks.append(
                {
                    **base_check,
                    "state": "attention",
                    "summary": f"sequence-view contract probe failed {dataset_id}: {exc}",
                    "errors": [str(exc)],
                    "probe_error": str(exc),
                    "generated_artifact_freshness": "stale_or_incomplete",
                }
            )

    ok_count = sum(1 for check in checks if check.get("state") == "ok")
    required_failures = sum(1 for check in checks if check.get("state") != "ok" and check.get("required") is True)
    optional_failures = sum(1 for check in checks if check.get("state") != "ok" and check.get("required") is not True)
    state = "attention" if required_failures or optional_failures else "ok"
    return {
        "state": state,
        "drives_top_level_attention": required_failures > 0,
        "checks": checks,
        "counts_by_product_kind": dict(sorted(product_counts.items())),
        "counts_by_orientation": dict(sorted(orientation_counts.items())),
        "counts_by_recommended_pooling": dict(sorted(pooling_counts.items())),
        "generated_artifact_freshness": {
            "state": "attention" if stale_or_incomplete else "ok",
            "stale_or_incomplete_datasets": _ordered_unique(stale_or_incomplete),
        },
        "summary": (
            f"sequence-view product contracts {ok_count}/{len(checks)} ok; "
            f"required_failures={required_failures} optional_failures={optional_failures}"
        ),
    }


def inspect_promoter_study_infer_feature_completion(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, object] | None:
    contract = study_context.ops_contract
    if contract is None or study_context.study_repo_root is None:
        return None
    specs = _preflight_specs_by_kind(contract=contract, kind="infer_sequence_view_completion")
    if not specs:
        return None

    checks: list[dict[str, object]] = []
    all_plans: list[dict[str, object]] = []
    for spec in specs:
        check_id = str(spec.get("check_id") or "").strip()
        surface_id = str(spec.get("surface") or "").strip()
        required = bool(spec.get("required", True))
        base_check = {
            "check_id": check_id,
            "surface": surface_id,
            "required": required,
        }
        try:
            config_path, job = _infer_completion_surface_config(
                surface_payload=contract.execution_surfaces.get(surface_id) or {},
                repo_root=study_context.study_repo_root,
            )
            from dnadesign.infer import plan_sequence_view_feature_inventory_completion_from_config

            plans = [
                dict(plan)
                for plan in plan_sequence_view_feature_inventory_completion_from_config(
                    config_path,
                    job=job,
                )
            ]
            aggregate = _aggregate_infer_completion_plans(plans)
            expectation = _infer_completion_expectation_from_payload(spec.get("expected"))
            violations = _infer_completion_threshold_violations(aggregate=aggregate, expectation=expectation)
            state = "attention" if violations else "ok"
            all_plans.extend(plans)
            checks.append(
                {
                    **base_check,
                    "state": state,
                    "config_path": str(config_path),
                    "job": job,
                    "thresholds": expectation,
                    "violations": violations,
                    "plans": plans,
                    **aggregate,
                    "summary": (
                        f"{str(spec.get('summary') or '').rstrip('.')}. "
                        f"reusable_vectors={aggregate['reusable_vectors']} stale_vectors={aggregate['stale_vectors']} "
                        f"missing_vectors={aggregate['missing_vectors']} "
                        f"reusable_scalars={aggregate['reusable_scalars']} "
                        f"stale_scalars={aggregate['stale_scalars']} "
                        f"missing_scalars={aggregate['missing_scalars']} "
                        f"missing_products={aggregate['missing_products']}."
                    ),
                }
            )
        except Exception as exc:
            checks.append(
                {
                    **base_check,
                    "state": "attention",
                    "summary": f"infer sequence-view completion probe failed {check_id}: {exc}",
                    "probe_error": str(exc),
                }
            )

    aggregate = _aggregate_infer_completion_plans(all_plans) if all_plans else _empty_infer_completion_aggregate()
    ok_count = sum(1 for check in checks if check.get("state") == "ok")
    required_failures = sum(1 for check in checks if check.get("state") != "ok" and check.get("required") is True)
    optional_failures = sum(1 for check in checks if check.get("state") != "ok" and check.get("required") is not True)
    return {
        "state": "attention" if required_failures or optional_failures else "ok",
        "drives_top_level_attention": required_failures > 0,
        "checks": checks,
        "aggregate": aggregate,
        "summary": (
            "infer sequence-view feature completion "
            f"checks {ok_count}/{len(checks)} ok; "
            f"reusable_vectors={aggregate['reusable_vectors']} stale_vectors={aggregate['stale_vectors']} "
            f"missing_vectors={aggregate['missing_vectors']} "
            f"reusable_scalars={aggregate['reusable_scalars']} stale_scalars={aggregate['stale_scalars']} "
            f"missing_scalars={aggregate['missing_scalars']} missing_products={aggregate['missing_products']}; "
            f"required_failures={required_failures} optional_failures={optional_failures}"
        ),
    }


def _overlay_guardrail_state(*, root: Path, dataset_id: str, namespace: str) -> dict[str, object]:
    dataset = Dataset(root, dataset_id)
    overlay = next((item for item in dataset.list_overlays() if item.namespace == namespace), None)
    if overlay is None:
        return {
            "state": "attention",
            "dataset": dataset_id,
            "namespace": namespace,
            "overlay_present": False,
            "overlay_compact": False,
            "summary": f"source overlay guardrail missing {dataset_id}:{namespace}",
        }
    overlay_path = Path(overlay.path)
    overlay_compact = overlay_path.is_file()
    return {
        "state": "ok" if overlay_compact else "attention",
        "dataset": dataset_id,
        "namespace": namespace,
        "overlay_present": True,
        "overlay_compact": overlay_compact,
        "overlay_path": str(overlay_path),
        "summary": (
            f"source overlay compact {dataset_id}:{namespace}"
            if overlay_compact
            else f"source overlay needs compaction {dataset_id}:{namespace}"
        ),
    }


def _densegen_metadata_projection_state(
    *,
    root: Path,
    dataset_id: str,
    expected_rows: int,
    label: str,
) -> dict[str, object]:
    required_columns = ("densegen__plan", "densegen__required_regulators")
    dataset = Dataset(root, dataset_id)
    schema = dataset.schema()
    missing_columns = [column for column in required_columns if column not in schema.names]
    if missing_columns:
        return {
            "state": "attention",
            "dataset": dataset_id,
            "label": label,
            "required_columns": list(required_columns),
            "missing_columns": missing_columns,
            "non_null_counts": {},
            "expected_rows": expected_rows,
            "summary": f"{label} DenseGen metadata columns missing {dataset_id}: {', '.join(missing_columns)}",
        }

    counts = {column: 0 for column in required_columns}
    for batch in dataset.scan(columns=list(required_columns), include_overlays=True, batch_size=65_536):
        for column in required_columns:
            array = batch.column(batch.schema.get_field_index(column))
            counts[column] += int(batch.num_rows - array.null_count)

    complete = all(count >= expected_rows for count in counts.values())
    min_count = min(counts.values()) if counts else 0
    return {
        "state": "ok" if complete else "attention",
        "dataset": dataset_id,
        "label": label,
        "required_columns": list(required_columns),
        "missing_columns": [],
        "non_null_counts": counts,
        "expected_rows": expected_rows,
        "missing_densegen_rows": max(expected_rows - min_count, 0),
        "summary": (
            f"{label} DenseGen metadata ready {dataset_id} {min_count}/{expected_rows}"
            if complete
            else f"{label} DenseGen metadata incomplete {dataset_id} {min_count}/{expected_rows}"
        ),
    }


def _preflight_specs_by_kind(*, contract: object, kind: str) -> list[dict[str, object]]:
    preflight = getattr(contract, "preflight", None)
    check_specs = getattr(preflight, "check_specs", {}) if preflight is not None else {}
    specs: list[dict[str, object]] = []
    if not isinstance(check_specs, Mapping):
        return specs
    for phase_specs in check_specs.values():
        for spec in phase_specs:
            if isinstance(spec, Mapping) and str(spec.get("kind") or "").strip() == kind:
                specs.append(dict(spec))
    return specs


def _sequence_view_expectation_from_payload(payload: object) -> SequenceViewContractExpectation:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("sequence_view_contract expected payload must be a mapping.")
    return SequenceViewContractExpectation(
        total_records=_optional_int(payload.get("total_records")),
        total_views=_optional_int(payload.get("total_views")),
        counts_by_product_kind=_string_int_mapping(payload.get("counts_by_product_kind")),
        counts_by_orientation=_string_int_mapping(payload.get("counts_by_orientation")),
        counts_by_context_kind=_string_int_mapping(payload.get("counts_by_context_kind")),
        counts_by_recommended_pooling=_string_int_mapping(payload.get("counts_by_recommended_pooling")),
        exact_lengths_by_product_kind=_string_int_mapping(payload.get("exact_lengths_by_product_kind")),
    )


def _infer_completion_surface_config(
    *,
    surface_payload: Mapping[str, object],
    repo_root: Path,
) -> tuple[Path, str | None]:
    argv = tuple(str(token) for token in surface_payload.get("argv") or ())
    config = _argv_option(argv, "--config")
    if config is None:
        raise ValueError("infer_sequence_view_completion surface must define --config")
    job = _argv_option(argv, "--job")
    return (_resolve_command_config_path(raw_path=config, surface_payload=surface_payload, repo_root=repo_root), job)


def _resolve_command_config_path(
    *,
    raw_path: str,
    surface_payload: Mapping[str, object],
    repo_root: Path,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or raw_path.startswith(("repo:", "manifest:", "cwd:")):
        return resolve_repo_relative_path(
            repo_root=repo_root,
            raw_path=raw_path,
            status_kind="promoter-study-status",
        )
    cwd_ref = str(surface_payload.get("cwd_ref") or "").strip()
    cwd = (
        resolve_repo_relative_path(
            repo_root=repo_root,
            raw_path=cwd_ref,
            status_kind="promoter-study-status",
        )
        if cwd_ref
        else repo_root
    )
    return (cwd / path).resolve()


def _argv_option(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            text = str(argv[index + 1]).strip()
            return text or None
        prefix = f"{flag}="
        if token.startswith(prefix):
            text = token[len(prefix) :].strip()
            return text or None
    return None


def _aggregate_infer_completion_plans(plans: Sequence[Mapping[str, object]]) -> dict[str, object]:
    aggregate = _empty_infer_completion_aggregate()
    product_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    pooling_counts: Counter[str] = Counter()
    for plan in plans:
        for field in _INFER_COMPLETION_SCALAR_FIELDS:
            aggregate[field] = int(aggregate[field]) + _required_int(plan.get(field, 0))
        _update_counter(product_counts, plan.get("by_product_kind"))
        _update_counter(orientation_counts, plan.get("by_orientation"))
        _update_counter(pooling_counts, plan.get("by_pooling_operation"))
        aggregate["plans_count"] = int(aggregate["plans_count"]) + 1
    aggregate["counts_by_product_kind"] = dict(sorted(product_counts.items()))
    aggregate["counts_by_orientation"] = dict(sorted(orientation_counts.items()))
    aggregate["counts_by_pooling_operation"] = dict(sorted(pooling_counts.items()))
    return aggregate


_INFER_COMPLETION_SCALAR_FIELDS = (
    "required_views",
    "required_vectors",
    "required_scalars",
    "existing_vectors",
    "existing_scalars",
    "reusable_vectors",
    "reusable_scalars",
    "stale_vectors",
    "stale_scalars",
    "missing_vectors",
    "missing_scalars",
    "missing_products",
    "persisted_vector_reusable",
    "persisted_scalar_reusable",
    "existing_aliases",
    "existing_scalar_aliases",
)


def _empty_infer_completion_aggregate() -> dict[str, object]:
    return {
        "plans_count": 0,
        **{field: 0 for field in _INFER_COMPLETION_SCALAR_FIELDS},
        "counts_by_product_kind": {},
        "counts_by_orientation": {},
        "counts_by_pooling_operation": {},
    }


def _infer_completion_expectation_from_payload(payload: object) -> dict[str, int]:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("infer_sequence_view_completion expected payload must be a mapping.")
    return {
        "max_missing_vectors": _optional_int(payload.get("max_missing_vectors")) or 0,
        "max_missing_scalars": _optional_int(payload.get("max_missing_scalars")) or 0,
        "max_stale_vectors": _optional_int(payload.get("max_stale_vectors")) or 0,
        "max_stale_scalars": _optional_int(payload.get("max_stale_scalars")) or 0,
        "max_missing_products": _optional_int(payload.get("max_missing_products")) or 0,
    }


def _infer_completion_threshold_violations(
    *,
    aggregate: Mapping[str, object],
    expectation: Mapping[str, int],
) -> list[str]:
    violations: list[str] = []
    for observed_key, threshold_key in (
        ("missing_vectors", "max_missing_vectors"),
        ("missing_scalars", "max_missing_scalars"),
        ("stale_vectors", "max_stale_vectors"),
        ("stale_scalars", "max_stale_scalars"),
        ("missing_products", "max_missing_products"),
    ):
        observed = _required_int(aggregate.get(observed_key, 0))
        threshold = _required_int(expectation.get(threshold_key, 0))
        if observed > threshold:
            violations.append(f"{observed_key}={observed} exceeds {threshold_key}={threshold}")
    return violations


def _update_counter(counter: Counter[str], payload: object) -> None:
    if not isinstance(payload, Mapping):
        return
    for key, value in payload.items():
        normalized_key = str(key or "").strip()
        if normalized_key:
            counter[normalized_key] += _required_int(value)


def _string_int_mapping(payload: object) -> dict[str, int]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("expected a mapping of string keys to integer counts")
    return {str(key): _required_int(value) for key, value in payload.items()}


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def _required_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("expected an integer, not a boolean")
    if isinstance(value, int):
        return int(value)
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    raise ValueError(f"expected an integer, got {value!r}")


def _ordered_unique(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _missing_promoter_study_result(
    *,
    context: PromoterStudyResolvedContext,
    evidence: dict[str, object],
) -> tuple[str, str, dict[str, object]] | None:
    if not context.study_dir_exists:
        return ("missing", "promoter study directory not found", evidence)
    missing_required_files = list(context.missing_required_files)
    missing_declared_present = list(context.missing_declared_present)
    missing_execution_surfaces = list(context.missing_execution_surfaces)
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
    return None


def _study_adapter_context(context: StudyStatusContext) -> PromoterStatusAdapterContext:
    if not isinstance(context.adapter_context, PromoterStatusAdapterContext):
        raise ValueError("promoter status context has invalid adapter_context payload")
    return context.adapter_context


__all__ = ["PromoterStudyStatusAdapter", "PromoterStatusAdapterContext", "STUDY_STATUS_ADAPTER"]
