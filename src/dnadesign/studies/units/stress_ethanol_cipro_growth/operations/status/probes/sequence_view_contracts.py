"""
Sequence-view contract probes for stress_ethanol_cipro_growth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Sequence

from dnadesign.ops.status import string_or_none
from dnadesign.usr import Dataset, SequenceViewContractExpectation, validate_sequence_view_contract

from ..record_normalizer import StressEthanolCiproGrowthResolvedContext


def inspect_stress_ethanol_cipro_growth_sequence_view_contracts(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
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


__all__ = ["inspect_stress_ethanol_cipro_growth_sequence_view_contracts"]
