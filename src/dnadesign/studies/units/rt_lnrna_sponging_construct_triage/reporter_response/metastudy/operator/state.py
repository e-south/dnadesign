"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/operator/state.py

Strict codec and structural validation for source-controlled meta-study state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import yaml

from ..acquisition_projection import validate_acquisition_projection_payload
from ..contracts._values import MetastudyContractError
from ..contracts.decision_codec import validate_decision_payload
from ..contracts.materialization import materialization_attempt_from_payload
from ..contracts.objective import DEFAULT_OBJECTIVE_READINESS, objective_readiness_from_payload
from ..contracts.protocol import DEFAULT_PROTOCOL
from ..sensitivity import parse_sensitivity_evaluations
from ..sensitivity_coverage import validate_sensitivity_coverage_receipt_payloads

ROUTE_ID = "rt_lnrna_reporter_response_metastudy"
ROUTE_REGISTRY_PATH = ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
STATE_FILE = "metastudy-state.yaml"
READINESS_SCHEMA_ID = "rt_lnrna_reporter_response_readiness_snapshot.v1"
STATE_SCHEMA_ID = "rt_lnrna_reporter_response_metastudy_state.v7"
RECEIPT_NORMALIZATION = "omit environment-specific reader_command before canonical JSON hashing"
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise MetastudyContractError(f"duplicate YAML key in combined meta-study state: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def validate_source_controlled_state(path: Path, *, phd_root: Path) -> dict[str, object]:
    """Load and validate one atomic source-controlled state generation."""

    payload = load_state_yaml(Path(path))
    route_registry = route_registry_for_phd_root(phd_root)
    validate_state_payload(payload, route_registry=route_registry)
    return payload


def validate_state_payload(payload: object, *, route_registry: Path | None = None) -> None:
    """Validate the exact combined state contract without resolving live records."""

    if not isinstance(payload, dict):
        raise MetastudyContractError("combined meta-study state fields do not match the exact contract")
    expected_fields = {
        "schema_id",
        "generation_digest",
        "readiness",
        "decision",
        "objective_readiness",
        "sensitivity_evaluations",
        "sensitivity_coverage_receipts",
        "acquisition_projection",
    }
    if set(payload) != expected_fields:
        raise MetastudyContractError("combined meta-study state fields do not match the exact contract")
    if payload["schema_id"] != STATE_SCHEMA_ID:
        raise MetastudyContractError("combined meta-study state schema_id changed")
    require_sha256_digest(payload["generation_digest"], label="state generation digest")
    body = {
        key: payload[key]
        for key in (
            "readiness",
            "decision",
            "objective_readiness",
            "sensitivity_evaluations",
            "sensitivity_coverage_receipts",
            "acquisition_projection",
        )
        if key in payload
    }
    if payload["generation_digest"] != canonical_digest(body):
        raise MetastudyContractError("combined meta-study state generation digest changed")
    decision = payload["decision"]
    objective_readiness = payload["objective_readiness"]
    readiness = payload["readiness"]
    if not isinstance(decision, dict) or not isinstance(objective_readiness, dict) or not isinstance(readiness, dict):
        raise MetastudyContractError("combined meta-study state projections must be objects")
    if set(readiness) != {
        "schema_id",
        "source_identity",
        "last_verified",
        "selected_experiment_count",
        "related_experiment_count",
        "related_experiment_ids",
        "ready_experiment_count",
        "ready_experiment_ids",
        "blocked_experiment_ids",
    }:
        raise MetastudyContractError("combined meta-study readiness fields changed")
    source_identity = readiness["source_identity"]
    if not isinstance(source_identity, dict) or set(source_identity) != {
        "route_id",
        "route_registry_path",
        "route_registry_digest",
        "normalized_full_receipt_digest",
        "normalization",
    }:
        raise MetastudyContractError("combined meta-study readiness source identity changed")
    _validate_readiness_snapshot(
        readiness,
        source_identity=source_identity,
        route_registry=route_registry,
    )
    validate_decision_payload(decision)
    if objective_readiness_from_payload(objective_readiness) != DEFAULT_OBJECTIVE_READINESS:
        raise MetastudyContractError("combined meta-study objective readiness changed")
    parse_sensitivity_evaluations(payload["sensitivity_evaluations"])
    attempts_payload = decision["materialization_attempts"]
    if not isinstance(attempts_payload, list):
        raise MetastudyContractError("combined meta-study attempts must be an array")
    attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts_payload)
    )
    validate_sensitivity_coverage_receipt_payloads(
        payload["sensitivity_coverage_receipts"],
        attempts=attempts,
    )
    acquisition_payload = payload["acquisition_projection"]
    selected_reduction = decision["selected_reduction"]
    if selected_reduction is not None:
        projection = validate_acquisition_projection_payload(acquisition_payload)
        if projection.selected_reduction != tuple(selected_reduction):
            raise MetastudyContractError("acquisition projection differs from the selected reduction")
    elif acquisition_payload is not None:
        raise MetastudyContractError("state without a selected reduction cannot contain a acquisition projection")
    expected_readiness = {
        "selected_experiment_count": readiness.get("selected_experiment_count"),
        "ready_experiment_count": readiness.get("ready_experiment_count"),
        "ready_experiment_ids": readiness.get("ready_experiment_ids"),
        "blocked_experiment_ids": readiness.get("blocked_experiment_ids"),
        "receipt_digest": source_identity["normalized_full_receipt_digest"],
    }
    if decision["readiness"] != expected_readiness:
        raise MetastudyContractError("combined meta-study readiness and decision generations differ")


def _validate_readiness_snapshot(
    readiness: dict[str, object],
    *,
    source_identity: dict[str, object],
    route_registry: Path | None,
) -> None:
    if readiness["schema_id"] != READINESS_SCHEMA_ID:
        raise MetastudyContractError("combined meta-study readiness schema_id changed")
    if source_identity["route_id"] != ROUTE_ID:
        raise MetastudyContractError("combined meta-study readiness route_id changed")
    if source_identity["route_registry_path"] != ROUTE_REGISTRY_PATH:
        raise MetastudyContractError("combined meta-study readiness route registry path changed")
    if source_identity["normalization"] != RECEIPT_NORMALIZATION:
        raise MetastudyContractError("combined meta-study readiness normalization changed")
    require_sha256_digest(
        source_identity["route_registry_digest"],
        label="readiness route registry digest",
    )
    if route_registry is not None:
        expected_registry_digest = "sha256:" + hashlib.sha256(route_registry.read_bytes()).hexdigest()
        if source_identity["route_registry_digest"] != expected_registry_digest:
            raise MetastudyContractError("combined meta-study readiness route registry digest changed")
    require_sha256_digest(
        source_identity["normalized_full_receipt_digest"],
        label="readiness normalized receipt digest",
    )
    verified_on = readiness["last_verified"]
    if not isinstance(verified_on, str):
        raise MetastudyContractError("combined meta-study readiness last_verified must be an ISO date")
    try:
        parsed_verified_on = date.fromisoformat(verified_on)
    except ValueError as exc:
        raise MetastudyContractError("combined meta-study readiness last_verified must be an ISO date") from exc
    if parsed_verified_on.isoformat() != verified_on:
        raise MetastudyContractError("combined meta-study readiness last_verified must be an ISO date")

    selected_count = readiness["selected_experiment_count"]
    if type(selected_count) is not int or selected_count != DEFAULT_PROTOCOL.planned_kinetic_experiments:
        raise MetastudyContractError("combined meta-study readiness selected experiment count changed")
    related_ids = readiness["related_experiment_ids"]
    related_count = readiness["related_experiment_count"]
    expected_related_ids = list(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids)
    if (
        type(related_count) is not int
        or related_count != len(expected_related_ids)
        or not isinstance(related_ids, list)
        or related_ids != expected_related_ids
    ):
        raise MetastudyContractError("combined meta-study readiness related experiment cohort changed")

    ready_ids = readiness["ready_experiment_ids"]
    blocked_ids = readiness["blocked_experiment_ids"]
    ready_count = readiness["ready_experiment_count"]
    if (
        type(ready_count) is not int
        or not isinstance(ready_ids, list)
        or not isinstance(blocked_ids, list)
        or not all(isinstance(value, str) for value in (*ready_ids, *blocked_ids))
    ):
        raise MetastudyContractError("combined meta-study readiness ready experiment count or cohort changed")
    ready_set = set(ready_ids)
    blocked_set = set(blocked_ids)
    planned_ids = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
    if (
        ready_count != len(ready_ids)
        or ready_count + len(blocked_ids) != selected_count
        or bool(ready_set & blocked_set)
        or ready_set | blocked_set != set(planned_ids)
        or ready_ids != [value for value in planned_ids if value in ready_set]
        or blocked_ids != [value for value in planned_ids if value in blocked_set]
    ):
        raise MetastudyContractError("combined meta-study readiness selected experiment cohort changed")


def load_state_yaml(path: Path) -> object:
    """Read one state YAML document with duplicate-key rejection."""

    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except MetastudyContractError:
        raise
    except yaml.YAMLError as exc:
        raise MetastudyContractError(f"cannot parse combined meta-study state YAML: {exc}") from exc


def route_registry_for_phd_root(phd_root: Path) -> Path:
    root = Path(phd_root).expanduser().resolve()
    route_registry = root / ROUTE_REGISTRY_PATH
    if not route_registry.is_file():
        raise MetastudyContractError("PhD root does not contain the canonical route registry")
    return route_registry


def require_sha256_digest(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_DIGEST.fullmatch(value) is None:
        raise MetastudyContractError(f"combined meta-study {label} is not a canonical SHA-256 digest")


def canonical_digest(payload: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def digest_file(path: Path) -> str:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise MetastudyContractError(f"cannot read canonical route registry: {exc}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()
