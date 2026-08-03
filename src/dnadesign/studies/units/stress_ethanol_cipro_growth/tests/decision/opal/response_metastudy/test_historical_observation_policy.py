"""Fail-closed tests for the response metastudy's frozen policy snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    load_multistate_behavior_protocol,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime.historical import (
    HistoricalObservationPolicyV2Error,
    load_historical_observation_policy_v2,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime.historical import (
    source_files as historical_source_files,
)

PACKAGE = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy")
PROTOCOL_PATH = PACKAGE / "config/multistate_response_behavior_shadow_v2.yaml"
FROZEN_PROTOCOL_PATH = PACKAGE / "config/multistate_response_behavior_shadow_v1.yaml"


def test_frozen_source_loader_does_not_import_the_active_observation_policy() -> None:
    source = (PACKAGE / "runtime/multistate_behavior_sources.py").read_text(encoding="utf-8")
    assert "response_window_observations.policy" not in source
    assert "load_historical_observation_policy_v2" in source


def test_activation_receipt_protocol_remains_byte_stable_and_separate() -> None:
    frozen = load_multistate_behavior_protocol(FROZEN_PROTOCOL_PATH)
    current = load_multistate_behavior_protocol(PROTOCOL_PATH)

    assert frozen.source_sha256 == (
        "0428656248dbaae8a917f7757eb9bd045cb5a37e547fa6a39de292ac0f789bfd"  # pragma: allowlist secret
    )
    assert frozen.protocol_id == "secg_multistate_response_behavior_shadow_v1"
    assert current.protocol_id == "secg_multistate_response_behavior_shadow_v2"
    assert current.source_sha256 != frozen.source_sha256


def test_repo_snapshot_projects_the_exact_approved_historical_policy() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    source = protocol.source_equivalence
    policy = load_historical_observation_policy_v2(
        Path(source.prior_observation_policy_repo_path),
        expected_sha256=source.prior_observation_policy_sha256,
        expected_reader_bundle_sha256=source.prior_observation_reader_bundle_sha256,
        expected_candidate_bindings_sha256=source.prior_candidate_bindings_manifest_sha256,
        expected_approval_sha256=source.prior_observation_approval_sha256,
        expected_primary_reduction_id=protocol.primary_reduction_id,
    )

    assert policy.approval_status == "approved"
    assert policy.approved_by == "Eric J. South"
    assert policy.approved_at == "2026-07-15T19:33:54-04:00"
    assert policy.config_sha256 == source.prior_observation_policy_sha256
    assert policy.reader_bundle_sha256 == source.prior_observation_reader_bundle_sha256
    assert policy.candidate_bindings_sha256 == source.prior_candidate_bindings_manifest_sha256
    assert policy.aggregation.primary_reduction_id == protocol.primary_reduction_id
    assert not policy.repeat_decisions.empty


def test_historical_policy_rejects_missing_and_tampered_snapshots(tmp_path: Path) -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    source = protocol.source_equivalence
    missing = tmp_path / "missing.yaml"
    with pytest.raises(HistoricalObservationPolicyV2Error, match="missing"):
        _load(missing, protocol=protocol)

    tampered = tmp_path / "policy.yaml"
    tampered.write_bytes(Path(source.prior_observation_policy_repo_path).read_bytes() + b"\n# drift\n")
    with pytest.raises(HistoricalObservationPolicyV2Error, match="digest mismatch"):
        _load(tampered, protocol=protocol)


def test_historical_policy_rejects_approval_and_source_identity_drift() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    source = protocol.source_equivalence
    path = Path(source.prior_observation_policy_repo_path)

    with pytest.raises(HistoricalObservationPolicyV2Error, match="approval identity"):
        load_historical_observation_policy_v2(
            path,
            expected_sha256=source.prior_observation_policy_sha256,
            expected_reader_bundle_sha256=source.prior_observation_reader_bundle_sha256,
            expected_candidate_bindings_sha256=source.prior_candidate_bindings_manifest_sha256,
            expected_approval_sha256="0" * 64,
            expected_primary_reduction_id=protocol.primary_reduction_id,
        )

    with pytest.raises(HistoricalObservationPolicyV2Error, match="source identities"):
        load_historical_observation_policy_v2(
            path,
            expected_sha256=source.prior_observation_policy_sha256,
            expected_reader_bundle_sha256="0" * 64,
            expected_candidate_bindings_sha256=source.prior_candidate_bindings_manifest_sha256,
            expected_approval_sha256=source.prior_observation_approval_sha256,
            expected_primary_reduction_id=protocol.primary_reduction_id,
        )
    with pytest.raises(HistoricalObservationPolicyV2Error, match="source identities"):
        load_historical_observation_policy_v2(
            path,
            expected_sha256=source.prior_observation_policy_sha256,
            expected_reader_bundle_sha256=source.prior_observation_reader_bundle_sha256,
            expected_candidate_bindings_sha256="0" * 64,
            expected_approval_sha256=source.prior_observation_approval_sha256,
            expected_primary_reduction_id=protocol.primary_reduction_id,
        )


def test_protocol_declared_historical_request_is_present_and_digest_bound(tmp_path: Path) -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    source = protocol.source_equivalence

    request = historical_source_files._source_path(
        Path.cwd(),
        relative_path=source.prior_observation_request_repo_path,
        expected_sha256=source.prior_observation_request_sha256,
        label="historical Reader request",
    )
    assert request == Path(source.prior_observation_request_repo_path).resolve()

    tampered = tmp_path / "request.yaml"
    tampered.write_bytes(request.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        historical_source_files._source_path(
            tmp_path,
            relative_path=tampered.name,
            expected_sha256=source.prior_observation_request_sha256,
            label="historical Reader request",
        )


def _load(path: Path, *, protocol: object):
    source = protocol.source_equivalence
    return load_historical_observation_policy_v2(
        path,
        expected_sha256=source.prior_observation_policy_sha256,
        expected_reader_bundle_sha256=source.prior_observation_reader_bundle_sha256,
        expected_candidate_bindings_sha256=source.prior_candidate_bindings_manifest_sha256,
        expected_approval_sha256=source.prior_observation_approval_sha256,
        expected_primary_reduction_id=protocol.primary_reduction_id,
    )
