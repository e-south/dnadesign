"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_publication.py

Immutable bundle publication and offline-verification tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.trijunction import api as api_module
from dnadesign.trijunction import build, verify
from dnadesign.trijunction.contracts import parse_request
from dnadesign.trijunction.contracts.identity import canonical_json_bytes, sha256_bytes
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.errors import TriJunctionBundleError
from dnadesign.trijunction.publication import verify as verifier_module
from dnadesign.trijunction.publication import writer as writer_module
from dnadesign.trijunction.publication.writer import _publish_bundle
from dnadesign.trijunction.tests.test_planner import _request_mapping


def test_publish_and_verify_complete_create_only_bundle(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    plan = design_trijunction(request)
    destination = tmp_path / "runs" / "design-v1"

    published = build(request, destination=destination)
    verified = verify(destination)

    assert published.path == destination.absolute()
    assert verified.status == "verified"
    assert verified.plan_id == plan.plan_id
    assert verified.artifact_count == 5
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "dnadesign.trijunction.bundle.v1"
    assert set(manifest["artifacts"]) == {"checks", "orders", "plan", "request", "review"}
    assert (
        (destination / "orders" / "oligos.tsv").read_text(encoding="utf-8").startswith("order_id\ttarget_ids\tpool_id")
    )
    review_rows = json.loads((destination / "views" / "three_way_junction_review.v1.json").read_text())
    assert [row["target"]["target_id"] for row in review_rows] == [target.target_id for target in plan.targets]
    assert {row["search"]["thermodynamic_screening"] for row in review_rows} == {"not_run"}


def test_build_runs_one_design_and_one_post_install_semantic_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    design_calls = 0
    replay_calls = 0
    original_design = api_module.design_trijunction
    original_replay = verifier_module.design_trijunction

    def counted_design(value):  # type: ignore[no-untyped-def]
        nonlocal design_calls
        design_calls += 1
        return original_design(value)

    def counted_replay(value):  # type: ignore[no-untyped-def]
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(value)

    monkeypatch.setattr(api_module, "design_trijunction", counted_design)
    monkeypatch.setattr(verifier_module, "design_trijunction", counted_replay)

    build(request, destination=destination)

    assert design_calls == 1
    assert replay_calls == 1


def test_existing_bundle_is_not_replaced(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    before = (destination / "manifest.json").read_bytes()

    with pytest.raises(TriJunctionBundleError, match="already exists and is immutable"):
        build(request, destination=destination)

    assert (destination / "manifest.json").read_bytes() == before


def test_tampered_artifact_fails_offline_verification(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    (destination / "plan.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(TriJunctionBundleError, match="content identity does not match"):
        verify(destination)


def test_oversized_integer_in_manifest_is_reported_as_a_sanitized_bundle_error(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    oversized_integer = "9" * 5_000
    (destination / "manifest.json").write_text(
        f'{{"oversized":{oversized_integer}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(TriJunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == f"TriJunction manifest is not valid UTF-8 JSON: {destination / 'manifest.json'}"
    assert oversized_integer not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_manifest_recursion_failure_is_reported_as_a_sanitized_bundle_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)

    def recursion_failure(*_args: object, **_kwargs: object) -> None:
        raise RecursionError

    monkeypatch.setattr(verifier_module.json, "loads", recursion_failure)

    with pytest.raises(TriJunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == f"TriJunction manifest is not valid UTF-8 JSON: {destination / 'manifest.json'}"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_oversized_integer_in_embedded_request_is_reported_as_a_sanitized_bundle_error(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    oversized_integer = "9" * 5_000
    request_content = f'{{"seed":{oversized_integer}}}\n'.encode()
    (destination / "request.json").write_bytes(request_content)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"]["request"]["bytes"] = len(request_content)
    manifest["artifacts"]["request"]["sha256"] = sha256_bytes(request_content)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(TriJunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == "Bundle request cannot reproduce a valid TriJunction plan."
    assert oversized_integer not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_embedded_request_recursion_failure_is_reported_as_a_sanitized_bundle_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    original_loads = verifier_module.json.loads
    calls = 0

    def recurse_on_request(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_loads(*args, **kwargs)
        raise RecursionError

    monkeypatch.setattr(verifier_module.json, "loads", recurse_on_request)

    with pytest.raises(TriJunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == "Bundle request cannot reproduce a valid TriJunction plan."
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert calls == 2


def test_request_plan_mismatch_fails_before_payload_work_or_filesystem_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_request = parse_request(_request_mapping())
    second_mapping = _request_mapping()
    second_mapping["seed"] = 99
    second_request = parse_request(second_mapping)
    mismatched_plan = replace(design_trijunction(first_request), seed=99)
    destination = tmp_path / "parent" / "design-v1"

    def fail_if_serialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("mismatched inputs must fail before payload construction")

    monkeypatch.setattr(writer_module, "bundle_payloads", fail_if_serialized)

    with pytest.raises(TriJunctionBundleError, match="request does not match"):
        _publish_bundle(second_request, mismatched_plan, destination)

    assert not destination.parent.exists()
