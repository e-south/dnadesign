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

from dnadesign.trijunction import build, verify
from dnadesign.trijunction.contracts import parse_request
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.errors import TriJunctionBundleError
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
