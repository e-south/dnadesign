"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_publication.py

Create-only bundle publication and offline-verification tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import fields, replace
from pathlib import Path

import pytest

from dnadesign.junction import api as api_module
from dnadesign.junction import build, verify
from dnadesign.junction.contracts import parse_request
from dnadesign.junction.contracts.identity import canonical_json_bytes, sha256_bytes
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.errors import JunctionBundleError
from dnadesign.junction.publication import payloads as payloads_module
from dnadesign.junction.publication import snapshot as snapshot_module
from dnadesign.junction.publication import verify as verifier_module
from dnadesign.junction.publication import writer as writer_module
from dnadesign.junction.publication.payloads import ARTIFACT_PATHS
from dnadesign.junction.publication.snapshot import (
    _DirectoryIdentity,
    _RetainedSnapshotRead,
    open_bundle_snapshot,
)
from dnadesign.junction.publication.writer import _publish_bundle
from dnadesign.junction.tests.test_planner import _request_mapping


def test_publish_and_verify_complete_create_only_bundle(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    plan = design_junction(request)
    destination = tmp_path / "runs" / "design-v1"

    published = build(request, destination=destination)
    verified = verify(destination)

    assert published.path == destination.absolute()
    assert verified.status == "verified"
    assert verified.plan_id == plan.plan_id
    assert verified.artifact_count == 5
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "dnadesign.junction.bundle.v1"
    assert set(manifest["artifacts"]) == {"checks", "orders", "plan", "request", "review"}
    assert (
        (destination / "orders" / "oligos.tsv")
        .read_text(encoding="utf-8")
        .startswith("order_id\ttarget_ids\tassembly_group_id")
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
    original_design = api_module.design_junction
    original_replay = verifier_module.design_junction

    def counted_design(value):  # type: ignore[no-untyped-def]
        nonlocal design_calls
        design_calls += 1
        return original_design(value)

    def counted_replay(value):  # type: ignore[no-untyped-def]
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(value)

    monkeypatch.setattr(api_module, "design_junction", counted_design)
    monkeypatch.setattr(verifier_module, "design_junction", counted_replay)

    build(request, destination=destination)

    assert design_calls == 1
    assert replay_calls == 1


def test_publication_and_replay_retain_at_most_one_rendered_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class TrackedPayload(bytes):
        live = 0
        peak = 0

        def __new__(cls, content: bytes):
            instance = super().__new__(cls, content)
            cls.live += 1
            cls.peak = max(cls.peak, cls.live)
            return instance

        def __del__(self) -> None:
            type(self).live -= 1

    original_renderer = payloads_module.render_artifact_bytes
    rendered_keys: list[str] = []

    def tracked_renderer(key, request, plan):  # type: ignore[no-untyped-def]
        assert TrackedPayload.live == 0
        rendered_keys.append(key)
        return TrackedPayload(original_renderer(key, request, plan))

    monkeypatch.setattr(writer_module, "render_artifact_bytes", tracked_renderer)
    monkeypatch.setattr(verifier_module, "render_artifact_bytes", tracked_renderer)
    request = parse_request(_request_mapping())

    build(request, destination=tmp_path / "design-v1")
    gc.collect()

    assert rendered_keys == [*ARTIFACT_PATHS, *ARTIFACT_PATHS]
    assert TrackedPayload.peak == 1
    assert TrackedPayload.live == 0


def test_build_rejects_a_different_valid_bundle_swapped_into_the_final_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alternate_mapping = _request_mapping()
    alternate_mapping["seed"] = 99
    alternate = tmp_path / "alternate"
    alternate_publication = build(parse_request(alternate_mapping), destination=alternate)
    destination = tmp_path / "design-v1"
    displaced = tmp_path / "displaced-intended"
    original_verify = writer_module._verify_published_bundle

    def swap_then_verify(path: Path):  # type: ignore[no-untyped-def]
        path.rename(displaced)
        shutil.copytree(alternate, path)
        return original_verify(path)

    monkeypatch.setattr(writer_module, "_verify_published_bundle", swap_then_verify)

    with pytest.raises(JunctionBundleError, match="does not match the supplied plan and request"):
        build(parse_request(_request_mapping()), destination=destination)

    assert displaced.is_dir()
    assert verify(destination).plan_id == alternate_publication.plan_id


def test_build_rejects_an_equivalent_bundle_copied_over_the_final_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "design-v1"
    displaced = tmp_path / "displaced-intended"
    original_verify = writer_module._verify_published_bundle

    def swap_then_verify(path: Path):  # type: ignore[no-untyped-def]
        path.rename(displaced)
        shutil.copytree(displaced, path)
        return original_verify(path)

    monkeypatch.setattr(writer_module, "_verify_published_bundle", swap_then_verify)

    with pytest.raises(JunctionBundleError, match="path identity changed after publication"):
        build(parse_request(_request_mapping()), destination=destination)

    assert displaced.is_dir()
    assert verify(destination).status == "verified"


def test_snapshot_retains_file_identity_without_retaining_payload_bytes() -> None:
    assert "content" not in {field.name for field in fields(_RetainedSnapshotRead)}
    assert "ctime_ns" in {field.name for field in fields(_DirectoryIdentity)}


def test_snapshot_accepts_metadata_timestamp_change_when_bytes_are_stable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.bin"
    artifact.write_bytes(b"stable")

    with open_bundle_snapshot(
        bundle,
        expected_files=frozenset({"artifact.bin"}),
        reject_undeclared_entries=True,
    ) as snapshot:
        snapshot.read_file(
            Path("artifact.bin"),
            limit=64,
            context="test artifact",
            retain_content=False,
        )
        initial = artifact.stat()
        os.utime(artifact, ns=(initial.st_atime_ns - 1_000_000_000, initial.st_mtime_ns))
        assert artifact.stat().st_ctime_ns != initial.st_ctime_ns

        snapshot.assert_stable()


def test_snapshot_rehash_rejects_same_size_content_change_with_restored_mtime(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.bin"
    artifact.write_bytes(b"before")

    with open_bundle_snapshot(
        bundle,
        expected_files=frozenset({"artifact.bin"}),
        reject_undeclared_entries=True,
    ) as snapshot:
        snapshot.read_file(
            Path("artifact.bin"),
            limit=64,
            context="test artifact",
            retain_content=False,
        )
        initial = artifact.stat()
        artifact.write_bytes(b"after!")
        os.utime(artifact, ns=(initial.st_atime_ns, initial.st_mtime_ns))

        with pytest.raises(JunctionBundleError, match="changed after it was verified"):
            snapshot.assert_stable()


def test_snapshot_revalidates_declared_path_after_final_content_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.bin"
    artifact.write_bytes(b"verified")
    displaced = tmp_path / "displaced-original.bin"

    with open_bundle_snapshot(
        bundle,
        expected_files=frozenset({"artifact.bin"}),
        reject_undeclared_entries=True,
    ) as snapshot:
        snapshot.read_file(
            Path("artifact.bin"),
            limit=64,
            context="test artifact",
            retain_content=False,
        )
        original_read = os.read
        swapped = False

        def swap_declared_path(descriptor: int, length: int) -> bytes:
            nonlocal swapped
            if not swapped:
                artifact.replace(displaced)
                artifact.write_bytes(b"attacker")
                swapped = True
            return original_read(descriptor, length)

        monkeypatch.setattr(os, "read", swap_declared_path)

        with pytest.raises(JunctionBundleError, match="changed after it was verified"):
            snapshot.assert_stable()

    assert swapped
    assert artifact.read_bytes() == b"attacker"


def test_snapshot_rejects_same_inode_rewrite_after_final_content_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.bin"
    artifact.write_bytes(b"verified")

    with open_bundle_snapshot(
        bundle,
        expected_files=frozenset({"artifact.bin"}),
        reject_undeclared_entries=True,
    ) as snapshot:
        snapshot.read_file(
            Path("artifact.bin"),
            limit=64,
            context="test artifact",
            retain_content=False,
        )
        original_rehash = type(snapshot)._assert_retained_contents
        initial = artifact.stat()
        mutated = False

        def mutate_after_rehash(instance):  # type: ignore[no-untyped-def]
            nonlocal mutated
            checkpoints = original_rehash(instance)
            artifact.write_bytes(b"attacker")
            os.utime(artifact, ns=(initial.st_atime_ns, initial.st_mtime_ns))
            mutated = True
            return checkpoints

        monkeypatch.setattr(type(snapshot), "_assert_retained_contents", mutate_after_rehash)

        with pytest.raises(JunctionBundleError, match="changed after it was verified"):
            snapshot.assert_stable()

    assert mutated
    assert artifact.stat().st_ino == initial.st_ino
    assert artifact.read_bytes() == b"attacker"


def test_snapshot_rejects_undeclared_entry_added_after_final_inventory_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.bin"
    artifact.write_bytes(b"verified")
    undeclared = bundle / "undeclared.bin"

    with open_bundle_snapshot(
        bundle,
        expected_files=frozenset({"artifact.bin"}),
        reject_undeclared_entries=True,
    ) as snapshot:
        snapshot.read_file(
            Path("artifact.bin"),
            limit=64,
            context="test artifact",
            retain_content=False,
        )
        original_inventory_check = snapshot_module._reject_undeclared_entries
        initial = bundle.stat()
        checks = 0

        def add_after_final_scan(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal checks
            original_inventory_check(*args, **kwargs)
            checks += 1
            if checks == 2:
                undeclared.write_bytes(b"attacker")
                os.utime(bundle, ns=(initial.st_atime_ns, initial.st_mtime_ns))

        monkeypatch.setattr(snapshot_module, "_reject_undeclared_entries", add_after_final_scan)

        with pytest.raises(JunctionBundleError, match="Bundle directory changed during verification"):
            snapshot.assert_stable()

    assert checks == 2
    assert undeclared.read_bytes() == b"attacker"


def test_existing_bundle_is_not_replaced(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    before = (destination / "manifest.json").read_bytes()

    with pytest.raises(JunctionBundleError, match="already exists; publication is create-only"):
        build(request, destination=destination)

    assert (destination / "manifest.json").read_bytes() == before


def test_tampered_artifact_fails_offline_verification(tmp_path: Path) -> None:
    request = parse_request(_request_mapping())
    destination = tmp_path / "design-v1"
    build(request, destination=destination)
    (destination / "plan.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(JunctionBundleError, match="content identity does not match"):
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

    with pytest.raises(JunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == f"junction manifest is not valid UTF-8 JSON: {destination / 'manifest.json'}"
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

    with pytest.raises(JunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == f"junction manifest is not valid UTF-8 JSON: {destination / 'manifest.json'}"
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

    with pytest.raises(JunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == "Bundle request cannot reproduce a valid junction plan."
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

    with pytest.raises(JunctionBundleError) as raised:
        verify(destination)

    assert str(raised.value) == "Bundle request cannot reproduce a valid junction plan."
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
    mismatched_plan = replace(design_junction(first_request), seed=99)
    destination = tmp_path / "parent" / "design-v1"

    def fail_if_serialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("mismatched inputs must fail before payload construction")

    monkeypatch.setattr(writer_module, "render_artifact_bytes", fail_if_serialized)

    with pytest.raises(JunctionBundleError, match="request does not match"):
        _publish_bundle(second_request, mismatched_plan, destination)

    assert not destination.parent.exists()
