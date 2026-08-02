"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_hardening_contracts.py

Adversarial contracts at TriJunction's public and publication boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.trijunction import build, parse_request, plan, verify
from dnadesign.trijunction.errors import (
    TriJunctionBundleError,
    TriJunctionConfigError,
    TriJunctionDesignError,
)
from dnadesign.trijunction.sequence import reverse_complement


def _target_sequence(*, offset: int = 0, length: int = 72) -> str:
    motif = "ACGATTCGGTACCTGATGCACTGA"
    rotated = motif[offset:] + motif[:offset]
    repeats = (length + len(rotated) - 1) // len(rotated)
    return (rotated * repeats)[:length]


def _target(
    *,
    target_id: str,
    pool_id: str,
    sequence: str,
) -> dict[str, object]:
    return {
        "id": target_id,
        "pool_id": pool_id,
        "sequence": sequence,
        "recovery_primers": {
            "mode": "target_specific",
            "forward": {
                "binding_sequence": sequence[:8],
                "five_prime_extension": "",
            },
            "reverse": {
                "binding_sequence": reverse_complement(sequence[-8:]),
                "five_prime_extension": "",
            },
        },
    }


def _request_mapping(*, targets: list[dict[str, object]] | None = None) -> dict[str, object]:
    if targets is None:
        sequence = _target_sequence()
        targets = [
            _target(
                target_id="target-a",
                pool_id="pool-a",
                sequence=sequence,
            )
        ]
    return {
        "schema": "dnadesign.trijunction.request.v1",
        "seed": 17,
        "planning": {
            "oligo_length": 46,
            "barcode_length": 16,
            "toehold_length": 8,
            "search_range": 2,
            "toehold_search_iterations": 12,
            "barcode_pool_factor": 5,
            "barcode_generation_attempts": 100_000,
            "barcode_toehold_k": 4,
            "barcode_pair_k": 5,
            "barcode_subset_iterations": 12,
            "matching_iterations": 50,
            "barcode_gc_min": 0.25,
            "barcode_gc_max": 0.75,
            "barcode_max_homopolymer": 3,
        },
        "targets": targets,
        "order_policy": {
            "synthesis_scale": "declared test scale",
            "barcode_bearing_purification": "declared test purification",
            "complement_purification": "declared test purification",
            "primer_purification": "declared test purification",
            "complement_end_preparation": "vendor_5_prime_phosphate",
            "max_oligo_length": 64,
        },
    }


def _publish_test_bundle(destination: Path) -> None:
    request = parse_request(_request_mapping())
    build(request, destination=destination)


def test_three_way_junction_rejects_targets_below_junction_geometry() -> None:
    sequence = _target_sequence(length=24)
    request = parse_request(
        _request_mapping(
            targets=[
                _target(
                    target_id="short-junction",
                    pool_id="junction-pool",
                    sequence=sequence,
                )
            ]
        )
    )

    with pytest.raises(TriJunctionDesignError, match="no complete toehold locus"):
        plan(request)


def test_barcode_ids_are_pool_qualified_and_globally_unique() -> None:
    targets = [
        _target(
            target_id="target-a",
            pool_id="pool-a",
            sequence=_target_sequence(),
        ),
        _target(
            target_id="target-b",
            pool_id="pool-b",
            sequence=_target_sequence(offset=3),
        ),
    ]

    result = plan(parse_request(_request_mapping(targets=targets)))
    barcode_ids = [junction.barcode_id for pool in result.pools for junction in pool.junctions]

    assert barcode_ids
    assert len(barcode_ids) == len(set(barcode_ids))
    assert all(
        junction.barcode_id.startswith(f"{pool.pool_id}:barcode-")
        for pool in result.pools
        for junction in pool.junctions
    )


def test_search_receipt_records_every_stage_seed() -> None:
    result = plan(parse_request(_request_mapping()))
    search = result.pools[0].search
    seed_fields = (
        "toehold_seed",
        "barcode_generation_seed",
        "barcode_subset_seed",
        "matching_seed",
    )

    assert all(isinstance(getattr(search, field), int) for field in seed_fields)


def test_published_bundle_uses_private_permissions(tmp_path: Path) -> None:
    destination = tmp_path / "runs" / "design-v1"

    _publish_test_bundle(destination)

    directories = [destination, *(path for path in destination.rglob("*") if path.is_dir())]
    files = [path for path in destination.rglob("*") if path.is_file()]
    assert directories
    assert files
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)


@pytest.mark.parametrize(
    ("entry_kind", "match"),
    [
        ("file", "undeclared file"),
        ("directory", "undeclared directory"),
        ("symlink", "must not contain symlinks"),
    ],
)
def test_verification_rejects_undeclared_entries(
    tmp_path: Path,
    entry_kind: str,
    match: str,
) -> None:
    destination = tmp_path / entry_kind / "design-v1"
    _publish_test_bundle(destination)
    extra = destination / "undeclared"
    if entry_kind == "file":
        extra.write_text("not in manifest\n", encoding="utf-8")
    elif entry_kind == "directory":
        extra.mkdir()
    else:
        extra.symlink_to("request.json")

    with pytest.raises(TriJunctionBundleError, match=match):
        verify(destination)


def test_verification_rejects_artifact_replaced_by_symlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "design-v1"
    _publish_test_bundle(destination)
    request_path = destination / "request.json"
    external_request = tmp_path / "external-request.json"
    external_request.write_bytes(request_path.read_bytes())
    original_path_open = Path.open
    original_os_open = os.open
    replaced = False

    def replace_request_once() -> None:
        nonlocal replaced
        if replaced:
            return
        request_path.unlink()
        request_path.symlink_to(external_request)
        replaced = True

    def racing_path_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == request_path and mode == "rb":
            replace_request_once()
        return original_path_open(path, *args, **kwargs)

    def racing_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == "request.json" and dir_fd is not None:
            replace_request_once()
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "open", racing_path_open)
    monkeypatch.setattr(os, "open", racing_os_open)

    with pytest.raises(TriJunctionBundleError):
        verify(destination)

    assert request_path.is_symlink()


def test_verification_rejects_artifact_replaced_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "design-v1"
    _publish_test_bundle(destination)
    plan_path = destination / "plan.json"
    initial_stat = plan_path.stat()
    replacement = tmp_path / "replacement-plan.json"
    replacement.write_bytes(plan_path.read_bytes())
    original_os_read = os.read
    replaced = False

    def racing_os_read(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        chunk = original_os_read(descriptor, length)
        descriptor_stat = os.fstat(descriptor)
        if (
            not chunk
            and not replaced
            and descriptor_stat.st_dev == initial_stat.st_dev
            and descriptor_stat.st_ino == initial_stat.st_ino
        ):
            plan_path.unlink()
            replacement.replace(plan_path)
            replaced = True
        return chunk

    monkeypatch.setattr(os, "read", racing_os_read)

    with pytest.raises(TriJunctionBundleError):
        verify(destination)

    assert replaced


def test_verification_rejects_undeclared_file_added_after_initial_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "design-v1"
    _publish_test_bundle(destination)
    late_file = destination / "late-undeclared.txt"
    original_os_open = os.open
    introduced = False

    def racing_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal introduced
        if os.fspath(path) == "manifest.json" and dir_fd is not None and not introduced:
            introduced = True
            late_file.write_text("introduced after inventory\n", encoding="utf-8")
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_os_open)

    with pytest.raises(TriJunctionBundleError, match="undeclared file"):
        verify(destination)

    assert introduced


def test_verification_rejects_artifact_mutated_during_late_path_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "design-v1"
    _publish_test_bundle(destination)
    plan_path = destination / "plan.json"
    original_os_open = os.open
    checks_open_count = 0
    mutated = False

    def racing_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal checks_open_count, mutated
        if os.fspath(path) == "checks.json" and dir_fd is not None:
            checks_open_count += 1
            if checks_open_count == 2:
                plan_path.write_text("{}", encoding="utf-8")
                mutated = True
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_os_open)

    with pytest.raises(TriJunctionBundleError):
        verify(destination)

    assert mutated


def test_public_dataclass_construction_rejects_untrimmed_strings() -> None:
    request = parse_request(_request_mapping())
    target = request.targets[0]

    with pytest.raises(TriJunctionConfigError, match="leading or trailing whitespace"):
        replace(target, id=f" {target.id}")
    with pytest.raises(TriJunctionConfigError, match="leading or trailing whitespace"):
        replace(target, pool_id=f"{target.pool_id} ")
    with pytest.raises(TriJunctionConfigError, match="leading or trailing whitespace"):
        replace(request.order_policy, synthesis_scale=f" {request.order_policy.synthesis_scale}")
    with pytest.raises(TriJunctionConfigError):
        replace(target.recovery_primers, mode=f"{target.recovery_primers.mode} ")
    with pytest.raises(TriJunctionConfigError):
        replace(target.recovery_primers, mode=[])  # type: ignore[arg-type]
    with pytest.raises(TriJunctionConfigError):
        replace(request.order_policy, complement_end_preparation=[])  # type: ignore[arg-type]
    with pytest.raises(TriJunctionConfigError):
        replace(request, schema=f"{request.schema} ")


def test_complete_plan_is_stable_across_hash_seeds_and_fresh_processes(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    code = (
        "import json, sys; "
        "from dnadesign.trijunction import plan; "
        "from dnadesign.trijunction.contracts.request import load_request; "
        "print(json.dumps(plan(load_request(sys.argv[1])).to_mapping(), "
        "sort_keys=True, separators=(',', ':')))"
    )
    repo_root = Path(__file__).resolve().parents[4]
    outputs: list[str] = []
    for hash_seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", code, str(request_path)],
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
