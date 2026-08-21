"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/test_assessment_publication.py

Atomic, isolated publication of digest-addressed structure assessments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import dnadesign.artifacts.publication as artifact_publication
import dnadesign.folding.src.assessment.api as assessment_api
import dnadesign.folding.src.assessment.publication as assessment_publication
from dnadesign.folding import (
    FoldingConfigError,
    FoldingExecutionError,
    load_published_assessment,
    publish_structure_assessment,
)
from dnadesign.folding.tests._assessment_fixtures import (
    assessment_request as _request,
)
from dnadesign.folding.tests._assessment_fixtures import (
    install_fake_rna_module as _install_fake_rna_module,
)


def _optional_missing_request():
    base_request = _request()
    return base_request.model_copy(
        update={
            "backend": base_request.backend.model_copy(update={"python_module": "dnadesign_missing_rna_backend"}),
            "policy": base_request.policy.model_copy(update={"required": False}),
        },
    )


def _optional_missing_cli_request():
    base_request = _request()
    payload = base_request.model_dump(mode="python")
    payload["backend"].update(
        {
            "interface": "cli",
            "executable": "dnadesign-missing-rnafold",
            "python_module": None,
        }
    )
    payload["policy"]["required"] = False
    return type(base_request).model_validate(payload)


def _json_content(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _content_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def test_structure_assessment_publication_round_trips_exact_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"

    published = publish_structure_assessment(_request(), output_dir=output)
    replayed = load_published_assessment(output)

    assert published == replayed
    assert published.record.authority == "advisory"
    assert published.record.target.state_type == "hairpin_encoding_insert"
    assert published.record.prediction.status == "ok"
    assert published.record.prediction.backend is not None
    assert published.record.prediction.backend.version == "test-1.0"
    preflight = json.loads((output / "prediction/folding_preflight.json").read_text(encoding="utf-8"))
    assert preflight["output_dir"] == "."


def test_structure_assessment_timeout_leaves_no_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch, delay_seconds=2.0)
    output = tmp_path / "timed-out-assessment"

    with pytest.raises(FoldingExecutionError, match="timed out"):
        publish_structure_assessment(_request(timeout_seconds=0.1), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_backend_target_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_mutate(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        target = request_path.parent.parent / "assessment-target-sequence.json"
        target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-target-assessment"

    with pytest.raises(FoldingExecutionError, match="target artifact changed"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_named_worker_artifact_symlink_before_supervisor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    outside = tmp_path / "large-outside-request"
    with outside.open("wb") as handle:
        handle.truncate(4 * 1024 * 1024 * 1024)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_replace_request(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        request_path.unlink()
        request_path.symlink_to(outside)

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_replace_request)
    output = tmp_path / "linked-worker-request-assessment"

    with pytest.raises(ValueError, match="worker request.*symbolic link"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_verifies_the_copied_snapshot_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_copy_directory = artifact_publication._copy_directory
    mutated = False

    def mutate_stage_then_copy(source: Path | int, parent_descriptor: int, name: str) -> None:
        nonlocal mutated
        if isinstance(source, int) and not mutated:
            descriptor = os.open(
                "assessment-record.json",
                os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW,
                dir_fd=source,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"{}\n")
            mutated = True
        original_copy_directory(source, parent_descriptor, name)

    monkeypatch.setattr(artifact_publication, "_copy_directory", mutate_stage_then_copy)
    output = tmp_path / "mutated-during-copy-assessment"

    with pytest.raises(ValueError):
        publish_structure_assessment(_request(), output_dir=output)

    assert mutated
    assert not output.exists()


def test_structure_assessment_rejects_worker_request_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_mutate(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["policy"]["required"] = False
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-request-assessment"

    with pytest.raises(ValueError, match="worker request does not match"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_staged_symlink_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    outside = tmp_path / "large-outside-artifact"
    with outside.open("wb") as handle:
        handle.truncate(4 * 1024 * 1024 * 1024)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_link(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        (output_path / "backend-extra").symlink_to(outside)

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_link)
    output = tmp_path / "linked-artifact-assessment"

    with pytest.raises(ValueError, match="artifact inventory cannot use a symbolic link"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_preflight_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_mutate(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        preflight_path = output_path / "folding_preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["backend"]["version"] = "fabricated-version"
        preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-preflight-assessment"

    with pytest.raises(ValueError, match="preflight backend version"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_prediction_execution_metadata_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_mutate(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        prediction_path = output_path / "secondary_structure_prediction_v2.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        prediction["backend"]["parameters"] = {"temperature_c": 25.0}
        prediction_path.write_text(json.dumps(prediction, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-execution-metadata-assessment"

    with pytest.raises(ValueError, match="execution metadata"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_prediction_log_reference_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_mutate(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        prediction_path = output_path / "secondary_structure_prediction_v2.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        prediction["artifacts"] = {
            "stdout": "folding_preflight.json",
            "stderr": "folding_preflight.json",
        }
        prediction_path.write_text(json.dumps(prediction, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-log-reference-assessment"

    with pytest.raises(ValueError, match="log references"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rejects_prediction_result_without_log_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    original_run_worker = assessment_api.run_worker

    def run_worker_then_mutate(
        request_path: Path,
        output_path: Path,
        *,
        artifact_root_descriptor: int,
        timeout_seconds: float,
    ) -> None:
        original_run_worker(
            request_path,
            output_path,
            artifact_root_descriptor=artifact_root_descriptor,
            timeout_seconds=timeout_seconds,
        )
        prediction_path = output_path / "secondary_structure_prediction_v2.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        prediction["result"]["mfe_kcal_mol"] = -9.9
        prediction_path.write_text(json.dumps(prediction, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-result-assessment"

    with pytest.raises(ValueError, match="backend output evidence"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_publication_is_create_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)

    with pytest.raises(FoldingConfigError, match="already exists|create-only"):
        publish_structure_assessment(_request(), output_dir=output)


def test_structure_assessment_final_replay_failure_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    original_verify = assessment_api.verify_publication
    verification_count = 0

    def corrupt_then_verify(*args, **kwargs):
        nonlocal verification_count
        verification_count += 1
        if verification_count == 3:
            prediction = output / "prediction/secondary_structure_prediction_v2.json"
            prediction.write_bytes(prediction.read_bytes() + b" ")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(assessment_api, "verify_publication", corrupt_then_verify)

    with pytest.raises(ValueError, match="prediction digest"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_final_replay_must_equal_the_verified_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    original_verify = assessment_api.verify_publication
    verification_count = 0

    def verify_different_valid_object(*args, **kwargs):
        nonlocal verification_count
        verification_count += 1
        loaded = original_verify(*args, **kwargs)
        if verification_count != 3:
            return loaded
        return type(loaded)(
            manifest=loaded.manifest,
            request=loaded.request,
            record=loaded.record.model_copy(update={"assessment_id": "different-assessment"}),
        )

    monkeypatch.setattr(assessment_api, "verify_publication", verify_different_valid_object)

    with pytest.raises(FoldingExecutionError, match="does not match the verified staging assessment"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rechecks_path_identity_after_final_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    displaced = tmp_path / "displaced-assessment"
    original_verify = assessment_api.verify_publication
    verification_count = 0

    def replace_after_publication(*args, **kwargs):
        nonlocal verification_count
        verification_count += 1
        if verification_count == 3:
            output.rename(displaced)
            shutil.copytree(displaced, output)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(assessment_api, "verify_publication", replace_after_publication)

    with pytest.raises(FoldingConfigError, match="path identity changed"):
        publish_structure_assessment(_request(), output_dir=output)


def test_assessment_inventory_bounds_entry_count(tmp_path: Path) -> None:
    root = tmp_path / "assessment"
    root.mkdir()
    for index in range(assessment_publication.ARTIFACT_ENTRY_COUNT_LIMIT + 1):
        (root / f"artifact-{index:03d}").touch()

    with pytest.raises(ValueError, match=rf"{assessment_publication.ARTIFACT_ENTRY_COUNT_LIMIT}-entry limit"):
        assessment_publication.artifact_digests(root)


def test_assessment_inventory_bounds_aggregate_bytes(tmp_path: Path) -> None:
    root = tmp_path / "assessment"
    root.mkdir()
    artifact_size = 1_000_000
    artifact_count = (assessment_publication.ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES // artifact_size) + 1
    for index in range(artifact_count):
        with (root / f"artifact-{index:03d}").open("wb") as handle:
            handle.truncate(artifact_size)

    with pytest.raises(
        ValueError,
        match=rf"{assessment_publication.ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES}-byte aggregate limit",
    ):
        assessment_publication.artifact_digests(root)


def test_structure_assessment_loader_rejects_prediction_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    prediction = output / "prediction/secondary_structure_prediction_v2.json"
    prediction.write_bytes(prediction.read_bytes() + b" ")

    with pytest.raises(ValueError, match="prediction digest"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_target_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    target = output / "assessment-target-sequence.json"
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ValueError, match="target-sequence artifact digest"):
        load_published_assessment(output)


def test_structure_assessment_loader_replays_target_artifact_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    target_path = output / "assessment-target-sequence.json"
    manifest_path = output / "manifest.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["sequence"]["sequence"] = "ACATGC"
    target["sequence"]["sha256"] = hashlib.sha256(target["sequence"]["sequence"].encode()).hexdigest()
    target_content = (json.dumps(target, indent=2, sort_keys=True) + "\n").encode()
    target_path.write_bytes(target_content)
    target_digest = f"sha256:{hashlib.sha256(target_content).hexdigest()}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_sequence_artifact_digest"] = target_digest
    manifest["artifact_digests"]["assessment-target-sequence.json"] = target_digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target artifact does not match"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_unlisted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    (output / "unlisted.txt").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact inventory"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_staging_owner_in_published_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    (output / ".dnadesign-publication-owner.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="transaction metadata"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_inventoried_staging_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    owner_path = output / ".dnadesign-publication-owner.json"
    owner_content = b"{}\n"
    owner_path.write_bytes(owner_content)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_digests"][owner_path.name] = f"sha256:{hashlib.sha256(owner_content).hexdigest()}"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="transaction metadata"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_portable_staging_owner_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    owner_alias = output / ".DNADESIGN-PUBLICATION-OWNER.JSON"
    owner_content = b"{}\n"
    owner_alias.write_bytes(owner_content)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_digests"][owner_alias.name] = _content_digest(owner_content)
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="reserved publication owner metadata"):
        load_published_assessment(output)


@pytest.mark.skipif(os.name != "posix", reason="FIFO contract is POSIX-specific")
def test_structure_assessment_loader_rejects_special_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    os.mkfifo(output / "unexpected.fifo")

    with pytest.raises(ValueError, match="unsupported filesystem entry"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_symlinked_nested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    prediction = output / "prediction"
    relocated = output / "prediction-data"
    prediction.rename(relocated)
    prediction.symlink_to(relocated.name, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        load_published_assessment(output)


def test_structure_assessment_loader_anchors_every_publication_root_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    published_parent = tmp_path / "published-parent"
    replacement_parent = tmp_path / "replacement-parent"
    output = published_parent / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    publish_structure_assessment(_request(), output_dir=replacement_parent / "assessment")
    retained_parent = tmp_path / "retained-parent"
    original_open = assessment_publication.os.open
    swapped = False

    def open_after_ancestor_swap(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o600,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == published_parent.name and dir_fd is not None and not swapped:
            published_parent.rename(retained_parent)
            published_parent.symlink_to(replacement_parent, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(assessment_publication.os, "open", open_after_ancestor_swap)

    with pytest.raises(ValueError, match="publication root is missing or unsafe"):
        load_published_assessment(output)

    assert swapped


def test_structure_assessment_loader_reads_and_validates_one_anchored_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    request_path = output / "assessment-request.json"
    original_path = output / "assessment-request.original.json"
    outside = tmp_path / "replacement-request.json"
    outside.write_text("{}\n", encoding="utf-8")
    original_open_file = assessment_publication._AnchoredPublicationReader._open_file
    replaced = False

    def open_file_then_replace_path(
        reader: assessment_publication._AnchoredPublicationReader,
        relative: str,
        *,
        label: str,
    ) -> int:
        nonlocal replaced
        descriptor = original_open_file(reader, relative, label=label)
        if relative == "assessment-request.json" and not replaced:
            request_path.rename(original_path)
            request_path.symlink_to(outside)
            replaced = True
        return descriptor

    monkeypatch.setattr(
        assessment_publication._AnchoredPublicationReader,
        "_open_file",
        open_file_then_replace_path,
    )

    with pytest.raises(ValueError, match="artifact inventory cannot use a symbolic link"):
        load_published_assessment(output)

    assert replaced


def test_structure_assessment_loader_enumerates_the_anchored_publication_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    replacement = tmp_path / "replacement-assessment"
    detached = tmp_path / "detached-assessment"
    publish_structure_assessment(_request(), output_dir=output)
    publish_structure_assessment(_request(), output_dir=replacement)
    (output / "hidden.txt").write_text("not inventoried\n", encoding="utf-8")
    original_read_bytes = assessment_publication._AnchoredPublicationReader.read_bytes
    swapped = False

    def read_bytes_then_replace_publication(
        reader: assessment_publication._AnchoredPublicationReader,
        relative: str,
        *,
        label: str,
    ) -> bytes:
        nonlocal swapped
        content = original_read_bytes(reader, relative, label=label)
        if relative == "assessment-record.json" and not swapped:
            output.rename(detached)
            replacement.rename(output)
            swapped = True
        return content

    monkeypatch.setattr(
        assessment_publication._AnchoredPublicationReader,
        "read_bytes",
        read_bytes_then_replace_publication,
    )

    with pytest.raises(ValueError, match="artifact inventory does not match"):
        load_published_assessment(output)

    assert swapped


def test_structure_assessment_loader_rejects_record_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["assessment_id"] = "different-assessment"
    record_content = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    record_path.write_bytes(record_content)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_digest = f"sha256:{hashlib.sha256(record_content).hexdigest()}"
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="assessment_id must match"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_optional_status_for_required_policy(
    tmp_path: Path,
) -> None:
    output = tmp_path / "assessment"
    publish_structure_assessment(_optional_missing_request(), output_dir=output)

    request_path = output / "assessment-request.json"
    worker_request_path = output / "prediction/prediction-request.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    worker_request = json.loads(worker_request_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request["policy"]["required"] = True
    worker_request["policy"]["required"] = True
    request_content = _json_content(request)
    worker_request_content = _json_content(worker_request)
    request_digest = _content_digest(request_content)
    worker_request_digest = _content_digest(worker_request_content)
    request_path.write_bytes(request_content)
    worker_request_path.write_bytes(worker_request_content)
    record["request_digest"] = request_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    record_path.write_bytes(record_content)
    manifest["request_digest"] = request_digest
    manifest["worker_request_digest"] = worker_request_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["assessment-request.json"] = request_digest
    manifest["artifact_digests"]["prediction/prediction-request.json"] = worker_request_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required assessment cannot replay non-ok status"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_version_for_unavailable_backend(tmp_path: Path) -> None:
    output = tmp_path / "assessment"
    publish_structure_assessment(_optional_missing_cli_request(), output_dir=output)
    preflight_path = output / "prediction/folding_preflight.json"
    manifest_path = output / "manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight["backend"]["version"] = "fabricated-version"
    preflight_content = _json_content(preflight)
    preflight_path.write_bytes(preflight_content)
    manifest["artifact_digests"]["prediction/folding_preflight.json"] = _content_digest(preflight_content)
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="preflight blocker does not match"):
        load_published_assessment(output)


def test_structure_assessment_loader_derives_optional_missing_status_from_policy(tmp_path: Path) -> None:
    output = tmp_path / "assessment"
    publish_structure_assessment(_optional_missing_request(), output_dir=output)
    preflight_path = output / "prediction/folding_preflight.json"
    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight["status"] = "blocker_required_missing"
    prediction["status"] = "blocker_required_missing"
    record["status"] = "blocker_required_missing"
    record["prediction"]["status"] = "blocker_required_missing"
    preflight_content = _json_content(preflight)
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    preflight_path.write_bytes(preflight_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/folding_preflight.json"] = _content_digest(preflight_content)
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="preflight blocker does not match"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_derived_qa_when_backend_did_not_run(tmp_path: Path) -> None:
    output = tmp_path / "assessment"
    publish_structure_assessment(_optional_missing_request(), output_dir=output)
    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction["qa"]["pairing_summary"] = {
        "predicted_pair_count": 0,
        "cross_copy_pair_count": 0,
        "intended_pairing_count": 0,
        "intended_recovered_count": 0,
        "intended_partially_recovered_count": 0,
        "intended_missed_count": 0,
    }
    record["prediction"] = prediction
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="derived QA without backend execution"):
        load_published_assessment(output)


def test_structure_assessment_loader_derives_missing_backend_diagnostic_from_request(tmp_path: Path) -> None:
    output = tmp_path / "assessment"
    publish_structure_assessment(_optional_missing_request(), output_dir=output)
    preflight_path = output / "prediction/folding_preflight.json"
    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight["warnings"] = []
    prediction["qa"]["warnings"] = []
    record["prediction"] = prediction
    preflight_content = _json_content(preflight)
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    preflight_path.write_bytes(preflight_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/folding_preflight.json"] = _content_digest(preflight_content)
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="derived QA without backend execution"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_missing_status_after_successful_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    request = _request().model_copy(
        update={"policy": _request().policy.model_copy(update={"required": False})},
    )
    output = tmp_path / "assessment"
    publish_structure_assessment(request, output_dir=output)
    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction["status"] = "warning_optional_missing"
    prediction["result"] = None
    record["status"] = "warning_optional_missing"
    record["prediction"]["status"] = "warning_optional_missing"
    record["prediction"]["result"] = None
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="successful assessment preflight.*impossible prediction status"):
        load_published_assessment(output)


def test_structure_assessment_loader_replays_claimed_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    request = _request().model_copy(
        update={"policy": _request().policy.model_copy(update={"required": False})},
    )
    output = tmp_path / "assessment"
    publish_structure_assessment(request, output_dir=output)
    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction["status"] = "error"
    prediction["result"] = None
    prediction["failure"] = {
        "kind": "output_parse_exception",
        "message": "fabricated parse failure",
        "exception_type": "FoldingMalformedOutputError",
        "returncode": None,
    }
    prediction["qa"] = {
        "cross_copy_pairings": [],
        "errors": ["fabricated parse failure"],
        "intended_pairings": [],
        "length_matches_input": None,
        "pairing_summary": None,
        "warnings": [],
    }
    record["status"] = "error"
    record["prediction"] = prediction
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="parse-failure claim contradicts successful backend output replay"):
        load_published_assessment(output)


def test_structure_assessment_loader_requires_diagnostics_for_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    (tmp_path / "fake-backend/RNA.py").write_text(
        "__version__ = 'test-1.0'\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        "        return '.....', -1.2\n"
        "def fold_compound(sequence):\n"
        "    return Compound()\n",
        encoding="utf-8",
    )
    request = _request().model_copy(
        update={"policy": _request().policy.model_copy(update={"required": False, "fail_on_length_mismatch": False})},
    )
    output = tmp_path / "assessment"
    published = publish_structure_assessment(request, output_dir=output)
    assert published.record.status == "error"

    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction["qa"]["errors"] = []
    record["prediction"]["qa"]["errors"] = []
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="execution error lacks canonical diagnostic evidence"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_derived_qa_for_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    (tmp_path / "fake-backend/RNA.py").write_text(
        "__version__ = 'test-1.0'\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        "        return '.....', -1.2\n"
        "def fold_compound(sequence):\n"
        "    return Compound()\n",
        encoding="utf-8",
    )
    request = _request().model_copy(
        update={"policy": _request().policy.model_copy(update={"required": False, "fail_on_length_mismatch": False})},
    )
    output = tmp_path / "assessment"
    published = publish_structure_assessment(request, output_dir=output)
    assert published.record.status == "error"

    prediction_path = output / "prediction/secondary_structure_prediction_v2.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction["qa"]["pairing_summary"] = {
        "predicted_pair_count": 0,
        "cross_copy_pair_count": 0,
        "intended_pairing_count": 0,
        "intended_recovered_count": 0,
        "intended_partially_recovered_count": 0,
        "intended_missed_count": 0,
    }
    record["prediction"] = prediction
    prediction_content = _json_content(prediction)
    prediction_digest = _content_digest(prediction_content)
    record["prediction_digest"] = prediction_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    prediction_path.write_bytes(prediction_content)
    record_path.write_bytes(record_content)
    manifest["prediction_digest"] = prediction_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["prediction/secondary_structure_prediction_v2.json"] = prediction_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="execution error lacks canonical diagnostic evidence"):
        load_published_assessment(output)


def test_structure_assessment_loader_enforces_parse_failure_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    (tmp_path / "fake-backend/RNA.py").write_text(
        "__version__ = 'test-1.0'\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        "        return '.....', -1.2\n"
        "def fold_compound(sequence):\n"
        "    return Compound()\n",
        encoding="utf-8",
    )
    request = _request().model_copy(
        update={"policy": _request().policy.model_copy(update={"required": False, "fail_on_length_mismatch": False})},
    )
    output = tmp_path / "assessment"
    published = publish_structure_assessment(request, output_dir=output)
    assert published.record.prediction.failure is not None
    assert published.record.prediction.failure.kind == "output_parse_exception"

    request_path = output / "assessment-request.json"
    worker_request_path = output / "prediction/prediction-request.json"
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    worker_request = json.loads(worker_request_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request_payload["policy"]["fail_on_length_mismatch"] = True
    worker_request["policy"]["fail_on_length_mismatch"] = True
    request_content = _json_content(request_payload)
    worker_request_content = _json_content(worker_request)
    request_digest = _content_digest(request_content)
    worker_request_digest = _content_digest(worker_request_content)
    record["request_digest"] = request_digest
    record_content = _json_content(record)
    record_digest = _content_digest(record_content)
    request_path.write_bytes(request_content)
    worker_request_path.write_bytes(worker_request_content)
    record_path.write_bytes(record_content)
    manifest["request_digest"] = request_digest
    manifest["worker_request_digest"] = worker_request_digest
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["assessment-request.json"] = request_digest
    manifest["artifact_digests"]["prediction/prediction-request.json"] = worker_request_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_bytes(_json_content(manifest))

    with pytest.raises(ValueError, match="parse-failure status contradicts the persisted failure policy"):
        load_published_assessment(output)
