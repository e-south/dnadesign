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

import dnadesign.folding.src.assessment.api as assessment_api
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
        timeout_seconds: float,
    ) -> None:
        original_run_worker(request_path, output_path, timeout_seconds=timeout_seconds)
        target = request_path.parent.parent / "assessment-target-sequence.json"
        target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(assessment_api, "run_worker", run_worker_then_mutate)
    output = tmp_path / "mutated-target-assessment"

    with pytest.raises(FoldingExecutionError, match="target artifact changed"):
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
    original_loader = assessment_api.load_published_assessment

    def corrupt_then_load(output_dir: str | Path):
        root = Path(output_dir)
        prediction = root / "prediction/secondary_structure_prediction_v1.json"
        prediction.write_bytes(prediction.read_bytes() + b" ")
        return original_loader(root)

    monkeypatch.setattr(assessment_api, "load_published_assessment", corrupt_then_load)

    with pytest.raises(ValueError, match="prediction digest"):
        publish_structure_assessment(_request(), output_dir=output)

    assert not output.exists()


def test_structure_assessment_rechecks_path_identity_after_final_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    displaced = tmp_path / "displaced-assessment"
    original_loader = assessment_api.load_published_assessment

    def replace_after_publication(output_dir: str | Path):
        root = Path(output_dir)
        root.rename(displaced)
        shutil.copytree(displaced, root)
        return original_loader(root)

    monkeypatch.setattr(assessment_api, "load_published_assessment", replace_after_publication)

    with pytest.raises(FoldingConfigError, match="path identity changed"):
        publish_structure_assessment(_request(), output_dir=output)


def test_structure_assessment_loader_rejects_prediction_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    prediction = output / "prediction/secondary_structure_prediction_v1.json"
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

    with pytest.raises(ValueError, match="artifact inventory"):
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
