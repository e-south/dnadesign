"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/test_manifest.py

Tests for the study-owned projection of canonical Reader plot records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence import (
    READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
    READER_EVIDENCE_SCHEMA_VERSION,
    ReaderPromoterEvidenceError,
    materialize_reader_promoter_evidence_manifest,
    preview_reader_promoter_evidence_manifest,
    verify_reader_promoter_evidence_manifest,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence import (
    manifest as manifest_module,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence import (
    verification as source_verification,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence.cli import (
    main,
)

from ._fixtures import sha256, verified_source, write_candidate_bindings


def test_preview_projects_one_exact_reader_diagnostic_without_study_math(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = verified_source(tmp_path)
    _use_verified_source(monkeypatch, source)

    payload = _preview(tmp_path)

    assert payload["schema_version"] == READER_EVIDENCE_SCHEMA_VERSION
    assert payload["summary"] == {
        "rows": 1,
        "distinct_ids": 1,
        "reader_experiments": 1,
        "artifact_count": 1,
        "missing_artifact_rows": 0,
    }
    row = payload["rows"][0]
    assert set(row) == {
        "id",
        "candidate_id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        "evidence_role",
        "claim_status",
        "non_claim_boundary",
        "selected_binding",
        "sources",
        "artifacts",
    }
    assert row["claim_status"] == "objective_neutral"
    assert "objective_overlay" not in row
    assert "baserender" not in row["sources"]
    response = row["sources"]["response_window"]
    assert response["catalog"]["schema_version"] == 4
    assert response["records"]["designs"]["schema_version"] == 6
    assert response["records"]["traces"]["schema_version"] == 6
    assert response["records"]["diagnostic"]["record_id"] == READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID
    artifact = row["artifacts"][0]
    assert artifact["kind"] == "reader_record_projection"
    assert artifact["record_id"] == READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID
    assert artifact["source_record_revision_digest"] == source.display.record.revision_digest
    assert artifact["source_file_path"] == source.display.selected_file.reader_path
    assert artifact["path"].startswith(
        "reader_evidence_media/" + source.display.record.revision_digest.removeprefix("sha256:")
    )
    serialized = json.dumps(payload)
    assert "reader.response_window.bundle" not in serialized
    assert "promoter_evidence_bundle" not in serialized


def test_materialize_stages_portable_verified_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = verified_source(tmp_path)
    _use_verified_source(monkeypatch, source)
    out_dir = tmp_path / "campaign" / "inputs" / "r0"

    result = _materialize(tmp_path, out_dir=out_dir)
    payload = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    artifact = payload["rows"][0]["artifacts"][0]
    staged = result.manifest_json.parent / artifact["path"]

    assert staged.is_file()
    assert sha256(staged) == artifact["sha256"]
    assert verify_reader_promoter_evidence_manifest(result.manifest_json).artifact_count == 1
    source.display.selected_file.path.write_bytes(b"changed after publication")
    assert verify_reader_promoter_evidence_manifest(result.manifest_json).row_count == 1


def test_materialize_refuses_existing_manifest_and_preserves_it_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = verified_source(tmp_path)
    _use_verified_source(monkeypatch, source)
    out_dir = tmp_path / "campaign" / "inputs" / "r0"
    result = _materialize(tmp_path, out_dir=out_dir)
    before = result.manifest_json.read_bytes()

    with pytest.raises(ReaderPromoterEvidenceError, match="already exists"):
        _materialize(tmp_path, out_dir=out_dir)
    assert result.manifest_json.read_bytes() == before

    real_publish_media = manifest_module._publish_media
    monkeypatch.setattr(
        manifest_module,
        "_publish_media",
        lambda **_: (_ for _ in ()).throw(ReaderPromoterEvidenceError("publish failed")),
    )
    with pytest.raises(ReaderPromoterEvidenceError, match="publish failed"):
        _materialize(tmp_path, out_dir=out_dir, overwrite=True)
    assert result.manifest_json.read_bytes() == before

    monkeypatch.setattr(manifest_module, "_publish_media", real_publish_media)
    real_replace = manifest_module.os.replace

    def fail_manifest_commit(source_path: object, destination_path: object) -> None:
        if Path(destination_path) == result.manifest_json:
            raise OSError("commit failed")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(manifest_module.os, "replace", fail_manifest_commit)
    with pytest.raises(ReaderPromoterEvidenceError, match="Could not publish.*commit failed"):
        _materialize(tmp_path, out_dir=out_dir, overwrite=True)
    assert result.manifest_json.read_bytes() == before


def test_materialize_rejects_symlinked_media_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = verified_source(tmp_path)
    _use_verified_source(monkeypatch, source)
    out_dir = tmp_path / "campaign" / "inputs" / "r0"
    outside = tmp_path / "outside"
    outside.mkdir()
    out_dir.mkdir(parents=True)
    (out_dir / "reader_evidence_media").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReaderPromoterEvidenceError, match="must not be a symlink"):
        _materialize(tmp_path, out_dir=out_dir)

    assert list(outside.iterdir()) == []
    assert not (out_dir / "reader_evidence_promoter_response.json").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["rows"][0]["sources"]["response_window"]["records"]["diagnostic"].__setitem__(
                "revision_digest", "sha256:" + "f" * 64
            ),
            "source binding",
        ),
        (
            lambda payload: payload["rows"][0]["artifacts"][0].__setitem__("source_file_path", "plots/different.png"),
            "file-evidence",
        ),
        (
            lambda payload: payload["rows"][0]["selected_binding"].__setitem__("candidate_id", "different-candidate"),
            "selected binding",
        ),
        (
            lambda payload: payload["rows"][0]["sources"]["response_window"]["records"]["diagnostic"].__setitem__(
                "config_digest", "sha256:" + "f" * 64
            ),
            "one config digest",
        ),
        (
            lambda payload: payload["rows"][0]["sources"]["response_window"]["records"]["diagnostic"][
                "producer"
            ].__setitem__("plugin", "plot/different"),
            "producer identity",
        ),
        (
            lambda payload: payload["rows"][0]["sources"]["response_window"]["records"]["diagnostic"]["inputs"][
                0
            ].__setitem__("record_revision_digest", "sha256:" + "f" * 64),
            "designs input revision",
        ),
    ],
)
def test_display_verifier_rejects_provenance_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    source = verified_source(tmp_path)
    _use_verified_source(monkeypatch, source)
    result = _materialize(tmp_path, out_dir=tmp_path / "campaign" / "inputs" / "r0")
    payload = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    mutate(payload)
    result.manifest_json.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match=message):
        verify_reader_promoter_evidence_manifest(result.manifest_json)


def test_source_verification_joins_canonical_display_to_exact_study_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = verified_source(tmp_path)
    bindings = write_candidate_bindings(tmp_path / "bindings")
    monkeypatch.setattr(source_verification, "load_reader_response_records", lambda **_: source.records)
    monkeypatch.setattr(
        source_verification,
        "load_reader_response_display_record",
        lambda records, **_: source.display,
    )

    resolved = source_verification.verify_reader_promoter_evidence_source(
        reader_root=tmp_path / "reader",
        experiment_root=source.records.experiment_root,
        projection_path=source.records.projection_path,
        bindings_bundle=bindings,
    )

    assert resolved.design_id == "pDual-10-ES1p"
    assert resolved.candidate_id == "candidate-1"
    assert resolved.selected_binding["binding_method"] == "exact_alias"
    assert resolved.binding_source["schema_id"] == "dnadesign.study.promoter_candidate_bindings.v1"


def test_cli_uses_canonical_reader_source_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = verified_source(tmp_path)
    _use_verified_source(monkeypatch, source)
    common = [
        "--reader-root",
        str(tmp_path / "reader"),
        "--experiment-root",
        str(source.records.experiment_root),
        "--projection",
        str(source.records.projection_path),
        "--bindings-bundle",
        str(tmp_path / "bindings"),
    ]
    assert main(["preview", *common]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["rows"] == 1
    out_dir = tmp_path / "campaign" / "inputs" / "r0"
    assert main(["materialize", *common, "--out-dir", str(out_dir)]) == 0
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["schema_version"] == READER_EVIDENCE_SCHEMA_VERSION
    assert main(["verify", materialized["manifest_json"]]) == 0
    assert json.loads(capsys.readouterr().out)["artifact_count"] == 1


def _use_verified_source(monkeypatch: pytest.MonkeyPatch, source: object) -> None:
    monkeypatch.setattr(manifest_module, "verify_reader_promoter_evidence_source", lambda **_: source)


def _preview(tmp_path: Path) -> dict[str, object]:
    return preview_reader_promoter_evidence_manifest(
        reader_root=tmp_path / "reader",
        experiment_root=tmp_path / "experiment",
        projection_path=tmp_path / "projection.yaml",
        bindings_bundle=tmp_path / "bindings",
    )


def _materialize(
    tmp_path: Path,
    *,
    out_dir: Path,
    overwrite: bool = False,
):
    return materialize_reader_promoter_evidence_manifest(
        reader_root=tmp_path / "reader",
        experiment_root=tmp_path / "experiment",
        projection_path=tmp_path / "projection.yaml",
        bindings_bundle=tmp_path / "bindings",
        out_dir=out_dir,
        overwrite=overwrite,
    )
