"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/test_manifest.py

Tests the study-owned Reader promoter-evidence display manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence import (
    ReaderPromoterEvidenceError,
    materialize_reader_promoter_evidence_manifest,
    preview_reader_promoter_evidence_manifest,
    verify_reader_promoter_evidence_manifest,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence import (
    manifest as manifest_module,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence.cli import (
    main,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    materialize_promoter_candidate_bindings,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.tests.promoter_candidate_bindings.test_artifact import (
    preview as candidate_binding_preview,
)

from ._fixtures import sha256 as _sha256
from ._fixtures import write_candidate_bindings as _write_candidate_bindings
from ._fixtures import write_reader_bundle as _write_reader_bundle


def test_preview_builds_display_only_rows_from_verified_reader_bundles(tmp_path: Path) -> None:
    bindings = _write_candidate_bindings(
        tmp_path / "bindings",
        [
            ("candidate-densegen", "pDual-10-ES1p", "densegen_tfbs"),
            ("candidate-genbank", "pDual-10-spyp", "usr_genbank_annotations_v1"),
        ],
    )
    densegen, _ = _write_reader_bundle(
        tmp_path / "densegen",
        candidate_id="candidate-densegen",
        design_id="pDual-10-ES1p",
        experiment_id="20260619_sfxi",
        bindings_bundle=bindings,
    )
    genbank, _ = _write_reader_bundle(
        tmp_path / "genbank",
        candidate_id="candidate-genbank",
        design_id="pDual-10-spyp",
        experiment_id="20260622_sfxi",
        adapter_kind="usr_genbank_annotations_v1",
        bindings_bundle=bindings,
    )

    payload = preview_reader_promoter_evidence_manifest(
        [densegen, genbank],
        bindings_bundle=bindings,
    )

    assert payload["schema_version"] == "stress_ethanol_cipro_growth.reader_promoter_evidence.v1"
    assert payload["campaign_slug"] == "secg_msrb_greedy"
    assert payload["round"] == "r0"
    assert "observed_round" not in payload
    assert "label_input" not in payload
    assert payload["summary"] == {
        "rows": 2,
        "distinct_ids": 2,
        "reader_experiments": 2,
        "artifact_count": 4,
        "missing_artifact_rows": 0,
    }
    densegen_row = payload["rows"][0]
    assert densegen_row == {
        "id": "candidate-densegen",
        "candidate_id": "candidate-densegen",
        "design_id": "pDual-10-ES1p",
        "reader_experiment_id": "20260619_sfxi",
        "reduction_id": "event_logmean_6_12h_post",
        "evidence_role": "display_only",
        "claim_status": "objective_neutral",
        "selected_binding": densegen_row["selected_binding"],
        "binding_source": densegen_row["binding_source"],
        "artifacts": densegen_row["artifacts"],
    }
    assert "time_selected_h" not in densegen_row
    assert "label_input" not in densegen_row
    assert "observed_round" not in densegen_row
    assert densegen_row["selected_binding"]["binding_status"] == "resolved"
    assert densegen_row["selected_binding"]["binding_method"] == "exact_alias"
    assert densegen_row["selected_binding"]["densegen_run_id"] == "reader_sfxi_pdual10_archive_port"
    source_manifest = densegen / "manifest.json"
    for artifact in densegen_row["artifacts"]:
        source_artifact = densegen / Path(artifact["path"]).name
        assert artifact["semantic_kind"] == "promoter_response_evidence"
        assert artifact["kind"] == "reader_publication"
        assert artifact["record_id"] == "reader.response_window.promoter_evidence_bundle.v3"
        assert artifact["scope"] == "design_reduction"
        assert artifact["exists"] is True
        assert artifact["bytes"] == source_artifact.stat().st_size
        assert artifact["sha256"] == _sha256(source_artifact)
        assert not Path(artifact["path"]).is_absolute()
        assert "source_manifest_path" not in artifact
        assert artifact["source_manifest_sha256"] == _sha256(source_manifest)


def test_preview_rejects_reader_claim_that_disagrees_with_explicit_study_bindings(tmp_path: Path) -> None:
    allowed_root = tmp_path / "study"
    bindings = allowed_root / "bindings"
    materialize_promoter_candidate_bindings(
        candidate_binding_preview(one_row=True),
        out_dir=bindings,
        allowed_output_root=allowed_root,
    )
    reader_bundle, _ = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-other",
        design_id="pDual-10-A",
        experiment_id="20260713_sfxi",
    )

    with pytest.raises(ReaderPromoterEvidenceError, match="candidate identity"):
        preview_reader_promoter_evidence_manifest(
            [reader_bundle],
            bindings_bundle=bindings,
        )


def test_preview_accepts_reader_screen_only_overlay_without_promoting_a_score(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "screen-only",
        candidate_id="candidate-screen",
        design_id="pDual-10-screen",
        experiment_id="20260713_sfxi",
        claim_status="screen_only",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objective_overlay"] = {
        "schema_version": "reader.response_window.objective_display_overlay.v1",
        "objective_id": "response_magnitude_feasibility_v1",
        "claim_status": "screen_only",
        "experiment_id": "20260713_sfxi",
        "reader_design_id": "pDual-10-screen",
        "reduction_id": "event_logmean_6_12h_post",
        "manifest_sha256": "sha256:" + "5" * 64,
        "components": [
            {
                "component_id": "on_fluorescence_floor",
                "label": "ON fluorescence floor",
                "value": 1.25,
                "unit": "log2 ratio",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    payload = preview_reader_promoter_evidence_manifest([bundle], bindings_bundle=bindings)

    assert payload["rows"][0]["claim_status"] == "screen_only"
    assert "rmf" not in payload["rows"][0]
    assert "score" not in payload["rows"][0]


def test_materialize_atomically_writes_and_verifies_the_default_manifest(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    output_dir = tmp_path / "campaign" / "inputs" / "r0"

    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        bindings_bundle=bindings,
        out_dir=output_dir,
    )

    assert result.manifest_json == output_dir / "reader_evidence_promoter_response.json"
    assert result.manifest_json.is_file()
    verification = verify_reader_promoter_evidence_manifest(result.manifest_json)
    assert verification.row_count == 1
    assert verification.artifact_count == 2
    published = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    for artifact in published["rows"][0]["artifacts"]:
        staged_media = result.manifest_json.parent / artifact["path"]
        assert staged_media.is_file()
        assert _sha256(staged_media) == artifact["sha256"]
    original = result.manifest_json.read_bytes()
    with pytest.raises(ReaderPromoterEvidenceError, match="already exists"):
        materialize_reader_promoter_evidence_manifest(
            [bundle],
            bindings_bundle=bindings,
            out_dir=output_dir,
        )
    assert result.manifest_json.read_bytes() == original


def test_preview_uses_portable_content_addressed_media_references(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )

    payload = preview_reader_promoter_evidence_manifest([bundle], bindings_bundle=bindings)

    artifacts = payload["rows"][0]["artifacts"]
    for artifact in artifacts:
        assert not Path(artifact["path"]).is_absolute()
        assert artifact["path"].startswith("reader_evidence_media/")
        assert "source_manifest_path" not in artifact


def test_published_evidence_verifies_after_source_bundles_are_removed(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        bindings_bundle=bindings,
        out_dir=tmp_path / "campaign" / "inputs" / "r0",
    )
    for source_bundle in (bundle, bindings):
        for path in source_bundle.iterdir():
            path.unlink()
        source_bundle.rmdir()

    verified = verify_reader_promoter_evidence_manifest(result.manifest_json)

    assert verified.row_count == 1
    assert verified.artifact_count == 2


def test_preview_rejects_screen_overlay_with_more_than_six_components(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "too-many-components",
        candidate_id="candidate-screen",
        design_id="pDual-10-screen",
        experiment_id="20260713_sfxi",
        claim_status="screen_only",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objective_overlay"] = {
        "schema_version": "reader.response_window.objective_display_overlay.v1",
        "objective_id": "response_magnitude_feasibility_v1",
        "claim_status": "screen_only",
        "experiment_id": "20260713_sfxi",
        "reader_design_id": "pDual-10-screen",
        "reduction_id": "event_logmean_6_12h_post",
        "manifest_sha256": "sha256:" + "5" * 64,
        "components": [
            {"component_id": f"c{index}", "label": f"C {index}", "value": 1.0, "unit": "log2 ratio"}
            for index in range(7)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="between one and six"):
        preview_reader_promoter_evidence_manifest([bundle], bindings_bundle=bindings)


def test_cli_previews_materializes_and_verifies_without_label_mutation(tmp_path: Path, capsys) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    out_dir = tmp_path / "campaign" / "inputs" / "r0"

    assert main(["preview", "--bindings-bundle", str(bindings), str(bundle)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["summary"]["rows"] == 1
    assert (
        main(
            [
                "materialize",
                "--bindings-bundle",
                str(bindings),
                "--out-dir",
                str(out_dir),
                str(bundle),
            ]
        )
        == 0
    )
    written = json.loads(capsys.readouterr().out)
    assert written["row_count"] == 1
    manifest_path = Path(written["manifest_json"])
    assert main(["verify", str(manifest_path)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == {
        "artifact_count": 2,
        "manifest_json": str(manifest_path),
        "row_count": 1,
        "schema_version": "stress_ethanol_cipro_growth.reader_promoter_evidence.v1",
    }


@pytest.mark.parametrize(
    ("adapter_kind", "field", "value", "message"),
    [
        ("densegen_tfbs", "reader_design_id", "pDual-10-other", "selection design_id"),
        ("densegen_tfbs", "candidate_id", "candidate-other", "selection candidate_id"),
        ("densegen_tfbs", "binding_method", "prefix_alias", "exact-binding provenance"),
        ("densegen_tfbs", "densegen_plan", None, "requires selected_binding plan"),
        ("usr_genbank_annotations_v1", "densegen_run_id", "unexpected-run", "requires null DenseGen"),
        ("usr_genbank_annotations_v1", "source_class", "", "provenance is malformed"),
    ],
)
def test_preview_rejects_malformed_selected_binding_provenance(
    tmp_path: Path,
    adapter_kind: str,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
        adapter_kind=adapter_kind,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_binding"][field] = value
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match=message):
        preview_reader_promoter_evidence_manifest([bundle], bindings_bundle=bindings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "different-experiment"),
        ("reduction_id", "different-reduction"),
    ],
)
def test_preview_rejects_response_source_identity_that_disagrees_with_selection(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"]["response_window"][field] = value
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match=rf"{field}.*selection {field}"):
        preview_reader_promoter_evidence_manifest([bundle], bindings_bundle=bindings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "different-experiment"),
        ("reader_design_id", "pDual-10-other"),
        ("reduction_id", "different-reduction"),
    ],
)
def test_preview_rejects_objective_overlay_identity_that_disagrees_with_selection(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "screen-only",
        candidate_id="candidate-screen",
        design_id="pDual-10-screen",
        experiment_id="20260713_sfxi",
        claim_status="screen_only",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objective_overlay"] = {
        "schema_version": "reader.response_window.objective_display_overlay.v1",
        "objective_id": "response_magnitude_feasibility_v1",
        "claim_status": "screen_only",
        "experiment_id": "20260713_sfxi",
        "reader_design_id": "pDual-10-screen",
        "reduction_id": "event_logmean_6_12h_post",
        "manifest_sha256": "sha256:" + "5" * 64,
        "components": [
            {
                "component_id": "on_fluorescence_floor",
                "label": "ON fluorescence floor",
                "value": 1.25,
                "unit": "log2 ratio",
            }
        ],
    }
    manifest["objective_overlay"][field] = value
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match=rf"{field}.*selection"):
        preview_reader_promoter_evidence_manifest([bundle], bindings_bundle=bindings)


def test_preview_rejects_artifact_digest_and_signature_tampering(tmp_path: Path) -> None:
    digest_bundle, digest_bindings = _write_reader_bundle(
        tmp_path / "digest",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    (digest_bundle / "promoter_evidence.png").write_bytes(b"\x89PNG\r\n\x1a\ntampered")
    with pytest.raises(ReaderPromoterEvidenceError, match="digest or size mismatch"):
        preview_reader_promoter_evidence_manifest(
            [digest_bundle],
            bindings_bundle=digest_bindings,
        )

    signature_bundle, signature_bindings = _write_reader_bundle(
        tmp_path / "signature",
        candidate_id="candidate-2",
        design_id="pDual-10-2",
        experiment_id="20260713_sfxi",
    )
    png_path = signature_bundle / "promoter_evidence.png"
    png_path.write_bytes(b"not-a-png-signature")
    manifest_path = signature_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][png_path.name] = {
        "path": png_path.name,
        "bytes": png_path.stat().st_size,
        "sha256": _sha256(png_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ReaderPromoterEvidenceError, match="PNG signature is invalid"):
        preview_reader_promoter_evidence_manifest(
            [signature_bundle],
            bindings_bundle=signature_bindings,
        )


def test_preview_rejects_duplicate_selection_identity(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )

    with pytest.raises(ReaderPromoterEvidenceError, match="duplicate selection identity"):
        preview_reader_promoter_evidence_manifest(
            [bundle, bundle],
            bindings_bundle=bindings,
        )


def test_materialize_preserves_prior_manifest_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    out_dir = tmp_path / "campaign" / "inputs" / "r0"
    first = materialize_reader_promoter_evidence_manifest(
        [bundle],
        bindings_bundle=bindings,
        out_dir=out_dir,
    )
    original = first.manifest_json.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        materialize_reader_promoter_evidence_manifest(
            [bundle],
            bindings_bundle=bindings,
            out_dir=out_dir,
            overwrite=True,
        )

    assert first.manifest_json.read_bytes() == original
    assert not list(out_dir.glob(".*.staging-*"))
    assert all(path.is_dir() and any(path.iterdir()) for path in out_dir.glob("reader_evidence_media"))


def test_display_verifier_rejects_artifact_metadata_tampering(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        bindings_bundle=bindings,
        out_dir=tmp_path / "campaign" / "inputs" / "r0",
    )
    display = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    display["rows"][0]["artifacts"][0]["sha256"] = "sha256:" + "0" * 64
    result.manifest_json.write_text(json.dumps(display, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="digest or size mismatch"):
        verify_reader_promoter_evidence_manifest(result.manifest_json)


def test_display_verifier_rejects_selected_binding_identity_tampering(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        bindings_bundle=bindings,
        out_dir=tmp_path / "campaign" / "inputs" / "r0",
    )
    display = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    display["rows"][0]["selected_binding"]["candidate_id"] = "candidate-other"
    result.manifest_json.write_text(json.dumps(display, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="identity disagrees with its row"):
        verify_reader_promoter_evidence_manifest(result.manifest_json)


def test_display_verifier_rejects_a_different_campaign_destination(tmp_path: Path) -> None:
    bundle, bindings = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        bindings_bundle=bindings,
        out_dir=tmp_path / "campaign" / "inputs" / "r0",
    )
    display = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    display["campaign_slug"] = "secg_cipro_rf_sfxi_topn"
    result.manifest_json.write_text(json.dumps(display, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="secg_msrb_greedy"):
        verify_reader_promoter_evidence_manifest(result.manifest_json)
