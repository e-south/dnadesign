"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/test_manifest.py

Tests the study-owned Reader promoter-evidence display manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
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


def test_preview_builds_display_only_rows_from_verified_reader_bundles(tmp_path: Path) -> None:
    densegen = _write_reader_bundle(
        tmp_path / "densegen",
        candidate_id="candidate-densegen",
        design_id="pDual-10-ES1p",
        experiment_id="20260619_sfxi",
    )
    genbank = _write_reader_bundle(
        tmp_path / "genbank",
        candidate_id="candidate-genbank",
        design_id="pDual-10-spyp",
        experiment_id="20260622_sfxi",
        adapter_kind="usr_genbank_annotations_v1",
    )

    payload = preview_reader_promoter_evidence_manifest([densegen, genbank])

    assert payload["schema_version"] == "stress_ethanol_cipro_growth.reader_evidence.v1"
    assert payload["campaign_slug"] == "secg_rmf_greedy"
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
        assert artifact["semantic_kind"] == "promoter_response_evidence"
        assert artifact["kind"] == "reader_publication"
        assert artifact["record_id"] == "reader.response_window.promoter_evidence_bundle.v1"
        assert artifact["scope"] == "design_reduction"
        assert artifact["exists"] is True
        assert artifact["bytes"] == Path(artifact["path"]).stat().st_size
        assert artifact["sha256"] == _sha256(Path(artifact["path"]))
        assert artifact["source_manifest_path"] == str(source_manifest.resolve())
        assert artifact["source_manifest_sha256"] == _sha256(source_manifest)


def test_preview_accepts_reader_screen_only_overlay_without_promoting_a_score(tmp_path: Path) -> None:
    bundle = _write_reader_bundle(
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

    payload = preview_reader_promoter_evidence_manifest([bundle])

    assert payload["rows"][0]["claim_status"] == "screen_only"
    assert "rmf" not in payload["rows"][0]
    assert "score" not in payload["rows"][0]


def test_materialize_atomically_writes_and_verifies_the_default_manifest(tmp_path: Path) -> None:
    bundle = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    output_dir = tmp_path / "campaign" / "inputs" / "r0"

    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        out_dir=output_dir,
    )

    assert result.manifest_json == output_dir / "reader_evidence_promoter_response.json"
    assert result.manifest_json.is_file()
    verification = verify_reader_promoter_evidence_manifest(result.manifest_json)
    assert verification.row_count == 1
    assert verification.artifact_count == 2
    original = result.manifest_json.read_bytes()
    with pytest.raises(ReaderPromoterEvidenceError, match="already exists"):
        materialize_reader_promoter_evidence_manifest(
            [bundle],
            out_dir=output_dir,
        )
    assert result.manifest_json.read_bytes() == original


def test_preview_rejects_screen_overlay_with_more_than_six_components(tmp_path: Path) -> None:
    bundle = _write_reader_bundle(
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
        "manifest_sha256": "sha256:" + "5" * 64,
        "components": [
            {"component_id": f"c{index}", "label": f"C {index}", "value": 1.0, "unit": "log2 ratio"}
            for index in range(7)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="between one and six"):
        preview_reader_promoter_evidence_manifest([bundle])


def test_cli_previews_materializes_and_verifies_without_label_mutation(tmp_path: Path, capsys) -> None:
    bundle = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    out_dir = tmp_path / "campaign" / "inputs" / "r0"

    assert main(["preview", str(bundle)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["summary"]["rows"] == 1
    assert (
        main(
            [
                "materialize",
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
        "schema_version": "stress_ethanol_cipro_growth.reader_evidence.v1",
    }


@pytest.mark.parametrize(
    ("adapter_kind", "field", "value", "message"),
    [
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
    bundle = _write_reader_bundle(
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
        preview_reader_promoter_evidence_manifest([bundle])


def test_preview_rejects_artifact_digest_and_signature_tampering(tmp_path: Path) -> None:
    digest_bundle = _write_reader_bundle(
        tmp_path / "digest",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    (digest_bundle / "promoter_evidence.png").write_bytes(b"\x89PNG\r\n\x1a\ntampered")
    with pytest.raises(ReaderPromoterEvidenceError, match="digest or size mismatch"):
        preview_reader_promoter_evidence_manifest([digest_bundle])

    signature_bundle = _write_reader_bundle(
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
        preview_reader_promoter_evidence_manifest([signature_bundle])


def test_preview_rejects_duplicate_selection_identity(tmp_path: Path) -> None:
    bundle = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )

    with pytest.raises(ReaderPromoterEvidenceError, match="duplicate selection identity"):
        preview_reader_promoter_evidence_manifest([bundle, bundle])


def test_materialize_preserves_prior_manifest_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    out_dir = tmp_path / "campaign" / "inputs" / "r0"
    first = materialize_reader_promoter_evidence_manifest(
        [bundle],
        out_dir=out_dir,
    )
    original = first.manifest_json.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        materialize_reader_promoter_evidence_manifest(
            [bundle],
            out_dir=out_dir,
            overwrite=True,
        )

    assert first.manifest_json.read_bytes() == original
    assert not list(out_dir.glob(".*.staging-*"))


def test_display_verifier_rejects_artifact_metadata_tampering(tmp_path: Path) -> None:
    bundle = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        out_dir=tmp_path / "campaign" / "inputs" / "r0",
    )
    display = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    display["rows"][0]["artifacts"][0]["sha256"] = "sha256:" + "0" * 64
    result.manifest_json.write_text(json.dumps(display, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="metadata disagrees"):
        verify_reader_promoter_evidence_manifest(result.manifest_json)


def test_display_verifier_rejects_a_different_campaign_destination(tmp_path: Path) -> None:
    bundle = _write_reader_bundle(
        tmp_path / "reader-bundle",
        candidate_id="candidate-1",
        design_id="pDual-10-1",
        experiment_id="20260713_sfxi",
    )
    result = materialize_reader_promoter_evidence_manifest(
        [bundle],
        out_dir=tmp_path / "campaign" / "inputs" / "r0",
    )
    display = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    display["campaign_slug"] = "secg_cipro_rf_sfxi_topn"
    result.manifest_json.write_text(json.dumps(display, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReaderPromoterEvidenceError, match="secg_rmf_greedy"):
        verify_reader_promoter_evidence_manifest(result.manifest_json)


def _write_reader_bundle(
    root: Path,
    *,
    candidate_id: str,
    design_id: str,
    experiment_id: str,
    claim_status: str = "objective_neutral",
    adapter_kind: str = "densegen_tfbs",
) -> Path:
    root.mkdir(parents=True)
    pdf = root / "promoter_evidence.pdf"
    png = root / "promoter_evidence.png"
    pdf.write_bytes(b"%PDF-1.7\nreader evidence\n")
    png.write_bytes(b"\x89PNG\r\n\x1a\nreader evidence\n")
    manifest = {
        "schema_version": "reader.response_window.promoter_evidence_bundle.v1",
        "created_at": "2026-07-13T12:00:00+00:00",
        "claim_status": claim_status,
        "non_claim_boundary": (
            "Reader presents response-window evidence and sequence context; it does not calculate, calibrate, "
            "or promote an RMF objective."
        ),
        "selection": {
            "experiment_id": experiment_id,
            "design_id": design_id,
            "candidate_id": candidate_id,
            "reduction_id": "event_logmean_6_12h_post",
        },
        "selected_binding": {
            "sequence_sha256": "sha256:" + "6" * 64,
            "sequence_authority_dataset_id": "usr_sfxi_pdual10_densegen_promoters",
            "sequence_authority_id": candidate_id,
            "sequence_authority_sha256": "sha256:" + "7" * 64,
            "source_class": "densegen" if adapter_kind == "densegen_tfbs" else "construct_derived",
            "design_family": "ethanol_ciprofloxacin" if adapter_kind == "densegen_tfbs" else "control",
            "binding_status": "resolved",
            "binding_method": "exact_alias",
            "densegen_plan": "ethanol_ciprofloxacin" if adapter_kind == "densegen_tfbs" else None,
            "densegen_run_id": "reader_sfxi_pdual10_archive_port" if adapter_kind == "densegen_tfbs" else None,
            "densegen_sampling_library_hash": "archive_library_hash" if adapter_kind == "densegen_tfbs" else None,
        },
        "sources": {
            "response_window": {
                "schema_version": "reader.response_window.bundle.v3",
                "request_id": "stress_ethanol_cipro_growth.response_window.v2",
                "manifest_sha256": "sha256:" + "1" * 64,
            },
            "candidate_bindings": {
                "schema_id": "dnadesign.study.promoter_candidate_bindings.v1",
                "schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "manifest_sha256": "sha256:" + "2" * 64,
                "records_sha256": "sha256:" + "3" * 64,
                "candidate_table_id": "usr_prom_eth_cip_opal_candidates",
                "candidate_selection_sha256": "sha256:" + "4" * 64,
            },
            "baserender": {
                "contract_id": "dnadesign.baserender.sequence_panel.v1",
                "contract_version": "1",
                "style_profile": "promoter_compact_slide.v1",
                "renderer_name": "sequence_rows",
                "adapter_kind": adapter_kind,
                "sequence_length_bp": 60,
                "feature_count": 2,
                "strand_count": 2,
                "legend_entries": ["tf:CpxR"],
                "image_width_px": 2200,
                "image_height_px": 430,
            },
        },
        "objective_overlay": None,
        "artifacts": {
            pdf.name: {"path": pdf.name, "bytes": pdf.stat().st_size, "sha256": _sha256(pdf)},
            png.name: {"path": png.name, "bytes": png.stat().st_size, "sha256": _sha256(png)},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
