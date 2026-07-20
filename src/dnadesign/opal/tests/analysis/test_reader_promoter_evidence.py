"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_reader_promoter_evidence.py

Tests static OPAL display of Reader promoter-response evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dnadesign.opal.api.reader_evidence import READER_EVIDENCE_MANIFEST_ADAPTER
from dnadesign.opal.src.analysis.notebook_components import reader_evidence_preview
from dnadesign.opal.src.analysis.notebook_components.reader_evidence import (
    build_notebook_reader_evidence_surface,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_visual,
)
from dnadesign.opal.src.analysis.notebook_components.reader_promoter_evidence import (
    READER_PROMOTER_EVIDENCE_MAX_BYTES,
    verify_reader_promoter_evidence_artifact,
)


def test_promoter_evidence_discovery_preserves_display_only_provenance(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    evidence_manifest = workdir / "inputs" / "r0" / "reader_evidence_promoter_response.json"
    evidence_manifest.parent.mkdir(parents=True)
    source_manifest_sha256 = "sha256:" + "a" * 64
    relative_media = Path("reader_evidence_media") / ("a" * 64) / "promoter_evidence.png"
    png_path = evidence_manifest.parent / relative_media
    png_path.parent.mkdir(parents=True)
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nreader evidence")
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema_version": "example_study.reader_promoter_evidence.v2",
                "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
                "created_at": "2026-07-13T12:00:00+00:00",
                "campaign_slug": "secg_msrb_greedy",
                "round": "r0",
                "summary": {
                    "rows": 1,
                    "distinct_ids": 1,
                    "reader_experiments": 1,
                    "artifact_count": 1,
                    "missing_artifact_rows": 0,
                },
                "rows": [
                    {
                        "id": "candidate-1",
                        "candidate_id": "candidate-1",
                        "design_id": "pDual-10-1",
                        "reader_experiment_id": "20260713_sfxi",
                        "reduction_id": "event_logmean_6_12h_post",
                        "evidence_role": "display_only",
                        "claim_status": "objective_neutral",
                        "non_claim_boundary": _non_claim_boundary(),
                        "selected_binding": {
                            "reader_design_id": "pDual-10-1",
                            "candidate_id": "candidate-1",
                            "sequence_sha256": "sha256:" + "1" * 64,
                            "sequence_authority_dataset_id": "usr_sfxi_pdual10_densegen_promoters",
                            "sequence_authority_id": "candidate-1",
                            "sequence_authority_sha256": "sha256:" + "2" * 64,
                            "source_class": "densegen",
                            "design_family": "ethanol_ciprofloxacin",
                            "binding_status": "resolved",
                            "binding_method": "exact_alias",
                            "densegen_plan": "ethanol_ciprofloxacin",
                            "densegen_run_id": "run-1",
                            "densegen_sampling_library_hash": "library-1",
                        },
                        "sources": _sources(
                            experiment_id="20260713_sfxi",
                            reduction_id="event_logmean_6_12h_post",
                        ),
                        "objective_overlay": None,
                        "artifacts": [
                            {
                                "semantic_kind": "promoter_response_evidence",
                                "kind": "reader_publication",
                                "record_id": "reader.response_window.promoter_evidence_bundle.v5",
                                "scope": "design_reduction",
                                "path": relative_media.as_posix(),
                                "path_label": "20260713_sfxi/pDual-10-1/event_logmean_6_12h_post/promoter_evidence.png",
                                "exists": True,
                                "media_type": "image/png",
                                "bytes": png_path.stat().st_size,
                                "sha256": _sha256(png_path),
                                "source_manifest_sha256": source_manifest_sha256,
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = discover_reader_evidence_artifacts(workdir)
    surface = build_notebook_reader_evidence_surface(
        {
            "campaign": {"workdir": str(workdir)},
            "reader_evidence": discover_reader_evidence_manifests(workdir),
            "reader_evidence_artifacts": artifacts,
        }
    )

    assert len(artifacts) == 1
    assert artifacts[0]["candidate_id"] == "candidate-1"
    assert artifacts[0]["reduction_id"] == "event_logmean_6_12h_post"
    assert artifacts[0]["evidence_role"] == "display_only"
    assert artifacts[0]["claim_status"] == "objective_neutral"
    assert artifacts[0]["selected_binding"]["binding_status"] == "resolved"
    assert artifacts[0]["selected_binding"]["binding_method"] == "exact_alias"
    assert artifacts[0]["bytes"] == png_path.stat().st_size
    assert artifacts[0]["sha256"] == _sha256(png_path)
    assert artifacts[0]["path"] == relative_media.as_posix()
    assert artifacts[0]["manifest_path"] == str(evidence_manifest)
    assert "source_manifest_path" not in artifacts[0]
    assert artifacts[0]["source_manifest_sha256"] == source_manifest_sha256
    assert artifacts[0]["sources"]["candidate_bindings"]["candidate_table_id"] == ("usr_prom_eth_cip_opal_candidates")
    assert artifacts[0]["sources"]["baserender"]["adapter_kind"] == "densegen_tfbs"
    assert artifacts[0]["objective_overlay"] is None
    assert surface["media_plot_type_labels"] == ["Promoter response evidence"]
    assert "time_selected_h" not in artifacts[0]


def test_promoter_evidence_verifies_staged_relative_media_without_reader_source(tmp_path: Path) -> None:
    manifest_path = tmp_path / "campaign" / "inputs" / "r0" / "reader_evidence_promoter_response.json"
    media_path = manifest_path.parent / "reader_evidence_media" / ("a" * 64) / "promoter_evidence.png"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"\x89PNG\r\n\x1a\nstaged evidence")
    row = {
        "semantic_kind": "promoter_response_evidence",
        "kind": "reader_publication",
        "artifact_record_id": "reader.response_window.promoter_evidence_bundle.v5",
        "scope": "design_reduction",
        "id": "candidate-1",
        "candidate_id": "candidate-1",
        "design_id": "pDual-10-1",
        "reader_experiment_id": "20260713_sfxi",
        "reduction_id": "event_logmean_6_12h_post",
        "evidence_role": "display_only",
        "claim_status": "objective_neutral",
        "non_claim_boundary": _non_claim_boundary(),
        "selected_binding": {
            "reader_design_id": "pDual-10-1",
            "candidate_id": "candidate-1",
            "sequence_sha256": "sha256:" + "1" * 64,
            "sequence_authority_dataset_id": "usr_sfxi_pdual10_densegen_promoters",
            "sequence_authority_id": "candidate-1",
            "sequence_authority_sha256": "sha256:" + "2" * 64,
            "source_class": "densegen",
            "design_family": "ethanol_ciprofloxacin",
            "binding_status": "resolved",
            "binding_method": "exact_alias",
            "densegen_plan": "ethanol_ciprofloxacin",
            "densegen_run_id": "run-1",
            "densegen_sampling_library_hash": "library-1",
        },
        "sources": _sources(
            experiment_id="20260713_sfxi",
            reduction_id="event_logmean_6_12h_post",
        ),
        "objective_overlay": None,
        "path": f"reader_evidence_media/{'a' * 64}/promoter_evidence.png",
        "path_label": "20260713_sfxi/pDual-10-1/event_logmean_6_12h_post/promoter_evidence.png",
        "manifest_path": str(manifest_path),
        "exists": True,
        "media_type": "image/png",
        "bytes": media_path.stat().st_size,
        "sha256": _sha256(media_path),
        "source_manifest_sha256": "sha256:" + "a" * 64,
    }
    _write_publication_manifest(manifest_path, row)

    assert verify_reader_promoter_evidence_artifact(row) == media_path


def test_promoter_evidence_render_rejects_artifact_changed_after_discovery(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    png_path = Path(str(row["manifest_path"])).parent / str(row["path"])
    png_path.write_bytes(b"\x89PNG\r\n\x1a\ntampered")

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "md"
    assert "verification failed" in rendered["text"].lower()


def test_promoter_evidence_render_rejects_authentic_media_relabeling(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    surface["media_rows"][0]["candidate_id"] = "different-candidate"
    surface["media_rows"][0]["id"] = "different-candidate"

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "md"
    assert "identity is inconsistent" in rendered["text"]


def test_promoter_evidence_render_rejects_tilde_artifact_path(tmp_path: Path, monkeypatch) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    surface["media_rows"][0]["path"] = "~/reader-bundle/promoter_evidence.png"

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "md"
    assert "confined content-addressed" in rendered["text"]


def test_promoter_evidence_renders_verified_static_media(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "vstack"
    assert "Evidence details" in str(rendered)
    assert "Objective-neutral" in str(rendered)
    assert "DenseGen TFBS" in str(rendered)
    assert "across-well median" in str(rendered)
    assert "event is placed at either recorded timing bound" in str(rendered)


def test_promoter_evidence_discloses_typed_screen_only_overlay_below_the_visual(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    row["claim_status"] = "screen_only"
    row["objective_overlay"] = {
        "schema_version": "reader.response_window.objective_display_overlay.v2",
        "objective_id": "multistate_response_behavior_v1",
        "objective_display_label": "MSRB",
        "claim_status": "screen_only",
        "experiment_id": row["reader_experiment_id"],
        "reader_design_id": row["design_id"],
        "reduction_id": row["reduction_id"],
        "manifest_sha256": "sha256:" + "f" * 64,
        "components": [
            {
                "component_id": "response_ordering",
                "label": "Response ordering",
                "value": 0.42,
                "unit": "log2",
            }
        ],
    }
    _write_publication_manifest(Path(str(row["manifest_path"])), row)

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert "MSRB" in str(rendered)
    assert "Response ordering: 0.42 log2" in str(rendered)
    assert "screen only" in str(rendered)


def test_promoter_evidence_discloses_genbank_source_without_densegen_provenance(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    row["sources"]["baserender"]["adapter_kind"] = "usr_genbank_annotations_v1"
    for field in ("densegen_plan", "densegen_run_id", "densegen_sampling_library_hash"):
        row["selected_binding"][field] = None
    _write_publication_manifest(Path(str(row["manifest_path"])), row)

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert "GenBank source" in str(rendered)
    assert "not recorded" not in str(rendered)


def test_promoter_evidence_rejects_retired_v4_artifact_identity(tmp_path: Path) -> None:
    surface, _ = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    row["artifact_record_id"] = "reader.response_window.promoter_evidence_bundle.v4"
    _write_publication_manifest(Path(str(row["manifest_path"])), row)

    with pytest.raises(ValueError, match="publication identity"):
        verify_reader_promoter_evidence_artifact(row)


def test_promoter_evidence_rejects_cross_study_source_projection(tmp_path: Path) -> None:
    surface, _ = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    row["sources"]["candidate_bindings"]["study_id"] = "different_study"
    _write_publication_manifest(Path(str(row["manifest_path"])), row)

    with pytest.raises(ValueError, match="source study identities disagree"):
        verify_reader_promoter_evidence_artifact(row)


def test_promoter_evidence_render_enforces_media_size_ceiling(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    surface["media_rows"][0]["bytes"] = READER_PROMOTER_EVIDENCE_MAX_BYTES + 1

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "md"
    assert "render ceiling" in rendered["text"]


def test_reader_pdf_preview_has_a_bounded_subprocess_timeout(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "reader.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(
        reader_evidence_preview.shutil,
        "which",
        lambda command: "/usr/bin/gs" if command == "gs" else None,
    )

    def time_out(command, *, check, capture_output, text, timeout):
        assert timeout == 30
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(reader_evidence_preview.subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="timed out after 30 seconds"):
        reader_evidence_preview.reader_pdf_preview_path(pdf_path)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_promoter_surface(tmp_path: Path) -> tuple[dict[str, list[dict[str, object]]], str]:
    manifest_path = tmp_path / "campaign" / "inputs" / "r0" / "reader_evidence_promoter_response.json"
    source_manifest_sha256 = "sha256:" + "a" * 64
    relative_media = Path("reader_evidence_media") / ("a" * 64) / "promoter_evidence.png"
    png_path = manifest_path.parent / relative_media
    png_path.parent.mkdir(parents=True)
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nreader evidence")
    candidate_id = "candidate-1"
    design_id = "pDual-10-1"
    experiment_id = "20260713_sfxi"
    reduction_id = "event_logmean_6_12h_post"
    selected_binding = {
        "reader_design_id": design_id,
        "candidate_id": candidate_id,
        "sequence_sha256": "sha256:" + "1" * 64,
        "sequence_authority_dataset_id": "usr_sfxi_pdual10_densegen_promoters",
        "sequence_authority_id": candidate_id,
        "sequence_authority_sha256": "sha256:" + "2" * 64,
        "source_class": "densegen",
        "design_family": "ethanol_ciprofloxacin",
        "binding_status": "resolved",
        "binding_method": "exact_alias",
        "densegen_plan": "ethanol_ciprofloxacin",
        "densegen_run_id": "run-1",
        "densegen_sampling_library_hash": "library-1",
    }
    label = "r0 | 20260713_sfxi | pDual-10-1"
    row = {
        "label": label,
        "plot_type_label": "Promoter response evidence",
        "semantic_kind": "promoter_response_evidence",
        "kind": "reader_publication",
        "artifact_record_id": "reader.response_window.promoter_evidence_bundle.v5",
        "scope": "design_reduction",
        "id": candidate_id,
        "candidate_id": candidate_id,
        "design_id": design_id,
        "reader_experiment_id": experiment_id,
        "reduction_id": reduction_id,
        "evidence_role": "display_only",
        "claim_status": "objective_neutral",
        "non_claim_boundary": _non_claim_boundary(),
        "selected_binding": selected_binding,
        "sources": _sources(experiment_id=experiment_id, reduction_id=reduction_id),
        "objective_overlay": None,
        "path": relative_media.as_posix(),
        "path_label": f"{experiment_id}/{design_id}/{reduction_id}/promoter_evidence.png",
        "manifest_path": str(manifest_path.resolve()),
        "exists": True,
        "media_type": "image/png",
        "bytes": png_path.stat().st_size,
        "sha256": _sha256(png_path),
        "source_manifest_sha256": source_manifest_sha256,
    }
    _write_publication_manifest(manifest_path, row)
    return (
        {"media_rows": [row]},
        label,
    )


def _non_claim_boundary() -> str:
    return (
        "Reader presents response-window evidence and sequence context; downstream objective scoring, "
        "normalization or calibration, and promotion remain outside Reader."
    )


def _sources(*, experiment_id: str, reduction_id: str) -> dict[str, object]:
    return {
        "response_window": {
            "schema_version": "reader.response_window.bundle.v5",
            "study_id": "stress_ethanol_cipro_growth",
            "request_id": "stress-response-window-v1",
            "experiment_id": experiment_id,
            "reduction_id": reduction_id,
            "manifest_sha256": "sha256:" + "e" * 64,
        },
        "candidate_bindings": {
            "schema_id": "dnadesign.study.promoter_candidate_bindings.v1",
            "schema_version": "1",
            "study_id": "stress_ethanol_cipro_growth",
            "manifest_sha256": "sha256:" + "b" * 64,
            "records_sha256": "sha256:" + "c" * 64,
            "candidate_table_id": "usr_prom_eth_cip_opal_candidates",
            "candidate_selection_sha256": "sha256:" + "d" * 64,
        },
        "baserender": {
            "contract_id": "dnadesign.baserender.sequence_panel.v1",
            "contract_version": "1",
            "style_profile": "promoter_compact_slide.v1",
            "renderer_name": "sequence_rows",
            "adapter_kind": "densegen_tfbs",
            "sequence_length_bp": 60,
            "feature_count": 2,
            "strand_count": 2,
            "legend_entries": ["tf:CpxR"],
            "image_width_px": 2200,
            "image_height_px": 430,
        },
    }


def _write_publication_manifest(manifest_path: Path, row: dict[str, object]) -> None:
    publication_row = {
        field: row[field]
        for field in (
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
            "objective_overlay",
        )
    }
    publication_row["artifacts"] = [
        {
            "semantic_kind": row["semantic_kind"],
            "kind": row["kind"],
            "record_id": row["artifact_record_id"],
            "scope": row["scope"],
            "path": row["path"],
            "path_label": row["path_label"],
            "exists": row["exists"],
            "media_type": row["media_type"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "source_manifest_sha256": row["source_manifest_sha256"],
        }
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "example_study.reader_promoter_evidence.v2",
                "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
                "round": "r0",
                "summary": {
                    "rows": 1,
                    "distinct_ids": 1,
                    "reader_experiments": 1,
                    "artifact_count": 1,
                    "missing_artifact_rows": 0,
                },
                "rows": [publication_row],
            }
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeMo:
    def md(self, text: str) -> dict[str, str]:
        return {"kind": "md", "text": text}

    def Html(self, text: str) -> dict[str, str]:
        return {"kind": "html", "html": text}

    def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
        return {"kind": "vstack", "items": items, "gap": gap}

    def accordion(self, items: dict[str, object], **kwargs: object) -> dict[str, object]:
        return {"kind": "accordion", "items": items, **kwargs}
