"""Tests static OPAL display of Reader promoter-response evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dnadesign.opal.src.analysis.notebook_components import (
    reader_evidence_preview,
    reader_evidence_triptych,
)
from dnadesign.opal.src.analysis.notebook_components import (
    reader_evidence_visual as reader_evidence_visual_module,
)
from dnadesign.opal.src.analysis.notebook_components.reader_evidence import (
    build_notebook_reader_evidence_surface,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_visual,
)
from dnadesign.opal.src.analysis.notebook_components.reader_evidence_triptych import (
    render_notebook_reader_evidence_time_control,
)
from dnadesign.opal.src.analysis.notebook_components.reader_promoter_evidence import (
    READER_PROMOTER_EVIDENCE_MAX_BYTES,
)


def test_promoter_evidence_discovery_preserves_display_only_provenance(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    reader_bundle = tmp_path / "reader-bundle"
    reader_bundle.mkdir()
    png_path = reader_bundle / "promoter_evidence.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nreader evidence")
    source_manifest = reader_bundle / "manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    evidence_manifest = workdir / "inputs" / "r0" / "reader_evidence_promoter_response.json"
    evidence_manifest.parent.mkdir(parents=True)
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.reader_evidence.v1",
                "created_at": "2026-07-13T12:00:00+00:00",
                "campaign_slug": "secg_rmf_greedy",
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
                        "selected_binding": {
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
                        "artifacts": [
                            {
                                "semantic_kind": "promoter_response_evidence",
                                "kind": "reader_publication",
                                "record_id": "reader.response_window.promoter_evidence_bundle.v1",
                                "scope": "design_reduction",
                                "path": str(png_path.resolve()),
                                "path_label": "20260713_sfxi/pDual-10-1/event_logmean_6_12h_post/promoter_evidence.png",
                                "exists": True,
                                "media_type": "image/png",
                                "bytes": png_path.stat().st_size,
                                "sha256": _sha256(png_path),
                                "source_manifest_path": str(source_manifest.resolve()),
                                "source_manifest_sha256": _sha256(source_manifest),
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
    assert artifacts[0]["source_manifest_path"] == str(source_manifest.resolve())
    assert artifacts[0]["source_manifest_sha256"] == _sha256(source_manifest)
    assert surface["media_plot_type_labels"] == ["Promoter response evidence"]
    assert "time_selected_h" not in artifacts[0]


def test_promoter_evidence_render_rejects_artifact_changed_after_discovery(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    png_path = Path(str(surface["media_rows"][0]["path"]))
    png_path.write_bytes(b"\x89PNG\r\n\x1a\ntampered")

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        selected_time_h=12.0,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "md"
    assert "verification failed" in rendered["text"].lower()


def test_promoter_evidence_render_rejects_authentic_media_relabeling(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    surface["media_rows"][0]["candidate_id"] = "different-candidate"

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "md"
    assert "source selection" in rendered["text"]


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
    assert "exact absolute path" in rendered["text"]


def test_promoter_evidence_does_not_invoke_sfxi_reconstruction(tmp_path: Path, monkeypatch) -> None:
    surface, label = _valid_promoter_surface(tmp_path)

    def fail_sfxi(*args, **kwargs):
        raise AssertionError("SFXI reconstruction must not run for promoter-response evidence")

    monkeypatch.setattr(reader_evidence_triptych, "reader_sfxi_triptych_time_metadata", fail_sfxi)
    monkeypatch.setattr(reader_evidence_visual_module, "render_reader_sfxi_triptych_visual", fail_sfxi)

    time_control = render_notebook_reader_evidence_time_control(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )
    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        selected_time_h=12.0,
        mo=_FakeMo(),
    )

    assert time_control is None
    assert rendered["kind"] == "vstack"


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
    png_path = tmp_path / "reader-bundle" / "promoter_evidence.png"
    png_path.parent.mkdir()
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nreader evidence")
    candidate_id = "candidate-1"
    design_id = "pDual-10-1"
    experiment_id = "20260713_sfxi"
    reduction_id = "event_logmean_6_12h_post"
    selected_binding = {
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
    source_manifest = png_path.parent / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "reader.response_window.promoter_evidence_bundle.v1",
                "claim_status": "objective_neutral",
                "selection": {
                    "candidate_id": candidate_id,
                    "design_id": design_id,
                    "experiment_id": experiment_id,
                    "reduction_id": reduction_id,
                },
                "selected_binding": selected_binding,
                "artifacts": {
                    png_path.name: {
                        "path": png_path.name,
                        "bytes": png_path.stat().st_size,
                        "sha256": _sha256(png_path),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    label = "r0 | 20260713_sfxi | pDual-10-1"
    return (
        {
            "media_rows": [
                {
                    "label": label,
                    "plot_type_label": "Promoter response evidence",
                    "semantic_kind": "promoter_response_evidence",
                    "kind": "reader_publication",
                    "artifact_record_id": "reader.response_window.promoter_evidence_bundle.v1",
                    "scope": "design_reduction",
                    "id": candidate_id,
                    "candidate_id": candidate_id,
                    "design_id": design_id,
                    "reader_experiment_id": experiment_id,
                    "reduction_id": reduction_id,
                    "evidence_role": "display_only",
                    "claim_status": "objective_neutral",
                    "selected_binding": selected_binding,
                    "path": str(png_path.resolve()),
                    "exists": True,
                    "media_type": "image/png",
                    "bytes": png_path.stat().st_size,
                    "sha256": _sha256(png_path),
                    "source_manifest_path": str(source_manifest.resolve()),
                    "source_manifest_sha256": _sha256(source_manifest),
                }
            ]
        },
        label,
    )


class _FakeMo:
    def md(self, text: str) -> dict[str, str]:
        return {"kind": "md", "text": text}

    def Html(self, text: str) -> dict[str, str]:
        return {"kind": "html", "html": text}

    def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
        return {"kind": "vstack", "items": items, "gap": gap}
