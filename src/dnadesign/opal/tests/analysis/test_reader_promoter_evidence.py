"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_reader_promoter_evidence.py

Tests for OPAL display of canonical Reader diagnostic projections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dnadesign.opal.api.reader_evidence import (
    READER_EVIDENCE_MANIFEST_ADAPTER,
    reader_evidence_artifact_adapter,
)
from dnadesign.opal.src.analysis.notebook_components import reader_evidence_preview
from dnadesign.opal.src.analysis.notebook_components.reader_evidence import (
    build_notebook_reader_evidence_surface,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_visual,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence import (
    notebook_adapter,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence.contracts import (
    PROMOTER_RESPONSE_SEMANTIC_KIND,
    canonical_json_sha256,
)


def test_discovery_preserves_exact_record_provenance_and_notebook_routing(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]

    assert row["label"] == label
    assert row["artifact_record_id"] == "plot:four_state_event_window_diagnostic"
    assert row["kind"] == "reader_record_projection"
    assert row["source_record_revision_digest"] == "sha256:" + "a" * 64
    assert row["source_file_path"] == "plots/four_state_event_window_diagnostic.png"
    assert row["source_receipt_sha256"] == canonical_json_sha256(row["sources"]["response_window"])
    assert row["sources"]["response_window"]["catalog"]["schema_version"] == 4
    assert set(row["sources"]) == {"response_window", "candidate_bindings"}
    assert surface["media_plot_type_labels"] == ["Promoter response evidence"]
    assert "time_selected_h" not in row


def test_verification_uses_staged_bytes_without_reader_checkout(tmp_path: Path) -> None:
    surface, _ = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]

    adapter = reader_evidence_artifact_adapter(PROMOTER_RESPONSE_SEMANTIC_KIND)
    assert adapter.verify_artifact(row).is_file()
    assert adapter.verify_artifact.__module__.startswith("dnadesign.studies.units.stress_ethanol_cipro_growth")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row.__setitem__("source_record_revision_digest", "sha256:" + "f" * 64),
            "verified publication manifest",
        ),
        (
            lambda row: row.__setitem__("source_receipt_sha256", "sha256:" + "f" * 64),
            "verified publication manifest",
        ),
        (
            lambda row: row.__setitem__("source_file_path", "plots/different.png"),
            "verified publication manifest",
        ),
        (
            lambda row: row.__setitem__("artifact_record_id", "reader.response_window.promoter_evidence_bundle.v5"),
            "verified publication manifest",
        ),
        (
            lambda row: row["sources"]["candidate_bindings"].__setitem__("study_id", "different_study"),
            "verified publication manifest",
        ),
    ],
)
def test_verification_rejects_provenance_drift(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    surface, _ = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    mutate(row)

    adapter = reader_evidence_artifact_adapter(PROMOTER_RESPONSE_SEMANTIC_KIND)
    with pytest.raises(ValueError, match=message):
        adapter.verify_artifact(row)


def test_render_rejects_media_changed_after_discovery(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    path = Path(str(row["manifest_path"])).parent / str(row["path"])
    path.write_bytes(b"\x89PNG\r\n\x1a\ntampered")

    rendered = _render(surface, label=label)

    assert rendered["kind"] == "md"
    assert "verification failed" in rendered["text"].lower()


def test_render_rejects_relabeling_and_unconfined_paths(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    row = surface["media_rows"][0]
    row["candidate_id"] = "different-candidate"
    row["id"] = "different-candidate"
    assert "verified publication manifest" in _render(surface, label=label)["text"]

    surface, label = _valid_promoter_surface(tmp_path / "second")
    surface["media_rows"][0]["path"] = "~/four_state_event_window_diagnostic.png"
    assert "verified publication manifest" in _render(surface, label=label)["text"]


def test_render_shows_the_four_panel_diagnostic_and_exact_receipts(tmp_path: Path) -> None:
    surface, label = _valid_promoter_surface(tmp_path)

    rendered = _render(surface, label=label)

    assert rendered["kind"] == "vstack"
    text = str(rendered)
    assert "Evidence details" in text
    assert "Objective-neutral display evidence" in text
    assert "growth trajectory" in text
    assert "reference-relative magnitude" in text
    assert "Reader catalog" in text
    assert "Exact records" in text
    assert "objective overlay" not in text.lower()


def test_render_enforces_media_size_ceiling(tmp_path: Path, monkeypatch) -> None:
    surface, label = _valid_promoter_surface(tmp_path)
    monkeypatch.setattr(notebook_adapter, "_MAX_NOTEBOOK_MEDIA_BYTES", 1)

    rendered = _render(surface, label=label)

    assert rendered["kind"] == "md"
    assert "file evidence" in rendered["text"].lower() or "render ceiling" in rendered["text"].lower()


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


def _valid_promoter_surface(tmp_path: Path) -> tuple[dict[str, object], str]:
    workdir = tmp_path / "campaign"
    manifest_path = workdir / "inputs" / "r0" / "reader_evidence_promoter_response.json"
    manifest_path.parent.mkdir(parents=True)
    revision_digest = "sha256:" + "a" * 64
    source_file_path = "plots/four_state_event_window_diagnostic.png"
    relative_media = Path("reader_evidence_media") / ("a" * 64) / Path(source_file_path).name
    media_path = manifest_path.parent / relative_media
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"\x89PNG\r\n\x1a\ncanonical Reader diagnostic")
    response = _response_source(
        media_path=media_path,
        revision_digest=revision_digest,
        source_file_path=source_file_path,
    )
    row = _publication_row(
        response=response,
        relative_media=relative_media,
        media_path=media_path,
        revision_digest=revision_digest,
        source_file_path=source_file_path,
    )
    manifest_path.write_text(json.dumps(_publication(row), indent=2) + "\n", encoding="utf-8")
    artifacts = discover_reader_evidence_artifacts(workdir)
    surface = build_notebook_reader_evidence_surface(
        {
            "campaign": {"workdir": str(workdir)},
            "reader_evidence": discover_reader_evidence_manifests(workdir),
            "reader_evidence_artifacts": artifacts,
        }
    )
    assert len(surface["media_rows"]) == 1
    label = str(surface["media_rows"][0]["label"])
    return surface, label


def _response_source(*, media_path: Path, revision_digest: str, source_file_path: str) -> dict[str, object]:
    config_digest = "sha256:" + "0" * 64

    def dataframe_record(record_id: str, contract_id: str, character: str) -> dict[str, object]:
        return {
            "record_id": record_id,
            "kind": "dataframe_artifact",
            "schema_version": 6,
            "revision": 2,
            "revision_digest": "sha256:" + character * 64,
            "config_digest": config_digest,
            "producer_config_digest": "sha256:" + "9" * 64,
            "producer": {
                "kind": "pipeline",
                "id": "four_state_event_window",
                "plugin": "protocol/plate_reader_four_state_event_window",
            },
            "inputs": [],
            "contract_id": contract_id,
            "path": f"artifacts/{Path(record_id).name}.parquet",
            "size_bytes": 10,
            "content_digest": "sha256:" + character * 64,
        }

    designs = dataframe_record(
        "four_state_event_window/designs",
        "plate_reader.four_state_event_window.designs.v4",
        "1",
    )
    traces = dataframe_record(
        "four_state_event_window/traces",
        "plate_reader.four_state_event_window.traces.v3",
        "2",
    )

    return {
        "schema_version": "stress_ethanol_cipro_growth.reader_response_record_source.v1",
        "output_experiment_id": "20260717_stress_response_window_aggregate",
        "source_experiment_id": "20260619_sfxi_sensor-panel-m9-glu-1-10",
        "design_id": "pDual-10-ES1p",
        "reduction_id": "event_logmean_4_8h_post",
        "protocol_id": "plate_reader/four_state_event_window",
        "projection_sha256": "sha256:" + "4" * 64,
        "catalog": {
            "schema_version": 4,
            "provenance_epoch_id": "123e4567-e89b-42d3-a456-426614174000",
            "sha256": "sha256:" + "5" * 64,
        },
        "records": {
            "designs": designs,
            "traces": traces,
            "diagnostic": {
                "record_id": "plot:four_state_event_window_diagnostic",
                "kind": "file_bundle",
                "schema_version": 6,
                "revision": 3,
                "revision_digest": revision_digest,
                "config_digest": config_digest,
                "producer_config_digest": "sha256:" + "3" * 64,
                "producer": {
                    "kind": "plot",
                    "id": "four_state_event_window_diagnostic",
                    "plugin": "plot/four_state_event_window_diagnostic",
                },
                "inputs": [
                    {
                        "label": label,
                        "kind": "record",
                        "record": record["record_id"],
                        "discovery_policy": "record",
                        "record_revision_digest": record["revision_digest"],
                    }
                    for label, record in (("designs", designs), ("traces", traces))
                ],
                "file_evidence": [
                    {
                        "path": source_file_path,
                        "size_bytes": media_path.stat().st_size,
                        "content_digest": _sha256(media_path),
                    }
                ],
            },
        },
    }


def _publication_row(
    *,
    response: dict[str, object],
    relative_media: Path,
    media_path: Path,
    revision_digest: str,
    source_file_path: str,
) -> dict[str, object]:
    return {
        "id": "candidate-1",
        "candidate_id": "candidate-1",
        "design_id": "pDual-10-ES1p",
        "reader_experiment_id": "20260619_sfxi_sensor-panel-m9-glu-1-10",
        "reduction_id": "event_logmean_4_8h_post",
        "evidence_role": "display_only",
        "claim_status": "objective_neutral",
        "non_claim_boundary": (
            "Reader publishes verified response-window records and diagnostic media; the stress study binds "
            "candidate identity and display meaning. Objective scoring, label promotion, and campaign state "
            "are separate."
        ),
        "selected_binding": {
            "reader_design_id": "pDual-10-ES1p",
            "candidate_id": "candidate-1",
            "sequence_sha256": "sha256:" + "6" * 64,
            "sequence_authority_dataset_id": "reader-test-authority",
            "sequence_authority_id": "authority:pDual-10-ES1p",
            "sequence_authority_sha256": "sha256:" + "7" * 64,
            "source_class": "densegen",
            "design_family": "ethanol_ciprofloxacin",
            "binding_status": "resolved",
            "binding_method": "exact_alias",
            "densegen_plan": "ethanol_ciprofloxacin",
            "densegen_run_id": "run-1",
            "densegen_sampling_library_hash": "library-1",
        },
        "sources": {
            "response_window": response,
            "candidate_bindings": {
                "schema_id": "dnadesign.study.promoter_candidate_bindings.v1",
                "schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "manifest_sha256": "sha256:" + "8" * 64,
                "records_sha256": "sha256:" + "9" * 64,
                "candidate_table_id": "usr_prom_eth_cip_opal_candidates",
                "candidate_selection_sha256": "sha256:" + "b" * 64,
            },
        },
        "artifacts": [
            {
                "semantic_kind": "promoter_response_evidence",
                "kind": "reader_record_projection",
                "record_id": "plot:four_state_event_window_diagnostic",
                "scope": "design_reduction",
                "path": relative_media.as_posix(),
                "path_label": (
                    "20260619_sfxi_sensor-panel-m9-glu-1-10/pDual-10-ES1p/"
                    "event_logmean_4_8h_post/four_state_event_window_diagnostic.png"
                ),
                "exists": True,
                "media_type": "image/png",
                "bytes": media_path.stat().st_size,
                "sha256": _sha256(media_path),
                "source_record_revision_digest": revision_digest,
                "source_file_path": source_file_path,
                "source_receipt_sha256": canonical_json_sha256(response),
            }
        ],
    }


def _publication(row: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "stress_ethanol_cipro_growth.reader_promoter_evidence.v3",
        "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
        "created_at": "2026-07-29T12:00:00+00:00",
        "campaign_slug": "secg_msrb_greedy",
        "round": "r0",
        "summary": {
            "rows": 1,
            "distinct_ids": 1,
            "reader_experiments": 1,
            "artifact_count": 1,
            "missing_artifact_rows": 0,
        },
        "rows": [row],
    }


def _render(surface: dict[str, object], *, label: str):
    return render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Promoter response evidence",
        selected_artifact_label=label,
        mo=_FakeMo(),
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeMo:
    def md(self, text: str) -> dict[str, str]:
        return {"kind": "md", "text": text}

    def Html(self, text: str) -> dict[str, str]:
        return {"kind": "html", "html": text}

    def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
        return {"kind": "vstack", "items": items, "gap": gap}

    def accordion(self, items: dict[str, object], **kwargs: object) -> dict[str, object]:
        return {"kind": "accordion", "items": items, **kwargs}
