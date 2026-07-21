"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_reader_evidence.py

Tests Reader evidence notebook component contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

from dnadesign.opal.api.reader_evidence import READER_EVIDENCE_MANIFEST_ADAPTER
from dnadesign.opal.src.analysis.notebook_components import reader_evidence_visual as reader_evidence_visual_module
from dnadesign.opal.src.analysis.notebook_components.reader_evidence import (
    build_notebook_reader_evidence_artifact_options,
    build_notebook_reader_evidence_plot_type_options,
    build_notebook_reader_evidence_record_memory_key,
    build_notebook_reader_evidence_surface,
    build_notebook_reader_evidence_visual_choices,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_control,
    render_notebook_reader_evidence_artifact_visual,
    render_notebook_reader_evidence_plot_type_control,
    render_notebook_reader_evidence_record_control,
    resolve_notebook_reader_evidence_preferred_record_label,
)
from dnadesign.opal.src.analysis.notebook_components.reader_evidence_media import (
    preferred_reader_media_rows,
)


def test_reader_media_grouping_distinguishes_zero_hour_from_missing_time() -> None:
    base = {
        "manifest_path": "reader-evidence.json",
        "round": 0,
        "id": "candidate-1",
        "design_id": "design-1",
        "reader_experiment_id": "experiment-1",
        "reduction_id": "snapshot",
        "semantic_kind": "promoter_response_evidence",
        "media_type": "image/png",
    }

    rows = preferred_reader_media_rows(
        [
            {**base, "time_selected_h": 0.0, "path": "zero-hour.png"},
            {**base, "path": "time-not-recorded.png"},
        ]
    )

    assert len(rows) == 2
    assert rows[0]["time_selected_h"] == 0.0
    assert "time_selected_h" not in rows[1]


def test_reader_evidence_surface_groups_media_by_plot_type(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    raw_pdf = tmp_path / "reader" / "raw.pdf"
    triptych_pdf = tmp_path / "reader" / "triptych.pdf"
    heatmap_png = tmp_path / "reader" / "heatmap.png"
    raw_pdf.parent.mkdir(parents=True)
    raw_pdf.write_bytes(b"%PDF-1.4\n")
    triptych_pdf.write_bytes(b"%PDF-1.4\n")
    heatmap_png.write_bytes(b"png")
    manifest = workdir / "inputs" / "r0" / "reader_evidence_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "example_study.reader_evidence.v1",
                "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
                "round": "r0",
                "summary": {
                    "rows": 1,
                    "distinct_ids": 1,
                    "reader_experiments": 1,
                    "artifact_count": 3,
                    "missing_artifact_rows": 0,
                },
                "rows": [
                    {
                        "id": "candidate-1",
                        "design_id": "pDual-10-SECG-B0-ETH-01",
                        "reader_experiment_id": "20260706_sfxi",
                        "time_selected_h": 12.041,
                        "artifacts": [
                            {
                                "semantic_kind": "raw_kinetics",
                                "kind": "reader_plot",
                                "record_id": "plot:raw_kinetics",
                                "scope": "design",
                                "path": str(raw_pdf),
                                "exists": True,
                                "media_type": "application/pdf",
                            },
                            {
                                "semantic_kind": "intensity_overview",
                                "kind": "reader_plot",
                                "record_id": "plot:intensity_overview",
                                "scope": "design",
                                "path": str(triptych_pdf),
                                "exists": True,
                                "media_type": "application/pdf",
                            },
                            {
                                "semantic_kind": "sfxi_vec8_heatmap",
                                "kind": "reader_plot",
                                "record_id": "plot:sfxi_vec8_heatmap",
                                "scope": "design",
                                "path": str(heatmap_png),
                                "exists": True,
                                "media_type": "image/png",
                            },
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    view_model = {
        "campaign": {"workdir": str(workdir)},
        "reader_evidence": discover_reader_evidence_manifests(workdir),
        "reader_evidence_artifacts": discover_reader_evidence_artifacts(workdir),
    }

    surface = build_notebook_reader_evidence_surface(view_model)

    assert build_notebook_reader_evidence_plot_type_options(surface) == [
        "Plate-reader time series",
        "Time series + snapshot",
        "SFXI vec8 heatmap",
    ]
    assert build_notebook_reader_evidence_artifact_options(
        surface,
        selected_plot_type_label="SFXI vec8 heatmap",
    ) == ["Round 0 · 2026-07-06 · SFXI · pDual-10-SECG-B0-ETH-01 · 12.04 h"]


def test_reader_evidence_labels_disambiguate_media_without_exposing_paths(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    png = tmp_path / "reader" / "promoter_evidence.png"
    pdf = tmp_path / "reader" / "promoter_evidence.pdf"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png")
    pdf.write_bytes(b"pdf")
    manifest = workdir / "inputs" / "r0" / "reader_evidence_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "example_study.reader_evidence.v1",
                "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
                "round": "r0",
                "summary": {
                    "rows": 1,
                    "distinct_ids": 1,
                    "reader_experiments": 1,
                    "artifact_count": 2,
                    "missing_artifact_rows": 0,
                },
                "rows": [
                    {
                        "id": "candidate-1",
                        "design_id": "design-1",
                        "reader_experiment_id": "20260706_response-window-opal-20-28",
                        "reduction_id": "event_logmean_4_8h_post",
                        "artifacts": [
                            {
                                "semantic_kind": "promoter_response_evidence",
                                "kind": "reader_publication",
                                "record_id": "plot:pdf",
                                "scope": "design_reduction",
                                "path": str(pdf),
                                "exists": True,
                                "media_type": "application/pdf",
                            },
                            {
                                "semantic_kind": "promoter_response_evidence",
                                "kind": "reader_publication",
                                "record_id": "plot:png",
                                "scope": "design_reduction",
                                "path": str(png),
                                "exists": True,
                                "media_type": "image/png",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    surface = build_notebook_reader_evidence_surface(
        {
            "campaign": {"workdir": str(workdir)},
            "reader_evidence": discover_reader_evidence_manifests(workdir),
            "reader_evidence_artifacts": discover_reader_evidence_artifacts(workdir),
        }
    )

    assert build_notebook_reader_evidence_artifact_options(
        surface,
        selected_plot_type_label="Promoter response evidence",
    ) == ["Round 0 · 2026-07-06 · Response window OPAL 20–28 · design-1 · 4–8 h post-event"]
    assert surface["media_rows"][0]["media_type"] == "image/png"
    assert [item["media_type"] for item in surface["media_rows"][0]["available_media"]] == [
        "image/png",
        "application/pdf",
    ]
    assert len(surface["artifact_rows"]) == 2
    assert all("/reader/" not in label for label in surface["media_labels"])


def test_reader_evidence_discovery_routes_by_public_adapter_not_study_schema(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    manifest = workdir / "inputs" / "r0" / "reader_evidence_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "unrelated.study_specific_schema.v9",
                "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
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
                        "design_id": "design-1",
                        "reader_experiment_id": "experiment-1",
                        "artifacts": [
                            {
                                "semantic_kind": "example_evidence",
                                "kind": "reader_plot",
                                "record_id": "plot:example",
                                "scope": "design",
                                "path": "evidence/example.png",
                                "exists": True,
                                "media_type": "image/png",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = discover_reader_evidence_manifests(workdir)

    assert rows[0]["status"] == "ready"
    assert rows[0]["rows"] == 1


def test_reader_evidence_discovery_rejects_adapter_summary_drift(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    manifest = workdir / "inputs" / "r0" / "reader_evidence_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "example.reader_evidence.v1",
                "opal_adapter": READER_EVIDENCE_MANIFEST_ADAPTER,
                "round": "r0",
                "summary": {
                    "rows": 1,
                    "distinct_ids": 1,
                    "reader_experiments": 1,
                    "artifact_count": 0,
                    "missing_artifact_rows": 0,
                },
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    assert discover_reader_evidence_manifests(workdir)[0]["status"] == "schema_attention"
    assert discover_reader_evidence_artifacts(workdir) == []


def test_reader_evidence_discovery_rejects_manifest_without_public_adapter(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    manifest = workdir / "inputs" / "r0" / "reader_evidence_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.reader_promoter_evidence.v1",
                "round": "r0",
                "summary": {"rows": 1},
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    assert discover_reader_evidence_manifests(workdir)[0]["status"] == "schema_attention"
    assert discover_reader_evidence_artifacts(workdir) == []


def test_reader_evidence_visual_choices_join_deliverable_universe() -> None:
    surface = {"media_plot_type_labels": ["Plate-reader time series", "Time series + snapshot"]}

    assert build_notebook_reader_evidence_visual_choices(surface) == [
        {
            "label": "Reader evidence | Plate-reader time series",
            "title": "Plate-reader time series",
            "surface_kind": "reader_evidence",
            "selection_scope": "campaign",
            "reader_plot_type_label": "Plate-reader time series",
        },
        {
            "label": "Reader evidence | Time series + snapshot",
            "title": "Time series + snapshot",
            "surface_kind": "reader_evidence",
            "selection_scope": "campaign",
            "reader_plot_type_label": "Time series + snapshot",
        },
    ]


def test_reader_evidence_controls_use_reader_scoped_plot_labels() -> None:
    surface = {
        "media_plot_type_labels": ["Time series + snapshot"],
        "media_rows": [
            {
                "label": "r0 | exp | design | 12.00 h",
                "plot_type_label": "Time series + snapshot",
                "semantic_kind": "intensity_overview",
                "exists": True,
                "media_type": "application/pdf",
                "path": "/tmp/plot.pdf",
            }
        ],
    }
    mo = _FakeMo()

    plot_ui = render_notebook_reader_evidence_plot_type_control(surface, mo=mo)
    remembered = "r1 | exp | design | 12.00 h"
    on_change = object()
    surface["media_rows"].append(
        {
            **surface["media_rows"][0],
            "label": remembered,
        }
    )
    artifact_ui = render_notebook_reader_evidence_artifact_control(
        surface,
        selected_plot_type_label="Time series + snapshot",
        preferred_record_label=remembered,
        on_change=on_change,
        mo=mo,
    )

    assert plot_ui["label"] == "Reader plot type"
    assert artifact_ui["label"] == "Reader record"
    assert artifact_ui["value"] == remembered
    assert artifact_ui["on_change"] is on_change


def test_reader_evidence_record_memory_key_is_campaign_and_deliverable_scoped() -> None:
    key = build_notebook_reader_evidence_record_memory_key(
        campaign_slug="secg_msrb_greedy",
        reader_plot_type_label="Promoter response evidence",
    )

    assert key == (
        'reader_evidence_record_v1:{"campaign_slug":"secg_msrb_greedy",'
        '"reader_plot_type_label":"Promoter response evidence"}'
    )
    assert key != build_notebook_reader_evidence_record_memory_key(
        campaign_slug="other_campaign",
        reader_plot_type_label="Promoter response evidence",
    )
    assert key != build_notebook_reader_evidence_record_memory_key(
        campaign_slug="secg_msrb_greedy",
        reader_plot_type_label="SFXI vec8 heatmap",
    )


def test_reader_evidence_record_memory_restores_only_current_membership() -> None:
    options = ["Round 0 · first", "Round 1 · second"]

    assert (
        resolve_notebook_reader_evidence_preferred_record_label(
            options,
            preferred_record_label="Round 1 · second",
        )
        == "Round 1 · second"
    )
    assert (
        resolve_notebook_reader_evidence_preferred_record_label(
            options,
            preferred_record_label="Round 2 · stale",
        )
        == "Round 0 · first"
    )
    assert (
        resolve_notebook_reader_evidence_preferred_record_label(
            options,
            preferred_record_label=None,
        )
        == "Round 0 · first"
    )


def test_reader_evidence_record_control_updates_campaign_deliverable_memory() -> None:
    surface = {
        "media_rows": [
            {
                "label": "Round 0 · first",
                "plot_type_label": "Promoter response evidence",
            },
            {
                "label": "Round 0 · second",
                "plot_type_label": "Promoter response evidence",
            },
        ]
    }
    key = build_notebook_reader_evidence_record_memory_key(
        campaign_slug="secg_msrb_greedy",
        reader_plot_type_label="Promoter response evidence",
    )
    state = {key: "Round 0 · second"}

    def memory() -> dict[str, str]:
        return state

    def set_memory(value: dict[str, str]) -> None:
        state.clear()
        state.update(value)

    control = render_notebook_reader_evidence_record_control(
        surface,
        campaign_slug="secg_msrb_greedy",
        selected_plot_type_label="Promoter response evidence",
        memory=memory,
        set_memory=set_memory,
        mo=_FakeMo(),
    )

    assert control["value"] == "Round 0 · second"
    control["on_change"]("Round 0 · first")
    assert state == {key: "Round 0 · first"}


def test_reader_sfxi_triptych_pdf_renders_as_completed_static_artifact(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "reader.pdf"
    preview_path = tmp_path / "reader-preview.png"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    preview_path.write_bytes(b"png-preview")
    monkeypatch.setattr(reader_evidence_visual_module, "reader_pdf_preview_path", lambda path: preview_path)
    surface = {
        "media_rows": [
            {
                "label": "r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h",
                "plot_type_label": "Time series + snapshot",
                "semantic_kind": "intensity_overview",
                "path": str(pdf_path),
                "exists": True,
                "media_type": "application/pdf",
            }
        ]
    }

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Time series + snapshot",
        selected_artifact_label="r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h",
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "vstack"
    frame_html = rendered["items"][0]["html"]
    assert "<iframe" in frame_html
    assert "Fit image to panel" in frame_html
    assert "Zoom in" in frame_html
    assert "data:image/png;base64" in frame_html
    assert "max-width:100%" in frame_html
    assert "overflow-wrap:anywhere" in rendered["items"][1]["html"]


def test_reader_evidence_png_renders_zoomable_visual(tmp_path: Path) -> None:
    image_path = tmp_path / "reader-heatmap.png"
    image_path.write_bytes(b"png")
    surface = {
        "media_rows": [
            {
                "label": "r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h",
                "plot_type_label": "SFXI vec8 heatmap",
                "path": str(image_path),
                "exists": True,
                "media_type": "image/png",
            }
        ]
    }

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="SFXI vec8 heatmap",
        selected_artifact_label="r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h",
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "vstack"
    frame_html = rendered["items"][0]["html"]
    assert "<iframe" in frame_html
    assert "Fit image to panel" in frame_html
    assert "data:image/png;base64" in frame_html
    assert "max-width:100%" in frame_html
    assert "overflow-wrap:anywhere" in rendered["items"][1]["html"]


class _FakeMo:
    def __init__(self) -> None:
        self.ui = _FakeUi()

    def md(self, text: str) -> dict[str, str]:
        return {"kind": "md", "text": text}

    def Html(self, text: str) -> dict[str, str]:
        return {"kind": "html", "html": text}

    def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
        return {"kind": "vstack", "items": items, "gap": gap}


class _FakeUi:
    def dropdown(
        self,
        options: list[str],
        *,
        value: str,
        label: str,
        searchable: bool = False,
        full_width: bool = False,
        on_change: object | None = None,
    ) -> dict[str, object]:
        return {
            "kind": "dropdown",
            "options": options,
            "value": value,
            "label": label,
            "searchable": searchable,
            "full_width": full_width,
            "on_change": on_change,
        }

    def slider(
        self,
        *,
        start: float,
        stop: float,
        value: float,
        step: float,
        debounce: bool,
        show_value: bool,
        label: str,
        full_width: bool,
    ) -> dict[str, object]:
        return {
            "kind": "slider",
            "start": start,
            "stop": stop,
            "value": value,
            "step": step,
            "debounce": debounce,
            "show_value": show_value,
            "label": label,
            "full_width": full_width,
        }
