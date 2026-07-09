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

from dnadesign.opal.src.analysis.notebook_components import reader_evidence_triptych
from dnadesign.opal.src.analysis.notebook_components import reader_evidence_visual as reader_evidence_visual_module
from dnadesign.opal.src.analysis.notebook_components.reader_evidence import (
    build_notebook_reader_evidence_artifact_options,
    build_notebook_reader_evidence_plot_type_options,
    build_notebook_reader_evidence_surface,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_control,
    render_notebook_reader_evidence_artifact_visual,
    render_notebook_reader_evidence_plot_type_control,
)
from dnadesign.opal.src.analysis.notebook_components.reader_evidence_triptych import (
    render_notebook_reader_evidence_time_control,
)


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
                "schema_version": "stress_ethanol_cipro_growth.reader_evidence.v1",
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
                        "design_id": "pDual-10-SECG-B0-ETH-01",
                        "reader_experiment_id": "20260706_sfxi",
                        "time_selected_h": 12.041,
                        "artifacts": [
                            {
                                "semantic_kind": "raw_kinetics",
                                "path": str(raw_pdf),
                                "exists": True,
                                "media_type": "application/pdf",
                            },
                            {
                                "semantic_kind": "intensity_overview",
                                "path": str(triptych_pdf),
                                "exists": True,
                                "media_type": "application/pdf",
                            },
                            {
                                "semantic_kind": "sfxi_vec8_heatmap",
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
    ) == ["r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h"]


def test_reader_evidence_controls_use_generic_plot_labels() -> None:
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
    artifact_ui = render_notebook_reader_evidence_artifact_control(
        surface,
        selected_plot_type_label="Time series + snapshot",
        mo=mo,
    )

    assert plot_ui["label"] == "Plot type"
    assert artifact_ui["label"] == "Plot instance"


def test_reader_evidence_time_control_uses_triptych_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        reader_evidence_triptych,
        "reader_sfxi_triptych_time_metadata",
        lambda row: {"start": 0.0, "stop": 20.0, "value": 12.0, "step": 0.25, "ground_truth_time_h": 12.0},
    )
    surface = {
        "media_rows": [
            {
                "label": "r0 | exp | design | 12.00 h",
                "plot_type_label": "Time series + snapshot",
                "semantic_kind": "intensity_overview",
                "exists": True,
                "media_type": "application/pdf",
                "path": "/tmp/plot.pdf",
            }
        ]
    }

    rendered = render_notebook_reader_evidence_time_control(
        surface,
        selected_plot_type_label="Time series + snapshot",
        selected_artifact_label="r0 | exp | design | 12.00 h",
        mo=_FakeMo(),
    )

    assert rendered["kind"] == "slider"
    assert rendered["label"] == "Time (h)"
    assert rendered["value"] == 12.0


def test_reader_evidence_pdf_renders_zoomable_image_preview(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "reader.pdf"
    preview_path = tmp_path / "reader-preview.png"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    preview_path.write_bytes(b"png-preview")
    monkeypatch.setattr(reader_evidence_visual_module, "reader_pdf_preview_path", lambda path: preview_path)
    surface = {
        "media_rows": [
            {
                "label": "r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h",
                "plot_type_label": "Plate-reader time series",
                "path": str(pdf_path),
                "exists": True,
                "media_type": "application/pdf",
            }
        ]
    }

    rendered = render_notebook_reader_evidence_artifact_visual(
        surface,
        selected_plot_type_label="Plate-reader time series",
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
    ) -> dict[str, object]:
        return {
            "kind": "dropdown",
            "options": options,
            "value": value,
            "label": label,
            "searchable": searchable,
            "full_width": full_width,
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
