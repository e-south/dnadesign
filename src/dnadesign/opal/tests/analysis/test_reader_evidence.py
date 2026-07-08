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

from dnadesign.opal.src.analysis.notebook_components.reader_evidence import (
    READER_EVIDENCE_PDF_HEIGHT,
    build_notebook_reader_evidence_artifact_options,
    build_notebook_reader_evidence_plot_type_options,
    build_notebook_reader_evidence_surface,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_visual,
)


def test_reader_evidence_surface_groups_media_by_plot_type(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    raw_pdf = tmp_path / "reader" / "raw.pdf"
    heatmap_png = tmp_path / "reader" / "heatmap.png"
    raw_pdf.parent.mkdir(parents=True)
    raw_pdf.write_bytes(b"%PDF-1.4\n")
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
        "SFXI vec8 heatmap",
    ]
    assert build_notebook_reader_evidence_artifact_options(
        surface,
        selected_plot_type_label="SFXI vec8 heatmap",
    ) == ["r0 | 20260706_sfxi | pDual-10-SECG-B0-ETH-01 | 12.04 h"]


def test_reader_evidence_pdf_uses_compact_notebook_height(tmp_path: Path) -> None:
    pdf_path = tmp_path / "reader.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
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

    assert rendered == {"kind": "pdf", "path": pdf_path, "width": "100%", "height": READER_EVIDENCE_PDF_HEIGHT}


class _FakeMo:
    def md(self, text: str) -> dict[str, str]:
        return {"kind": "md", "text": text}

    def pdf(self, path: Path, *, width: str, height: str) -> dict[str, object]:
        return {"kind": "pdf", "path": path, "width": width, "height": height}
