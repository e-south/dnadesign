"""
Tests for visual-only Snapback specs and rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.snapback_visual_workflow import run_snapback_visual
from dnadesign.cruncher.snapback import visual_plot
from dnadesign.cruncher.snapback.errors import SnapbackSpecError
from dnadesign.cruncher.snapback.load import load_snapback_visual_spec


def _visual_payload() -> dict[str, object]:
    return {
        "snapback_visual": {
            "schema_version": 1,
            "contract": "single_nick_snapback_visual_v1",
            "name": "msd-HOPV5",
        },
        "input": {"precursor_top_strand": "CCTCAGCCCGCTGA"},
        "nick": {
            "label": "Nt.Bpu10I",
            "site_sequence": "CCTNAGC",
            "site_span": {"start": 0, "end": 7},
            "nick_boundary": 2,
            "nicked_strand": "top",
        },
        "product": {
            "active_strand": "bottom",
            "active_label": "Retained Bottom",
            "upstream_context_nt": 2,
            "stem_sequence": "AGTC",
            "cap_sequence": "GGGC",
            "foldback_sequence": "GACT",
        },
        "output": {"run_dir": "outputs/msd-HOPV5_visual", "render_format": "pdf"},
    }


def _write_visual_spec(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    spec_path = (
        tmp_path / "workspaces" / "msd-HOPV5_snapback" / "configs" / "snapback" / "msd-HOPV5.visual.snapback.yaml"
    )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(payload or _visual_payload(), sort_keys=False), encoding="utf-8")
    return spec_path


def test_load_snapback_visual_spec_accepts_msd_hopv5_geometry(tmp_path: Path) -> None:
    spec_path = _write_visual_spec(tmp_path)

    spec, resolved, workspace_root = load_snapback_visual_spec(spec_path)

    assert resolved == spec_path.resolve()
    assert workspace_root == (tmp_path / "workspaces" / "msd-HOPV5_snapback").resolve()
    assert spec.name == "msd-HOPV5"
    assert spec.active_product_sequence == "GGAGTCGGGCGACT"
    assert spec.effective_stem_bp == 6
    assert spec.product.cap_sequence == "GGGC"


def test_load_snapback_visual_spec_rejects_decomposition_drift(tmp_path: Path) -> None:
    payload = _visual_payload()
    payload["product"]["foldback_sequence"] = "GACC"
    spec_path = _write_visual_spec(tmp_path, payload)

    with pytest.raises(SnapbackSpecError, match="reverse complement"):
        load_snapback_visual_spec(spec_path)


def test_run_snapback_visual_writes_isolated_plot_bundle(tmp_path: Path) -> None:
    spec_path = _write_visual_spec(tmp_path)

    run_dir, report = run_snapback_visual(spec_path)

    assert report.status == "rendered"
    assert report.nick_label == "Nt.Bpu10I"
    assert report.nick_boundary_from_left == 2
    assert report.plot_data_path.endswith("snapback_visual_plot_data.json")
    assert report.effective_stem_bp == 6
    assert report.stem_sequence == "AGTC"
    assert report.cap_sequence == "GGGC"
    assert report.foldback_sequence == "GACT"
    assert (run_dir / "plots" / "msd-HOPV5.snapback_visual.pdf").read_bytes().startswith(b"%PDF")
    context = json.loads((run_dir / "analysis" / "snapback_visual_plot_data.json").read_text(encoding="utf-8"))
    assert context["released_product"]["released_strand_label"] == "Exposed Top"
    assert context["released_product"]["active_label"] == "Retained Bottom"
    assert context["released_product"]["active_product_sequence"] == "GGAGTCGGGCGACT"
    assert context["precursor"]["nick_site_sequence"] == "CCTNAGC"
    assert context["precursor"]["nick_site_orientation"] == "forward"
    assert context["precursor"]["top_sequence"] == "CCTNAGCCCGCTGA"
    assert context["precursor"]["bottom_sequence"] == "GGANTCGGGCGACT"
    assert context["precursor"]["top_assignable_base_positions"] == [3]
    assert context["precursor"]["bottom_assignable_base_positions"] == [3]
    assert context["released_product"]["active_assignable_base_positions"] == [3]
    assert context["released_product"]["stem_span"] == {"start": 2, "end": 6}
    assert context["released_product"]["cap_span"] == {"start": 6, "end": 10}
    assert context["released_product"]["foldback_span"] == {"start": 10, "end": 14}
    assert context["foldback"]["foldback_sequence"] == "GACT"
    assert context["foldback"]["foldback_partner_sequence"] == "TCAG"
    assert context["foldback"]["origin_boundary_from_left"] == 2
    assert context["foldback"]["upstream_context_span"] == {"start": 0, "end": 2}
    assert context["foldback"]["top_row"]["label"] == "Foldback Stem"
    assert context["foldback"]["bottom_row"]["label"] == "Stem"
    assert context["foldback"]["top_row"]["sequence"] == "CCTCAG"
    assert context["foldback"]["bottom_row"]["sequence"] == "GGAGTC"
    assert context["foldback"]["bottom_row"]["assignable_base_positions"] == [3]
    manifest = json.loads((run_dir / "meta" / "snapback_visual_manifest.json").read_text(encoding="utf-8"))
    assert "context" not in manifest["artifacts"]
    assert "plot_context_json" not in manifest["artifacts"]
    assert manifest["artifacts"]["plot_data_json"].endswith("snapback_visual_plot_data.json")
    assert (run_dir / "meta" / "snapback_visual_status.json").exists()


def test_visual_precursor_bolds_only_canonical_site_orientation_strand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = load_snapback_visual_spec(_write_visual_spec(tmp_path))[0]
    emphasized_segments_by_row: dict[str, list[tuple[int, int]]] = {}

    def record_sequence(*_args, row_label: str, emphasis_segments=None, **_kwargs) -> None:
        emphasized_segments_by_row[row_label] = list(emphasis_segments or [])

    monkeypatch.setattr(visual_plot.plot_common, "draw_sequence", record_sequence)
    context = visual_plot.build_snapback_visual_plot_context(spec)["precursor"]

    fig, ax = plt.subplots()
    visual_plot._render_precursor_panel(ax, context=context)
    plt.close(fig)

    assert emphasized_segments_by_row["Top"] == [(0, 7)]
    assert emphasized_segments_by_row["Bottom"] == []


def test_visual_foldback_carries_upstream_degenerate_pair_as_symbolic_context(tmp_path: Path) -> None:
    payload = _visual_payload()
    payload["nick"]["site_sequence"] = "CNTCAGC"
    spec = load_snapback_visual_spec(_write_visual_spec(tmp_path, payload))[0]

    context = visual_plot.build_snapback_visual_plot_context(spec)

    assert context["precursor"]["top_sequence"] == "CNTCAGCCCGCTGA"
    assert context["precursor"]["bottom_sequence"] == "GNAGTCGGGCGACT"
    assert context["released_product"]["released_strand_sequence"] == "CN"
    assert context["released_product"]["active_product_sequence"] == "GNAGTCGGGCGACT"
    assert context["foldback"]["top_row"]["sequence"] == "CNTCAG"
    assert context["foldback"]["bottom_row"]["sequence"] == "GNAGTC"
    assert context["foldback"]["top_row"]["assignable_base_positions"] == [1]
    assert context["foldback"]["bottom_row"]["assignable_base_positions"] == [1]


def test_visual_precursor_bolds_reverse_oriented_site_on_bottom_strand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _visual_payload()
    payload["input"]["precursor_top_strand"] = "GCTCAGGA"
    payload["nick"]["site_sequence"] = "CCTNAGC"
    payload["nick"]["site_span"] = {"start": 0, "end": 7}
    payload["nick"]["nick_boundary"] = 2
    payload["product"]["upstream_context_nt"] = 2
    payload["product"]["stem_sequence"] = "A"
    payload["product"]["cap_sequence"] = "GTCC"
    payload["product"]["foldback_sequence"] = "T"
    spec = load_snapback_visual_spec(_write_visual_spec(tmp_path, payload))[0]
    emphasized_segments_by_row: dict[str, list[tuple[int, int]]] = {}

    def record_sequence(*_args, row_label: str, emphasis_segments=None, **_kwargs) -> None:
        emphasized_segments_by_row[row_label] = list(emphasis_segments or [])

    monkeypatch.setattr(visual_plot.plot_common, "draw_sequence", record_sequence)
    context = visual_plot.build_snapback_visual_plot_context(spec)["precursor"]

    fig, ax = plt.subplots()
    visual_plot._render_precursor_panel(ax, context=context)
    plt.close(fig)

    assert context["nick_site_orientation"] == "reverse"
    assert emphasized_segments_by_row["Top"] == []
    assert emphasized_segments_by_row["Bottom"] == [(0, 7)]
