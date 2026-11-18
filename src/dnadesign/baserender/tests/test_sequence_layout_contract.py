"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_sequence_layout_contract.py

Public contracts for physical sequence layout without implicit fitting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import subprocess
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender import (
    Feature,
    LayoutError,
    Record,
    SchemaError,
    SequenceLayout,
    Span,
    render_record_figure,
    render_record_svg,
    validate_job,
)

from .conftest import densegen_job_payload, write_job, write_parquet

STYLE = {"legend": False, "font_size_seq": 12, "show_pair_rungs": False}


def _record(length=60):
    return Record(
        id=f"layout-{length}",
        alphabet="DNA",
        sequence=("ACGT" * 25)[:length],
        features=(
            Feature("left", "kmer", Span(4, 10, "fwd"), "ACGTAC"),
            Feature("right", "kmer", Span(28, 34, "rev"), "GTACGT"),
        ),
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"base_pitch_pt": 0},
        {"base_pitch_pt": True},
        {"base_pitch_pt": "12px"},
        {"canvas_size_pt": (600, float("inf"))},
        {"canvas_size_pt": (600,)},
        {"sequence_origin_pt": (-1, 100)},
        {"top_extent_pt": float("nan")},
        {"bottom_extent_pt": -1},
        {"extra_bottom_padding_pt": -1},
        {"base_pitch_pt": 10**1000},
    ],
)
def test_invalid_units_fail_when_layout_is_constructed(fields):
    with pytest.raises(LayoutError, match="layout\\."):
        SequenceLayout(**fields)


def test_mapping_contract_is_strict_and_detached():
    payload = {"base_pitch_pt": 10, "canvas_size_pt": [1200, 400], "sequence_origin_pt": [80, 200]}
    layout = SequenceLayout.from_mapping(payload)
    payload["canvas_size_pt"][0] = 1
    assert layout.canvas_size_pt == (1200, 400)
    assert SequenceLayout.from_mapping(layout.to_mapping()) == layout
    with pytest.raises(LayoutError, match="unknown.*width_px"):
        SequenceLayout.from_mapping({"width_px": 1200})


def test_fixed_layout_pins_grid_and_features_across_sequence_lengths():
    layout = SequenceLayout(base_pitch_pt=10, canvas_size_pt=(1200, 400), sequence_origin_pt=(80, 200))
    results = [render_record_svg(_record(n), style_overrides=STYLE, layout=layout) for n in (60, 100)]
    for result in results:
        assert (result.width_pt, result.height_pt) == pytest.approx((1200, 400))
        assert result.diagnostics.sequence_origin_pt == pytest.approx((80, 200))
        assert result.diagnostics.base_pitch_pt == pytest.approx(10)
    assert results[0].diagnostics.feature_bounds_pt == results[1].diagnostics.feature_bounds_pt
    assert results[0].diagnostics.nucleotide_box_bounds_pt == results[1].diagnostics.nucleotide_box_bounds_pt


def test_canvas_growth_keeps_the_original_top_left_grid_anchor():
    natural = render_record_svg(_record(), style_overrides=STYLE)
    expanded = render_record_svg(
        _record(),
        style_overrides=STYLE,
        layout=SequenceLayout(canvas_size_pt=(natural.width_pt + 100, natural.height_pt + 100)),
    )
    assert expanded.diagnostics.sequence_origin_pt == pytest.approx(natural.diagnostics.sequence_origin_pt)
    for key, bounds in natural.diagnostics.feature_bounds_pt.items():
        assert expanded.diagnostics.feature_bounds_pt[key] == pytest.approx(bounds, abs=1e-9)


@pytest.mark.parametrize(
    "layout",
    [
        SequenceLayout(base_pitch_pt=1),
        SequenceLayout(canvas_size_pt=(20, 20)),
        SequenceLayout(canvas_size_pt=(1200, 400), sequence_origin_pt=(0, 0)),
        SequenceLayout(canvas_size_pt=(1e308, 1e308)),
        SequenceLayout(base_pitch_pt=1e307),
    ],
)
def test_insufficient_pitch_or_canvas_fails_before_allocation(layout):
    before = plt.get_fignums()
    with pytest.raises(LayoutError, match="pitch|bounds|canvas"):
        render_record_svg(_record(), style_overrides=STYLE, layout=layout)
    assert plt.get_fignums() == before


def test_typed_layout_and_legacy_metadata_cannot_compete():
    record = replace(_record(), meta={"cell_width_scale": 1.1})
    with pytest.raises(LayoutError, match="cell_width_scale.*layout"):
        render_record_svg(record, style_overrides=STYLE, layout=SequenceLayout(base_pitch_pt=10))


@pytest.mark.parametrize("dpi", [72, 144])
def test_explicit_point_geometry_is_independent_of_output_dpi(dpi):
    result = render_record_svg(
        _record(),
        style_overrides={**STYLE, "dpi": dpi},
        layout=SequenceLayout(base_pitch_pt=10, canvas_size_pt=(1200, 400), sequence_origin_pt=(80, 200)),
    )
    assert result.diagnostics.base_pitch_pt == pytest.approx(10)
    assert result.diagnostics.sequence_origin_pt == pytest.approx((80, 200))
    assert (result.width_pt, result.height_pt) == pytest.approx((1200, 400))


@pytest.mark.parametrize("as_mapping", [False, True])
def test_figure_api_accepts_the_same_layout_contract(as_mapping):
    layout = SequenceLayout(base_pitch_pt=10, canvas_size_pt=(1200, 400), sequence_origin_pt=(80, 200))
    value = layout.to_mapping() if as_mapping else layout
    figure = render_record_figure(_record(), style_overrides=STYLE, renderer_options={"layout": value})
    try:
        assert tuple(figure.get_size_inches() * 72) == pytest.approx((1200, 400))
    finally:
        plt.close(figure)


@pytest.mark.parametrize("field", ["base_pitch_pt", "wrong_units_px"])
def test_job_admission_rejects_invalid_layout_settings(tmp_path, field):
    source = write_parquet(tmp_path / "input.parquet", [{"id": "r1", "sequence": "ACGT"}])
    payload = densegen_job_payload(
        parquet_path=source, bundle_path=tmp_path / "results", outputs=[{"kind": "images", "fmt": "svg"}]
    )
    payload["render"]["options"] = {"layout": {field: -1}}
    with pytest.raises(SchemaError, match=field):
        validate_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "results").exists()


def test_job_cannot_silently_ignore_explicit_layout_for_video(tmp_path):
    source = write_parquet(tmp_path / "input.parquet", [{"id": "r1", "sequence": "ACGT"}])
    payload = densegen_job_payload(
        parquet_path=source, bundle_path=tmp_path / "results", outputs=[{"kind": "video", "path": "video.mp4"}]
    )
    payload["render"]["options"] = {"layout": {"base_pitch_pt": 10}}
    with pytest.raises(SchemaError, match="layout.*video"):
        validate_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "results").exists()


def test_base_pitch_reserves_base_row_even_without_feature_rectangles():
    record = replace(_record(), features=())
    with pytest.raises(LayoutError, match="sequence/feature glyphs"):
        render_record_svg(record, style_overrides=STYLE, layout=SequenceLayout(base_pitch_pt=1))


def test_smaller_feature_text_does_not_allow_overlapping_sequence_letters():
    with pytest.raises(LayoutError, match="sequence/feature glyphs"):
        render_record_svg(
            _record(),
            style_overrides={**STYLE, "font_size_seq": 18, "font_size_feature_label": 8},
            layout=SequenceLayout(base_pitch_pt=8),
        )


def test_layout_job_validation_keeps_plotting_libraries_unloaded(tmp_path):
    source = write_parquet(tmp_path / "input.parquet", [{"id": "r1", "sequence": "ACGT"}])
    payload = densegen_job_payload(
        parquet_path=source, bundle_path=tmp_path / "results", outputs=[{"kind": "images", "fmt": "svg"}]
    )
    payload["render"]["options"] = {"layout": {"base_pitch_pt": 10}}
    job_path = write_job(tmp_path / "job.yaml", payload)
    code = (
        "import sys\nfrom dnadesign.baserender import SequenceLayout, validate_job\n"
        "validate_job(sys.argv[1])\nprint('matplotlib' in sys.modules)\nprint('numpy' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(job_path)], capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == ["False", "False"]
    assert not (tmp_path / "results").exists()
