"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_native_jobs.py

Native image-job contracts, public API parity and publication failure checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from dnadesign.baserender import (
    IllustrationCaption,
    RenderingError,
    SchemaError,
    SequenceLayout,
    adapt_record,
    render_record_svg,
    run_job,
    validate_job,
)
from dnadesign.baserender.src.cli.actions import _job_to_mapping

from .conftest import write_job


def _rows():
    rows = []
    for length in (60, 100):
        sequence = ("ACGT" * 25)[:length]
        rows.append(
            {
                "id": f"r{length}",
                "sequence": sequence,
                "features": [
                    {
                        "id": name,
                        "kind": "kmer",
                        "span": {"start": start, "end": start + 6, "strand": "fwd"},
                        "label": sequence[start : start + 6],
                    }
                    for name, start in (("left", 4), ("right", 28))
                ],
                "effects": [
                    {
                        "kind": "anchored_illustration",
                        "target": {
                            "bindings": [
                                {"anchor_id": "upstream", "feature_id": "left", "start": 4, "end": 10},
                                {"anchor_id": "downstream", "feature_id": "right", "start": 28, "end": 34},
                            ]
                        },
                        "params": {"asset_id": "rnap_sigma70", "width_px": 300},
                    }
                ],
            }
        )
    return rows


def _payload(tmp_path, *, fmt="svg"):
    source = tmp_path / "records.json"
    source.write_text(json.dumps(_rows()))
    return {
        "version": 4,
        "contract": {"kind": "sequence_rows_render_v3"},
        "bundle": {"path": str(tmp_path / "bundle")},
        "input": {
            "kind": "json",
            "path": str(source),
            "alphabet": "DNA",
            "adapter": {
                "kind": "generic_features",
                "columns": {"id": "id", "sequence": "sequence", "features": "features", "effects": "effects"},
                "policies": {},
            },
        },
        "render": {
            "renderer": "sequence_rows",
            "style": {"preset": None, "overrides": {"legend": False, "font_mono": "Arial", "font_label": "Arial"}},
            "options": {
                "layout": {"canvas_size_pt": [1300, 400], "sequence_origin_pt": [60, 280], "top_extent_pt": 104}
            },
            "native_svg": {
                "captions": {"rnap_sigma70": {"text": "RNAP", "family": "Arial", "size_pt": 18}},
                "rasterization": {"dpi": 144},
            },
        },
        "outputs": [{"kind": "images", "fmt": fmt}],
        "run": {"strict": True, "fail_on_skips": True},
    }


def test_native_svg_job_matches_the_public_api_and_records_portable_geometry(tmp_path):
    payload = _payload(tmp_path)
    report = run_job(payload, caller_root=tmp_path)
    bundle = tmp_path / "bundle"
    for row in _rows():
        record = adapt_record(
            row, adapter_kind="generic_features", adapter_columns=payload["input"]["adapter"]["columns"]
        )
        expected = render_record_svg(
            record,
            style_overrides=payload["render"]["style"]["overrides"],
            layout=SequenceLayout.from_mapping(payload["render"]["options"]["layout"]),
            captions={"rnap_sigma70": IllustrationCaption("RNAP", family="Arial", size_pt=18)},
        )
        target = bundle / "images" / f"{row['id']}.svg"
        assert target.read_text() == expected.svg
        detached = expected.diagnostics.to_mapping()
        detached["feature_bounds_pt"]["left"][0] = -1
        assert expected.diagnostics.feature_bounds_pt["left"][0] >= 0
        assert not ET.parse(target).findall(".//{http://www.w3.org/2000/svg}image")
    manifest_text = (bundle / "manifest.json").read_text()
    assert str(tmp_path) not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["render_spec"]["native_svg"]["captions"]["rnap_sigma70"]["size_pt"] == 18
    records = manifest["output_metrics"]["images"]["records"]
    assert len(records) == 2
    assert records[0]["diagnostics"]["sequence_origin_pt"] == [60, 280]
    assert "path" not in records[0]["diagnostics"]["fonts"]["sequence"]
    resolved = records[0]["resolved_illustrations"]["br-vector-0"]
    assert resolved["artwork"]["element_ids"] and "path" not in resolved["artwork"]
    assert resolved["caption"]["size_pt"] == 18
    assert report.outputs["images_dir"] == str(bundle / "images")


@pytest.mark.skipif(shutil.which("inkscape") is None, reason="native PNG integration requires Inkscape")
def test_native_png_job_uses_the_same_geometry_and_transparency(tmp_path):
    payload = _payload(tmp_path, fmt="png")
    run_job(payload, caller_root=tmp_path)
    for row in _rows():
        image = Image.open(tmp_path / "bundle" / "images" / f"{row['id']}.png")
        assert image.size == (2600, 800)
        assert image.mode == "RGBA" and image.getchannel("A").getextrema() == (0, 255)
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    record = manifest["output_metrics"]["images"]["records"][0]
    assert record["rasterizer"]["version"].startswith("Inkscape ")
    assert len(record["source_svg_sha256"]) == 64


def test_native_job_normalization_retains_settings_and_is_detached(tmp_path):
    from dnadesign.baserender import rasterize_svg
    from dnadesign.baserender.src.config.raster import RasterOptions

    defaults = RasterOptions().to_mapping()
    assert all(inspect.signature(rasterize_svg).parameters[key].default == value for key, value in defaults.items())
    payload = _payload(tmp_path)
    parsed = validate_job(payload, caller_root=tmp_path)
    normalized = _job_to_mapping(parsed)
    assert normalized["render"]["native_svg"]["rasterization"]["dpi"] == 144
    payload["render"]["native_svg"]["captions"]["rnap_sigma70"]["size_pt"] = 6
    assert parsed.render.native_svg.captions["rnap_sigma70"].size_pt == 18
    assert validate_job(normalized, caller_root=tmp_path).render.native_svg == parsed.render.native_svg


def test_native_batch_rejects_bad_second_record_before_any_figure_or_bundle(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    rows = _rows()
    rows[1]["effects"][0]["target"]["bindings"][0]["feature_id"] = "missing"
    Path(payload["input"]["path"]).write_text(json.dumps(rows))

    def forbidden(*args, **kwargs):
        pytest.fail("native batch drew before admitting every record")

    monkeypatch.setattr(plt, "figure", forbidden)
    with pytest.raises(RenderingError, match="missing"):
        run_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_missing_native_png_dependency_fails_before_source_capture(tmp_path, monkeypatch):
    from dnadesign.baserender.src.reporting import RunReport

    payload = _payload(tmp_path, fmt="png")
    payload["render"]["native_svg"]["rasterization"]["inkscape"] = "missing-native-job-inkscape"

    def forbidden(*args, **kwargs):
        pytest.fail("missing native dependency was deferred until input capture")

    monkeypatch.setattr(RunReport, "capture_source_evidence", forbidden)
    with pytest.raises(RenderingError, match="Inkscape.*unavailable"):
        run_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "native",
    [
        None,
        False,
        {"unknown": 1},
        {"rasterization": {"dpi": False}},
        {"captions": {"rnap_sigma70": {"text": "RNAP", "wrong": 1}}},
    ],
)
def test_invalid_native_config_fails_at_shared_admission(tmp_path, native):
    payload = _payload(tmp_path)
    payload["render"]["native_svg"] = native
    with pytest.raises(SchemaError, match="native_svg"):
        validate_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_native_job_validation_does_not_load_plotting_libraries(tmp_path):
    path = write_job(tmp_path / "job.yaml", _payload(tmp_path))
    code = (
        "import sys\nfrom dnadesign.baserender import validate_job\nvalidate_job(sys.argv[1])\n"
        "print('matplotlib' in sys.modules)\nprint('numpy' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ["False", "False"]


def _custom_art(payload, tmp_path):
    path = tmp_path / "glyph.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><g id="body"><path d="M0 0H100V40H0Z" fill="#777"/></g></svg>'
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["render"]["native_svg"]["illustrations"] = {
        "rnap_sigma70": {
            "artwork": {
                "path": str(path),
                "sha256": digest,
                "element_ids": ["body"],
                "view_box": [0, 0, 100, 40],
                "anchors": {"upstream": [0.25, 0.9], "downstream": [0.5, 0.9]},
            }
        }
    }
    return path, digest


def test_native_job_pins_custom_vector_artwork_without_local_paths_in_manifest(tmp_path):
    payload = _payload(tmp_path)
    path, digest = _custom_art(payload, tmp_path)
    run_job(payload, caller_root=tmp_path)
    text = (tmp_path / "bundle" / "manifest.json").read_text()
    assert str(path) not in text
    manifest = json.loads(text)
    source = manifest["render_spec"]["native_svg"]["illustrations"]["rnap_sigma70"]["artwork"]
    assert source["sha256"] == digest and source["element_ids"] == ["body"]
    assert "path" not in source
    assert (
        manifest["output_metrics"]["images"]["records"][0]["diagnostics"]["illustrations"][0]["source_sha256"] == digest
    )


def test_artwork_change_during_native_render_aborts_publication(tmp_path, monkeypatch):
    from dnadesign.baserender.src.outputs import native_images

    payload = _payload(tmp_path)
    path, _digest = _custom_art(payload, tmp_path)
    real_draw = native_images.draw_native_svg
    count = 0

    def draw(plan):
        nonlocal count
        count += 1
        if count == 2:
            path.write_text("new author revision")
        return real_draw(plan)

    monkeypatch.setattr(native_images, "draw_native_svg", draw)
    with pytest.raises(RenderingError, match="SHA-256"):
        run_job(payload, caller_root=tmp_path)
    assert count == 2 and not (tmp_path / "bundle").exists()


def test_late_native_draw_failure_leaves_no_partial_bundle(tmp_path, monkeypatch):
    from dnadesign.baserender.src.outputs import native_images

    payload = _payload(tmp_path)
    real_draw = native_images.draw_native_svg
    count = 0

    def draw(plan):
        nonlocal count
        count += 1
        if count == 2:
            raise RenderingError("intentional late native failure")
        return real_draw(plan)

    monkeypatch.setattr(native_images, "draw_native_svg", draw)
    with pytest.raises(RenderingError, match="intentional late"):
        run_job(payload, caller_root=tmp_path)
    assert count == 2 and not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("fmt", ["pdf", "video"])
def test_unsupported_native_outputs_fail_at_admission(tmp_path, fmt):
    payload = _payload(tmp_path, fmt=fmt)
    if fmt == "video":
        payload["outputs"] = [{"kind": "video", "path": "movie.mp4"}]
    with pytest.raises(SchemaError, match="native_svg.*SVG/PNG"):
        validate_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_native_single_file_rejects_implicit_raster_grid(tmp_path):
    payload = _payload(tmp_path)
    payload["outputs"] = [{"kind": "images", "path": "one.svg", "fmt": "svg"}]
    with pytest.raises(SchemaError, match="exactly one record"):
        run_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("fmt", ["svg", "png"])
def test_native_cli_run_publishes_real_images(tmp_path, fmt):
    if fmt == "png" and shutil.which("inkscape") is None:
        pytest.skip("native PNG integration requires Inkscape")
    path = write_job(tmp_path / "job.yaml", _payload(tmp_path, fmt=fmt))
    result = subprocess.run(
        [sys.executable, "-B", "-m", "dnadesign.baserender.src.cli.app", "job", "run", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "bundle" / "images" / f"r100.{fmt}").is_file()
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert manifest["output_metrics"]["images"]["representation"] == "native_svg"


def test_native_batch_can_mix_records_with_and_without_an_illustration(tmp_path):
    payload = _payload(tmp_path)
    rows = _rows()
    rows[1]["effects"] = []
    Path(payload["input"]["path"]).write_text(json.dumps(rows))
    run_job(payload, caller_root=tmp_path)
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert manifest["output_metrics"]["images"]["records"][1]["resolved_illustrations"] == {}


def test_unused_native_override_is_rejected_before_drawing(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    payload["render"]["native_svg"]["captions"]["typo"] = {"text": "unused"}

    def forbidden(*args, **kwargs):
        pytest.fail("unused native override reached drawing")

    monkeypatch.setattr(plt, "figure", forbidden)
    with pytest.raises(RenderingError, match="unused.*typo"):
        run_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()
