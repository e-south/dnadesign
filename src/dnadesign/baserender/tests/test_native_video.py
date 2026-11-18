"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_native_video.py

Qualify fixed native frames, encoding contracts and atomic video publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from PIL import Image, ImageChops

from dnadesign.baserender import RenderingError, SchemaError, run_job, validate_job
from dnadesign.baserender.src.cli.actions import _job_to_mapping

from .test_native_jobs import _payload

_LIVE = pytest.mark.skipif(
    any(shutil.which(name) is None for name in ("ffmpeg", "ffprobe", "inkscape")),
    reason="native video qualification needs existing FFmpeg and Inkscape",
)


def _video_payload(tmp_path, *, fmt="png"):
    payload = _payload(tmp_path, fmt=fmt)
    payload["outputs"].append(
        {
            "kind": "video",
            "path": "movie.mp4",
            "fps": 5,
            "frames_per_record": 2,
            "pauses": {"r60": 0.2},
            "background": "#e5eded",
            "encode_timeout_s": 30,
        }
    )
    return payload


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("background", None, "background"),
        ("background", "transparent", "background"),
        ("content_fit", "fill_width", "content_fit"),
        ("title_text", "second title", "title"),
        ("encode_timeout_s", 0, "encode_timeout_s"),
    ],
)
def test_native_video_rejects_ambiguous_or_ignored_settings(tmp_path, field, value, match):
    payload = _video_payload(tmp_path)
    payload["outputs"][1][field] = value
    with pytest.raises(SchemaError, match=match):
        validate_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_native_video_requires_fixed_canvas_origin_and_static_fallback(tmp_path):
    payload = _video_payload(tmp_path)
    for key in ("canvas_size_pt", "sequence_origin_pt"):
        removed = payload["render"]["options"]["layout"].pop(key)
        with pytest.raises(SchemaError, match=key):
            validate_job(payload, caller_root=tmp_path)
        payload["render"]["options"]["layout"][key] = removed
    payload["outputs"] = payload["outputs"][1:]
    with pytest.raises(SchemaError, match="SVG/PNG.*fallback"):
        validate_job(payload, caller_root=tmp_path)


def test_native_video_settings_roundtrip(tmp_path):
    payload = _video_payload(tmp_path)
    job = validate_job(payload, caller_root=tmp_path)
    normalized = _job_to_mapping(job)
    assert normalized["outputs"][1]["background"] == "#e5eded"
    assert normalized["outputs"][1]["encode_timeout_s"] == 30
    assert validate_job(normalized, caller_root=tmp_path).outputs == job.outputs


@_LIVE
@pytest.mark.parametrize("fps,fmt", [(5, "png"), (30, "svg")])
def test_native_video_keeps_pixels_geometry_and_exact_frame_timing(tmp_path, fps, fmt):
    payload = _video_payload(tmp_path, fmt=fmt)
    payload["outputs"][1]["fps"] = fps
    report = run_job(payload, caller_root=tmp_path)
    bundle = tmp_path / "bundle"
    manifest_text = (bundle / "manifest.json").read_text()
    assert str(tmp_path) not in manifest_text
    manifest = json.loads(manifest_text)
    video = manifest["output_metrics"]["video"]
    repeats = [2 + round(0.2 * fps), 2]
    assert video["planned_frame_count"] == sum(repeats)
    assert video["frame_repeats"] == repeats
    assert video["background"] == "#e5eded"
    assert video["width_px"] == 2600 and video["height_px"] == 800
    assert len(video["encoder"]["executable_sha256"]) == 64
    rows = manifest["output_metrics"]["images"]["records"]
    assert [row["diagnostics"]["sequence_origin_pt"] for row in rows] == [[60, 280], [60, 280]]
    assert len({row["diagnostics"]["base_pitch_pt"] for row in rows}) == 1
    if fmt == "png":
        assert all(Image.open(bundle / "images" / row["file"]).getchannel("A").getextrema() == (0, 255) for row in rows)
    else:
        assert all(
            ET.parse(bundle / "images" / row["file"]).getroot().tag == "{http://www.w3.org/2000/svg}svg" for row in rows
        )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(bundle / "movie.mp4"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["nb_read_frames"], stream["r_frame_rate"]) == (
        2600,
        800,
        str(sum(repeats)),
        f"{fps}/1",
    )
    assert report.outputs["video_path"] == str(bundle / "movie.mp4")


def test_video_background_and_timeout_are_not_silently_ignored_by_ordinary_jobs(tmp_path):
    payload = _video_payload(tmp_path)
    del payload["render"]["native_svg"]
    del payload["render"]["options"]["layout"]
    with pytest.raises(SchemaError, match="native_svg"):
        validate_job(payload, caller_root=tmp_path)


@_LIVE
def test_native_video_encoder_failure_rolls_back_fallbacks_too(tmp_path, monkeypatch):
    from dnadesign.baserender.src.outputs import native_video

    def fail(*args, **kwargs):
        raise RenderingError("intentional encode failure")

    monkeypatch.setattr(native_video, "encode_native_frames", fail)
    with pytest.raises(RenderingError, match="intentional encode"):
        run_job(_video_payload(tmp_path), caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_native_video_missing_encoder_fails_before_input_capture(tmp_path, monkeypatch):
    from dnadesign.baserender.src.outputs import native_video
    from dnadesign.baserender.src.reporting import RunReport

    def fail(*args, **kwargs):
        raise RenderingError("FFmpeg is unavailable")

    monkeypatch.setattr(native_video, "resolve_encoder", fail)
    monkeypatch.setattr(RunReport, "capture_source_evidence", lambda *args: pytest.fail("late dependency check"))
    with pytest.raises(RenderingError, match="FFmpeg.*unavailable"):
        run_job(_video_payload(tmp_path), caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_native_video_composites_alpha_without_resizing():
    from dnadesign.baserender.src.outputs.native_video import composite_frame

    source = Image.new("RGBA", (12, 6), (0, 0, 0, 0))
    source.putpixel((5, 3), (255, 0, 0, 128))
    result = composite_frame(source, "#e5eded")
    expected = Image.new("RGBA", source.size, "#e5eded")
    expected.alpha_composite(source)
    assert result.mode == "RGB" and result.size == source.size
    assert ImageChops.difference(result, expected.convert("RGB")).getbbox() is None


@_LIVE
@pytest.mark.parametrize("change,match", [("width", "match.*canvas"), ("odd", "even pixel"), ("record", "missing")])
def test_native_video_rejects_invalid_geometry_before_drawing(tmp_path, monkeypatch, change, match):
    import matplotlib.pyplot as plt

    payload = _video_payload(tmp_path)
    if change == "width":
        payload["outputs"][1]["width_px"] = 1280
    elif change == "odd":
        payload["render"]["options"]["layout"]["canvas_size_pt"][0] = 1300.5
    else:
        path = Path(payload["input"]["path"])
        rows = json.loads(path.read_text())
        rows[1]["effects"][0]["target"]["bindings"][0]["feature_id"] = "missing"
        path.write_text(json.dumps(rows))
    monkeypatch.setattr(plt, "figure", lambda *args, **kwargs: pytest.fail("invalid batch drew a figure"))
    with pytest.raises((SchemaError, RenderingError), match=match):
        run_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="encoder timing qualification needs existing FFmpeg")
@pytest.mark.parametrize("fps,repeats", [(5, [3, 2]), (30, [30, 30])])
def test_native_encoder_keeps_each_exact_hold_not_just_total_frames(tmp_path, fps, repeats):
    import numpy as np

    from dnadesign.baserender.src.outputs.native_video import encode_native_frames, resolve_encoder

    for index, color in enumerate(("red", "blue")):
        Image.new("RGB", (48, 24), color).save(tmp_path / f"frame{index:06d}.png")
    payload = _video_payload(tmp_path)
    payload["outputs"][1]["fps"] = fps
    output = validate_job(payload, caller_root=tmp_path).outputs[1]
    output.path.parent.mkdir(parents=True)
    encode_native_frames(resolve_encoder(), frames_dir=tmp_path, repeats=repeats, output=output)
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(output.path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 24, 48, 3)
    identities = [0 if frame[0, 0, 0] > frame[0, 0, 2] else 1 for frame in frames]
    assert identities == [0] * repeats[0] + [1] * repeats[1]
