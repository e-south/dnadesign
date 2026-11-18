"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_configuration_admission.py

Keep YAML, hot-edited styles and typed video jobs on explicit admission boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import re
from dataclasses import replace

import pytest

from dnadesign.baserender import SchemaError, resolve_style, run_job, validate_job

from .conftest import write_job
from .test_native_jobs import _payload
from .test_native_video import _video_payload


def test_duplicate_nested_job_key_is_rejected(tmp_path):
    path = write_job(tmp_path / "job.yaml", _payload(tmp_path))
    text = re.sub(r"(?m)^(\s*)dpi: 144$", r"\1dpi: 72\n\1dpi: 144", path.read_text())
    assert text != path.read_text()
    path.write_text(text)
    with pytest.raises(SchemaError, match="Duplicate YAML key.*dpi"):
        validate_job(path)


def test_duplicate_style_key_is_rejected(tmp_path):
    path = tmp_path / "style.yaml"
    path.write_text("font_size_seq: 12\nfont_size_seq: 18\n")
    with pytest.raises(SchemaError, match="Duplicate YAML key.*font_size_seq"):
        resolve_style(preset=path)


def test_style_file_edits_take_effect_without_a_process_restart(tmp_path):
    path = tmp_path / "style.yaml"
    path.write_text("font_size_seq: 12\n")
    assert resolve_style(preset=path).font_size_seq == 12
    path.write_text("font_size_seq: 18\n")
    assert resolve_style(preset=path).font_size_seq == 18
    path.write_text("font_size_seq: 12\nfont_size_seq: 18\n")
    with pytest.raises(SchemaError, match="Duplicate YAML key"):
        resolve_style(preset=path)


def test_cached_style_results_do_not_share_mutable_palette_state(tmp_path):
    path = tmp_path / "style.yaml"
    path.write_text("palette: {example: '#aabbcc'}\n")
    first = resolve_style(preset=path)
    first.palette["example"] = "#112233"
    assert resolve_style(preset=path).palette["example"] == "#aabbcc"


def test_yaml_merge_override_and_scalar_alias_remain_explicitly_supported(tmp_path):
    path = tmp_path / "style.yaml"
    path.write_text(
        "font_mono: &font Arial\nfont_label: *font\npalette:\n  <<: {example: '#aabbcc'}\n  example: '#112233'\n"
    )
    style = resolve_style(preset=path)
    assert style.font_mono == style.font_label == "Arial"
    assert style.palette["example"] == "#112233"


@pytest.mark.parametrize(
    "field,value",
    [
        ("fps", 0),
        ("fps", True),
        ("frames_per_record", 0),
        ("pauses", {"r60": -1}),
        ("pauses", {"r60": float("nan")}),
        ("total_duration", float("inf")),
        ("total_duration", -1),
        ("width_px", 0),
        ("height_px", True),
        ("aspect_ratio", 0),
        ("content_fit", "unknown"),
        ("title_font_size", 0),
    ],
)
def test_typed_video_settings_fail_before_dependencies_or_input_capture(tmp_path, monkeypatch, field, value):
    from dnadesign.baserender.src.outputs import native_video

    job = validate_job(_video_payload(tmp_path), caller_root=tmp_path)
    output = replace(job.outputs[1], **{field: value})
    job = replace(job, outputs=(job.outputs[0], output))
    monkeypatch.setattr(
        native_video.NativeVideoWriter, "resolve", lambda *args, **kwargs: pytest.fail("late video admission")
    )
    with pytest.raises(SchemaError, match=field):
        run_job(job)
    assert not (tmp_path / "bundle").exists()
