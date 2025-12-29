"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_style_presets.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.contracts import SchemaError
from dnadesign.baserender.src.presets.style_presets import (
    DEFAULT_PRESET_NAME,
    deep_merge,
    resolve_style,
    resolve_style_preset_path,
)


def test_deep_merge_nested_only_overrides_changed():
    base = {"kmer": {"fill_alpha": 0.88, "height_factor": 1.10}}
    overrides = {"kmer": {"fill_alpha": 0.70}}
    out = deep_merge(base, overrides)
    assert out["kmer"]["fill_alpha"] == 0.70
    assert out["kmer"]["height_factor"] == 1.10


def test_deep_merge_non_mapping_replaces():
    base = {"connector_dash": [1.0, 3.0]}
    overrides = {"connector_dash": [2.0, 2.0]}
    out = deep_merge(base, overrides)
    assert out["connector_dash"] == [2.0, 2.0]


def test_deep_merge_palette_merge_adds_keys():
    base = {"palette": {"sigma": "#969eba"}}
    overrides = {"palette": {"tf:cpxr": "#bf812e"}}
    out = deep_merge(base, overrides)
    assert out["palette"]["sigma"] == "#969eba"
    assert out["palette"]["tf:cpxr"] == "#bf812e"


def test_resolve_style_preset_by_name():
    p = resolve_style_preset_path(DEFAULT_PRESET_NAME)
    assert p.exists()
    assert p.name in {f"{DEFAULT_PRESET_NAME}.yml", f"{DEFAULT_PRESET_NAME}.yaml"}


def test_resolve_style_preset_by_path():
    p = resolve_style_preset_path("styles/presentation_default.yml")
    assert p.exists()
    assert p.name == "presentation_default.yml"


def test_missing_preset_raises_schemaerror():
    with pytest.raises(SchemaError):
        resolve_style_preset_path("definitely_not_a_real_preset_12345")


def test_resolve_style_default_applies_presentation_default():
    s = resolve_style()
    # presentation_default.yml sets these away from Style() legacy defaults
    assert s.track_spacing == 35.0
    assert s.baseline_spacing == 40.0
    assert s.sigma_link_inner_margin_bp == 0.50


def test_unknown_key_in_overrides_raises_schemaerror():
    with pytest.raises(SchemaError):
        _ = resolve_style(overrides={"not_a_style_key": 123})


def test_invalid_kmer_nested_key_raises_schemaerror():
    with pytest.raises(SchemaError):
        _ = resolve_style(overrides={"kmer": {"not_a_kmer_key": 1}})
