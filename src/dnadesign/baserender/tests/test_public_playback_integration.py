"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_public_playback_integration.py

Verify public BaseRender seams required by dense-array playback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib.resources import files

from dnadesign import baserender


def test_low_level_render_operations_are_public() -> None:
    """Producer integrations do not need to import BaseRender internals."""
    assert callable(baserender.initialize_runtime)
    assert callable(baserender.compute_layout)
    assert callable(baserender.render_record)


def test_rnap_overlay_is_a_packaged_resource() -> None:
    """The curated RNAP illustration is available after package installation."""
    asset = files("dnadesign.baserender").joinpath(
        "assets",
        "overlays",
        "rnap_sigma70.png",
    )
    assert asset.is_file()
    assert asset.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
