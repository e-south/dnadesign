"""Tests for bounded, descriptor-captured render sources."""

from pathlib import Path

import pytest

from dnadesign.baserender.src.io.captured_source import CapturedSource


def test_verify_unchanged_reuses_capture_limit_after_content_release(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"data")
    captured = CapturedSource.capture(source, max_bytes=4).without_content()
    source.write_bytes(b"oversized")

    with pytest.raises(ValueError, match="Render source changed during execution") as error:
        captured.verify_unchanged()

    assert str(error.value.__cause__) == f"Render source exceeds the maximum of 4 bytes: {source}"
