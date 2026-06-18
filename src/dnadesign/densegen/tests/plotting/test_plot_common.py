"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_common.py

Regression tests for plot common DenseGen plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dnadesign.densegen.src.viz.plot_common import _rename_artifact_path, _save_figure


def test_save_figure_writes_svg_sibling_for_non_svg_output(tmp_path) -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1])
        pdf_path = tmp_path / "demo.pdf"
        _save_figure(fig, pdf_path, style={})
        assert pdf_path.exists()
        assert pdf_path.with_suffix(".svg").exists()
    finally:
        plt.close(fig)


def test_save_figure_does_not_duplicate_svg_output(tmp_path) -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [1, 0])
        svg_path = tmp_path / "demo.svg"
        _save_figure(fig, svg_path, style={})
        assert svg_path.exists()
        assert list(tmp_path.glob("demo*.svg")) == [svg_path]
    finally:
        plt.close(fig)


def test_rename_artifact_path_moves_svg_sibling_with_primary_output(tmp_path) -> None:
    source = tmp_path / "legacy.pdf"
    svg_source = source.with_suffix(".svg")
    source.write_bytes(b"pdf")
    svg_source.write_text("<svg />", encoding="utf-8")

    target = tmp_path / "renamed.pdf"
    result = _rename_artifact_path(source, target)

    assert result == target
    assert target.exists()
    assert target.with_suffix(".svg").exists()
    assert not source.exists()
    assert not svg_source.exists()
