"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_notebook_render_contract.py

Tests for the DenseGen notebook render contract used by BaseRender integration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors

from dnadesign.baserender import DENSEGEN_TFBS_REQUIRED_KEYS, render_parquet_record_figure
from dnadesign.densegen.src.core.record_values import normalize_used_tfbs_entries
from dnadesign.densegen.src.integrations.baserender.notebook_contract import (
    REQUIRED_TFBS_ENTRY_KEYS,
    densegen_notebook_render_contract,
)


def test_notebook_render_contract_renders_without_optional_details_column(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAAA",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
                ],
            }
        ]
    ).to_parquet(records_path)

    contract = densegen_notebook_render_contract()
    fig = render_parquet_record_figure(
        dataset_path=records_path,
        record_id="row1",
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
        style_preset=contract.style_preset,
        style_overrides=contract.style_overrides,
    )
    assert fig is not None
    plt.close(fig)


def test_notebook_render_contract_is_explicit_and_complete() -> None:
    contract = densegen_notebook_render_contract()
    assert contract.adapter_kind == "densegen_tfbs"
    assert contract.adapter_columns == {
        "id": "id",
        "sequence": "sequence",
        "annotations": "densegen__used_tfbs_detail",
    }
    assert contract.adapter_policies == {"on_invalid_row": "error"}
    assert contract.style_preset == "presentation_default"
    assert isinstance(contract.style_overrides, dict)
    palette = contract.style_overrides.get("palette")
    assert isinstance(palette, dict)
    assert palette.get("tf:lexA") == "#5DADE2"
    assert palette.get("tf:cpxR") == "#2D9B66"
    assert palette.get("tf:baeR") == "#E58A2B"
    assert palette.get("tf:background") == "#C3CAD3"
    assert palette.get("promoter:sigma70_core:upstream") == "#7D86D1"
    assert palette.get("promoter:sigma70_core:downstream") == "#C886D1"
    assert contract.style_overrides.get("font_size_seq") == 24
    assert contract.style_overrides.get("font_size_label") == 24
    assert contract.style_overrides.get("legend_mode") == "bottom"
    assert contract.style_overrides.get("legend_height_px") == 136.0
    assert contract.style_overrides.get("legend_font_size") == 24
    assert contract.style_overrides.get("legend_patch_w") == 88.0
    assert contract.style_overrides.get("legend_patch_h") == 34.0
    assert contract.style_overrides.get("legend_gap_patch_text") == 22.0
    assert contract.style_overrides.get("legend_gap_x") == 44.0
    assert contract.style_overrides.get("legend_vertical_align") == 1.0
    assert contract.style_overrides.get("uniform_display_font_size") is True
    assert contract.record_window_limit == 500
    assert REQUIRED_TFBS_ENTRY_KEYS == DENSEGEN_TFBS_REQUIRED_KEYS


def test_notebook_render_contract_renders_extended_densegen_tf_tags(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAGGGGCCCC",
                "densegen__used_tfbs_detail": [
                    {
                        "regulator": "lexA_CTGTATAWAWWHACA",
                        "orientation": "fwd",
                        "sequence": "AAA",
                        "offset": 0,
                    },
                    {
                        "regulator": "cpxR_MANWWHTTTAM",
                        "orientation": "fwd",
                        "sequence": "GGG",
                        "offset": 4,
                    },
                    {
                        "regulator": "baeR_TTTCTSCVHNA",
                        "orientation": "fwd",
                        "sequence": "CCC",
                        "offset": 8,
                    },
                ],
            }
        ]
    ).to_parquet(records_path)

    contract = densegen_notebook_render_contract()
    fig = render_parquet_record_figure(
        dataset_path=records_path,
        record_id="row1",
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
        style_preset=contract.style_preset,
        style_overrides=contract.style_overrides,
    )
    assert fig is not None
    plt.close(fig)


def test_notebook_render_contract_bottom_legend_keeps_densegen_parts_compact(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "ACGATTGACATGCTCCATTATAATGGGG",
                "densegen__used_tfbs_detail": [
                    {
                        "regulator": "lexA_CTGTATAWAWWHACA",
                        "orientation": "fwd",
                        "sequence": "ACGA",
                        "offset": 0,
                    },
                    {
                        "regulator": "cpxR_MANWWHTTTAM",
                        "orientation": "fwd",
                        "sequence": "TGCT",
                        "offset": 10,
                    },
                    {
                        "regulator": "baeR_TTTCTSCVHNA",
                        "orientation": "fwd",
                        "sequence": "CCAT",
                        "offset": 14,
                    },
                    {
                        "part_kind": "fixed_element",
                        "role": "upstream",
                        "constraint_name": "sigma70_core",
                        "sequence": "TTGACA",
                        "offset": 4,
                        "length": 6,
                        "spacer_length": 8,
                    },
                    {
                        "part_kind": "fixed_element",
                        "role": "downstream",
                        "constraint_name": "sigma70_core",
                        "sequence": "TATAAT",
                        "offset": 18,
                        "length": 6,
                        "spacer_length": 8,
                    },
                ],
            }
        ]
    ).to_parquet(records_path)

    contract = densegen_notebook_render_contract()
    fig = render_parquet_record_figure(
        dataset_path=records_path,
        record_id="row1",
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
        style_preset=contract.style_preset,
        style_overrides=contract.style_overrides,
    )
    try:
        axis = fig.axes[0]
        fig.canvas.draw()
        legend_texts = [text for text in axis.texts if float(text.get_zorder()) == 10.0]
        assert len(legend_texts) >= 5
        assert len({round(float(text.get_position()[1]), 3) for text in legend_texts}) <= 2
        legend_labels = {text.get_text() for text in legend_texts}
        assert "LexA sites" in legend_labels
        assert "CpxR sites" in legend_labels
        assert "BaeR sites" in legend_labels
        assert "-35 sites" in legend_labels
        assert "-10 sites" in legend_labels
    finally:
        plt.close(fig)


def test_notebook_render_contract_uses_variant_specific_promoter_annotation_labels(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
                "densegen__used_tfbs_detail": [
                    {
                        "regulator": "lexA_CTGTATAWAWWHACA",
                        "orientation": "fwd",
                        "sequence": "AAA",
                        "offset": 0,
                    },
                    {
                        "part_kind": "fixed_element",
                        "role": "upstream",
                        "constraint_name": "sigma70_core",
                        "sequence": "TTTGGG",
                        "offset": 5,
                        "length": 6,
                        "spacer_length": 6,
                        "variant_id": "f",
                    },
                    {
                        "part_kind": "fixed_element",
                        "role": "downstream",
                        "constraint_name": "sigma70_core",
                        "sequence": "AAAATT",
                        "offset": 17,
                        "length": 6,
                        "spacer_length": 6,
                        "variant_id": "B",
                    },
                ],
            }
        ]
    ).to_parquet(records_path)

    contract = densegen_notebook_render_contract()
    fig = render_parquet_record_figure(
        dataset_path=records_path,
        record_id="row1",
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
        style_preset=contract.style_preset,
        style_overrides=contract.style_overrides,
    )
    try:
        axis = fig.axes[0]
        fig.canvas.draw()
        text_labels = {text.get_text() for text in axis.texts}
        legend_labels = {text.get_text() for text in axis.texts if float(text.get_zorder()) == 10.0}
        assert "-35 site (f)" in text_labels
        assert "-10 site (B)" in text_labels
        assert "-35 sites" in legend_labels
        assert "-10 sites" in legend_labels
    finally:
        plt.close(fig)


def test_notebook_render_contract_keeps_background_cpxr_baer_visually_separated() -> None:
    contract = densegen_notebook_render_contract()
    palette = contract.style_overrides.get("palette") or {}

    def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    background = mcolors.to_rgb(str(palette.get("tf:background")))
    cpxr = mcolors.to_rgb(str(palette.get("tf:cpxR")))
    baer = mcolors.to_rgb(str(palette.get("tf:baeR")))

    assert _distance(background, cpxr) >= 0.20
    assert _distance(background, baer) >= 0.20
    assert _distance(cpxr, baer) >= 0.20


def test_notebook_render_contract_renders_legacy_tf_tfbs_keys(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAAA",
                "densegen__used_tfbs_detail": normalize_used_tfbs_entries(
                    [{"tf": "lexA", "orientation": "fwd", "tfbs": "AAA", "offset": 0}]
                ),
            }
        ]
    ).to_parquet(records_path)

    contract = densegen_notebook_render_contract()
    fig = render_parquet_record_figure(
        dataset_path=records_path,
        record_id="row1",
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
        style_preset=contract.style_preset,
        style_overrides=contract.style_overrides,
    )
    assert fig is not None
    plt.close(fig)
