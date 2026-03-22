"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_read_views_head_order.py

Tests for human-label-first ordering in USR head views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.usr.src.cli_commands.read_views import (
    _reorder_head_columns_for_display,
    _select_head_columns_for_rich_display,
)


def test_reorder_head_columns_prioritizes_usr_labels() -> None:
    df = pd.DataFrame(
        [
            {
                "sequence": "ACGT",
                "construct_seed__label": "demo-seed",
                "id": "abc",
                "usr_label__primary": "pDual-10",
                "length": 4,
            }
        ]
    )

    reordered = _reorder_head_columns_for_display(df, explicit_columns=None)

    assert list(reordered.columns) == [
        "id",
        "usr_label__primary",
        "construct_seed__label",
        "sequence",
        "length",
    ]


def test_reorder_head_columns_respects_explicit_selection() -> None:
    df = pd.DataFrame([{"sequence": "ACGT", "usr_label__primary": "pDual-10", "id": "abc"}])

    reordered = _reorder_head_columns_for_display(df, explicit_columns=["sequence", "id"])

    assert list(reordered.columns) == ["sequence", "usr_label__primary", "id"]


def test_select_head_columns_for_rich_display_focuses_wide_frames() -> None:
    df = pd.DataFrame(
        [
            {
                "sequence": "ACGT",
                "construct_seed__label": "demo-seed",
                "construct_seed__role": "anchor",
                "construct_seed__topology": "linear",
                "id": "abc",
                "usr_label__primary": "pDual-10",
                "usr_label__aliases": ["pDual10"],
                "length": 4,
                "source": "seed",
            }
        ]
    )

    focused = _select_head_columns_for_rich_display(df, explicit_columns=None)

    assert list(focused.columns) == [
        "id",
        "usr_label__primary",
        "usr_label__aliases",
        "sequence",
        "construct_seed__role",
        "construct_seed__topology",
    ]


def test_select_head_columns_for_rich_display_respects_explicit_selection() -> None:
    df = pd.DataFrame(
        [{"sequence": "ACGT", "usr_label__primary": "pDual-10", "usr_label__aliases": ["pDual10"], "id": "abc"}]
    )

    focused = _select_head_columns_for_rich_display(df, explicit_columns=["id", "sequence"])

    assert list(focused.columns) == ["sequence", "usr_label__primary", "usr_label__aliases", "id"]
