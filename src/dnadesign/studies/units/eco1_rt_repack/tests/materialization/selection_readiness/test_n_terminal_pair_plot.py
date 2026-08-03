"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_n_terminal_pair_plot.py

Focused N-terminal comparison tests for the selected Eco1 distal pair.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.n_terminal_pair_plot import (
    build_n_terminal_pair_comparison,
    write_n_terminal_pair_comparison_plot,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_METADATA,
    SELECTION_PLOT_PLAIN_TITLES,
)


def test_n_terminal_pair_comparison_uses_canonical_sequences_and_counts_charge_proxy() -> None:
    sequences = {
        "wild_type": _canonical("MKSAEYLNTFRLRNLG"),
        "candidate_d01": _canonical("MKSASVLRNEKLKKKG"),
        "candidate_d02": _canonical("MKAREKEREERLREMG"),
    }
    panel_rows = [
        {"candidate_id": "candidate_d01", "variant_id": "Eco1RT-G3-D01"},
        {"candidate_id": "candidate_d02", "variant_id": "Eco1RT-G3-D02"},
    ]

    rows = build_n_terminal_pair_comparison(
        panel_rows=panel_rows,
        canonical_sequences_by_id=sequences,
    )

    assert [row["label"] for row in rows] == ["WT", "D01", "D02"]
    assert [row["sequence_window"][:16] for row in rows] == [
        "MKSAEYLNTFRLRNLG",
        "MKSASVLRNEKLKKKG",
        "MKAREKEREERLREMG",
    ]
    assert [(row["basic_count_4_16"], row["acidic_count_4_16"], row["net_charge_proxy_4_16"]) for row in rows] == [
        (2, 1, 1),
        (5, 1, 4),
        (5, 5, 0),
    ]
    assert rows[0]["changed_positions"] == ()
    assert rows[1]["changed_positions"] == (5, 6, 8, 9, 10, 11, 13, 14, 15)
    assert rows[2]["changed_positions"] == (3, 4, 6, 7, 8, 9, 10, 14, 15)


def test_n_terminal_pair_plot_marks_alpha1_contact_positions_and_review_limit(tmp_path: Path) -> None:
    sequences = {
        "wild_type": _canonical("MKSAEYLNTFRLRNLG"),
        "candidate_d01": _canonical("MKSASVLRNEKLKKKG"),
        "candidate_d02": _canonical("MKAREKEREERLREMG"),
    }
    panel_rows = [
        {"candidate_id": "candidate_d01", "variant_id": "Eco1RT-G3-D01"},
        {"candidate_id": "candidate_d02", "variant_id": "Eco1RT-G3-D02"},
    ]

    row = write_n_terminal_pair_comparison_plot(
        tmp_path,
        panel_rows=panel_rows,
        canonical_sequences_by_id=sequences,
        input_hashes={"foldcheck_input_sequences": "sha256:test"},
    )

    svg_text = Path(str(row["path"])).read_text(encoding="utf-8")
    assert row["plot_id"] == "selection_distal_pair_n_terminal_comparison"
    assert row["role"] == "review_only"
    assert "descriptive" in str(row["interpretation_limit"]).lower()
    assert row["title"] == SELECTION_PLOT_PLAIN_TITLES[row["plot_id"]]
    assert "WT, D01, and D02 N-terminal sequence and charge-proxy comparison" in svg_text
    assert "α1 helix · residues 1–14" in svg_text
    assert "F10" in svg_text
    assert "R13" in svg_text
    assert "residues 4–16" in svg_text
    assert "Basic" in svg_text
    assert "Acidic" in svg_text
    assert "Net" in svg_text
    assert "K/R/H = +1" in svg_text
    assert "D/E = −1" in svg_text
    assert "descriptive, not causal" in svg_text


def test_n_terminal_pair_plot_derives_accessible_prose_from_plotted_rows(tmp_path: Path) -> None:
    sequences = {
        "wild_type": _canonical("MAAAAAAAAAAAAAAA"),
        "candidate_d01": _canonical("MAAKKKEAAAAAAAAA"),
        "candidate_d02": _canonical("MAADEAAAAAAAAAAA"),
    }
    panel_rows = [
        {"candidate_id": "candidate_d01", "variant_id": "Eco1RT-G3-D01"},
        {"candidate_id": "candidate_d02", "variant_id": "Eco1RT-G3-D02"},
    ]

    row = write_n_terminal_pair_comparison_plot(
        tmp_path,
        panel_rows=panel_rows,
        canonical_sequences_by_id=sequences,
        input_hashes={"foldcheck_input_sequences": "sha256:test"},
    )

    svg_text = Path(str(row["path"])).read_text(encoding="utf-8")
    expected_counts = (
        "WT has 0 basic residues and 0 acidic residues, D01 has 3 basic residues and 1 acidic residue, "
        "and D02 has 0 basic residues and 2 acidic residues"
    )
    assert expected_counts in row["alt_text"]
    assert expected_counts in svg_text
    assert "WT, D01, and D02 N-terminal sequence and charge-proxy comparison" in svg_text
    assert "D01 retains a more basic N-terminal patch than D02" not in svg_text
    assert "WT has 2 basic and 1 acidic residue" not in svg_text


def test_n_terminal_pair_plot_uses_measured_signal_visual_contract(tmp_path: Path) -> None:
    sequences = {
        "wild_type": _canonical("MKSAEYLNTFRLRNLG"),
        "candidate_d01": _canonical("MKSASVLRNEKLKKKG"),
        "candidate_d02": _canonical("MKAREKEREERLREMG"),
    }
    panel_rows = [
        {"candidate_id": "candidate_d01", "variant_id": "Eco1RT-G3-D01"},
        {"candidate_id": "candidate_d02", "variant_id": "Eco1RT-G3-D02"},
    ]

    row = write_n_terminal_pair_comparison_plot(
        tmp_path,
        panel_rows=panel_rows,
        canonical_sequences_by_id=sequences,
        input_hashes={"foldcheck_input_sequences": "sha256:test"},
    )

    svg_text = Path(str(row["path"])).read_text(encoding="utf-8")
    normalized = svg_text.lower()
    for color in ("#34495e", "#008c95", "#b96b72", "#eaf3f5"):
        assert color in normalized
    assert "font-family: 'arial'" in normalized
    assert "dejavu sans" not in normalized
    assert "fill-opacity" not in normalized
    assert "Charge proxy · residues 4–16" in svg_text
    assert "Outlined cells = substitutions from WT" in svg_text
    assert "Letter color:" not in svg_text


def test_n_terminal_pair_plot_is_review_only_context() -> None:
    metadata = SELECTION_PLOT_METADATA["selection_distal_pair_n_terminal_comparison"]

    assert metadata["role"] == "review_only"
    assert metadata["notebook_group"] == "context_checks"
    assert "does not select rows" in metadata["not_a_selector_reason"]


def _canonical(prefix: str) -> str:
    return prefix + "A" * (320 - len(prefix))
