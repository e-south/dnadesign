"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_sequence_dissimilarity_plot.py

Tests for Junction-owned sequence-comparison plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
import yaml

from dnadesign.junction.contracts import parse_request
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.errors import JunctionConfigError
from dnadesign.junction.presentation.sequence_review import (
    JunctionSequenceDissimilarityV1,
    plot_sequence_dissimilarity,
    render_sequence_dissimilarity_svg,
    sequence_dissimilarity_contracts,
)
from dnadesign.junction.presentation.sequence_review import plot as plot_module
from dnadesign.junction.sequence import (
    levenshtein_distance,
    longest_common_substring_length,
    position_weighted_levenshtein,
)


def _dna_code(value: int, *, length: int) -> str:
    bases = "ACGT"
    return "".join(bases[(value >> (2 * offset)) & 3] for offset in range(length))


def _review(*, junction_count: int = 3) -> JunctionSequenceDissimilarityV1:
    return JunctionSequenceDissimilarityV1.model_validate(
        {
            "contract_kind": "junction_sequence_dissimilarity_v1",
            "source": {
                "plan_schema": "dnadesign.junction.plan.v1",
                "plan_id": f"sha256:{'a' * 64}",
                "request_sha256": f"sha256:{'b' * 64}",
                "algorithm": "dnadesign.junction.string.v1",
            },
            "assembly_group_id": "assembly-a",
            "junctions": [
                {
                    "junction_id": f"target-a:junction-{index + 1:04d}",
                    "target_id": "target-a",
                    "toehold_sequence_5to3": _dna_code(index + 1, length=10),
                    "barcode_sequence_5to3": _dna_code(index + 101, length=22),
                }
                for index in range(junction_count)
            ],
            "thermodynamic_screening": "not_run",
        }
    )


def test_plot_uses_the_exact_junction_string_metrics() -> None:
    review = _review()
    figure = plot_sequence_dissimilarity(review)
    try:
        axes = {axis.get_gid(): axis for axis in figure.axes if axis.get_gid()}
        assert set(axes) == {
            "junction-sequence-dissimilarity:toeholds",
            "junction-sequence-dissimilarity:barcodes",
            "junction-sequence-dissimilarity:combined",
        }
        first, second = review.junctions[:2]
        assert float(
            axes["junction-sequence-dissimilarity:toeholds"].collections[0].get_array()[0, 1]
        ) == pytest.approx(
            position_weighted_levenshtein(
                first.toehold_sequence_5to3,
                second.toehold_sequence_5to3,
            )
        )
        assert int(axes["junction-sequence-dissimilarity:barcodes"].collections[0].get_array()[0, 1]) == (
            levenshtein_distance(first.barcode_sequence_5to3, second.barcode_sequence_5to3)
        )
        assert int(axes["junction-sequence-dissimilarity:combined"].collections[0].get_array()[0, 1]) == (
            longest_common_substring_length(
                first.toehold_sequence_5to3 + first.barcode_sequence_5to3,
                second.toehold_sequence_5to3 + second.barcode_sequence_5to3,
            )
        )
    finally:
        plt.close(figure)


def test_large_group_requires_an_explicit_bounded_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    review = _review(junction_count=25)

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized selections must fail before matrix or figure allocation")

    monkeypatch.setattr(plot_module, "pairwise_matrices", fail_if_allocated)
    monkeypatch.setattr(plot_module, "Figure", fail_if_allocated)
    with pytest.raises(JunctionConfigError, match="has 25 junctions.*choose at most 24"):
        plot_sequence_dissimilarity(review)

    monkeypatch.undo()
    selected = [f"target-a:junction-{index + 1:04d}" for index in range(12)]
    figure = plot_sequence_dissimilarity(review, junction_ids=selected)
    try:
        assert len(figure.axes[0].get_xticklabels()) == 12
        assert "12 of 25 junctions" in figure.texts[1].get_text()
    finally:
        plt.close(figure)


def test_long_sequences_fail_before_pairwise_or_figure_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _review().model_dump(mode="json")
    payload["junctions"] = payload["junctions"][:2]
    for index, junction in enumerate(payload["junctions"]):
        junction["toehold_sequence_5to3"] = "A" * 2_999 + "CG"[index]
        junction["barcode_sequence_5to3"] = "AG"[index]
    review = JunctionSequenceDissimilarityV1.model_validate(payload)

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized pairwise work must fail before allocation")

    monkeypatch.setattr(plot_module, "pairwise_matrices", fail_if_allocated)
    monkeypatch.setattr(plot_module, "Figure", fail_if_allocated)
    with pytest.raises(JunctionConfigError, match="pairwise work requires.*dynamic-programming cells"):
        plot_sequence_dissimilarity(review)


def test_import_and_svg_render_preserve_the_selected_matplotlib_backend() -> None:
    original_backend = matplotlib.get_backend()

    render_sequence_dissimilarity_svg(_review())

    assert matplotlib.get_backend() == original_backend


def test_contract_rejects_thermodynamic_overclaim_and_duplicate_barcodes() -> None:
    overclaim = _review().model_dump(mode="json")
    overclaim["thermodynamic_screening"] = "passed"
    with pytest.raises(ValueError, match="thermodynamic_screening"):
        JunctionSequenceDissimilarityV1.model_validate(overclaim)

    duplicate = copy.deepcopy(_review().model_dump(mode="json"))
    duplicate["junctions"][1]["barcode_sequence_5to3"] = duplicate["junctions"][0]["barcode_sequence_5to3"]
    with pytest.raises(ValueError, match="barcode sequences must be unique"):
        JunctionSequenceDissimilarityV1.model_validate(duplicate)


def test_checked_in_sequence_comparison_matches_the_demo_request() -> None:
    package_root = Path(__file__).resolve().parents[1]
    request_path = package_root / "examples" / "three-fragment-review" / "request.yaml"
    asset_path = package_root / "docs" / "assets" / "sequence-dissimilarity.svg"
    plan = design_junction(parse_request(yaml.safe_load(request_path.read_text(encoding="utf-8"))))
    [review] = sequence_dissimilarity_contracts(plan)

    with matplotlib.rc_context({"font.size": 37, "axes.linewidth": 4.0}):
        first = render_sequence_dissimilarity_svg(review)
    second = render_sequence_dissimilarity_svg(review)

    assert first == second == asset_path.read_bytes()
    assert b"data:image/png" not in first
