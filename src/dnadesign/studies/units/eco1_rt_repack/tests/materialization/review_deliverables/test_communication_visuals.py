"""Scientific-communication plot contracts for the Eco1 review notebook."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.constraint_map import (  # noqa: E501
    _motif_anchor_segments,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.structural_screen import (  # noqa: E501
    _assert_marginal_axes_aligned,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    load_review_manifest,
    review_lane_lookup,
    visual_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_visuals import (
    render_video,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)

_ALWAYS_AVAILABLE_VISUAL_IDS = {
    "communication_design_space_map",
    "communication_structural_screen",
    "communication_selected_panel",
}


def test_communication_visuals_are_additive_file_backed_notebook_options(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    _manifest, deliverables, _manifest_path, manifest_root = load_review_manifest(str(result.notebook_path))

    assert review_lane_lookup(deliverables)["Communication visuals"] == "communication"
    communication_rows = visual_deliverables(deliverables, selected_lane="communication")
    communication_ids = {str(row["deliverable_id"]) for row in communication_rows}
    assert _ALWAYS_AVAILABLE_VISUAL_IDS.issubset(communication_ids)
    assert "communication_electrostatic_surface" not in communication_ids
    assert "communication_structure_story_movie" not in communication_ids
    assert "communication_candidate_cycle_movie" not in communication_ids
    assert all("slide" not in deliverable_id for deliverable_id in communication_ids)
    for row in communication_rows:
        assert row["role"] == "communication_facing"
        assert (manifest_root / str(row["path"])).exists(), row["deliverable_id"]

    core_ids = {str(row["deliverable_id"]) for row in visual_deliverables(deliverables)}
    assert "selection_hypothesis_panel_flow" in core_ids
    assert "selection_mutation_set_dissimilarity" in core_ids
    assert "communication_structure_story_browser" not in core_ids


def test_communication_svg_outputs_are_accessible_and_use_declared_aspect_ratios(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["deliverable_id"]): row for row in manifest["deliverables"]}
    expected_aspect_ranges = {
        "communication_design_space_map": (2.20, float("inf")),
        "communication_structural_screen": (0.90, 1.15),
        "communication_selected_panel": (2.00, float("inf")),
    }
    for deliverable_id, (minimum_aspect, maximum_aspect) in expected_aspect_ranges.items():
        path = result.manifest_path.parent / str(rows[deliverable_id]["path"])
        root = ElementTree.parse(path).getroot()
        view_box = [float(value) for value in str(root.attrib["viewBox"]).split()]
        aspect_ratio = view_box[2] / view_box[3]
        assert minimum_aspect <= aspect_ratio <= maximum_aspect, deliverable_id
        svg_text = path.read_text(encoding="utf-8")
        assert "<title" in svg_text
        assert "<desc" in svg_text
        assert 'role="img"' in svg_text


def test_communication_design_space_map_names_motifs_sources_and_threshold(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["deliverables"] if item["deliverable_id"] == "communication_design_space_map")
    svg_text = (result.manifest_path.parent / row["path"]).read_text(encoding="utf-8")

    assert "NAxxH" in svg_text
    assert "YADD" in svg_text
    assert "VTG" in svg_text
    assert "Wang et al." in svg_text
    assert "Inouye et al." in svg_text
    assert "WT is clade-9 plurality at \N{GREATER-THAN OR EQUAL TO}25%" in svg_text
    assert "Open: distal scaffold &gt;10 \N{LATIN CAPITAL LETTER A WITH RING ABOVE}" in svg_text
    assert (
        "Open: peripheral shell &gt;5 to \N{LESS-THAN OR EQUAL TO}10 \N{LATIN CAPITAL LETTER A WITH RING ABOVE}"
    ) in svg_text
    assert "Fixed motif neighborhoods" in svg_text


def test_communication_design_space_map_uses_exact_motif_anchor_spans() -> None:
    segments = dict(
        _motif_anchor_segments(
            [
                {"canonical_position": position, "manual_mask_reason": reason}
                for reason, positions in (
                    ("retron_x_naxxh", range(105, 110)),
                    ("catalytic_yadd", range(195, 199)),
                    ("retron_y_vtg", range(243, 246)),
                )
                for position in positions
            ]
        )
    )

    assert segments == {
        "NAxxH": {105, 106, 107, 108, 109},
        "YADD": {195, 196, 197, 198},
        "VTG": {243, 244, 245},
    }


def test_communication_structural_screen_has_marginal_counts(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["deliverables"] if item["deliverable_id"] == "communication_structural_screen")
    svg_text = (result.manifest_path.parent / row["path"]).read_text(encoding="utf-8")

    assert svg_text.count("Sequence count") >= 2
    assert ("Maximum local C\N{GREEK SMALL LETTER ALPHA} RMSD (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})") in svg_text
    assert "Mean ColabFold pLDDT" in svg_text


def test_communication_selected_panel_keeps_alpha1_burden_in_evidence_not_the_main_figure(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["deliverables"] if item["deliverable_id"] == "communication_selected_panel")
    svg_text = (result.manifest_path.parent / row["path"]).read_text(encoding="utf-8")

    assert "Alpha-1" not in svg_text
    assert "non-R13" not in svg_text
    assert "Shell charge" in svg_text
    assert svg_text.count("ProteinMPNN") >= 4
    assert "Distal policy" in svg_text
    assert "Peripheral policy" in svg_text
    assert "Combined policy" in svg_text
    assert "Jaccard distance" in svg_text


def test_communication_visual_materialization_removes_unmanifested_visuals(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    panel_root = tmp_path / "review_deliverables" / "communication_visuals"
    panel_root.mkdir(parents=True, exist_ok=True)
    stale_svg_path = panel_root / "stale.svg"
    stale_png_path = panel_root / "stale.png"
    stale_svg_path.write_text("<svg/>", encoding="utf-8")
    stale_png_path.write_bytes(b"png")

    materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )

    assert not stale_svg_path.exists()
    assert not stale_png_path.exists()


def test_communication_video_is_centered_and_bounded_for_notebook_review(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mp4"
    movie_path.write_bytes(b"mp4")

    rendered = render_video(
        {"title": "Test movie", "deliverable_id": "test_movie"},
        mo=FakeMo,
        media_path=movie_path,
    )

    video_stack = rendered["items"][0]
    assert video_stack["kind"] == "hstack"
    assert video_stack["kwargs"]["justify"] == "center"
    assert video_stack["items"][0]["kwargs"]["width"] == "min(100%, 960px)"
    assert video_stack["items"][0]["kwargs"]["height"] == "auto"


def test_structural_screen_marginal_axes_share_main_plot_bounds() -> None:
    fig = plt.figure(figsize=(8, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5))
    top_ax = fig.add_subplot(grid[0, 0])
    main_ax = fig.add_subplot(grid[1, 0])
    right_ax = fig.add_subplot(grid[1, 1])
    _assert_marginal_axes_aligned(main_ax=main_ax, top_ax=top_ax, right_ax=right_ax)
    right_ax.set_position([0.8, 0.2, 0.1, 0.3])
    with pytest.raises(RuntimeError, match="marginal axes are misaligned"):
        _assert_marginal_axes_aligned(main_ax=main_ax, top_ax=top_ax, right_ax=right_ax)
    plt.close(fig)
