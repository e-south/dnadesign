"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_workflow.py

Bundle and show-path tests for released-product snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import pytest
import yaml

import dnadesign.cruncher.snapback.released_target_search as released_target_search
from dnadesign.cruncher.app.snapback_released_show import released_show_payload
from dnadesign.cruncher.app.snapback_released_solve_workflow import run_released_snapback_solve
from dnadesign.cruncher.app.snapback_released_target_search_workflow import run_released_snapback_target_search
from dnadesign.cruncher.app.snapback_released_workflow import (
    run_released_snapback_design,
    validate_released_snapback_spec,
)
from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes
from dnadesign.cruncher.release_enzymes.catalog import load_merged_release_enzyme_catalog
from dnadesign.cruncher.snapback import (
    released_plot_common,
    released_plot_foldback,
    released_plot_precursor,
    released_plot_released,
)
from dnadesign.cruncher.snapback.errors import SnapbackSpecError
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_hit_plot import (
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _SITE_FOOTPRINT_VERTICAL_PAD,
    _site_footprint_bounds,
    build_released_hit_plot_context,
    build_released_hit_plot_model,
    render_released_hit_plot,
)
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedSolveOutputConfig,
    ReleasedTargetSearchConfig,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_plot_models import (
    PlotFoldbackPanelContext,
    PlotFoldbackRow,
    PlotFragmentRow,
    PlotPrecursorPanelContext,
    PlotReleasedProductContext,
    PlotSpan,
)
from dnadesign.cruncher.tests.released_snapback.builders import write_released_workspace


def test_released_hit_plot_site_footprint_bounds_track_the_duplex_band() -> None:
    fill_y0, fill_y1 = _site_footprint_bounds()

    assert fill_y0 == pytest.approx(_ROW_BOTTOM_Y - _SITE_FOOTPRINT_VERTICAL_PAD)
    assert fill_y1 == pytest.approx(_ROW_TOP_Y + _SITE_FOOTPRINT_VERTICAL_PAD)
    assert fill_y0 > (_ROW_BOTTOM_Y - 0.08)
    assert fill_y1 < (_ROW_TOP_Y + 0.08)


def test_draw_sequence_uses_one_short_centered_pastel_fill_for_assignable_bases() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots()
    released_plot_common.draw_sequence(
        ax,
        sequence="AC",
        y=0.5,
        row_label="Row",
        start_terminal=None,
        end_terminal=None,
        assignable_base_positions=[1],
    )
    plt.close(fig)

    assignable_patches = [patch for patch in ax.patches if isinstance(patch, FancyBboxPatch)]
    assert len(assignable_patches) == 1
    patch = assignable_patches[0]
    assert "Round" in type(patch.get_boxstyle()).__name__
    assert patch.get_height() <= 0.06
    assert patch.get_y() + (patch.get_height() / 2.0) == pytest.approx(0.5)


def test_snapback_plot_canvas_is_pure_white() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    released_plot_common.configure_axis(ax, x_min=0, x_max=8, title="Panel")
    fig.patch.set_facecolor(released_plot_common._FIGURE_FACE)
    plt.close(fig)

    assert released_plot_common._FIGURE_FACE == "#FFFFFF"
    assert to_hex(fig.get_facecolor(), keep_alpha=False).upper() == "#FFFFFF"
    assert to_hex(ax.get_facecolor(), keep_alpha=False).upper() == "#FFFFFF"


def test_draw_sequence_keeps_long_row_labels_clear_of_terminal_annotations() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 1.4), dpi=170)
    released_plot_common.configure_axis(ax, x_min=0, x_max=8, title="Foldback")
    released_plot_common.draw_sequence(
        ax,
        sequence="CCTCAG",
        y=0.70,
        row_label="Foldback Stem",
        start_terminal="5'",
        end_terminal=None,
    )
    released_plot_common.draw_sequence(
        ax,
        sequence="GGAGTC",
        y=0.54,
        row_label="Stem",
        start_terminal="3'",
        end_terminal=None,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_by_label = {text.get_text(): text for text in ax.texts}
    foldback_label_box = text_by_label["Foldback Stem"].get_window_extent(renderer=renderer)
    top_terminal_box = text_by_label["5'"].get_window_extent(renderer=renderer)
    stem_label_box = text_by_label["Stem"].get_window_extent(renderer=renderer)
    bottom_terminal_box = text_by_label["3'"].get_window_extent(renderer=renderer)
    plt.close(fig)

    assert foldback_label_box.x1 + 3.0 < top_terminal_box.x0
    assert stem_label_box.x1 + 3.0 < bottom_terminal_box.x0


def test_foldback_panel_draws_tighter_cap_arc_and_top_active_action_arrow_down() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, FancyArrowPatch

    context = PlotFoldbackPanelContext(
        origin_boundary_from_left=0,
        stem_sequence="GAG",
        cap_sequence="AAA",
        foldback_sequence="CTC",
        foldback_partner_sequence="CTC",
        upstream_context_span=PlotSpan(start=0, end=0),
        nicked_strand="bottom",
        top_row=PlotFoldbackRow(
            role="active_stem",
            label="Stem",
            sequence="GAG",
            span=PlotSpan(start=0, end=3),
            left_terminal="5'",
        ),
        bottom_row=PlotFoldbackRow(
            role="foldback_return",
            label="Foldback Stem",
            sequence="CTC",
            span=PlotSpan(start=0, end=3),
            left_terminal="3'",
        ),
        foldback_mismatch_positions=[],
    )

    assert released_plot_foldback._foldback_action_direction(context) == "down"  # noqa: SLF001

    fig, ax = plt.subplots()
    released_plot_foldback.render_foldback_panel(ax, context=context)
    plt.close(fig)

    arcs = [patch for patch in ax.patches if isinstance(patch, Arc)]
    action_arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    assert len(arcs) == 1
    cap_arc = arcs[0]
    cap_left_edge = cap_arc.center[0] - (cap_arc.width / 2.0)
    assert cap_left_edge - released_plot_common.x_for_boundary(3) <= 0.06
    assert len(action_arrows) == 1
    arrow = action_arrows[0]
    action_path = released_plot_foldback._foldback_action_arrow_path(  # noqa: SLF001
        cap_arc_center_x=cap_arc.center[0],
        cap_arc_center_y=cap_arc.center[1],
        arc_width=cap_arc.width,
        arc_height=cap_arc.height,
        direction="down",
    )
    assert action_path.vertices[0][0] - cap_arc.center[0] >= 0.82
    assert action_path.vertices[-1][0] > action_path.vertices[0][0]
    assert action_path.vertices[-1][1] < action_path.vertices[0][1]
    assert arrow.get_mutation_scale() >= 14


def test_foldback_panel_action_arrow_points_up_when_bottom_active_folds_up() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, FancyArrowPatch

    context = PlotFoldbackPanelContext(
        origin_boundary_from_left=0,
        stem_sequence="GAG",
        cap_sequence="AAA",
        foldback_sequence="CTC",
        foldback_partner_sequence="CTC",
        upstream_context_span=PlotSpan(start=0, end=0),
        nicked_strand="top",
        top_row=PlotFoldbackRow(
            role="foldback_return",
            label="Foldback Stem",
            sequence="CTC",
            span=PlotSpan(start=0, end=3),
            left_terminal="5'",
        ),
        bottom_row=PlotFoldbackRow(
            role="active_stem",
            label="Stem",
            sequence="GAG",
            span=PlotSpan(start=0, end=3),
            left_terminal="3'",
        ),
        foldback_mismatch_positions=[],
    )

    assert released_plot_foldback._foldback_action_direction(context) == "up"  # noqa: SLF001

    fig, ax = plt.subplots()
    released_plot_foldback.render_foldback_panel(ax, context=context)
    plt.close(fig)

    [cap_arc] = [patch for patch in ax.patches if isinstance(patch, Arc)]
    [arrow] = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    action_path = released_plot_foldback._foldback_action_arrow_path(  # noqa: SLF001
        cap_arc_center_x=cap_arc.center[0],
        cap_arc_center_y=cap_arc.center[1],
        arc_width=cap_arc.width,
        arc_height=cap_arc.height,
        direction="up",
    )
    assert action_path.vertices[-1][1] > action_path.vertices[0][1]
    assert action_path.vertices[-1][0] > action_path.vertices[0][0]
    assert arrow.get_mutation_scale() >= 14


def test_foldback_color_segments_do_not_count_negative_degenerate_prefix_as_stem() -> None:
    assert released_plot_foldback._foldback_color_segments(  # noqa: SLF001
        role="active_stem",
        span_start=-2,
        span_end=3,
        origin_boundary=0,
    ) == [(0, 3, released_plot_common._STEM)]
    assert released_plot_foldback._foldback_color_segments(  # noqa: SLF001
        role="foldback_return",
        span_start=-2,
        span_end=3,
        origin_boundary=0,
    ) == [(0, 3, released_plot_common._FOLDBACK)]


def test_precursor_panel_hues_inert_non_product_bases_gray(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_segments_by_row: dict[str, list[tuple[int, int, str]]] = {}

    def record_sequence(*_args, row_label: str, color_segments=None, **_kwargs) -> None:
        color_segments_by_row[row_label] = list(color_segments or [])

    monkeypatch.setattr(released_plot_precursor, "draw_sequence", record_sequence)
    context = PlotPrecursorPanelContext(
        top_sequence="AAAACCCCGGGG",
        bottom_sequence=complement_sequence("AAAACCCCGGGG"),
        nick_site={"orientation": "forward"},
        nick_event={},
        nicked_strand="top",
        release_site={"orientation": "forward"},
        release_event={"top_cut_boundary": 8, "bottom_cut_boundary": 10},
        top_span=PlotSpan(start=0, end=12),
        bottom_span=PlotSpan(start=0, end=12),
        nick_boundary=0,
        nick_site_span=PlotSpan(start=0, end=2),
        release_site_span=PlotSpan(start=8, end=10),
        retained_partner_span=PlotSpan(start=0, end=0),
        active_product_span=PlotSpan(start=0, end=8),
        sacrificial_top_tail_span=PlotSpan(start=8, end=12),
        sacrificial_bottom_tail_span=PlotSpan(start=10, end=12),
    )

    fig, ax = plt.subplots()
    released_plot_precursor.render_precursor_panel(
        ax,
        context=context,
        nickase_variant_id="Nt.Test",
        release_variant_id="Re.Test",
    )
    plt.close(fig)

    assert color_segments_by_row["Top"] == [(10, 12, released_plot_common._TAIL)]
    assert color_segments_by_row["Bottom"] == [(10, 12, released_plot_common._TAIL)]


def test_precursor_panel_bolds_canonical_site_orientation_strand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emphasized_segments_by_row: dict[str, list[tuple[int, int]]] = {}
    assignable_positions_by_row: dict[str, list[int]] = {}

    def record_sequence(
        *_args,
        row_label: str,
        assignable_base_positions=None,
        emphasis_segments=None,
        **_kwargs,
    ) -> None:
        assignable_positions_by_row[row_label] = list(assignable_base_positions or [])
        emphasized_segments_by_row[row_label] = list(emphasis_segments or [])

    monkeypatch.setattr(released_plot_precursor, "draw_sequence", record_sequence)
    context = PlotPrecursorPanelContext(
        top_sequence="CCTCAGCCCGCTGA",
        bottom_sequence=complement_sequence("CCTCAGCCCGCTGA"),
        top_assignable_base_positions=[10],
        bottom_assignable_base_positions=[0, 1],
        nick_site={"orientation": "reverse"},
        nick_event={},
        nicked_strand="top",
        release_site={"orientation": "forward"},
        release_event={"top_cut_boundary": 12, "bottom_cut_boundary": 11},
        top_span=PlotSpan(start=0, end=14),
        bottom_span=PlotSpan(start=0, end=14),
        nick_boundary=2,
        nick_site_span=PlotSpan(start=0, end=7),
        release_site_span=PlotSpan(start=10, end=14),
        retained_partner_span=PlotSpan(start=0, end=2),
        active_product_span=PlotSpan(start=2, end=14),
        sacrificial_top_tail_span=PlotSpan(start=14, end=14),
        sacrificial_bottom_tail_span=PlotSpan(start=14, end=14),
    )

    fig, ax = plt.subplots()
    released_plot_precursor.render_precursor_panel(
        ax,
        context=context,
        nickase_variant_id="Nt.Bpu10I",
        release_variant_id="BsaI-HFv2",
    )
    plt.close(fig)

    assert emphasized_segments_by_row["Top"] == [(10, 14)]
    assert emphasized_segments_by_row["Bottom"] == [(0, 7)]
    assert assignable_positions_by_row["Top"] == [10]
    assert assignable_positions_by_row["Bottom"] == [0, 1]


def test_bstnbi_reverse_complemented_precursor_bolds_bottom_canonical_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True)
    report = run_released_snapback_target_search(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
            release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
            search=ReleasedTargetSearchConfig(
                max_results=16,
                near_boundary_search_limit=8,
                allow_precut_footprint_outside_active_product=True,
                allowed_active_strands=["top", "bottom"],
                allowed_route_families=["bottom_active_from_top_nick", "top_active_from_bottom_nick"],
            ),
        ),
        workspace_root=workspace_root,
    )
    assert report.exact_hits
    emphasized_segments_by_row: dict[str, list[tuple[int, int]]] = {}

    def record_sequence(*_args, row_label: str, emphasis_segments=None, **_kwargs) -> None:
        emphasized_segments_by_row[row_label] = list(emphasis_segments or [])

    monkeypatch.setattr(released_plot_precursor, "draw_sequence", record_sequence)

    checked_reverse_bstnbi = False
    for hit in report.exact_hits:
        context = build_released_hit_plot_model(hit).precursor
        emphasized_segments_by_row.clear()

        fig, ax = plt.subplots()
        released_plot_precursor.render_precursor_panel(
            ax,
            context=context,
            nickase_variant_id=hit.nickase_variant_id,
            release_variant_id=hit.release_variant_id,
        )
        plt.close(fig)

        expected_top: list[tuple[int, int]] = []
        expected_bottom: list[tuple[int, int]] = []
        for site, span in (
            (context.nick_site, context.nick_site_span),
            (context.release_site, context.release_site_span),
        ):
            expected = (span.start, span.end)
            if site["orientation"] == "forward":
                expected_top.append(expected)
            else:
                expected_bottom.append(expected)

        assert emphasized_segments_by_row["Top"] == expected_top
        assert emphasized_segments_by_row["Bottom"] == expected_bottom
        if hit.nickase_variant_id == "Nt.BstNBI":
            checked_reverse_bstnbi = True
            assert context.nick_site["orientation"] == "reverse"
            assert (context.nick_site_span.start, context.nick_site_span.end) in expected_bottom

    assert checked_reverse_bstnbi is True


def test_precursor_panel_moves_overlapping_top_nick_site_label_below(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    placements: dict[str, str] = {}

    def record_site_footprint(*_args, label: str, label_placement: str = "above", **_kwargs) -> None:
        placements[label] = label_placement

    monkeypatch.setattr(released_plot_precursor, "draw_site_footprint", record_site_footprint)
    context = PlotPrecursorPanelContext(
        top_sequence="CCTCAGCCCGCTGA",
        bottom_sequence=complement_sequence("CCTCAGCCCGCTGA"),
        nick_site={"orientation": "forward"},
        nick_event={},
        nicked_strand="top",
        release_site={"orientation": "forward"},
        release_event={"top_cut_boundary": 12, "bottom_cut_boundary": 11},
        top_span=PlotSpan(start=0, end=14),
        bottom_span=PlotSpan(start=0, end=14),
        nick_boundary=2,
        nick_site_span=PlotSpan(start=0, end=7),
        release_site_span=PlotSpan(start=10, end=14),
        retained_partner_span=PlotSpan(start=0, end=2),
        active_product_span=PlotSpan(start=2, end=14),
        sacrificial_top_tail_span=PlotSpan(start=14, end=14),
        sacrificial_bottom_tail_span=PlotSpan(start=14, end=14),
    )

    fig, ax = plt.subplots()
    released_plot_precursor.render_precursor_panel(
        ax,
        context=context,
        nickase_variant_id="Nt.Bpu10I",
        release_variant_id="BsaI-HFv2",
    )
    plt.close(fig)

    assert placements == {"Nt.Bpu10I": "below", "BsaI-HFv2": "above"}


def test_released_panel_omits_release_cut_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boundary_labels: list[str] = []
    span_labels: list[str] = []
    region_label_y: dict[str, float] = {}

    def record_boundary(*_args, label: str, **_kwargs) -> None:
        boundary_labels.append(label)

    def record_span(*_args, label: str, **_kwargs) -> None:
        span_labels.append(label)

    def record_region(*_args, label: str, y: float, **_kwargs) -> None:
        region_label_y[label] = y

    monkeypatch.setattr(released_plot_released, "draw_strand_boundary", record_boundary)
    monkeypatch.setattr(released_plot_released, "draw_region_label", record_region)
    monkeypatch.setattr(released_plot_released, "draw_span", record_span, raising=False)
    context = PlotReleasedProductContext(
        retained_partner_sequence="CC",
        active_product_sequence="GGATTCGTAAT",
        nick_boundary=2,
        release_top_cut_boundary=1,
        release_bottom_cut_boundary=5,
        upstream_context_span=PlotSpan(start=0, end=2),
        retained_partner_span=PlotSpan(start=0, end=2),
        active_product_span=PlotSpan(start=0, end=11),
        nicked_strand="top",
        top_row=PlotFragmentRow(
            role="retained_partner",
            physical_state="released",
            strand="top",
            label="Top",
            sequence="CC",
            span=PlotSpan(start=0, end=2),
            start_terminal="5'",
            end_terminal="3'",
        ),
        bottom_row=PlotFragmentRow(
            role="active_product",
            physical_state="retained",
            strand="bottom",
            label="Exposed Bottom",
            sequence="GGATTCGTAAT",
            span=PlotSpan(start=0, end=11),
            start_terminal="3'",
            end_terminal="5'",
        ),
        duplex_overlap_span=PlotSpan(start=0, end=2),
        duplex_top_sequence="CC",
        duplex_bottom_sequence="GG",
        duplex_mismatch_positions=[],
        bottom_only_overhang_span=PlotSpan(start=2, end=11),
        active_prefix_span=PlotSpan(start=0, end=2),
        stem_span=PlotSpan(start=2, end=5),
        cap_span=PlotSpan(start=5, end=8),
        foldback_span=PlotSpan(start=8, end=11),
        nickase_site_survives_post_release=False,
        release_site_survives_post_release=False,
    )

    fig, ax = plt.subplots()
    released_plot_released.render_released_panel(ax, context=context)
    plt.close(fig)

    assert boundary_labels == ["Nick"]
    assert span_labels == []
    assert region_label_y["Stem"] > region_label_y["Cap"]


def test_released_design_writes_bundle_and_released_show_revalidates_it(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, report = run_released_snapback_design(fixture.spec_path)

    assert report.status == "satisfied"
    assert (run_dir / "meta" / "released_snapback_manifest.json").exists()
    assert (run_dir / "meta" / "released_snapback_status.json").exists()
    assert (run_dir / "analysis" / "report.json").exists()
    assert (run_dir / "analysis" / "released_product_projection.json").exists()
    assert (run_dir / "analysis" / "pre_nick_site.json").exists()
    assert (run_dir / "analysis" / "release_site.json").exists()
    assert (run_dir / "export" / "released_design_summary.csv").exists()

    payload = released_show_payload(run_dir)

    assert payload["kind"] == "released_explicit"
    assert payload["status"] == "satisfied"
    projection_payload = json.loads(
        (run_dir / "analysis" / "released_product_projection.json").read_text(encoding="utf-8")
    )
    assert projection_payload["release_top_cut_precursor"] == 10
    assert projection_payload["release_bottom_cut_precursor"] == 9


def test_released_design_rejects_left_of_origin_outside_site_exact_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "de033"
    spec_path = workspace / "configs" / "snapback" / "de033.released.snapback.yaml"
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.BsmAI",
                            "specificity_id": "BsmAI",
                            "motif_top_5to3": "GTCTC",
                            "top_cut_offset": 6,
                            "selection": {"outside_site": True},
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "released_snapback": {
                    "schema_version": 1,
                    "kind": "single_nick_released_snapback_v1",
                    "name": "de033_left_prefix",
                },
                "input": {
                    "precursor_top_strand": "GTCTCAAACGTTGTTCCAA",
                },
                "nick_stage": {
                    "nickase_variant_id": "Nt.BsmAI",
                    "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                    "intended_site_sequence": "GTCTC",
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "intended_site_sequence": "CCAA",
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                },
                "final_target": {
                    "nick_boundary_from_left": 0,
                    "paired_bp": 3,
                    "cap_nt": 3,
                },
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_release_site_downstream_of_nick": True,
                    "require_complete_downstream_fragment_separation": True,
                },
                "output": {"run_dir": "outputs/released_design"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir, report = run_released_snapback_design(spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in report.issues)
    payload = released_show_payload(run_dir)
    assert payload["status"] == "invalid_precursor"


def test_released_solve_materializes_hits_and_emits_per_hit_plots(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    workspace_root = fixture.workspace_root

    run_dir, report = run_released_snapback_solve(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
            release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
            search=ReleasedTargetSearchConfig(max_results=2, near_boundary_search_limit=2),
        ),
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=2,
            render_format="pdf",
            emit_renders=True,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert report.status == "exact_hits_materialized"
    assert report.metadata.materialized_hit_count == 1
    assert report.metadata.selected_hit_kind == "exact"
    assert report.metadata.evaluated_pair_count > 0
    assert report.issues == []
    assert (run_dir / "meta" / "released_solve_manifest.json").exists()
    assert (run_dir / "meta" / "released_solve_status.json").exists()
    assert (run_dir / "analysis" / "solve_report.json").exists()
    assert (run_dir / "export" / "table__hits.csv").exists()
    for hit in report.hits:
        hit_run_dir = workspace_root / hit.materialized_run_dir
        assert hit.render_job_path is None
        assert hit.rendered_plot_path is not None
        assert hit_run_dir.exists()
        assert (workspace_root / hit.rendered_plot_path).exists()
        assert (workspace_root / hit.rendered_plot_path).read_bytes().startswith(b"%PDF")
        assert (hit_run_dir / "analysis" / "target_search_hit.json").exists()
        assert (hit_run_dir / "analysis" / "released_hit_plot_context.json").exists()
        assert (hit_run_dir / "analysis" / "released_product_projection.json").exists()
        assert (hit_run_dir / "analysis" / "pre_nick_site.json").exists()
        assert (hit_run_dir / "analysis" / "release_site.json").exists()
    first_context = json.loads(
        (
            workspace_root / report.hits[0].materialized_run_dir / "analysis" / "released_hit_plot_context.json"
        ).read_text(encoding="utf-8")
    )
    assert first_context["foldback"]["foldback_sequence"] == "CAA"
    assert first_context["foldback"]["foldback_partner_sequence"] == "AAC"


def test_released_design_rejects_frequent_cutter_nickase_by_default(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    nick_payload = yaml.safe_load(fixture.nick_catalog_path.read_text(encoding="utf-8"))
    nick_payload["nickases"]["entries"][0]["selection"] = {"warning_codes": ["FREQUENT_CUTTER"]}
    fixture.nick_catalog_path.write_text(yaml.safe_dump(nick_payload, sort_keys=False), encoding="utf-8")

    report = validate_released_snapback_spec(fixture.spec_path)

    assert report.status == "invalid_catalog"
    assert report.issues[0].code == "DISALLOWED_NICKASE_WARNING_CODE"
    assert report.metadata.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]


def test_released_design_rejects_unknown_nick_stage_key(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    spec_payload = yaml.safe_load(fixture.spec_path.read_text(encoding="utf-8"))
    spec_payload["nick_stage"]["unexpected_unknown_key"] = True
    fixture.spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(SnapbackSpecError, match="unexpected_unknown_key"):
        validate_released_snapback_spec(fixture.spec_path)


def test_released_design_fails_closed_on_ambiguous_precursor_origin(tmp_path: Path) -> None:
    fixture = write_released_workspace(
        tmp_path,
        precursor_top_strand="AACGTTGAACGTTGTTCCAA",
    )

    report = validate_released_snapback_spec(fixture.spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRECURSOR_ORIGIN_AMBIGUOUS" for issue in report.issues)
    assert report.projection is None
    assert report.candidate is None


def test_checked_in_de033_released_design_fixture_stays_invalid() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    spec_path = (
        repo_root
        / "src"
        / "dnadesign"
        / "cruncher"
        / "workspaces"
        / "de033"
        / "configs"
        / "snapback"
        / "de033.released.snapback.yaml"
    )

    report = validate_released_snapback_spec(spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in report.issues)


def test_released_hit_plot_context_marks_degenerate_assignments_without_losing_resolved_bases(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)
    search_report = run_released_snapback_target_search(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
            release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
            search=ReleasedTargetSearchConfig(
                max_results=16,
                near_boundary_search_limit=8,
                allow_precut_footprint_outside_active_product=True,
                allowed_active_strands=["top", "bottom"],
                allowed_route_families=["bottom_active_from_top_nick", "top_active_from_bottom_nick"],
            ),
        ),
        workspace_root=workspace_root,
    )
    hit = next(hit for hit in search_report.exact_hits if hit.nickase_variant_id == "Nt.BstNBI")
    active_row_key = "top_row" if hit.active_strand == "top" else "bottom_row"
    degenerate_active_indexes = sorted(
        base.active_index
        for base in hit.projection.active_product_provenance
        if base.source_constraint == "degenerate_motif_base"
    )

    plot_context = build_released_hit_plot_context(hit)

    assert degenerate_active_indexes
    assert plot_context["precursor"]["top_sequence"].startswith("NNNNNGACTC")
    assert plot_context["precursor"]["bottom_sequence"].startswith("NNNNNCTGAG")
    assert plot_context["precursor"]["top_assignable_base_positions"][:5] == [0, 1, 2, 3, 4]
    assert plot_context["precursor"]["bottom_assignable_base_positions"][:5] == [0, 1, 2, 3, 4]
    active_row = plot_context["released_product"][active_row_key]
    assert active_row["sequence"].startswith("N")
    assert "N" not in active_row["sequence"][1:]
    assert -1 in active_row["assignable_base_positions"]
    assert set(degenerate_active_indexes).issubset(set(active_row["assignable_base_positions"]))
    assert plot_context["foldback"]["top_row"]["span"]["start"] < 0
    assert plot_context["foldback"]["bottom_row"]["span"]["start"] < 0
    assert plot_context["foldback"]["top_row"]["sequence"].startswith("N")
    assert plot_context["foldback"]["bottom_row"]["sequence"].startswith("N")
    assert "N" not in plot_context["foldback"]["cap_sequence"]
    assert (
        plot_context["foldback"]["top_row"]["assignable_base_positions"]
        or plot_context["foldback"]["bottom_row"]["assignable_base_positions"]
        or plot_context["foldback"]["assignable_cap_base_positions"]
    )


def test_released_solve_real_presets_materializes_exact_hits_with_bottom_strand_context(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
        search=ReleasedTargetSearchConfig(max_results=8, near_boundary_search_limit=8),
    )

    search_report = run_released_snapback_target_search(
        request=request,
        workspace_root=workspace_root,
    )
    nick_catalog, _nick_paths = load_merged_nickase_catalog(
        preset_id="neb_nicking_v1",
        additional_preset_ids=["thermo_nicking_v1"],
        additional_paths=[],
        workspace_root=workspace_root,
    )
    release_catalog, _release_paths = load_merged_release_enzyme_catalog(
        preset_id="type_iis_release_v1",
        additional_paths=[],
        workspace_root=workspace_root,
    )
    disallowed_nick_placement_count = len(
        [
            placement
            for placement in released_target_search._nick_placements(
                nick_catalog,
                physical_nicked_strand="top",
            )
            if matching_nickase_warning_codes(
                placement.entry,
                warning_codes=request.search.disallowed_nickase_warning_codes,
            )
        ]
    )
    release_placement_count = len(
        released_target_search._release_placements(
            release_catalog,
            target=request.target,
        )
    )
    assert search_report.status == "exact_hits_found"
    assert search_report.metadata.pre_truncation_exact_hit_count == 2
    assert search_report.metadata.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]
    assert (
        search_report.metadata.blocker_counts["DISALLOWED_NICKASE_WARNING_CODE"]
        == disallowed_nick_placement_count * release_placement_count
    )
    assert {hit.nickase_variant_id for hit in search_report.exact_hits} == {"Nb.BsrDI", "Nb.BtsI"}

    run_dir, solve_report = run_released_snapback_solve(
        request=request,
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=8,
            render_format="pdf",
            emit_renders=False,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert run_dir.exists()
    assert solve_report.status == "exact_hits_materialized"
    assert solve_report.metadata.materialized_hit_count == len(search_report.exact_hits)
    assert solve_report.metadata.available_exact_hit_count == 2
    assert solve_report.metadata.selected_hit_kind == "exact"
    assert solve_report.hits[0].nickase_variant_id != "Nt.BspQI"
    assert solve_report.hits[0].release_variant_id == "BsaI-HFv2"
    assert solve_report.hits[0].target_search_hit.sacrificial_downstream_tail_nt == 7
    assert solve_report.hits[0].target_search_hit.upstream_retained_duplex_bp == 0
    assert solve_report.hits[0].target_search_hit.effective_stem_bp == 3
    plot_context = build_released_hit_plot_context(solve_report.hits[0].target_search_hit)
    assert plot_context["precursor"]["nick_site"]["local_start"] >= 0
    assert plot_context["precursor"]["nick_site"]["local_end"] >= 0
    assert (
        plot_context["precursor"]["nick_boundary"]
        == solve_report.hits[0].target_search_hit.pre_nick_event.boundary_context
    )
    assert plot_context["precursor"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert plot_context["released_product"]["retained_partner_span"]["start"] >= 0
    assert plot_context["released_product"]["active_product_span"]["start"] >= 0
    assert (
        plot_context["released_product"]["nicked_strand"]
        == solve_report.hits[0].target_search_hit.physical_nicked_strand
    )
    assert plot_context["released_product"]["duplex_overlap_span"] is None
    assert plot_context["released_product"]["duplex_top_sequence"] == ""
    assert plot_context["released_product"]["duplex_bottom_sequence"] == ""
    assert plot_context["released_product"]["duplex_mismatch_positions"] == []
    assert plot_context["released_product"]["top_only_overhang_span"] is None
    assert plot_context["released_product"]["bottom_only_overhang_span"] == {
        "start": 0,
        "end": plot_context["released_product"]["bottom_row"]["span"]["end"],
    }
    assert plot_context["foldback"]["foldback_mismatch_positions"] == []
    assert plot_context["foldback"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert (
        plot_context["labels"]["orientation_note"]
        == "Rows stay on physical top/bottom lanes; foldback includes retained upstream duplex before the nick."
    )
    assert plot_context["labels"]["active_start_terminal"] == "3'"
    assert plot_context["labels"]["active_end_terminal"] == "5'"
    assert plot_context["released_product"]["top_row"]["role"] == "retained_partner"
    assert plot_context["released_product"]["top_row"]["strand"] == "top"
    assert plot_context["released_product"]["top_row"]["physical_state"] == "released"
    assert plot_context["released_product"]["top_row"]["label"] == "Exposed Top"
    assert plot_context["released_product"]["top_row"]["sequence"] == ""
    assert plot_context["released_product"]["top_row"]["start_terminal"] == "5'"
    assert plot_context["released_product"]["top_row"]["end_terminal"] == "3'"
    assert plot_context["released_product"]["bottom_row"]["role"] == "active_product"
    assert plot_context["released_product"]["bottom_row"]["strand"] == "bottom"
    assert plot_context["released_product"]["bottom_row"]["physical_state"] == "retained"
    assert plot_context["released_product"]["bottom_row"]["label"] == "Retained Bottom"
    assert plot_context["released_product"]["bottom_row"]["sequence"] in {"GTAACGTAC", "GTGACGCAC"}
    assert plot_context["precursor"]["retained_partner_span"] == {
        "start": 0,
        "end": plot_context["precursor"]["nick_boundary"],
    }
    assert plot_context["precursor"]["active_product_span"] == {
        "start": (
            solve_report.hits[0].target_search_hit.projection.nick_coordinate_precursor
            - solve_report.hits[0].target_search_hit.projection.rebased_nick_boundary
        ),
        "end": (
            solve_report.hits[0].target_search_hit.projection.nick_coordinate_precursor
            - solve_report.hits[0].target_search_hit.projection.rebased_nick_boundary
            + solve_report.hits[0].target_search_hit.projection.active_product_length_nt
        ),
    }
    assert plot_context["foldback"]["top_row"]["role"] == "foldback_return"
    assert plot_context["foldback"]["bottom_row"]["role"] == "active_stem"
    assert plot_context["foldback"]["origin_boundary_from_left"] == 0
    assert plot_context["foldback"]["upstream_context_span"] == {"start": 0, "end": 0}
    assert plot_context["foldback"]["top_row"]["sequence"] in {"CAT", "CAC"}
    assert plot_context["foldback"]["bottom_row"]["sequence"] in {"GTA", "GTG"}
    first_context = json.loads(
        (
            workspace_root / solve_report.hits[0].materialized_run_dir / "analysis" / "released_hit_plot_context.json"
        ).read_text(encoding="utf-8")
    )
    assert first_context["precursor"]["nick_site"]["local_start"] >= 0
    assert first_context["precursor"]["nick_site"]["local_end"] >= 0
    assert (
        first_context["precursor"]["nick_boundary"]
        == solve_report.hits[0].target_search_hit.pre_nick_event.boundary_context
    )
    assert first_context["precursor"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert first_context["released_product"]["retained_partner_span"]["start"] >= 0
    assert first_context["released_product"]["active_product_span"]["start"] >= 0
    assert first_context["released_product"]["duplex_overlap_span"] is None
    assert first_context["released_product"]["duplex_top_sequence"] == ""
    assert first_context["released_product"]["duplex_bottom_sequence"] == ""
    assert first_context["released_product"]["duplex_mismatch_positions"] == []
    assert first_context["released_product"]["top_only_overhang_span"] is None
    assert first_context["released_product"]["bottom_only_overhang_span"] == {
        "start": 0,
        "end": first_context["released_product"]["bottom_row"]["span"]["end"],
    }
    assert first_context["released_product"]["nick_boundary"] >= 0
    assert (
        first_context["released_product"]["nicked_strand"]
        == solve_report.hits[0].target_search_hit.physical_nicked_strand
    )
    assert (
        first_context["released_product"]["retained_partner_span"]["end"]
        == first_context["released_product"]["nick_boundary"]
    )
    assert first_context["released_product"]["nickase_site_survives_post_release"] is False
    assert first_context["labels"]["active_start_terminal"] == "3'"
    assert first_context["labels"]["active_end_terminal"] == "5'"
    assert first_context["released_product"]["top_row"]["role"] == "retained_partner"
    assert first_context["released_product"]["top_row"]["physical_state"] == "released"
    assert first_context["released_product"]["top_row"]["label"] == "Exposed Top"
    assert first_context["released_product"]["top_row"]["sequence"] == ""
    assert first_context["released_product"]["bottom_row"]["role"] == "active_product"
    assert first_context["released_product"]["bottom_row"]["physical_state"] == "retained"
    assert first_context["released_product"]["bottom_row"]["label"] == "Retained Bottom"
    assert first_context["released_product"]["bottom_row"]["sequence"] in {"GTAACGTAC", "GTGACGCAC"}
    assert first_context["precursor"]["retained_partner_span"] == {
        "start": 0,
        "end": first_context["precursor"]["nick_boundary"],
    }
    assert first_context["foldback"]["top_row"]["role"] == "foldback_return"
    assert first_context["foldback"]["bottom_row"]["role"] == "active_stem"
    assert first_context["foldback"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert first_context["foldback"]["origin_boundary_from_left"] == 0
    assert first_context["foldback"]["upstream_context_span"] == {"start": 0, "end": 0}
    assert first_context["foldback"]["top_row"]["sequence"] in {"CAT", "CAC"}
    assert first_context["foldback"]["bottom_row"]["sequence"] in {"GTA", "GTG"}


def test_released_solve_real_presets_materializes_retained_active_hits_with_route_metadata(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
        search=ReleasedTargetSearchConfig(
            max_results=16,
            near_boundary_search_limit=8,
            allow_precut_footprint_outside_active_product=True,
            allowed_active_strands=["top", "bottom"],
            allowed_route_families=["bottom_active_from_top_nick", "top_active_from_bottom_nick"],
        ),
    )

    search_report = run_released_snapback_target_search(
        request=request,
        workspace_root=workspace_root,
    )

    assert search_report.status == "exact_hits_found"
    assert search_report.metadata.route_policy_final_geometry_source == "retained_active_strand"
    exact_hits_by_id = {hit.nickase_variant_id: hit for hit in search_report.exact_hits}
    assert {"Nt.BsmAI", "Nt.BstNBI", "Nt.AlwI", "Nb.BsrDI", "Nb.BtsI"}.issubset(exact_hits_by_id)
    bsrdi_context = build_released_hit_plot_context(exact_hits_by_id["Nb.BsrDI"])
    assert bsrdi_context["precursor"]["top_sequence"].startswith("NN")
    assert bsrdi_context["precursor"]["bottom_sequence"].startswith("NN")
    assert bsrdi_context["released_product"]["top_row"]["sequence"].startswith("NN")
    assert bsrdi_context["released_product"]["bottom_row"]["sequence"].startswith("NN")
    assert bsrdi_context["released_product"]["duplex_top_sequence"] == "NN"
    assert bsrdi_context["released_product"]["duplex_bottom_sequence"] == "NN"
    assert bsrdi_context["released_product"]["duplex_mismatch_positions"] == []
    assert {-2, -1}.issubset(set(bsrdi_context["released_product"]["top_row"]["assignable_base_positions"]))
    assert {-2, -1}.issubset(set(bsrdi_context["released_product"]["bottom_row"]["assignable_base_positions"]))
    assert bsrdi_context["foldback"]["origin_boundary_from_left"] == 0
    assert bsrdi_context["foldback"]["upstream_context_span"] == {"start": -2, "end": 0}
    assert bsrdi_context["foldback"]["top_row"]["sequence"] == "NNCAT"
    assert bsrdi_context["foldback"]["bottom_row"]["sequence"] == "NNGTA"
    assert bsrdi_context["foldback"]["top_row"]["span"] == {"start": -2, "end": 3}
    assert bsrdi_context["foldback"]["bottom_row"]["span"] == {"start": -2, "end": 3}
    assert {-2, -1}.issubset(set(bsrdi_context["foldback"]["top_row"]["assignable_base_positions"]))
    assert {-2, -1}.issubset(set(bsrdi_context["foldback"]["bottom_row"]["assignable_base_positions"]))

    run_dir, solve_report = run_released_snapback_solve(
        request=request,
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=16,
            render_format="pdf",
            emit_renders=False,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert run_dir.exists()
    assert solve_report.status == "exact_hits_materialized"
    assert solve_report.metadata.route_policy_final_geometry_source == "retained_active_strand"
    assert solve_report.metadata.allowed_active_strands == ["top", "bottom"]
    assert solve_report.metadata.allowed_route_families == [
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    ]
    assert any(hit.target_search_hit.active_strand == "top" for hit in solve_report.hits)
    top_active_hits = [hit for hit in solve_report.hits if hit.target_search_hit.active_strand == "top"]
    assert any(hit.nickase_variant_id == "Nt.BstNBI" for hit in top_active_hits)
    top_active_overhang_hit = next(
        hit
        for hit in top_active_hits
        if hit.target_search_hit.projection.active_product_length_nt
        > hit.target_search_hit.projection.retained_partner_length_nt
    )
    assert any(
        base.source_constraint == "degenerate_motif_base"
        for hit in top_active_hits
        for base in hit.target_search_hit.projection.active_product_provenance
    )
    top_active_context = build_released_hit_plot_context(top_active_overhang_hit.target_search_hit)
    top_active_coordinate_offset = (
        top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor
        - top_active_overhang_hit.target_search_hit.projection.rebased_nick_boundary
    )
    assert (
        top_active_context["precursor"]["nick_boundary"]
        == top_active_overhang_hit.target_search_hit.pre_nick_event.boundary_context
    )
    assert (
        top_active_context["precursor"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert top_active_context["labels"]["active_start_terminal"] == "5'"
    assert top_active_context["labels"]["active_end_terminal"] == "3'"
    assert top_active_context["precursor"]["retained_partner_span"] == {
        "start": 0,
        "end": top_active_context["precursor"]["nick_boundary"],
    }
    assert top_active_context["precursor"]["active_product_span"] == {
        "start": (
            top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor
            - top_active_overhang_hit.target_search_hit.projection.rebased_nick_boundary
        ),
        "end": (
            top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor
            - top_active_overhang_hit.target_search_hit.projection.rebased_nick_boundary
            + top_active_overhang_hit.target_search_hit.projection.active_product_length_nt
        ),
    }
    assert top_active_context["released_product"]["top_row"]["role"] == "active_product"
    assert top_active_context["released_product"]["top_row"]["strand"] == "top"
    assert top_active_context["released_product"]["top_row"]["physical_state"] == "retained"
    assert (
        top_active_context["released_product"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert top_active_context["released_product"]["bottom_row"]["role"] == "retained_partner"
    assert top_active_context["released_product"]["bottom_row"]["strand"] == "bottom"
    assert top_active_context["released_product"]["bottom_row"]["physical_state"] == "released"
    assert (
        top_active_context["released_product"]["bottom_row"]["sequence"]
        == top_active_context["precursor"]["bottom_sequence"][:top_active_coordinate_offset]
    )
    assert top_active_context["released_product"]["top_row"]["label"] == "Retained Top"
    assert top_active_context["released_product"]["bottom_row"]["label"] == "Exposed Bottom"
    assert top_active_context["released_product"]["bottom_row"]["span"] == {
        "start": (-top_active_coordinate_offset),
        "end": (
            top_active_overhang_hit.target_search_hit.projection.retained_partner_length_nt
            - top_active_coordinate_offset
        ),
    }
    assert top_active_context["released_product"]["bottom_row"]["start_terminal"] == "3'"
    assert top_active_context["released_product"]["bottom_row"]["end_terminal"] == "5'"
    assert top_active_context["released_product"]["top_row"]["sequence"] == (
        top_active_context["precursor"]["top_sequence"][:top_active_coordinate_offset]
        + top_active_overhang_hit.target_search_hit.projection.active_product_sequence
    )
    assert top_active_context["released_product"]["top_row"]["span"] == {
        "start": -top_active_coordinate_offset,
        "end": top_active_overhang_hit.target_search_hit.projection.active_product_length_nt,
    }
    assert top_active_context["released_product"]["top_only_overhang_span"] == {
        "start": 0,
        "end": top_active_context["released_product"]["top_row"]["span"]["end"],
    }
    assert top_active_context["released_product"]["bottom_only_overhang_span"] is None
    assert top_active_context["released_product"]["duplex_overlap_span"] == {
        "start": -top_active_coordinate_offset,
        "end": 0,
    }
    assert (
        top_active_context["released_product"]["duplex_top_sequence"]
        == top_active_context["precursor"]["top_sequence"][:top_active_coordinate_offset]
    )
    assert (
        top_active_context["released_product"]["duplex_bottom_sequence"]
        == top_active_context["precursor"]["bottom_sequence"][:top_active_coordinate_offset]
    )
    assert top_active_context["released_product"]["duplex_mismatch_positions"] == []
    assert top_active_context["foldback"]["top_row"]["role"] == "active_stem"
    assert top_active_context["foldback"]["bottom_row"]["role"] == "foldback_return"
    assert (
        top_active_context["foldback"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert top_active_context["foldback"]["top_row"]["label"] == "Stem"
    assert top_active_context["foldback"]["bottom_row"]["label"] == "Foldback Stem"
    assert top_active_context["foldback"]["top_row"]["span"] == {
        "start": -top_active_coordinate_offset,
        "end": top_active_overhang_hit.target_search_hit.final_candidate.paired_bp,
    }
    assert top_active_context["foldback"]["bottom_row"]["span"] == {
        "start": -top_active_coordinate_offset,
        "end": top_active_overhang_hit.target_search_hit.final_candidate.paired_bp,
    }
    assert top_active_context["foldback"]["top_row"]["sequence"] == (
        top_active_context["precursor"]["top_sequence"][:top_active_coordinate_offset]
        + top_active_context["foldback"]["stem_sequence"]
    )
    assert (
        top_active_context["foldback"]["bottom_row"]["sequence"]
        == top_active_context["precursor"]["bottom_sequence"][:top_active_coordinate_offset]
        + top_active_context["foldback"]["foldback_partner_sequence"]
    )
    assert set(range(-top_active_coordinate_offset, 0)).issubset(
        set(top_active_context["foldback"]["top_row"]["assignable_base_positions"])
    )
    assert set(range(-top_active_coordinate_offset, 0)).issubset(
        set(top_active_context["foldback"]["bottom_row"]["assignable_base_positions"])
    )
    invalid_offset_hit = top_active_overhang_hit.target_search_hit.model_copy(
        update={
            "projection": top_active_overhang_hit.target_search_hit.projection.model_copy(
                update={
                    "rebased_nick_boundary": (
                        top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor + 1
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="nonnegative precursor nick offset"):
        build_released_hit_plot_context(invalid_offset_hit)

    rendered_top_active_path = workspace_root / "top_active_triptych.png"
    rendered_top_active_context = render_released_hit_plot(
        top_active_overhang_hit.target_search_hit, rendered_top_active_path
    )
    rendered_bytes = rendered_top_active_path.read_bytes()
    assert rendered_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", rendered_bytes[16:24])
    assert width > height
    assert rendered_top_active_context["released_product"]["top_row"]["role"] == "active_product"
    assert rendered_top_active_context["released_product"]["top_row"]["physical_state"] == "retained"
    assert (
        rendered_top_active_context["released_product"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert (
        rendered_top_active_context["released_product"]["bottom_row"]["sequence"]
        == rendered_top_active_context["precursor"]["bottom_sequence"][:top_active_coordinate_offset]
    )
    assert rendered_top_active_context["released_product"]["top_row"]["label"] == "Retained Top"
    assert rendered_top_active_context["released_product"]["bottom_row"]["physical_state"] == "released"
    assert rendered_top_active_context["released_product"]["bottom_row"]["label"] == "Exposed Bottom"
    assert rendered_top_active_context["released_product"]["bottom_row"]["span"] == {
        "start": (-top_active_coordinate_offset),
        "end": (
            top_active_overhang_hit.target_search_hit.projection.retained_partner_length_nt
            - top_active_coordinate_offset
        ),
    }
    assert rendered_top_active_context["released_product"]["top_row"]["sequence"] == (
        rendered_top_active_context["precursor"]["top_sequence"][:top_active_coordinate_offset]
        + top_active_overhang_hit.target_search_hit.projection.active_product_sequence
    )
    assert rendered_top_active_context["released_product"]["top_only_overhang_span"] == {
        "start": 0,
        "end": rendered_top_active_context["released_product"]["top_row"]["span"]["end"],
    }
    assert rendered_top_active_context["released_product"]["bottom_only_overhang_span"] is None
    assert rendered_top_active_context["released_product"]["duplex_overlap_span"] == {
        "start": -top_active_coordinate_offset,
        "end": 0,
    }
    assert rendered_top_active_context["foldback"]["top_row"]["label"] == "Stem"
    assert rendered_top_active_context["foldback"]["bottom_row"]["label"] == "Foldback Stem"
    assert (
        rendered_top_active_context["foldback"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert (
        rendered_top_active_context["foldback"]["top_row"]["sequence"]
        == rendered_top_active_context["precursor"]["top_sequence"][:top_active_coordinate_offset]
        + rendered_top_active_context["foldback"]["stem_sequence"]
    )
    assert (
        rendered_top_active_context["foldback"]["bottom_row"]["sequence"]
        == rendered_top_active_context["precursor"]["bottom_sequence"][:top_active_coordinate_offset]
        + rendered_top_active_context["foldback"]["foldback_partner_sequence"]
    )

    with (run_dir / "export" / "table__hits.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["final_geometry_source"] in {"exposed_bottom_strand", "retained_active_strand"}
    assert rows[0]["route_family"]
    assert rows[0]["active_strand"] in {"top", "bottom"}
    assert rows[0]["retained_partner_strand"] in {"top", "bottom"}
    assert rows[0]["physical_nicked_strand"] in {"top", "bottom"}
    assert rows[0]["active_product_input_length_nt"]
    assert rows[0]["active_product_length_nt"]
    assert rows[0]["retained_partner_length_nt"]
    assert rows[0]["upstream_retained_duplex_bp"]
    assert rows[0]["effective_stem_bp"]
