"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_three_way_junction_review.py

End-to-end tests for neutral three-way-junction QA rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import LineCollection

import dnadesign.baserender as baserender
import dnadesign.junction as junction
from dnadesign.baserender.src.config import ImagesOutputCfg, VideoOutputCfg
from dnadesign.baserender.src.core import RenderingError
from dnadesign.baserender.src.outputs.images import write_images
from dnadesign.baserender.src.outputs.names import _unique_stem
from dnadesign.baserender.src.outputs.video import write_video

from .three_way_junction_review.fixtures import (
    _adapt_payload,
    _payload,
    _payload_with_many_junctions,
    _reverse_complement,
)


def _junction_request() -> dict[str, object]:
    sequence = ("ACGATTCGGTACCTGATGCACTGA" * 3)[:72]
    return {
        "schema": "dnadesign.junction.request.v2",
        "seed": 17,
        "planning": {
            "nominal_fragment_oligo_length": 46,
            "barcode_length": 16,
            "toehold_length": 8,
            "search_range": 2,
            "toehold_search_iterations": 40,
            "barcode_pool_factor": 5,
            "barcode_generation_attempts": 100_000,
            "barcode_toehold_k": 4,
            "barcode_pair_k": 5,
            "barcode_subset_iterations": 40,
            "matching_iterations": 100,
            "barcode_gc_min": 0.25,
            "barcode_gc_max": 0.75,
            "barcode_max_homopolymer": 3,
        },
        "targets": [
            {
                "id": "target-a",
                "assembly_group_id": "assembly-a",
                "sequence": sequence,
                "recovery_primers": {
                    "mode": "target_specific",
                    "forward": {"binding_sequence": sequence[:8], "five_prime_extension": ""},
                    "reverse": {
                        "binding_sequence": _reverse_complement(sequence[-8:]),
                        "five_prime_extension": "",
                    },
                },
            }
        ],
        "order_policy": {
            "synthesis_scale": "declared-test-scale",
            "barcode_bearing_purification": "declared-test-purification",
            "complement_purification": "declared-test-purification",
            "primer_purification": "declared-test-purification",
            "complement_end_preparation": "vendor_5_prime_phosphate",
            "minimum_fragment_oligo_length": 1,
            "max_oligo_length": 64,
        },
    }


def _base_artists(axis, prefix: str):
    marker = f"{prefix}:base:"
    selected = [artist for artist in axis.texts if (artist.get_gid() or "").startswith(marker)]
    return sorted(selected, key=lambda artist: int((artist.get_gid() or "").split(marker, 1)[1].split(":", 1)[0]))


def _base_map(artists, *, coordinate: int) -> dict[float, str]:
    return {round(artist.get_position()[coordinate], 12): artist.get_text() for artist in artists}


def test_public_catalog_and_adapter_expose_the_review_contract() -> None:
    descriptor = baserender.get_adapter_descriptor("three_way_junction_review_v1")

    record = _adapt_payload(_payload())

    assert descriptor.owner_tool == "junction"
    assert descriptor.supported_renderers == ("junction_annealed_fragments", "junction_three_way_assembly")
    assert descriptor.output_kinds == ("images",)
    assert descriptor.image_output_modes == ("directory",)
    assert descriptor.max_grid_records == 1
    assert descriptor.validation_scope == "document"
    assert baserender.get_renderer_descriptor("junction_annealed_fragments").option_keys == ("fragment_ids",)
    assert baserender.get_renderer_descriptor("junction_three_way_assembly").option_keys == (
        "view",
        "junction_ids",
    )
    assert record.id == "target-01"
    assert record.meta["adapter"] == "three_way_junction_review_v1"
    assert record.meta["three_way_junction_review"]["search"]["thermodynamic_screening"] == "not_run"


def test_review_image_stems_are_collision_safe_on_case_insensitive_filesystems() -> None:
    used: set[str] = set()

    assert _unique_stem("Target-A", used) == "Target-A"
    assert _unique_stem("target-a", used) == "target-a_2"


def test_assembly_process_is_a_separate_target_scale_view() -> None:
    record = _adapt_payload(_payload())

    figure = baserender.render(record, renderer="junction_three_way_assembly")
    try:
        assert [axis.get_gid() for axis in figure.axes] == ["junction-three-way-assembly:assembly"]
        text = "\n".join(item.get_text() for axis in figure.axes for item in axis.texts)
        assert "target-01" in text
        assert "Oligo plan for target-01" in text
        assert "The oligos remain separate before annealing" not in text
        assert "Fragment oligos encode the target" in text
        assert "Annealing forms pre-ligation junctions" in text
        assert "PCR yields the expected linear duplex" in text
        assert "junction-01" in text
        assert "Expected sequence geometry" not in text
    finally:
        plt.close(figure)


def test_annealed_fragment_map_draws_every_declared_domain_pair() -> None:
    payload = _payload()
    figure = baserender.render(_adapt_payload(payload), renderer="junction_annealed_fragments")
    try:
        pair_collections = [item for item in figure.axes[0].collections if isinstance(item, LineCollection)]
        paired_fragment_bases = sum(
            fragment["domain_span"]["end"] - fragment["domain_span"]["start"]
            for fragment in payload["geometry"]["fragments"]
        )
        assert sum(len(item.get_segments()) for item in pair_collections) == paired_fragment_bases
        text = "\n".join(item.get_text() for item in figure.axes[0].texts)
        assert "2 fragment pairs show the expected annealing" in text
        assert "F01" in text
        assert "F02" in text
        assert "Expected sequence geometry" not in text
        top_steps: set[float] = set()
        for fragment in payload["geometry"]["fragments"]:
            top_bases = _base_artists(
                figure.axes[0],
                f"junction-annealed:{fragment['fragment_id']}:top",
            )
            top_steps.update(
                round(right.get_position()[0] - left.get_position()[0], 12)
                for left, right in zip(top_bases, top_bases[1:])
            )
        assert len(top_steps) == 1
        complement = str.maketrans("ACGT", "TGCA")
        for fragment in payload["geometry"]["fragments"]:
            fragment_id = fragment["fragment_id"]
            domain_start = fragment["domain_span"]["start"]
            domain_end = fragment["domain_span"]["end"]
            expected_domain = payload["target"]["sequence_5to3"][domain_start:domain_end]
            top = _base_map(
                _base_artists(figure.axes[0], f"junction-annealed:{fragment_id}:top"),
                coordinate=0,
            )
            bottom = _base_map(
                _base_artists(figure.axes[0], f"junction-annealed:{fragment_id}:bottom"),
                coordinate=0,
            )
            pairs = next(
                item for item in pair_collections if item.get_gid() == f"junction-annealed:{fragment_id}:watson-crick"
            )
            paired_top: list[str] = []
            paired_bottom: list[str] = []
            for segment in pairs.get_segments():
                top_x = round(segment[0][0], 12)
                bottom_x = round(segment[1][0], 12)
                assert top_x == bottom_x
                assert top_x in top and bottom_x in bottom
                assert top[top_x].translate(complement) == bottom[bottom_x]
                paired_top.append(top[top_x])
                paired_bottom.append(bottom[bottom_x])
            assert "".join(paired_top) == expected_domain
            assert "".join(paired_bottom) == expected_domain.translate(complement)
    finally:
        plt.close(figure)


def test_junction_detail_draws_one_shared_three_arm_node_and_nick() -> None:
    payload = _payload()
    junction = payload["geometry"]["junctions"][0]
    figure = baserender.render(
        _adapt_payload(payload),
        renderer="junction_three_way_assembly",
        options={"view": "junction_detail", "junction_ids": ["junction-01"]},
    )
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        gids = {artist.get_gid() for artist in (*axis.lines, *axis.collections) if artist.get_gid()}
        assert "junction:junction-01:left-and-barcode-arm" in gids
        assert "junction:junction-01:barcode-and-right-arm" in gids
        assert "junction:junction-01:nick" in gids
        assert "junction:junction-01:left-top-break:0" in gids
        assert "junction:junction-01:left-bottom-break:0" in gids
        assert "junction:junction-01:right-top-break:0" not in gids
        assert "junction:junction-01:right-bottom-break:0" not in gids
        barcode_pairs = next(
            collection
            for collection in axis.collections
            if collection.get_gid() == "junction:junction-01:barcode-pairs"
        )
        target_pairs = next(
            collection for collection in axis.collections if collection.get_gid() == "junction:junction-01:target-pairs"
        )
        assert all(segment[0][1] == segment[1][1] for segment in barcode_pairs.get_segments())
        assert all(segment[0][0] == segment[1][0] for segment in target_pairs.get_segments())
        text = "\n".join(item.get_text() for item in axis.texts)
        assert "junction-01 joins F01 to F02" in text
        assert "t1" in text and "t1*" in text
        assert "b1" in text and "b1*" in text
        assert sum(item.get_text() == "5′" for item in axis.texts) == 2
        assert sum(item.get_text() == "3′" for item in axis.texts) == 2
        left_barcode = _base_artists(axis, "junction:junction-01:barcode-b")
        right_barcode = _base_artists(axis, "junction:junction-01:barcode-b-star")
        assert "".join(item.get_text() for item in left_barcode) == junction["barcode"]
        assert "".join(item.get_text() for item in right_barcode) == junction["barcode_complement"]
        assert all(
            left.get_position()[1] < right.get_position()[1] for left, right in zip(left_barcode, left_barcode[1:])
        )
        assert all(
            left.get_position()[1] > right.get_position()[1] for left, right in zip(right_barcode, right_barcode[1:])
        )
        toehold_bases = _base_artists(axis, "junction:junction-01:toehold-top")
        horizontal_step = toehold_bases[1].get_position()[0] - toehold_bases[0].get_position()[0]
        vertical_step = left_barcode[1].get_position()[1] - left_barcode[0].get_position()[1]
        assert horizontal_step == pytest.approx(1.0)
        assert vertical_step == pytest.approx(horizontal_step)

        complement = str.maketrans("ACGT", "TGCA")
        top_target = _base_map(
            tuple(
                artist
                for role in ("left-top", "toehold-top", "right-top")
                for artist in _base_artists(axis, f"junction:junction-01:{role}")
            ),
            coordinate=0,
        )
        bottom_target = _base_map(
            tuple(
                artist
                for role in ("left-bottom", "toehold-bottom", "right-bottom")
                for artist in _base_artists(axis, f"junction:junction-01:{role}")
            ),
            coordinate=0,
        )
        for segment in target_pairs.get_segments():
            top_x = round(segment[0][0], 12)
            bottom_x = round(segment[1][0], 12)
            assert top_x == bottom_x
            assert top_x in top_target and bottom_x in bottom_target
            assert top_target[top_x].translate(complement) == bottom_target[bottom_x]

        left_by_y = _base_map(left_barcode, coordinate=1)
        right_by_y = _base_map(right_barcode, coordinate=1)
        for segment in barcode_pairs.get_segments():
            left_y = round(segment[0][1], 12)
            right_y = round(segment[1][1], 12)
            assert left_y == right_y
            assert left_y in left_by_y and right_y in right_by_y
            assert left_by_y[left_y].translate(complement) == right_by_y[right_y]
    finally:
        plt.close(figure)


def test_junction_detail_defaults_to_all_and_requires_a_subset_above_the_grid_limit() -> None:
    record = _adapt_payload(_payload_with_many_junctions(junction_count=12))
    figure = baserender.render(record, renderer="junction_three_way_assembly", options={"view": "junction_detail"})
    try:
        assert figure.texts[0].get_text() == "All 12 three-way junctions show the expected local annealing geometry"
    finally:
        plt.close(figure)

    oversized = _adapt_payload(_payload_with_many_junctions(junction_count=13))
    with pytest.raises(baserender.SchemaError, match="has 13 junctions.*must choose at most 12"):
        baserender.render(
            oversized,
            renderer="junction_three_way_assembly",
            options={"view": "junction_detail"},
        )


@pytest.mark.parametrize(
    "contract_kind",
    ["three_way_junction_review_render_v1", "render_job_v4"],
)
def test_review_jobs_require_per_target_image_directory(tmp_path: Path, contract_kind: str) -> None:
    source = tmp_path / "verified-source" / "target-01.review.json"
    source.parent.mkdir()
    source.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
    source_before = source.read_bytes()
    job = {
        "version": 4,
        "contract": {"kind": contract_kind},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": str(source.relative_to(tmp_path)),
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "junction_three_way_assembly",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "path": "three-way-junction-review.svg", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    with pytest.raises(baserender.SchemaError, match="requires a directory for images output"):
        baserender.run_job(job, caller_root=tmp_path)

    assert source.read_bytes() == source_before
    assert not (tmp_path / "review-render").exists()


def test_typed_review_job_cannot_bypass_image_directory_policy(tmp_path: Path) -> None:
    source = tmp_path / "verified-source" / "target-01.review.json"
    source.parent.mkdir()
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    mapping = {
        "version": 4,
        "contract": {"kind": "render_job_v4"},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": str(source.relative_to(tmp_path)),
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "render": {"renderer": "junction_three_way_assembly", "style": {}},
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }
    job = baserender.validate_job(mapping, caller_root=tmp_path)
    image_output = job.outputs[0]
    forged = replace(
        job,
        outputs=(
            replace(
                image_output,
                dir=None,
                path=job.bundle.path / "combined.svg",
            ),
        ),
    )

    with pytest.raises(baserender.SchemaError, match="requires a directory for images output"):
        baserender.run_job(forged)

    forged_video = replace(
        job,
        outputs=(
            VideoOutputCfg(
                kind="video",
                path=job.bundle.path / "review.mp4",
                fmt="mp4",
                fps=1,
                frames_per_record=1,
                pauses={},
                width_px=100,
                height_px=100,
                aspect_ratio=None,
                total_duration=None,
            ),
        ),
    )
    with pytest.raises(baserender.SchemaError, match="only supports output kinds: images"):
        baserender.run_job(forged_video)

    assert not job.bundle.path.exists()


def test_review_job_rejects_video_before_rendering(tmp_path: Path) -> None:
    source = tmp_path / "verified-source" / "target-01.review.json"
    source.parent.mkdir()
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    job = {
        "version": 4,
        "contract": {"kind": "render_job_v4"},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": str(source.relative_to(tmp_path)),
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "render": {"renderer": "junction_three_way_assembly", "style": {}},
        "outputs": [
            {
                "kind": "video",
                "path": "review.mp4",
                "width_px": 1_000_000,
                "height_px": 1_000_000,
                "frames_per_record": 2_147_483_647,
            }
        ],
        "run": {"strict": True, "fail_on_skips": True},
    }

    with pytest.raises(baserender.SchemaError, match="only supports output kinds: images"):
        baserender.validate_job(job, caller_root=tmp_path)

    assert not (tmp_path / "review-render").exists()


def test_review_renderer_rejects_multi_record_grids_on_public_and_writer_surfaces(tmp_path: Path) -> None:
    records = [
        _adapt_payload(_payload()),
        _adapt_payload(_payload()),
    ]

    with pytest.raises(baserender.SchemaError, match="at most 1 record per grid"):
        baserender.render(records, renderer="junction_three_way_assembly")
    with pytest.raises(baserender.RenderingError, match="not compatible with renderer 'sequence_rows'"):
        baserender.render(records, renderer="sequence_rows")

    style = baserender.resolve_style(preset=None, overrides=None)
    output_root = tmp_path / "unpublished"
    output = ImagesOutputCfg(kind="images", dir=None, path=output_root / "combined.svg", fmt="svg")
    with pytest.raises(baserender.SchemaError, match="requires a directory for images output"):
        write_images(
            records,
            output=output,
            renderer_name="junction_three_way_assembly",
            style=style,
            palette=baserender.Palette(style.palette),
        )
    with pytest.raises(baserender.RenderingError, match="not compatible with renderer 'sequence_rows'"):
        write_images(
            records,
            output=output,
            renderer_name="sequence_rows",
            style=style,
            palette=baserender.Palette(style.palette),
        )

    video_output = VideoOutputCfg(
        kind="video",
        path=output_root / "review.mp4",
        fmt="mp4",
        fps=1,
        frames_per_record=1,
        pauses={},
        width_px=100,
        height_px=100,
        aspect_ratio=None,
        total_duration=None,
    )
    with pytest.raises(baserender.SchemaError, match="only supports output kinds: images"):
        write_video(
            records[:1],
            output=video_output,
            renderer_name="junction_three_way_assembly",
            style=style,
            palette=baserender.Palette(style.palette),
        )

    assert not output_root.exists()


def test_svg_writer_omits_clock_derived_metadata(tmp_path: Path) -> None:
    record = _adapt_payload(_payload())
    style = baserender.resolve_style(preset=None, overrides=None)

    output_dir = write_images(
        [record],
        output=ImagesOutputCfg(kind="images", dir=tmp_path / "images", path=None, fmt="svg"),
        renderer_name="junction_three_way_assembly",
        style=style,
        palette=baserender.Palette(style.palette),
    )

    [output_path] = output_dir.glob("*.svg")
    svg = output_path.read_text(encoding="utf-8")
    assert "<dc:date>" not in svg
    assert not any(line.endswith((" ", "\t")) for line in svg.splitlines())

    second_dir = write_images(
        [record],
        output=ImagesOutputCfg(kind="images", dir=tmp_path / "second", path=None, fmt="svg"),
        renderer_name="junction_three_way_assembly",
        style=style,
        palette=baserender.Palette(style.palette),
    )
    [second_path] = second_dir.glob("*.svg")
    assert second_path.read_bytes() == output_path.read_bytes()


@pytest.mark.parametrize(
    "contract_kind",
    ["three_way_junction_review_render_v1", "render_job_v4"],
)
def test_run_job_publishes_review_bundle_recursively_owner_only(tmp_path: Path, contract_kind: str) -> None:
    source = tmp_path / "verified-source" / "target-01.review.json"
    source.parent.mkdir()
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    job = {
        "version": 4,
        "contract": {"kind": contract_kind},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": str(source.relative_to(tmp_path)),
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "junction_three_way_assembly",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    baserender.run_job(job, caller_root=tmp_path)

    bundle = tmp_path / "review-render"
    for path in (bundle, *bundle.rglob("*")):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected, path


def test_run_job_consumes_the_verified_junction_review_array(tmp_path: Path) -> None:
    request = junction.parse_request(_junction_request())
    source_bundle = tmp_path / "verified-design"
    junction.build(request, destination=source_bundle)
    junction.verify(source_bundle)
    source = source_bundle / "views" / "three_way_junction_review.v1.json"
    source_before = source.read_bytes()
    job = {
        "version": 4,
        "contract": {"kind": "three_way_junction_review_render_v1"},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": "verified-design/views/three_way_junction_review.v1.json",
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "junction_three_way_assembly",
            "options": {"view": "junction_detail", "junction_ids": ["target-a:junction-0001"]},
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    report = baserender.run_job(job, caller_root=tmp_path)

    assert Path(report.outputs["images_dir"]) == (tmp_path / "review-render" / "images").resolve()
    assert (tmp_path / "review-render" / "images" / "target-a.svg").is_file()
    manifest = json.loads((tmp_path / "review-render" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_spec"] == {
        "adapter_kind": "three_way_junction_review_v1",
        "alphabet": "DNA",
        "contract_kind": "three_way_junction_review_render_v1",
        "options": {"junction_ids": ["target-a:junction-0001"], "view": "junction_detail"},
        "renderer": "junction_three_way_assembly",
        "schema": "dnadesign.baserender.render_spec.v1",
        "style_sha256": manifest["render_spec"]["style_sha256"],
    }
    assert len(manifest["render_spec"]["style_sha256"]) == 64
    assert source.read_bytes() == source_before


def test_adapter_rejects_unvalidated_review_payloads() -> None:
    payload = _payload()
    payload["search"]["thermodynamic_screening"] = "passed"

    with pytest.raises(baserender.SchemaError, match="thermodynamic_screening"):
        _adapt_payload(payload)


def test_adapter_rejects_contradictory_thermodynamic_check_status() -> None:
    payload = _payload()
    payload["checks"][1]["status"] = "passed"

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


def test_adapter_rejects_missing_thermodynamic_check() -> None:
    payload = _payload()
    payload["checks"].pop()

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"toehold_min_distance": 10.0, "toehold_mean_distance": 1.0},
        {"barcode_rank_score": float("inf")},
    ],
)
def test_adapter_rejects_contradictory_or_nonfinite_search_metrics(updates: dict[str, float]) -> None:
    payload = _payload()
    payload["search"].update(updates)

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


def test_adapter_rejects_group_receipt_smaller_than_target_geometry() -> None:
    payload = _payload_with_many_junctions(junction_count=2)
    payload["search"]["locus_count"] = 1

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


def test_adapter_rejects_nonuniform_junction_sequence_lengths() -> None:
    payload = _payload_with_many_junctions(junction_count=2)
    payload["geometry"]["junctions"][1].update(
        {"barcode": "AACCGGT", "barcode_complement": _reverse_complement("AACCGGT")}
    )

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


def test_adapter_rejects_matching_count_above_multi_locus_permutation_space() -> None:
    payload = _payload_with_many_junctions(junction_count=2)
    payload["search"]["matchings_evaluated"] = 3

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"barcode_candidates_generated": 4},
        {"barcode_forbidden_toehold_k": 5},
        {"toehold_min_distance": 9.0, "toehold_mean_distance": 9.0},
        {"matching_max_pairwise_lcs": 13},
    ],
)
def test_adapter_rejects_impossible_search_evidence(updates: dict[str, float | int]) -> None:
    payload = _payload()
    payload["search"].update(updates)

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        _adapt_payload(payload)


def test_review_renderer_rejects_contradictory_thermodynamic_check_status() -> None:
    record = _adapt_payload(_payload())
    record.meta["three_way_junction_review"]["checks"][1]["status"] = "passed"

    with pytest.raises(RenderingError, match="invalid evidence"):
        baserender.render(record, renderer="junction_three_way_assembly")


def test_review_validation_errors_redact_raw_input_values() -> None:
    sentinel = "SENSITIVE-RAW-SEQUENCE-SENTINEL"
    payload = _payload()
    payload["target"]["sequence_5to3"] = sentinel

    with pytest.raises(baserender.SchemaError) as adapter_error:
        _adapt_payload(payload)

    assert sentinel not in str(adapter_error.value)
    assert adapter_error.value.__cause__ is None
    record = _adapt_payload(_payload())
    record.meta["three_way_junction_review"]["target"]["sequence_5to3"] = sentinel
    with pytest.raises(RenderingError) as renderer_error:
        baserender.render(record, renderer="junction_three_way_assembly")

    assert sentinel not in str(renderer_error.value)
    assert renderer_error.value.__cause__ is None


def test_review_job_rejects_oversized_input_before_capture_even_with_input_limit(tmp_path: Path) -> None:
    descriptor = baserender.get_render_contract_descriptor("three_way_junction_review_render_v1")
    assert descriptor.input_envelope is not None
    source = tmp_path / "oversized-review.json"
    with source.open("wb") as handle:
        handle.truncate(descriptor.input_envelope.max_bytes + 1)
    job = {
        "version": 4,
        "contract": {"kind": "three_way_junction_review_render_v1"},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": source.name,
            "limit": 1,
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "junction_three_way_assembly",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    with pytest.raises(baserender.SchemaError, match="maximum.*bytes"):
        baserender.run_job(job, caller_root=tmp_path)

    assert not (tmp_path / "review-render").exists()


def test_review_job_rejects_oversized_selection_before_rendering(tmp_path: Path) -> None:
    descriptor = baserender.get_render_contract_descriptor("three_way_junction_review_render_v1")
    assert descriptor.input_envelope is not None
    source = tmp_path / "target-01.review.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    selection = tmp_path / "oversized-selection.csv"
    with selection.open("wb") as handle:
        handle.write(b"id\n")
        handle.truncate(descriptor.input_envelope.max_bytes + 1)
    job = {
        "version": 4,
        "contract": {"kind": "three_way_junction_review_render_v1"},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": source.name,
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "selection": {"path": selection.name},
        "render": {
            "renderer": "junction_three_way_assembly",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    with pytest.raises(baserender.SchemaError, match="maximum.*bytes") as error:
        baserender.run_job(job, caller_root=tmp_path)

    assert str(selection) in str(error.value)
    assert not (tmp_path / "review-render").exists()


def test_review_job_rejects_selection_row_amplification_before_rendering(tmp_path: Path) -> None:
    descriptor = baserender.get_render_contract_descriptor("three_way_junction_review_render_v1")
    assert descriptor.input_envelope is not None
    source = tmp_path / "target-01.review.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    selection = tmp_path / "too-many-selection-rows.csv"
    selection.write_text(
        "id\n" + ("missing-target\n" * (descriptor.input_envelope.max_records + 1)),
        encoding="utf-8",
    )
    job = {
        "version": 4,
        "contract": {"kind": "three_way_junction_review_render_v1"},
        "bundle": {"path": "review-render"},
        "input": {
            "kind": "json",
            "path": source.name,
            "adapter": {"kind": "three_way_junction_review_v1"},
            "alphabet": "DNA",
        },
        "selection": {"path": selection.name},
        "render": {
            "renderer": "junction_three_way_assembly",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    with pytest.raises(baserender.SchemaError, match="maximum.*selection rows"):
        baserender.run_job(job, caller_root=tmp_path)

    assert not (tmp_path / "review-render").exists()
