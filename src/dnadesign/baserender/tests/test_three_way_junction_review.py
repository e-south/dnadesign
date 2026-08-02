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

import dnadesign.baserender as baserender
import dnadesign.trijunction as trijunction
from dnadesign.baserender.src.config import ImagesOutputCfg, VideoOutputCfg
from dnadesign.baserender.src.core import RenderingError
from dnadesign.baserender.src.outputs.images import write_images
from dnadesign.baserender.src.outputs.names import _unique_stem
from dnadesign.baserender.src.outputs.video import write_video

from .three_way_junction_review.fixtures import _payload, _payload_with_many_junctions, _reverse_complement


def _trijunction_request() -> dict[str, object]:
    sequence = ("ACGATTCGGTACCTGATGCACTGA" * 3)[:72]
    return {
        "schema": "dnadesign.trijunction.request.v1",
        "seed": 17,
        "planning": {
            "oligo_length": 46,
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
                "pool_id": "pool-a",
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
            "max_oligo_length": 64,
        },
    }


def test_public_catalog_and_adapter_expose_the_review_contract() -> None:
    descriptor = baserender.get_adapter_descriptor("three_way_junction_review_v1")

    record = baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1")

    assert descriptor.owner_tool == "trijunction"
    assert descriptor.supported_renderers == ("three_way_junction_review",)
    assert descriptor.output_kinds == ("images",)
    assert descriptor.image_output_modes == ("directory",)
    assert descriptor.max_grid_records == 1
    assert baserender.get_renderer_descriptor("three_way_junction_review").max_grid_records == 1
    assert record.id == "target-01"
    assert record.meta["adapter"] == "three_way_junction_review_v1"
    assert record.meta["three_way_junction_review"]["search"]["thermodynamic_screening"] == "not_run"


def test_review_image_stems_are_collision_safe_on_case_insensitive_filesystems() -> None:
    used: set[str] = set()

    assert _unique_stem("Target-A", used) == "Target-A"
    assert _unique_stem("target-a", used) == "target-a_2"


def test_review_renderer_emits_one_semantic_four_panel_figure() -> None:
    record = baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1")

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        assert [axis.get_gid() for axis in figure.axes] == [
            "three-way-junction-review:target-geometry",
            "three-way-junction-review:junction-assignments",
            "three-way-junction-review:strands-and-recovery",
            "three-way-junction-review:search-and-checks",
        ]
        text = "\n".join(item.get_text() for axis in figure.axes for item in axis.texts)
        assert "target-01" in text
        assert "universal" in text
        assert "THERMODYNAMIC SCREENING NOT RUN" in text
        assert "1 junction" in text
        assert "FWD bind · 4 nt" in text
        assert "FWD 5′ ext · 0 nt" in text
        assert "Primer sequence previews · digest = SHA-256[:12]" in text
        assert "AAAA" not in text
        assert "AA…A" in text
    finally:
        plt.close(figure)


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
            "renderer": "three_way_junction_review",
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
        "render": {"renderer": "three_way_junction_review", "style": {}},
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
        "render": {"renderer": "three_way_junction_review", "style": {}},
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
        baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1"),
        baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1"),
    ]

    with pytest.raises(baserender.SchemaError, match="at most 1 record per grid"):
        baserender.render(records, renderer="three_way_junction_review")
    with pytest.raises(baserender.SchemaError, match="at most 1 record per grid"):
        baserender.render(records, renderer="sequence_rows")

    style = baserender.resolve_style(preset=None, overrides=None)
    output_root = tmp_path / "unpublished"
    output = ImagesOutputCfg(kind="images", dir=None, path=output_root / "combined.svg", fmt="svg")
    with pytest.raises(baserender.SchemaError, match="requires a directory for images output"):
        write_images(
            records,
            output=output,
            renderer_name="three_way_junction_review",
            style=style,
            palette=baserender.Palette(style.palette),
        )
    with pytest.raises(baserender.SchemaError, match="requires a directory for images output"):
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
            renderer_name="three_way_junction_review",
            style=style,
            palette=baserender.Palette(style.palette),
        )

    assert not output_root.exists()


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
            "renderer": "three_way_junction_review",
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


def test_run_job_consumes_the_verified_trijunction_review_array(tmp_path: Path) -> None:
    request = trijunction.parse_request(_trijunction_request())
    source_bundle = tmp_path / "verified-design"
    trijunction.build(request, destination=source_bundle)
    trijunction.verify(source_bundle)
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
            "renderer": "three_way_junction_review",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    report = baserender.run_job(job, caller_root=tmp_path)

    assert Path(report.outputs["images_dir"]) == (tmp_path / "review-render" / "images").resolve()
    assert (tmp_path / "review-render" / "images" / "target-a.svg").is_file()
    assert source.read_bytes() == source_before


def test_adapter_rejects_unvalidated_review_payloads() -> None:
    payload = _payload()
    payload["search"]["thermodynamic_screening"] = "passed"

    with pytest.raises(baserender.SchemaError, match="thermodynamic_screening"):
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_contradictory_thermodynamic_check_status() -> None:
    payload = _payload()
    payload["checks"][1]["status"] = "passed"

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_missing_thermodynamic_check() -> None:
    payload = _payload()
    payload["checks"].pop()

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


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
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_pool_receipt_smaller_than_target_geometry() -> None:
    payload = _payload_with_many_junctions(junction_count=2)
    payload["search"]["locus_count"] = 1

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_nonuniform_junction_sequence_lengths() -> None:
    payload = _payload_with_many_junctions(junction_count=2)
    payload["geometry"]["junctions"][1].update(
        {"barcode": "AACCGGT", "barcode_complement": _reverse_complement("AACCGGT")}
    )

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_matching_count_above_multi_locus_permutation_space() -> None:
    payload = _payload_with_many_junctions(junction_count=2)
    payload["search"]["matchings_evaluated"] = 3

    with pytest.raises(baserender.SchemaError, match="Invalid three_way_junction_review_v1 contract"):
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


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
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")


def test_review_renderer_rejects_contradictory_thermodynamic_check_status() -> None:
    record = baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1")
    record.meta["three_way_junction_review"]["checks"][1]["status"] = "passed"

    with pytest.raises(RenderingError, match="invalid review evidence"):
        baserender.render(record, renderer="three_way_junction_review")


def test_review_validation_errors_redact_raw_input_values() -> None:
    sentinel = "SENSITIVE-RAW-SEQUENCE-SENTINEL"
    payload = _payload()
    payload["target"]["sequence_5to3"] = sentinel

    with pytest.raises(baserender.SchemaError) as adapter_error:
        baserender.adapt_record(payload, adapter_kind="three_way_junction_review_v1")

    assert sentinel not in str(adapter_error.value)
    assert adapter_error.value.__cause__ is None
    record = baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1")
    record.meta["three_way_junction_review"]["target"]["sequence_5to3"] = sentinel
    with pytest.raises(RenderingError) as renderer_error:
        baserender.render(record, renderer="three_way_junction_review")

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
            "renderer": "three_way_junction_review",
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
            "renderer": "three_way_junction_review",
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
            "renderer": "three_way_junction_review",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }

    with pytest.raises(baserender.SchemaError, match="maximum.*selection rows"):
        baserender.run_job(job, caller_root=tmp_path)

    assert not (tmp_path / "review-render").exists()
