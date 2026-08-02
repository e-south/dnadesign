"""End-to-end tests for neutral three-way-junction QA rendering."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
import dnadesign.trijunction as trijunction
from dnadesign.baserender.src.core import RenderingError
from dnadesign.baserender.src.outputs.names import _unique_stem


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _payload() -> dict[str, object]:
    target = "AAAACCCCGGGGTTTTAAAACCCC"
    toehold = target[10:14]
    reverse_binding = _reverse_complement(target[-4:])
    return {
        "contract_kind": "three_way_junction_review_v1",
        "source": {
            "plan_schema": "dnadesign.trijunction.plan.v1",
            "plan_id": f"sha256:{'a' * 64}",
            "request_sha256": f"sha256:{'b' * 64}",
            "algorithm": "trijunction.v1",
        },
        "target": {
            "target_id": "target-01",
            "pool_id": "pool-01",
            "sequence_5to3": target,
            "sequence_sha256": f"sha256:{hashlib.sha256(target.encode()).hexdigest()}",
        },
        "geometry": {
            "fragments": [
                {
                    "fragment_id": "fragment-01",
                    "index": 0,
                    "role": "first",
                    "domain_span": {"start": 0, "end": 10},
                },
                {
                    "fragment_id": "fragment-02",
                    "index": 1,
                    "role": "last",
                    "domain_span": {"start": 14, "end": len(target)},
                },
            ],
            "junctions": [
                {
                    "junction_id": "junction-01",
                    "toehold_span": {"start": 10, "end": 14},
                    "left_fragment_id": "fragment-01",
                    "right_fragment_id": "fragment-02",
                    "toehold": toehold,
                    "toehold_complement": _reverse_complement(toehold),
                    "barcode": "AACCGGTT",
                    "barcode_complement": "AACCGGTT",
                    "complement_nick_geometry_valid": True,
                    "complement_end_preparation": "vendor_5_prime_phosphate",
                }
            ],
        },
        "strands": [
            {
                "fragment_id": "fragment-01",
                "role": "first",
                "incoming_junction_id": None,
                "outgoing_junction_id": "junction-01",
                "barcode_bearing_sequence_5to3": target[:14] + "AACCGGTT",
                "complement_sequence_5to3": _reverse_complement(target[:10]),
            },
            {
                "fragment_id": "fragment-02",
                "role": "last",
                "incoming_junction_id": "junction-01",
                "outgoing_junction_id": None,
                "barcode_bearing_sequence_5to3": _reverse_complement("AACCGGTT") + target[14:],
                "complement_sequence_5to3": _reverse_complement(target[14:]) + _reverse_complement(toehold),
            },
        ],
        "recovery": {
            "mode": "universal",
            "forward": {
                "direction": "forward",
                "binding_sequence_5to3": target[:4],
                "five_prime_extension_5to3": "",
                "order_sequence_5to3": target[:4],
                "target_binding_span": {"start": 0, "end": 4},
            },
            "reverse": {
                "direction": "reverse",
                "binding_sequence_5to3": reverse_binding,
                "five_prime_extension_5to3": "",
                "order_sequence_5to3": reverse_binding,
                "target_binding_span": {"start": 20, "end": 24},
            },
            "first_fragment_id": "fragment-01",
            "last_fragment_id": "fragment-02",
            "expected_product_sequence_5to3": target,
            "extended_top_sequence_5to3": target,
            "extended_bottom_sequence_5to3": _reverse_complement(target),
        },
        "search": {
            "pool_id": "pool-01",
            "toehold_seed": 11,
            "barcode_generation_seed": 12,
            "barcode_subset_seed": 13,
            "matching_seed": 14,
            "locus_count": 1,
            "toehold_paths_evaluated": 20,
            "toehold_min_distance": 4.0,
            "toehold_mean_distance": 4.0,
            "toehold_rank_score": 1.0,
            "barcode_candidates_generated": 25,
            "barcode_forbidden_toehold_k": 3,
            "barcode_forbidden_barcode_k": 4,
            "barcode_subsets_evaluated": 20,
            "barcode_min_distance": 6.0,
            "barcode_mean_distance": 6.0,
            "barcode_rank_score": 1.0,
            "matchings_evaluated": 1,
            "matching_max_pairwise_lcs": 2,
            "thermodynamic_screening": "not_run",
        },
        "checks": [
            {
                "subject": {"kind": "target", "id": "target-01"},
                "check": "exact_target_reconstruction",
                "status": "passed",
                "detail": "exact",
            },
            {
                "subject": {"kind": "pool", "id": "pool-01"},
                "check": "thermodynamic_screening",
                "status": "not_run",
                "detail": "not performed",
            },
        ],
    }


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
    finally:
        plt.close(figure)


def test_run_job_writes_a_separate_create_only_review_bundle(tmp_path: Path) -> None:
    source = tmp_path / "verified-source" / "target-01.review.json"
    source.parent.mkdir()
    source.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
    source_before = source.read_bytes()
    job = {
        "version": 4,
        "contract": {"kind": "three_way_junction_review_render_v1"},
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

    report = baserender.run_job(job, caller_root=tmp_path)

    output = Path(report.outputs["images_path"])
    assert output == (tmp_path / "review-render" / "three-way-junction-review.svg").resolve()
    assert output.exists()
    assert source.read_bytes() == source_before
    with pytest.raises(baserender.SchemaError, match="already exists"):
        baserender.run_job(job, caller_root=tmp_path)


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
