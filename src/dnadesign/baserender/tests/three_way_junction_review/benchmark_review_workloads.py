"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/benchmark_review_workloads.py

Measure deterministic Junction review rendering on representative plans.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import dnadesign.baserender as baserender
from dnadesign.junction import parse_request, plan
from dnadesign.junction.presentation import review_contracts
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _request_mapping(workload: str) -> dict[str, object]:
    if workload == "demo_gene_scale":
        path = _repository_root() / "src/dnadesign/junction/examples/gene-scale/request.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if workload == "single_1kb":
        return scale_request_mapping(
            target_count=1,
            target_length=1_000,
            topology="shared",
            nominal_fragment_oligo_length=132,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    if workload == "pool_3x360bp":
        return scale_request_mapping(
            target_count=3,
            target_length=360,
            topology="shared",
            nominal_fragment_oligo_length=132,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    raise ValueError(f"unknown workload: {workload}")


def _records(workload: str):
    request = parse_request(_request_mapping(workload))
    started = time.perf_counter()
    result = plan(request)
    plan_seconds = time.perf_counter() - started
    reviews = review_contracts(result)
    review_records = baserender.adapt_records(
        [review.model_dump(mode="json") for review in reviews],
        adapter_kind="three_way_junction_review_v1",
    )
    return result, review_records, plan_seconds


def _render_all(review_records) -> int:
    svg_bytes = 0
    for record in review_records:
        jobs = (
            ("junction_annealed_fragments", None),
            ("junction_three_way_assembly", {"view": "assembly"}),
            ("junction_three_way_assembly", {"view": "junction_detail"}),
        )
        for renderer, options in jobs:
            figure = baserender.render(record, renderer=renderer, options=options)
            try:
                buffer = io.BytesIO()
                figure.savefig(buffer, format="svg", metadata={"Date": None})
                svg_bytes += buffer.tell()
            finally:
                plt.close(figure)
    return svg_bytes


def _measure(operation: Callable[[], int], *, runs: int) -> dict[str, object]:
    operation()
    durations: list[float] = []
    peaks: list[int] = []
    svg_bytes: list[int] = []
    for _ in range(runs):
        tracemalloc.start()
        started = time.perf_counter()
        size = operation()
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        svg_bytes.append(size)
    return {
        "runs": runs,
        "seconds": {
            "minimum": min(durations),
            "median": statistics.median(durations),
            "maximum": max(durations),
        },
        "peak_traced_bytes": {
            "minimum": min(peaks),
            "median": int(statistics.median(peaks)),
            "maximum": max(peaks),
        },
        "svg_bytes": svg_bytes[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("demo_gene_scale", "single_1kb", "pool_3x360bp"),
        required=True,
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    result, review_records, plan_seconds = _records(args.workload)
    fragment_counts = [len(target.fragments) for target in result.targets]
    junction_counts = [len(target.junctions) for target in result.targets]
    report = {
        "workload": args.workload,
        "targets": len(review_records),
        "assembly_groups": len(result.assembly_groups),
        "target_lengths": [len(record.sequence) for record in review_records],
        "fragment_counts": fragment_counts,
        "junction_counts": junction_counts,
        "plan_seconds": plan_seconds,
        "render_and_svg": _measure(
            lambda: _render_all(review_records),
            runs=args.runs,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
